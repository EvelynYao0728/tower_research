from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from filelock import FileLock

from quantaalpha.coder.costeer.task import CoSTEERTask
from quantaalpha.factors.coder.config import FACTOR_COSTEER_SETTINGS
from quantaalpha.core.exception import CodeFormatError, CustomRuntimeError, NoOutputError
from quantaalpha.core.experiment import Experiment, FBWorkspace
from quantaalpha.core.utils import cache_with_pickle
from quantaalpha.llm.client import md5_hash


def _importable_project_root() -> Path:
    """
    Directory that must be on PYTHONPATH so ``import quantaalpha`` works in a subprocess.

    Works for editable installs (…/QuantaAlpha/quantaalpha/…) and for ``site-packages``.
    """
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "quantaalpha" / "__init__.py").is_file():
            return p
    return here.parents[3]


def _python_executable() -> str:
    pb = (FACTOR_COSTEER_SETTINGS.python_bin or "").strip()
    if pb in ("", "python"):
        return sys.executable
    return pb


def _factor_py_uses_catalog_eval(code: str) -> bool:
    return "factor_eval" in code or "calculate_factor_to_parquet" in code


class FactorTask(CoSTEERTask):
    # TODO:  generalized the attributes into the Task
    # - factor_* -> *
    def __init__(
        self,
        factor_name,
        factor_description,
        factor_formulation,
        factor_expression = None,
        *args,
        variables: dict = {},
        resource: str = None,
        factor_implementation: bool = False,
        **kwargs,
    ) -> None:
        self.factor_name = (
            factor_name  # TODO: remove it in the later version. Keep it only for pickle version compatibility
        )
        self.factor_description = factor_description
        self.factor_formulation = factor_formulation
        self.factor_expression = factor_expression
        self.variables = variables
        self.factor_resources = resource
        self.factor_implementation = factor_implementation
        super().__init__(name=factor_name, *args, **kwargs)

    def get_task_information(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
variables: {str(self.variables)}"""
    

    def get_task_description(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}"""

    def get_task_information_and_implementation_result(self):
        return {
            "factor_name": self.factor_name,
            "factor_description": self.factor_description,
            "factor_formulation": self.factor_formulation,
            "factor_expression": self.factor_expression,
            "variables": str(self.variables),
            "factor_implementation": str(self.factor_implementation),
        }

    @staticmethod
    def from_dict(dict):
        return FactorTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.factor_name}]>"


def _resolve_expr_and_name(task: FactorTask, code: str) -> tuple[str | None, str | None]:
    expr = (getattr(task, "factor_expression", None) or "").strip()
    name = (getattr(task, "factor_name", None) or "").strip()
    if not expr:
        m = re.search(r'expr\s*=\s*["\'](.+?)["\']', code, re.DOTALL)
        if m:
            expr = m.group(1).strip()
    if not name:
        m = re.search(r'name\s*=\s*["\']([^"\']+)["\']', code)
        if m:
            name = m.group(1).strip()
    return expr or None, name or None


class FactorFBWorkspace(FBWorkspace):
    """
    This class is used to implement a factor by writing the code to a file.
    Input data and output factor value are also written to files.
    """

    # TODO: (Xiao) think raising errors may get better information for processing
    FB_EXEC_SUCCESS = "Execution succeeded without error."
    FB_CODE_NOT_SET = (
        "code is not set: code_dict 中缺少 factor.py，尚未有可执行实现。"
        "请先完成 factor_construct / factor_calculate（或由 LLM 写入 factor.py）再运行 PrivateFactorRunner。"
    )
    FB_EXECUTION_SUCCEEDED = "Execution succeeded without error."
    FB_OUTPUT_FILE_NOT_FOUND = "\nExpected output file not found."
    FB_OUTPUT_FILE_FOUND = "\nExpected output file found."

    def __init__(
        self,
        *args,
        raise_exception: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.raise_exception = raise_exception

    def hydrate_code_dict_from_disk(self) -> bool:
        """若磁盘上已有 ``factor.py`` 但内存 ``code_dict`` 丢失（pickle/多进程/会话），则回填。"""
        if self.code_dict is None:
            self.code_dict = {}
        if "factor.py" in self.code_dict:
            return False
        p = self.workspace_path / "factor.py"
        if not p.is_file():
            return False
        try:
            self.code_dict["factor.py"] = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
        return True

    def clear_execution_artifacts_only(self) -> None:
        """删除一次执行产物，保留 ``factor.py`` 与 ``code_dict``，供下游 PrivateFactorRunner 再跑。"""
        wp = self.workspace_path
        if not wp.is_dir():
            return
        for name in ("factor_output.parquet", "execution_stdout.txt", "execution.lock"):
            p = wp / name
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                pass
        cache_dir = wp / "_cache"
        try:
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir, ignore_errors=True)
        except OSError:
            pass

    def hash_func(self, data_type: str = "Debug") -> str:
        self.hydrate_code_dict_from_disk()
        return (
            md5_hash(data_type + self.code_dict["factor.py"])
            if ("factor.py" in self.code_dict and not self.raise_exception)
            else None
        )

    @cache_with_pickle(hash_func)
    def execute(self, data_type: str = "Debug") -> Tuple[str, pd.DataFrame]:
        """
        execute the implementation and get the factor value by the following steps:
        1. make the directory in workspace path
        2. write the code to the file in the workspace path
        3. link all the source data to the workspace path folder
        if call_factor_py is True:
            4. execute the code
        else:
            4. generate a script from template to import factor.py (legacy path)
        5. read factor values from factor_output.parquet in the workspace
        returns the execution feedback as a string and the factor value as a pandas dataframe


        Regarding the cache mechanism:
        1. We will store the function's return value to ensure it behaves as expected.
        - The cached information will include a tuple with the following: (execution_feedback, executed_factor_value_dataframe, Optional[Exception])

        """
        self.hydrate_code_dict_from_disk()
        super().execute()
        if self.code_dict is None or "factor.py" not in self.code_dict:
            if self.raise_exception:
                raise CodeFormatError(self.FB_CODE_NOT_SET)
            return self.FB_CODE_NOT_SET, None
        # 相对 workspace_path + cwd=相对路径 + argv=相对脚本时，Python 会按 cwd 解析脚本路径，易拼成错误路径找不到 factor.py
        ws_root = Path(self.workspace_path).expanduser().resolve(strict=False)
        if data_type == "Debug":
            try:
                (ws_root / ".full_panel_promoted").unlink(missing_ok=True)
            except OSError:
                pass
        with FileLock(ws_root / "execution.lock"):
            # Set data path for all versions
            source_data_path = (
                Path(
                    FACTOR_COSTEER_SETTINGS.data_folder_debug,
                )
                if data_type == "Debug"  # FIXME: (yx) don't think we should use a debug tag for this.
                else Path(
                    FACTOR_COSTEER_SETTINGS.data_folder,
                )
            )

            # Use absolute path
            if not source_data_path.is_absolute():
                source_data_path = ws_root.parent.parent.parent / source_data_path
            else:
                source_data_path = Path(source_data_path).absolute()

            source_data_path.mkdir(exist_ok=True, parents=True)
            code_path = ws_root / "factor.py"

            code_text = self.code_dict.get("factor.py", "")
            uses_catalog = _factor_py_uses_catalog_eval(code_text)
            skip_link = FACTOR_COSTEER_SETTINGS.skip_workspace_data_symlink and uses_catalog
            if not skip_link:
                if source_data_path.exists() and any(source_data_path.iterdir()):
                    self.link_all_files_in_folder_to_workspace(source_data_path, ws_root)
                else:
                    from quantaalpha.log import logger

                    logger.warning(
                        f"Data folder {source_data_path} does not exist or is empty. Skipping linking."
                    )

            execution_feedback = self.FB_EXECUTION_SUCCEEDED
            execution_success = False
            execution_error = None

            if self.target_task.version == 1:
                execution_code_path = code_path
            elif self.target_task.version == 2:
                execution_code_path = ws_root / f"{uuid.uuid4()}.py"
                execution_code_path.write_text((Path(__file__).parent / "factor_execution_template.txt").read_text())

            raw_out = b""
            inprocess_ok = False
            try:
                import os

                parq_path = ws_root / "factor_output.parquet"
                if (
                    FACTOR_COSTEER_SETTINGS.inprocess_factor_eval
                    and uses_catalog
                    and hasattr(self, "target_task")
                ):
                    expr, fname = _resolve_expr_and_name(self.target_task, code_text)
                    if expr and fname:
                        from quantaalpha.factors.coder.factor_eval import (
                            calculate_factor_to_parquet,
                            resolve_factor_eval_workers,
                        )

                        workers = resolve_factor_eval_workers()
                        if data_type == "Debug":
                            panel_n = int(FACTOR_COSTEER_SETTINGS.debug_panel_max_days or 0)
                            panel_arg = panel_n if panel_n > 0 else None
                        else:
                            panel_arg = 0
                        calculate_factor_to_parquet(
                            expr,
                            fname,
                            parq_path,
                            workers=workers,
                            panel_max_days=panel_arg,
                        )
                        raw_out = (
                            f"[inprocess factor_eval] rows written -> {parq_path.name}\n".encode()
                        )
                        execution_success = True
                        inprocess_ok = True

                if not inprocess_ok:
                    env = os.environ.copy()
                    project_root = _importable_project_root()
                    pythonpath = str(project_root)
                    if env.get("PYTHONPATH"):
                        env["PYTHONPATH"] = pythonpath + os.pathsep + env["PYTHONPATH"]
                    else:
                        env["PYTHONPATH"] = pythonpath
                    from quantaalpha.factors.coder.factor_eval import (
                        resolve_factor_eval_workers,
                    )

                    env["QUANTALPHA_FACTOR_EVAL_WORKERS"] = str(
                        resolve_factor_eval_workers()
                    )
                    if data_type == "Debug":
                        dbg = int(FACTOR_COSTEER_SETTINGS.debug_panel_max_days or 0)
                        if dbg > 0:
                            env["QUANTALPHA_TEMPLATE_PANEL_MAX_DAYS"] = str(dbg)
                    else:
                        env["QUANTALPHA_TEMPLATE_PANEL_MAX_DAYS"] = "0"

                    py_exe = _python_executable()
                    exec_script = execution_code_path.resolve(strict=False)
                    raw_out = subprocess.check_output(
                        [py_exe, str(exec_script)],
                        cwd=str(ws_root),
                        stderr=subprocess.STDOUT,
                        timeout=FACTOR_COSTEER_SETTINGS.file_based_execution_timeout,
                        env=env,
                    )
                    execution_success = True
            except subprocess.CalledProcessError as e:
                import site

                raw_out = e.output if e.output else b""
                sp0 = ""
                try:
                    sp0 = str(site.getsitepackages()[0])
                except (IndexError, OSError):
                    pass
                ws_tag = str(ws_root)
                execution_feedback = raw_out.decode(errors="replace").replace(ws_tag, "/path/to/workspace")
                if sp0:
                    execution_feedback = execution_feedback.replace(sp0, r"/path/to/site-packages")
                if len(execution_feedback) > 2000:
                    execution_feedback = (
                        execution_feedback[:1000] + "....hidden long error message...." + execution_feedback[-1000:]
                    )
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback)
                else:
                    execution_error = CustomRuntimeError(execution_feedback)
            except subprocess.TimeoutExpired as e:
                raw_out = e.output if getattr(e, "output", None) else b""
                execution_feedback += f"Execution timeout error and the timeout is set to {FACTOR_COSTEER_SETTINGS.file_based_execution_timeout} seconds."
                if self.raise_exception:
                    raise CustomRuntimeError(execution_feedback)
                else:
                    execution_error = CustomRuntimeError(execution_feedback)
            finally:
                try:
                    (ws_root / "execution_stdout.txt").write_bytes(raw_out)
                except OSError:
                    pass

            executed_factor_value_dataframe = None
            parq_path = ws_root / "factor_output.parquet"

            if execution_success:
                if parq_path.exists():
                    try:
                        executed_factor_value_dataframe = pd.read_parquet(parq_path)
                        execution_feedback += self.FB_OUTPUT_FILE_FOUND + " (factor_output.parquet)"
                    except Exception as e:
                        execution_feedback += f"Error found when reading parquet file: {e}"[:1000]
                    if executed_factor_value_dataframe is not None:
                        self._last_execute_df = executed_factor_value_dataframe

                if executed_factor_value_dataframe is None:
                    execution_feedback += self.FB_OUTPUT_FILE_NOT_FOUND
                    if self.raise_exception:
                        raise NoOutputError(execution_feedback)
                    execution_error = execution_error or NoOutputError(execution_feedback)
            elif not parq_path.exists():
                execution_feedback += self.FB_OUTPUT_FILE_NOT_FOUND
                if self.raise_exception:
                    raise NoOutputError(execution_feedback)
                execution_error = execution_error or NoOutputError(execution_feedback)

        return execution_feedback, executed_factor_value_dataframe

    def __str__(self) -> str:
        # NOTE:
        # If the code cache works, the workspace will be None.
        return f"File Factor[{self.target_task.factor_name}]: {self.workspace_path}"

    def __repr__(self) -> str:
        return self.__str__()

    @staticmethod
    def from_folder(task: FactorTask, path: Union[str, Path], **kwargs):
        path = Path(path)
        code_dict = {}
        for file_path in path.iterdir():
            if file_path.suffix == ".py":
                code_dict[file_path.name] = file_path.read_text()
        return FactorFBWorkspace(target_task=task, code_dict=code_dict, **kwargs)


FactorExperiment = Experiment
FeatureExperiment = Experiment
