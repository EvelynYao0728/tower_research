"""Standard argparse CLI for single-factor backtest.

Examples
--------
# 默认因子目录 simple_factors + 默认 label（可不写 -f/-l）
python -m single_factor_bt.cli

# 只跑指定因子
python -m single_factor_bt.cli -c imbalance_mean

# 显式指定路径
python -m single_factor_bt.cli \
    -f /home/yzyao.25/research/data/simple_factors \
    -l /home/yzyao.25/research/data/label

# 启用 PIT universe 过滤（需提供 JSON）
python -m single_factor_bt.cli \
    --universe-file /path/to/2025_universe.json

# 自定义输出目录 / 关闭缓存 / 8 进程
python -m single_factor_bt.cli \
    -o /tmp/bt_out --no-cache -w 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import run_backtest
from .safety import PUBLIC_ROOT, assert_safe_output

_DEFAULT_RESEARCH = Path("/home/yzyao.25/research")
_DEFAULT_DATA = _DEFAULT_RESEARCH / "data"

DEFAULT_OUTPUT = _DEFAULT_RESEARCH / "backtest" / "output"
DEFAULT_FEATURE = _DEFAULT_DATA / "simple_factors"
DEFAULT_LABEL = _DEFAULT_DATA / "label"
DEFAULT_TRADE_DATE_CSV = _DEFAULT_DATA / "trade_date.csv"
DEFAULT_UNIVERSE = _DEFAULT_DATA / "2025_universe.json"


HELP_EPILOG = """\
说明
----
因子与 label 默认读自 /home/yzyao.25/research/data/，结果输出到
/home/yzyao.25/research/backtest/output/<factor_name>/，互不覆盖。
禁止将输出目录设在本机只读数据树 research/public/ 下。

默认行为（相对旧版 backtest 的修正）
------------------------------------
* RankIC：因子与 label 在同一有效样本上 rank 后再相关（标准 Spearman）。
* 分层：decile_1 = 因子最低组，decile_10 = 因子最高组；long=D10，short=D1。
* 年化：按 [date_min, date_max] 日历跨度计算 obs_per_year，不再硬编码 9828。
* PIT universe：默认关闭（--no-universe）；有 universe JSON 时可显式开启。

输出文件
--------
<output>/summary.csv                              所有因子绩效汇总 (累积合并)
<output>/<factor>/metrics_per_minute.csv          每分钟横截面指标 + 各 decile 收益
<output>/<factor>/decile_inner_long_short.csv     每组组内多空收益
<output>/<factor>/decile_inner_long_short.png     组内多空累计曲线
<output>/<factor>/long_short_overall.png          整体 多 / 空 / 多空 累计收益
<output>/<factor>/intraday_ic_profile.csv         日内 IC / RankIC 跨日均值
<output>/<factor>/intraday_ic_profile.png         日内 IC 形态图
<output>/<factor>/_cache/<date>.parquet           每天分片缓存

summary.csv 列（每因子一行）
---------------------------
IC, ICIR, RankIC, RankICIR, long_ret, short_ret, long_short_ret,
volatility, sharpe, turnover
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="single-factor-bt",
        description="单因子横截面回测 — IC / ICIR / RankIC / RankICIR / 多空收益 "
                    "(parquet+feather, 长/宽表自动识别, 多进程加速).",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g_io = p.add_argument_group("数据输入 / 输出")
    g_io.add_argument(
        "-f", "--feature",
        type=Path, default=DEFAULT_FEATURE, metavar="PATH",
        help="因子文件或目录 (parquet/feather/.fea)。"
             f"默认 {DEFAULT_FEATURE}。",
    )
    g_io.add_argument(
        "-l", "--label",
        type=Path, default=DEFAULT_LABEL, metavar="PATH",
        help=f"label 文件或目录。默认 {DEFAULT_LABEL}",
    )
    g_io.add_argument(
        "-o", "--output",
        type=Path, default=DEFAULT_OUTPUT, metavar="DIR",
        help=f"输出根目录。默认 {DEFAULT_OUTPUT}。不得落在 {PUBLIC_ROOT} 下。",
    )

    g_col = p.add_argument_group("列名选择")
    g_col.add_argument(
        "-c", "--factor-col", default=None, metavar="COL",
        help="因子列名。**不指定时回测 feature 文件中所有可用因子列**。",
    )
    g_col.add_argument(
        "--label-col", default=None, metavar="COL",
        help="收益率列名。不传时自动探测："
             "ex_log_ret_10m → log_ret_10m → ex_ret_10m → ret_10m。",
    )

    g_calc = p.add_argument_group("计算参数")
    g_calc.add_argument(
        "-n", "--n-groups", type=int, default=10, metavar="K",
        help="按因子升序等分的组数 (默认 10)。多头=D10，空头=D1。",
    )
    g_calc.add_argument(
        "--inner-q", type=float, default=0.20, metavar="Q",
        help="每个组内做多 top Q / 做空 bottom Q 的比例 (默认 0.20)。",
    )
    g_calc.add_argument(
        "--sample-every-n-minutes", type=int, default=1, metavar="M",
        help="采样步长，每 M 分钟做一次横截面 (默认 1)。",
    )
    g_calc.add_argument(
        "--session-start", type=int, default=930, metavar="HHMM",
        help="交易时段开始分钟数 (默认 930 = 09:30)。",
    )
    g_calc.add_argument(
        "--session-end", type=int, default=1550, metavar="HHMM",
        help="交易时段结束分钟数 (默认 1550 = 15:50)。",
    )

    g_run = p.add_argument_group("运行控制")
    g_run.add_argument(
        "-w", "--workers", type=int, default=None, metavar="N",
        help="进程池大小，默认 min(8, ncpu-1)。",
    )
    g_run.add_argument(
        "--no-cache", action="store_true",
        help="关闭按天缓存 (output/<factor>/_cache/)。",
    )
    g_run.add_argument(
        "--trade-date-csv", type=Path, default=DEFAULT_TRADE_DATE_CSV, metavar="FILE",
        help=f"trade_date.csv，限定回测日期；默认 {DEFAULT_TRADE_DATE_CSV}，不存在则忽略。",
    )
    g_run.add_argument(
        "--universe-file", type=Path, default=DEFAULT_UNIVERSE, metavar="JSON",
        help="可选 PIT membership JSON；与 --no-universe 互斥。",
    )
    g_run.add_argument(
        "--no-universe", action="store_true",
        help="关闭 PIT 过滤（默认行为，与旧回测一致）。",
    )
    g_run.add_argument(
        "-v", "--verbose", action="store_true",
        help="打印更详细的执行信息。",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    out_root = assert_safe_output(args.output)
    universe_file = None
    if not args.no_universe and args.universe_file is not None:
        up = Path(args.universe_file)
        if up.is_file():
            universe_file = up
        elif args.verbose:
            print(f"[cfg] universe file not found, skipping PIT filter: {up}")

    if args.verbose:
        print(f"[cfg] feature : {args.feature}")
        print(f"[cfg] label   : {args.label}")
        print(f"[cfg] output  : {out_root}")
        print(f"[cfg] universe: {universe_file if universe_file else '(disabled)'}")
        print(f"[cfg] factor-col={args.factor_col}  label-col={args.label_col}")
        print(f"[cfg] n_groups={args.n_groups}  inner_q={args.inner_q}")
        print(f"[cfg] session=[{args.session_start},{args.session_end}] "
              f"every={args.sample_every_n_minutes}m  workers={args.workers}")

    trade_csv = args.trade_date_csv
    if trade_csv is not None and not Path(trade_csv).exists():
        trade_csv = None

    summary, out_dir = run_backtest(
        factor_path=args.feature,
        label_path=args.label,
        output_dir=out_root,
        factor_col=args.factor_col,
        label_col=args.label_col,
        n_groups=args.n_groups,
        inner_q=args.inner_q,
        sample_every_n_minutes=args.sample_every_n_minutes,
        session_start=args.session_start,
        session_end=args.session_end,
        workers=args.workers,
        use_cache=not args.no_cache,
        trade_date_csv=trade_csv,
        universe_file=universe_file,
    )

    print("\n========== ALL FACTORS SUMMARY ==========")
    from .metrics import CORE_SUMMARY_METRIC_COLUMNS

    cols_show = ["factor", *CORE_SUMMARY_METRIC_COLUMNS]
    cols_show = [c for c in cols_show if c in summary.columns]
    print(summary[cols_show].to_string(index=False))
    print(f"\n[ok] full summary  : {out_dir / 'summary.csv'}")
    print(f"[ok] per-factor dirs under: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
