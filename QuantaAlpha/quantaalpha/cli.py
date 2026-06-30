"""
QuantaAlpha CLI.

Commands:
  quantaalpha mine      - LLM agent factor mining (+ evolution when enabled in config)
  quantaalpha backtest  - backtest existing factors
  quantaalpha dry_run   - load private data and run research/backtest (no LLM)
"""

from pathlib import Path

from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parents[1]
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path, override=True)
else:
    load_dotenv(".env", override=True)

import fire


def dry_run():
    """Load a sample of private features and invoke research/backtest (no LLM)."""
    import tempfile
    from pathlib import Path

    import pandas as pd

    from quantaalpha.backtest import run_backtest
    from quantaalpha.data import PrivateDataConfig, load_feature_long

    cfg = PrivateDataConfig()
    cfg.validate_roots_exist()
    df = load_feature_long(["spread_mean"], dates=["20250102"])
    if df.empty:
        raise SystemExit("dry_run: no rows for spread_mean @ 20250102 — check data roots.")
    print(df.head())

    with tempfile.TemporaryDirectory() as td:
        tdc = Path(td) / "trade_date.csv"
        pd.DataFrame({"trade_date": ["20250102"]}).to_csv(tdc, index=False)
        bt = run_backtest(
            cfg.legacy_panel_root,
            factor_col="spread_mean",
            use_cache=True,
            trade_date_csv=tdc,
            workers=2,
        )
    row = bt.summary.loc[bt.summary["factor"] == "spread_mean"]
    print(row.to_string(index=False))


def app():
    def mine(*args, **kwargs):
        from quantaalpha.pipeline.factor_mining import main as _mine

        return _mine(*args, **kwargs)

    def backtest(*args, **kwargs):
        from quantaalpha.pipeline.factor_backtest import main as _backtest

        return _backtest(*args, **kwargs)

    fire.Fire(
        {
            "mine": mine,
            "backtest": backtest,
            "dry_run": dry_run,
        }
    )


if __name__ == "__main__":
    app()
