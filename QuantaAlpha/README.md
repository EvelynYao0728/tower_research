# QuantaAlpha

LLM-agent pipeline for automated factor mining and iteration on **private minute parquet** features, evaluated via `research/backtest`.

## Project layout

```
QuantaAlpha/
├── configs/                 # experiment YAML + .env.example
├── data/
│   ├── factorlib/           # mined factors (JSON)
│   └── results/             # per-run workspaces（本地运行时产物）
├── quantaalpha/
│   ├── cli.py               # entry: mine | backtest | dry_run
│   ├── pipeline/            # orchestration, planning, evolution
│   ├── factors/             # hypothesis, coder, runner, feedback, library
│   ├── coder/costeer/       # CoSTEER code-evolution framework
│   ├── data/                # private parquet catalog + loaders
│   ├── backtest/            # bridge to research/backtest
│   ├── llm/                 # API client + offline stub
│   ├── core/                # loop / scenario / experiment abstractions
│   ├── components/          # shared proposal & runner bases
│   └── utils/               # workflow session, templates
├── run.sh
└── pyproject.toml
```

## Quick start

```bash
cp configs/.env.example .env
# Edit QUANTALPHA_LEGACY_PANEL_ROOT / QUANTALPHA_PER_FEATURE_ROOT and LLM keys

pip install -e .
quantaalpha dry_run          # data + backtest only
./run.sh "价量因子挖掘"
```

### 本地因子 + 回测（不消耗 LLM token）

与模板 ``factor.py`` 相同 eval 语义，直接读 ``data/simple_factors`` 与 ``data/0511simple_factors``：

```bash
# 默认：按日并行 + 快路径（推荐，约 1–3 分钟写满 250 天）
python -m quantaalpha.backtest.local_factor_runner --skip-backtest -j 8

# 写完后只回测
python -m quantaalpha.backtest.local_factor_runner --only-backtest

# 其它表达式：按日 eval（-j 并行），语义同模板 factor.py
python -m quantaalpha.backtest.local_factor_runner --expr "..." --name MyFactor -j 8

# 极慢对照：全年整表 eval
python -m quantaalpha.backtest.local_factor_runner --bulk
```

实现见 ``quantaalpha/backtest/local_factor_runner.py``。

Offline smoke (no LLM API):

```bash
export QUANTALPHA_LLM_STUB=1
CONFIG=configs/experiment_venv_smoke.yaml ./run.sh "冒烟测试"
```

## Data paths

| Variable | Purpose |
|----------|---------|
| `QUANTALPHA_LEGACY_PANEL_ROOT` | daily parquet panels (`date=YYYYMMDD/*.parquet`) |
| `QUANTALPHA_PER_FEATURE_ROOT` | per-feature subdirs |
| `QUANTALPHA_RESEARCH_ROOT` | repo root for `research/backtest` (default: parent `research/`) |

## Agent loop

`mine` runs: **hypothesis → expression → CoSTEER code → private backtest → feedback** (×N loops). With `evolution.enabled: true` in `configs/experiment.yaml`, adds mutation/crossover across trajectories.
