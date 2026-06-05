# refute-inspect

Inspect AI adapter for **[REFUTE](https://huggingface.co/datasets/BGPT-OFFICIAL/refute)** — judge-free tasks for scientific critique and epistemic calibration on recent science paper summaries.

## Tasks

| Task | Description |
|------|-------------|
| `refute_forced_choice` | Pick the more flawed of twin summaries (contamination-proof, chance 50%) |
| `refute_soundness` | Binary sound/flawed classification |

## Install

```bash
pip install inspect-ai datasets
git clone https://github.com/connerlambden/refute-inspect.git
cd refute-inspect && pip install -e .
```

## Run

```bash
inspect eval src/refute_inspect/refute_inspect.py@refute_forced_choice --model openai/gpt-4o
inspect eval src/refute_inspect/refute_inspect.py@refute_soundness --model openai/gpt-4o
```

Dataset loads from Hugging Face `BGPT-OFFICIAL/refute` (config `refute_soundness`, revision pinned at runtime).

## Links

- Dataset: https://huggingface.co/datasets/BGPT-OFFICIAL/refute
- Technical report: https://huggingface.co/datasets/BGPT-OFFICIAL/refute/blob/main/TECHNICAL_REPORT.md
- Leaderboard: https://huggingface.co/spaces/BGPT-OFFICIAL/refute-leaderboard
