"""Inspect AI adapter for the REFUTE benchmark (judge-free core tasks).

REFUTE (Reasoning Over Evidence — Falsification, Uncertainty, Truth-grounding &
Epistemics) evaluates whether language models critique recent science papers with
calibrated, evidence-grounded judgment.

This adapter exposes the two *objective, judge-free* REFUTE tasks so any model can be
scored with a single command and no LLM judge:

  * refute_forced_choice  — show the SOUND and FLAWED twin of the same study and ask
                            which is more flawed. Contamination-proof (the flawed twin
                            only has to be worse than its OWN sound twin), per-vignette
                            difficulty cancels, chance = 50%.
  * refute_soundness      — binary sound/flawed classification over single summaries.

The generative critique tasks (refute_hard_60 / refute_120) are rubric-scored and need
an LLM judge; they are intentionally not included here so this adapter stays fully
objective and reproducible.

Usage
-----
    pip install inspect-ai datasets
    # set your provider key, e.g. export OPENAI_API_KEY=...
    inspect eval refute_inspect.py@refute_forced_choice --model openai/gpt-4o
    inspect eval refute_inspect.py@refute_soundness     --model openai/gpt-4o

By default the dataset is pulled from the Hugging Face Hub
(BGPT-OFFICIAL/refute, config refute_soundness, split train). Override with the
REFUTE_SOUNDNESS_JSONL environment variable to point at a local JSONL file.

Dataset: https://huggingface.co/datasets/BGPT-OFFICIAL/refute
Leaderboard: https://huggingface.co/spaces/BGPT-OFFICIAL/refute-leaderboard
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message

HF_DATASET = "BGPT-OFFICIAL/refute"
HF_CONFIG = "refute_soundness"

FORCED_CHOICE_SYSTEM = (
    "You are a rigorous methodological reviewer. You will see two study summaries, A and B, "
    "on the same topic. Exactly ONE has a more serious methodological flaw that undermines its "
    "central conclusion. Decide which. Respond on one line in EXACTLY this format: "
    "'ANSWER=<A or B>' then at most one short clause naming the flaw."
)

SOUNDNESS_SYSTEM = (
    "You are a rigorous methodological reviewer. You will see one study summary. Decide whether its "
    "central conclusion is methodologically SOUND or FLAWED given what is described. Respond on one "
    "line in EXACTLY this format: 'LABEL=<sound or flawed>' then at most one short clause naming the "
    "flaw if any."
)

_ANSWER_RE = re.compile(r"ANSWER\s*=\s*([AB])", re.I)
_FALLBACK_AB_RE = re.compile(r"\b([AB])\b")
_LABEL_RE = re.compile(r"LABEL\s*=\s*(sound|flawed)", re.I)
_FALLBACK_LABEL_RE = re.compile(r"\b(sound|flawed|unsound)\b", re.I)


def _base_of(item_id: str) -> str:
    """Strip the -clean / -flaw suffix to recover the shared twin id."""
    return item_id.rsplit("-clean", 1)[0].rsplit("-flaw", 1)[0]


def _load_soundness_items() -> list[dict[str, Any]]:
    """Load REFUTE soundness items from a local JSONL override or the HF Hub."""
    local = os.environ.get("REFUTE_SOUNDNESS_JSONL")
    if local:
        with open(local, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise RuntimeError(
            "Install `datasets` (pip install datasets) or set REFUTE_SOUNDNESS_JSONL "
            "to a local JSONL file."
        ) from exc
    ds = load_dataset(HF_DATASET, HF_CONFIG, split="train")
    return [dict(row) for row in ds]


def _build_forced_choice_samples(items: list[dict[str, Any]]) -> list[Sample]:
    by_base: dict[str, dict[str, dict]] = {}
    for it in items:
        by_base.setdefault(_base_of(it["id"]), {})[it["label"]] = it

    samples: list[Sample] = []
    for base in sorted(by_base):
        twin = by_base[base]
        if "sound" not in twin or "flawed" not in twin:
            continue
        # Deterministic flawed position from a stable hash of the base id (matches the
        # official REFUTE forced-choice protocol so results are directly comparable).
        flaw_first = int(hashlib.sha1(base.encode()).hexdigest(), 16) % 2 == 0
        a, b = (twin["flawed"], twin["sound"]) if flaw_first else (twin["sound"], twin["flawed"])
        flaw_letter = "A" if flaw_first else "B"
        prompt = (
            f"STUDY A:\n{a['summary']}\n\n"
            f"STUDY B:\n{b['summary']}\n\n"
            "Which study (A or B) has the more serious methodological flaw?"
        )
        samples.append(
            Sample(
                input=prompt,
                target=flaw_letter,
                id=base,
                metadata={"flaw_type": twin["flawed"].get("flaw_type", "")},
            )
        )
    return samples


def _build_soundness_samples(items: list[dict[str, Any]]) -> list[Sample]:
    samples: list[Sample] = []
    for it in items:
        label = str(it["label"]).lower()
        if label not in {"sound", "flawed"}:
            continue
        samples.append(
            Sample(
                input=f"STUDY SUMMARY:\n{it['summary']}\n\nIs this study's central conclusion sound or flawed?",
                target=label,
                id=it["id"],
                metadata={"flaw_type": it.get("flaw_type", "")},
            )
        )
    return samples


@scorer(metrics=[accuracy(), stderr()])
def _forced_choice_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        text = (state.output.completion or "").strip()
        m = _ANSWER_RE.search(text)
        if not m:
            m = _FALLBACK_AB_RE.search(text[:8])
        choice = m.group(1).upper() if m else None
        value = CORRECT if choice is not None and choice == target.text.upper() else INCORRECT
        return Score(value=value, answer=choice, explanation=text)

    return score


@scorer(metrics=[accuracy(), stderr()])
def _soundness_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        text = (state.output.completion or "").strip()
        m = _LABEL_RE.search(text)
        label = m.group(1).lower() if m else None
        if label is None:
            fm = _FALLBACK_LABEL_RE.search(text)
            if fm:
                token = fm.group(1).lower()
                label = "flawed" if token in {"flawed", "unsound"} else "sound"
        value = CORRECT if label is not None and label == target.text.lower() else INCORRECT
        return Score(value=value, answer=label, explanation=text)

    return score


@task
def refute_forced_choice() -> Task:
    """REFUTE forced-choice soundness task (contamination-proof; chance = 50%)."""
    samples = _build_forced_choice_samples(_load_soundness_items())
    return Task(
        dataset=MemoryDataset(samples, name="refute_forced_choice"),
        solver=[system_message(FORCED_CHOICE_SYSTEM), generate()],
        scorer=_forced_choice_scorer(),
        config={"temperature": 0.0, "max_tokens": 120},
    )


@task
def refute_soundness() -> Task:
    """REFUTE binary sound/flawed classification task."""
    samples = _build_soundness_samples(_load_soundness_items())
    return Task(
        dataset=MemoryDataset(samples, name="refute_soundness"),
        solver=[system_message(SOUNDNESS_SYSTEM), generate()],
        scorer=_soundness_scorer(),
        config={"temperature": 0.0, "max_tokens": 120},
    )
