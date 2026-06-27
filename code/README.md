# Multi-Modal Evidence Review — Solution

## Overview

This solution verifies damage claims by analyzing submitted images alongside claim conversations, user history, and evidence requirements. It uses **Google Gemini 3.1 Flash Lite** as the vision-language model (VLM) to inspect images and determine whether claims are supported, contradicted, or lack sufficient evidence. The model is driven by a structured **decision-tree prompt** that first checks whether the claimed part is visible (evidence sufficiency), then compares the visible damage to the customer's claim, and finally calibrates issue type and severity.

`gemini-3.1-flash-lite` was chosen because its **free-tier quota (500 requests/day)** comfortably covers the full sample + test workload (64 calls) at **zero cost**, while still providing strong multi-image vision.

## Architecture

The design separates the **VLM's visual judgement** from **deterministic
post-processing**. The model decides only what it can see; everything that can be made
reproducible (type coercion, label normalisation, severity calibration, history risk)
is handled in code that costs no API calls.

```
 dataset/claims.csv
        │
        ▼
 ┌─────────────────┐   per claim
 │  Build prompt   │   (decision-tree, all images batched into ONE call)
 └────────┬────────┘
          ▼
 ┌─────────────────┐   versioned SHA-256 cache  ┌──────────────────┐
 │   Gemini VLM    │◀──────────────────────────▶│ .vlm_cache.json  │
 │ (JSON, temp 0.1)│   (key includes prompt ver) └──────────────────┘
 └────────┬────────┘
          ▼   raw JSON
 ┌─────────────────────────────────────────────┐
 │  Layer 1  pydantic VlmPrediction  (coerce types, fail safe)
 │  Layer 2  calibration  (anchor issue/part to customer words; severity)
 │  Layer 3  clamp to allowed values + merge user-history risk flags
 └────────┬────────────────────────────────────┘
          ▼
     output.csv  (14-column schema)
```

## Prerequisites

- Python 3.10–3.12 (pinned via `.python-version`; [`uv`](https://docs.astral.sh/uv/) provisions it for you)
- A Google Gemini API key (free tier is sufficient)

## Setup

This project uses **[uv](https://docs.astral.sh/uv/)** for dependency management
(`pyproject.toml` + `uv.lock` are the source of truth). A pinned `requirements.txt`
is also provided for plain pip.

```bash
# With uv (recommended — provisions Python 3.12 and installs from the lockfile)
uv sync

# …or with plain pip
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt

# Configure the API key (copy the template, then edit)
cp .env.example .env      # then set GEMINI_API_KEY=...
```

## Running the Solution

Common tasks are wrapped in a `Makefile` (`make run`, `make eval`, `make test`).

### Generate predictions for the test set

```bash
uv run python code/main.py        # or: make run  /  python code/main.py
```

This reads `dataset/claims.csv`, calls the Gemini VLM for each claim, and writes `output.csv`.

**Options:**
```
--input        Input CSV path        (default: dataset/claims.csv)
--output       Output CSV path       (default: output.csv)
--model        Gemini model name     (default: gemini-3.1-flash-lite)
--cache        VLM cache file        (default: code/.vlm_cache.json)
--history      User history CSV      (default: dataset/user_history.csv)
--requirements Evidence reqs CSV     (default: dataset/evidence_requirements.csv)
```

### Evaluate on sample data

```bash
uv run python code/evaluation/main.py     # or: make eval
```

This runs the verifier against `dataset/sample_claims.csv`, compares predictions with
gold labels, prints an ablation (raw VLM vs. calibrated) and per-row diffs, and writes
`code/evaluation/evaluation_report.md` (metrics, ablation table, confusion matrices,
operational/cost analysis) plus `code/evaluation/sample_errors.csv`.

### Run the tests

```bash
uv run pytest        # or: make test
```

35 deterministic unit tests (no API/network) covering the calibration layer, customer-turn
parsing, post-processing/history merge, the versioned cache key, and the pydantic schema +
generated `output.csv`.

## Key Design Decisions

1. **Single VLM call per claim** — all images for a claim are sent together in one multi-modal Gemini call with structured JSON output, minimizing latency and cost.

2. **Structured JSON output** — `response_mime_type="application/json"` ensures Gemini returns parseable JSON matching the output schema, eliminating fragile regex parsing.

3. **Versioned deterministic caching** — a SHA-256 hash of (user_id, image_paths, user_claim, claim_object, model, **prompt version**) keys a local JSON cache. Re-runs cost zero API calls, and the prompt-version component auto-invalidates stale entries when the prompt changes (no silent stale answers).

4. **pydantic output schema** (`VlmPrediction`) — the raw model JSON is parsed through a typed, self-coercing schema (strings→bools, scalars/semicolon-strings→lists, missing fields→safe defaults) before post-processing, so shape drift from the VLM can't crash the pipeline.

5. **Three-layer post-processing** — the VLM handles visual analysis only; then (1) pydantic coerces types, (2) calibration normalises labels (below), (3) values are clamped to the allowed sets and user-history risk flags (`user_history_risk`, `manual_review_required`) are merged from `user_history.csv`.

6. **Exponential backoff** — API failures are retried up to 5 times with delays from 2s to 30s; real token usage (`usage_metadata`) is captured per call.

7. **Low temperature (0.1)** — reduces output variability across runs.

8. **Decision-tree prompt** — the prompt encodes an explicit, ordered procedure:
   (A) is the claimed part visible? → if not, `not_enough_information` + `false`
   evidence; (B) does the visible damage match the claim? → `supported` /
   `contradicted` (+ `claim_mismatch` / `wrong_object`); (C) report the *visible*
   issue/part; (D) estimate severity.

9. **Deterministic calibration layer** (`calibrate_issue_and_severity`, `calibrate_object_part`
   in `main.py`) — the VLM is reliable at the visual *judgement* but has stubborn,
   systematic biases that prompt instructions alone did not remove: it re-labels the
   customer's described damage with a more dramatic synonym (a described `crack` →
   `glass_shatter`, a bumper `scratch` → `dent`) and inflates `severity` by one notch.
   Because a **supported** claim by definition means the visible damage matches what the
   customer described, we deterministically anchor `issue_type` (and `object_part`, only
   when exactly one part family is named) to the customer's own words — parsed from the
   **customer turns only**, final turn first — and calibrate `severity` from the resolved
   `issue_type` (`scratch`/minor dent → `low`, `medium` default, `high` catastrophic-only).
   Runs on cached output (**zero API calls**), fully reproducible. Measured contribution
   (see the ablation in `evaluation/evaluation_report.md`): `issue_type` +20, `severity`
   +20, `object_part` +5, no regressions.

## Results on `sample_claims.csv` (20 labeled claims)

| Column | Accuracy | Macro-F1 | vs. VLM-only |
|:---|:---:|:---:|:---:|
| `claim_status` | 75.0% | 61.1% | — |
| `evidence_standard_met` | 90.0% | 72.2% | — |
| `valid_image` | 90.0% | 47.4% | — |
| `object_part` | **90.0%** | **85.2%** | +5.0 |
| `issue_type` | **70.0%** | **70.2%** | +20.0 |
| `severity` | **70.0%** | 39.8% | +20.0 |
| `risk_flags` (Jaccard) | 81.2% | — | — |

The calibration layer raised `issue_type` (50→70%), `severity` (50→70%) and `object_part`
(85→90%) over the raw VLM output with **no regressions** elsewhere. Residual errors are
entangled with the model's *visual* `claim_status` decision (e.g. it reads a bumper
scratch as a dent, or misses severe damage); these are left to the model rather than
overridden, since the images are the primary source of truth. See the auto-generated
`evaluation/evaluation_report.md` for the ablation, confusion matrices, and the
operational/cost analysis. Re-run `uv run python code/evaluation/main.py` to reproduce
(all sample responses are cached, so it costs zero API calls).

## Output Schema

| Column | Description |
|:---|:---|
| `evidence_standard_met` | Whether images are sufficient to evaluate the claim |
| `evidence_standard_met_reason` | Short reason for the evidence decision |
| `risk_flags` | Semicolon-separated risk flags, or `none` |
| `issue_type` | Visible issue type (dent, scratch, crack, etc.) |
| `object_part` | Relevant object part |
| `claim_status` | `supported`, `contradicted`, or `not_enough_information` |
| `claim_status_justification` | Concise image-grounded explanation |
| `supporting_image_ids` | Image IDs supporting the decision, or `none` |
| `valid_image` | Whether images are usable for automated review |
| `severity` | `none`, `low`, `medium`, `high`, or `unknown` |
