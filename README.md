# CrossRoute-Audit

CrossRoute-Audit is a Python framework for auditing the **faithfulness of
explanations in white-box vision-language models**. It compares where an
attribution method assigns importance with where causal interventions show that
visual or textual information actually affects a target answer.

The framework is designed for reproducible research on VLM explanation
faithfulness: adapter-based model support, schema-validated artifacts,
control-gated diagnosis, synthetic validation, and paper-ready analysis tools.

## Core idea

Attribution maps can show where a model appears to focus, but they do not prove
that the highlighted route causally drove the answer. CrossRoute-Audit anchors
explanation analysis with causal evidence:

```text
VQA sample
  -> target answer logit
  -> attribution mass by route/layer
  -> causal route intervention by layer
  -> rank/structural alignment metrics
  -> control-gated diagnosis
```

The primary signal is **RankAlignment**: a Spearman rank correlation between
layer-wise attribution mass and layer-wise causal effect on the same route. This
keeps the audit scale-invariant and avoids treating attribution magnitude alone
as evidence of faithfulness.

## Features

- Adapter contract for auditable white-box VLMs.
- BLIP-2, LLaVA, and InstructBLIP adapter code/tests.
- Layer Integrated Gradients attribution over audit-layer activations.
- Route ablation and activation patching for causal effects.
- Text-only, no-image, counterfactual, and negative-control gates.
- Rank, structural, fusion, sensitivity, and statistical metric utilities.
- JSON Schema validation for manifests and audit artifacts.
- Synthetic fault benchmark for validating diagnosis logic.
- Dataset manifest builders and final-analysis figure/table scripts.

## Installation

```bash
git clone https://github.com/umynameislove/crossroute-audit.git
cd crossroute-audit

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional dashboard support:

```bash
python -m pip install -e ".[dev,dashboard]"
```

## Quick checks

```bash
python -m pytest -q
python -m crossroute_audit.io.manifest data/manifest/samples.example.jsonl
python -m crossroute_audit.cli validate --out runs/synthetic_smoke --n 5
```

The CLI exposes artifact-oriented commands:

```bash
python -m crossroute_audit.cli --help
```

## Artifact workflow

CrossRoute-Audit separates GPU-heavy model runs from lightweight analysis.
Per-sample runs produce JSON artifacts such as:

- `control_status_<sample_id>.json`
- `causal_effect_<sample_id>.json`
- `attribution_mass_<sample_id>.json`
- `audit_report_<sample_id>.json`

Once control, causal, and attribution artifacts exist, batch reports can be
assembled with:

```bash
crossroute batch \
  --manifest data/manifest/samples.jsonl \
  --control-dir runs/control \
  --causal-dir runs/causal \
  --attr-dir runs/attr \
  --out runs/audit

crossroute report --run runs/audit --out runs/report.md
```

Multi-model comparison and paper-style figures:

```bash
python scripts/analyze_results.py \
  --models blip2=runs/blip2 llava=runs/llava \
  --out runs/figures
```

## Repository layout

```text
crossroute_audit/
  model_adapters/   Model adapter interface and VLM adapters
  instrumentation/  Hooks and activation capture
  interventions/    Route ablation and activation patching
  controls/         Baseline/control gates
  attribution/      Attribution and completeness utilities
  metrics/          Faithfulness, structure, fusion, and stats metrics
  io/               Manifest, schema, report, and analysis helpers
  synthetic/        Synthetic fault benchmark
  cli.py            Command-line interface

data/manifest/      Example and evaluation manifests
schemas/            JSON Schemas for artifacts
scripts/            Dataset and analysis scripts
tests/              Test suite
docs/SPEC.md        Technical specification
```

## Research scope

CrossRoute-Audit evaluates explanation faithfulness, not whether a model
“reasons correctly.” Layer-wise claims require white-box access to model
activations and interventions. Raw tensors, checkpoints, images, and run outputs
are intentionally excluded from version control.

## License

MIT. See `LICENSE`.
