# CrossRoute-Audit

CrossRoute-Audit is a research framework for testing whether explanations in
white-box vision-language models are faithful to the routes that actually drive
a target answer.

Attribution methods can tell us where a model appears to look. CrossRoute-Audit
adds the missing check: do those attribution scores agree with causal
interventions on the model's internal image/text routes?

The project is built around a simple audit contract:

1. choose a target VQA answer token,
2. measure attribution mass over image/text routes by layer,
3. ablate or patch those same routes to estimate causal effect,
4. compare the two signals with scale-invariant rank metrics,
5. gate any diagnosis behind text-only, no-image, counterfactual, and negative
   controls.

The goal is not to claim that a model “reasons incorrectly.” The goal is to
audit the faithfulness of explanation methods against causal evidence.

## Why this exists

Vision-language attribution maps are often used as evidence that a model relied
on the image. That is a stronger claim than attribution alone can support.
CrossRoute-Audit treats explanation faithfulness as an empirical question:

> If visual information is causally important at certain layers, attribution
> should rank those same layers as important.

This framing makes the audit useful for:

- checking whether attribution maps track causal routing rather than only visual
  salience,
- separating language-prior behavior from genuine visual grounding,
- comparing explanation faithfulness across models, attribution methods, and
  control conditions,
- producing reproducible JSON artifacts that can be reviewed, validated, and
  plotted without raw tensors.

## Current capabilities

The repository currently includes:

- an adapter-based model interface for white-box VLM auditing,
- BLIP-2, LLaVA, and InstructBLIP adapter implementations/tests,
- Layer Integrated Gradients attribution over audit-layer activations,
- image/text route ablation and activation-patching utilities,
- control-gated diagnosis logic,
- schema-validated manifest, control, causal, attribution, and audit-report
  artifacts,
- rank-alignment, structural-alignment, fusion, sensitivity, and statistical
  metric helpers,
- deterministic synthetic-fault validation,
- dataset manifest builders for pilot and N=100 VQA-style evaluation,
- final-analysis utilities for model comparison tables and paper-style figures.

Some commands are intentionally artifact-oriented. The single-sample
`crossroute audit` command is still a placeholder; the supported workflow is to
generate or provide per-sample artifacts, then use batch/report/analysis tools to
validate and summarize them.

## Repository layout

```text
crossroute_audit/
  model_adapters/   White-box model adapter contract and VLM adapters
  instrumentation/  Hooks, activation capture, and no-op controls
  interventions/    Route ablation and clean/corrupt activation patching
  controls/         Text-only, no-image, counterfactual, negative controls
  attribution/      Layer IG, completeness checks, method agreement
  metrics/          Rank, structure, causal, fusion, stats, sensitivity metrics
  io/               Manifest loading, schema validation, reports, analysis
  synthetic/        Synthetic fault generation and benchmark utilities
  dashboard/        Read-only Streamlit artifact viewer
  cli.py            `crossroute` command-line entry point

data/manifest/      Example, pilot, and N=100 manifest files
schemas/            JSON Schemas for public artifacts
scripts/            Dataset, smoke-test, and final-analysis scripts
tests/              Unit and integration-style test coverage
docs/SPEC.md        MVP technical specification and claim boundary
```

Images, model checkpoints, tensors, and run outputs are intentionally
gitignored. The repository tracks code, schemas, manifests, tests, and lightweight
metadata.

## Installation

Use Python 3.10 or newer. For development, an isolated virtual environment is
recommended.

```bash
git clone https://github.com/umynameislove/crossroute-audit.git
cd crossroute-audit

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional dashboard dependencies:

```bash
python -m pip install -e ".[dev,dashboard]"
```

The package exposes the `crossroute` console script after editable install.

```bash
crossroute --help
```

If your shell cannot see the console script yet, the module form is equivalent:

```bash
python -m crossroute_audit.cli --help
```

## Quick verification

Run the test suite:

```bash
python -m pytest -q
```

Validate the example manifest:

```bash
python -m crossroute_audit.io.manifest data/manifest/samples.example.jsonl
```

Run a small deterministic synthetic benchmark:

```bash
crossroute validate --out runs/synthetic_smoke --n 5
```

or:

```bash
python -m crossroute_audit.cli validate --out runs/synthetic_smoke --n 5
```

The synthetic benchmark writes `benchmark.csv` under the chosen run directory and
prints the measured accuracy over generated fault cases. These are logic tests;
they are not a substitute for real model artifacts.

## Data and manifests

The manifest schema is defined in `schemas/manifest.schema.json`. Each line is a
JSON object describing one audit sample.

Required fields include:

- `sample_id`
- `image_path`
- `question`
- `target_answer`
- `target_token_policy`
- `expected_visual_dependency`
- `text_only_answerable`
- `control_type`
- `notes`

Useful manifest files:

- `data/manifest/samples.example.jsonl` — tiny schema example,
- `data/manifest/samples.jsonl` — pilot manifest,
- `data/manifest/samples_n100.jsonl` — expanded VQA-style manifest.

Images are not committed by default. Place image files at the paths referenced by
the manifest before running GPU/model workflows.

To regenerate or inspect the N=100-style manifest builder:

```bash
python scripts/build_n100_dataset.py --help
```

Unit tests for the builder do not download data:

```bash
python -m pytest tests/test_build_n100.py -q
```

## Artifact workflow

CrossRoute-Audit is designed around JSON artifacts so that expensive GPU work
and lightweight analysis can be separated.

Typical artifact families:

- `control_status_<sample_id>.json`
- `causal_effect_<sample_id>.json`
- `attribution_mass_<sample_id>.json`
- `audit_report_<sample_id>.json`

After controls, causal effects, and attribution masses are available, combine
them into per-sample audit reports:

```bash
crossroute batch \
  --manifest data/manifest/samples.jsonl \
  --control-dir runs/control \
  --causal-dir runs/causal \
  --attr-dir runs/attr \
  --out runs/audit
```

Render a Markdown summary table:

```bash
crossroute report --run runs/audit --out runs/report.md
```

For final multi-model analysis and figures:

```bash
python scripts/analyze_results.py \
  --models blip2=runs/blip2 llava=runs/llava \
  --out runs/figures
```

The analysis script expects each model directory to contain matched
`attr/attribution_mass_*.json` and `causal/causal_effect_*.json` files.

## GPU smoke test

The BLIP-2 smoke script is meant for a machine with the required model weights
and enough GPU memory.

```bash
python scripts/smoke_blip2.py \
  --image data/images/example.jpg \
  --question "Is there a dog in the image?" \
  --target yes \
  --device cuda
```

It prints model metadata, target logit, token-group sizes, and captured
tensor-free summaries. This is a smoke check, not a full audit.

## Core metrics

The primary metric is RankAlignment: Spearman correlation between attribution
mass and causal effect on the same route/layer axis.

Additional metric families include:

- structural alignment: detrended rank alignment and top-k overlap,
- non-parametric statistics: bootstrap confidence intervals, Cliff's delta,
  sign test, Holm-Bonferroni, Benjamini-Hochberg,
- fusion score: combines alignment, completeness, and control cleanliness,
- sensitivity/adversarial utilities for deterministic stress testing.

Metric design principle: attribution magnitude alone is not treated as evidence
of faithfulness. Attribution is interpreted only in relation to causal effects
and control gates.

## Development workflow

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run focused tests while editing:

```bash
python -m pytest tests/test_rank_alignment.py -q
python -m pytest tests/test_structure_align.py -q
python -m pytest tests/test_cli.py -q
```

Run the full suite before opening a PR:

```bash
python -m pytest -q
```

Check the working tree:

```bash
git status --short
git diff --stat
```

## Contributing

This repository uses a branch-and-review workflow:

1. create a branch from the latest `main`,
2. keep changes scoped and reviewable,
3. run the relevant tests and full test suite,
4. open a pull request into `main`,
5. wait for review and merge by the repository lead.

Recommended branch examples:

```bash
git fetch origin
git checkout -b docs/readme-refresh origin/main
```

or, if you already have an up-to-date local `main`:

```bash
git checkout main
git pull origin main
git checkout -b docs/readme-refresh
```

Use commit authorship that is connected to your GitHub account. For example:

```bash
git config user.name "Bu0308"
git config user.email "187178852+Bu0308@users.noreply.github.com"
```

To check the author that Git will use:

```bash
git config --get user.name
git config --get user.email
```

For GitHub contribution credit, the commit author email must be associated with
your GitHub account, and the branch must eventually be merged into the default
branch of a non-fork repository. A normal merge commit preserves the original
commit author; squash merges may rewrite the final commit metadata depending on
how the maintainer merges.

## Citation and research status

CrossRoute-Audit is an active research codebase. The current repository is meant
to make explanation-faithfulness audits reproducible, inspectable, and easier to
extend across white-box VLMs. If you use it in research, cite the repository and
include the exact commit hash, manifest, model checkpoint, and generated
artifacts used in your run.

## License

MIT. See `LICENSE`.
