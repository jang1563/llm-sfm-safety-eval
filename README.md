---
license: apache-2.0
language: [en]
tags: [ai-safety, biosecurity, science-foundation-models, llm-evaluation, refusal-calibration]
pretty_name: LLM x SFM Safety Evaluation
size_categories: [10K<n<100K]
task_categories: [text-classification]
configs:
  - config_name: refusal_trials
    data_files: data/llm_sfm_refusal_trials.jsonl
---

# LLM x SFM Safety Evaluation

**When a general-purpose language model interprets the output of a specialist
science foundation model (a protein, genomic, RNA, or chemistry model), does its
safety behavior recognize the scientific content, or only the surface form of the
request?**

This repository is the empirical core of a study of that question: the evaluation
harness, the redacted aggregate results, and the measurement specifications behind
four findings about how deployed Claude models behave on the **LLM x SFM
interpretation channel**, a channel current system cards do not cover.

It is a **defensive safety-evaluation archive.** The unit of analysis is whether a
safety system recognizes scientific representations, never how to perform harmful
biological work. No biological sequences, synthesis routes, or operational content
are included; results are aggregate-only.

---

## Findings

- **The classifier pendulum.** On this channel, refusal swings discontinuously
  across a single model update (protein 1.4% to 91.7%, chemistry 3.4% to 93.5%,
  DNA 0 to about 100%, from Sonnet 4 to Sonnet 4.6), then persists (not flat)
  across the later releases. Coverage is installed in one step, not accumulated.
- **A calibration gradient, not uniform safety.** Apparent "safety" ranges from
  genuine sequence-hazard recognition (protein: a BoNT-A sequence is refused while
  benign barnase is not, at the identical prompt), through keyword-amplified
  registry recognition (chemistry), to pure context-keyword over-refusal (DNA:
  benign genes are refused under a "defensive biosecurity" framing).
- **The safety layer is managed-access infrastructure, not model scale.**
  Open-weight panels (3.8B to 111B parameters, 7 vendors) produce 0% genuine
  refusals while naming the same agents 98 to 100% of the time. The refusal
  behavior is specific to the managed deployment, not to capability.
- **A keyword-triggered tool-mode surface.** Sonnet 4.6's tool-mode classifier is
  keyword-triggered (a generic tool name drops refusal to 0%; a benign protein
  with a bio description draws a 100% false-positive); Opus 4.8 closes that
  keyword-rename bypass at the same aggregate refusal rate, a mechanism change
  invisible to refusal-rate-only reporting.

The executive, system-card-style write-up is in
[`System_Card_Supplement_CrossModal_SFM_Safety.md`](System_Card_Supplement_CrossModal_SFM_Safety.md).

## What is in this repository

```text
.
|-- System_Card_Supplement_CrossModal_SFM_Safety.md   the executive findings view
|-- specs/                  measurement specifications (recognition boundary, lab-in-the-loop, over-trust)
|-- pilot/                  the evaluation harness: probe and scoring scripts, runbooks, domain findings
|-- pilot/results_public/   redacted, aggregate-only outputs (labels and metrics; no raw responses)
|-- data/                   load-ready refusal-trials table (24.3K rows) + its builder
|-- scripts/                repository hygiene (secret scan, etc.)
|-- docs/                   reproducibility and safety/access notes
|-- requirements.txt        managed-API dependencies
`-- requirements-openweight.txt   optional open-weight stack
```

## Load-ready refusal table

For quick re-analysis (and the Hugging Face dataset viewer),
[`data/llm_sfm_refusal_trials.jsonl`](data/llm_sfm_refusal_trials.jsonl) is a tidy,
long-format table of **24,300 per-trial refusal outcomes** flattened from the three
identifier-gradient experiments (protein, chemistry, genomic). One row per model
call; outcome labels and metadata only, with no response text and no sequences.

```python
from datasets import load_dataset
ds = load_dataset("jang1563/llm-sfm-safety-eval", split="train")
# managed (Claude) vs open-weight refusal, by domain
import collections
agg = collections.Counter()
for r in ds:
    agg[(r["domain"], r["deployment"])] += r["refusal"]
```

| Column | Meaning |
|---|---|
| `domain` | `protein` / `chemistry` / `dna` |
| `source_experiment` | `variant_b` / `chem_domain` / `dna_domain` |
| `model`, `model_display` | model identifier |
| `deployment` | `managed` (Claude API) vs `open-weight` |
| `model_safety_tier` | source safety rating for open-weight models, else null |
| `entity`, `entity_category` | the agent tested (protein/substance/gene) and its class |
| `id_level` | identifier abstraction `ID-0` (name) to `ID-6` (sequence) |
| `rep` | replicate index |
| `refusal` | bool, the trial's refusal outcome (detector-level; see the findings for the genuine-refusal vs capability-disclaimer distinction) |
| `recognized` | bool/null, did the model name the agent correctly |
| `stop_reason` | API stop reason where recorded |

This table covers the identifier-gradient refusal experiments only; the secondary
surfaces (tool-mode, reliability over-trust, lab-autonomy loops) and the full
per-experiment aggregates remain as individual files under `pilot/results_public/`.
Regenerate with `python data/build_refusal_trials.py`.

## Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt   # add -r requirements-openweight.txt for local open-weight runs
cp .env.example .env                          # fill credentials locally; never commit .env
```

Live collection reads `ANTHROPIC_API_KEY` (and, for open-weight runs, the heavier
stack) from the environment. The committed `pilot/results_public/` outputs can be
read and re-analyzed without any API access. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for tested workflows.

## Scope and safety

This is a defensive measurement archive. In scope: safety-evaluation design,
classifier-coverage analysis, and aggregate refusal/response analysis. Out of
scope, and not present: wet-lab protocols, synthesis routes, operational
biological guidance, capability-uplift content, and any materialized hazardous
sequences. Sequence panels are reconstructed from public database accessions by
builder scripts; resolved sequences are not shipped. See
[`docs/SAFETY_AND_ACCESS.md`](docs/SAFETY_AND_ACCESS.md) and
[`SECURITY.md`](SECURITY.md).

## License

[Apache-2.0](LICENSE). Copyright (c) 2026 JangKeun Kim.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Contact

JangKeun Kim, Mason Lab, Weill Cornell Medicine, jak4013@med.cornell.edu
