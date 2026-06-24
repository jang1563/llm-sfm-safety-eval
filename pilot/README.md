# Pilot Scripts

This directory contains controlled safety-evaluation pilots and analysis tools for the LLM x SFM workstream.

## Main Script Families

- `d_spec_*.py`: safety-recognition boundary experiments across identifier abstraction levels.
- `v0_2_5_lal_*.py`: lab-in-the-loop trajectory pilots, judging, and analysis.
- `v0_2_6_graded_rejudge.py`: graded refusal re-judge infrastructure.
- `c_*.py`: natural homolog recognition and coverage probes.
- `*vllm*.py`, `*hf*.py`, `*openweight*.py`: open-weight model runners.
- `*.slurm`: cluster execution helpers.

## First Checks

```bash
python3 d_spec_config.py
python3 d_spec_variant_b.py --dry-run
```

Run dry paths first. Full pilot runs may call external APIs, consume credits, and produce raw outputs in `pilot/results/`, which is git-ignored; only redacted, aggregate-only outputs ship under `pilot/results_public/`.

## Credentials

Claude-backed pilots require `ANTHROPIC_API_KEY`. OpenAI judge comparisons require `OPENAI_API_KEY`. Gated Hugging Face models require `HF_TOKEN`.

Use environment variables only. Do not write credentials into scripts, command logs, or result files.

## Result Hygiene

Raw result files stay out of the tracked tree (git-ignored); only redacted, aggregate-only outputs ship under `pilot/results_public/`. When adding a new aggregate output, record enough provenance to reproduce the run:

- script name
- model and provider
- date/time
- sample size
- replicate count
- temperature and key parameters
- known failures or partial-run notes
