# Reproducibility

## Environment

Recommended baseline:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

For open-weight or HF model runners:

```bash
python3 -m pip install -r requirements-openweight.txt
```

Credentials are read from the environment. Start from `.env.example` and keep real values untracked.

## Non-Network Smoke Checks

Run these before any model API calls:

```bash
python3 pilot/d_spec_config.py
python3 pilot/d_spec_variant_b.py --dry-run
```

These confirm that the protein panel and prompt-generation path are intact without calling external services.

## Claude/API Pilots

Claude-backed scripts generally require:

```bash
export ANTHROPIC_API_KEY="..."
```

Common entry points:

```bash
python3 pilot/d_spec_variant_b.py --dry-run
python3 pilot/v0_2_6_graded_rejudge.py --help
python3 pilot/c_natural_homolog_probe.py --help
```

Prefer pilot modes or explicit limits before full replicate runs.

## Open-Weight Runs

Open-weight paths may require:

```bash
export HF_TOKEN="..."
```

Use the relevant `pilot/*vllm*.py`, `pilot/*hf*.py`, and cluster `.slurm` helpers. These runs are environment-dependent and should record model name, hardware, backend, and command line in the result metadata or handoff note.

## Result Provenance

When adding result files, include or document:

- script entry point
- commit hash
- model/provider
- date and timezone
- sample size and replicate count
- temperature and major model parameters
- known failures, retries, or partial runs
