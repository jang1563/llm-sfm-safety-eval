# Contributing

This is a defensive safety-evaluation repository. Contributions should preserve
that framing, keep changes auditable, and never add operational hazardous content
(see [`SECURITY.md`](SECURITY.md)).

## Change checklist

Before committing:

```bash
git status --short
git diff --check
bash scripts/secret_scan.sh
```

For script changes, run the nearest dry-run or non-network smoke test first:

```bash
python3 pilot/d_spec_config.py
python3 pilot/d_spec_variant_b.py --dry-run
```

## Commit style

Use small, descriptive commits when possible:

- `docs: clarify safety scope`
- `pilot: add graded rejudge aggregation`
- `specs: refine measurement design`

## Data and results

Keep raw and generated outputs out of the tracked tree (they are git-ignored).
Only redacted, aggregate-only outputs belong in `pilot/results_public/`. When
adding a new aggregate file, include enough context to reproduce or interpret it:

- script name and version or commit
- model and provider
- date and sample size
- key parameters such as temperature and replicate count
- known limitations or failed runs
