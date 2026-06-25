#!/usr/bin/env python3
"""Build a tidy, load-ready refusal-trials table from the redacted aggregates.

Flattens the per-trial outcomes of the three identifier-gradient refusal
experiments (protein / chemistry / genomic) in ``pilot/results_public/`` into a
single long-format JSONL with one row per model call. Only outcome labels and
metadata are emitted: there is NO response text and NO biological sequence in
this table (the source aggregates are already redacted to outcomes only).

Output: ``data/llm_sfm_refusal_trials.jsonl`` (HuggingFace dataset-viewer ready).
Re-run after updating ``pilot/results_public/`` to regenerate.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "pilot" / "results_public"
OUT = Path(__file__).resolve().parent / "llm_sfm_refusal_trials.jsonl"


def managed(model: str) -> str:
    return "managed" if "claude" in (model or "").lower() else "open-weight"


def rows():
    # Protein: D-spec Variant B (description-omitted identifier recognition)
    for fp in sorted(SRC.glob("variant_b_2026*.json")):
        d = json.loads(fp.read_text())
        model = d.get("model")
        for r in d.get("results", []):
            yield {
                "domain": "protein",
                "source_experiment": "variant_b",
                "model": model,
                "model_display": model,
                "deployment": managed(model),
                "model_safety_tier": None,
                "entity": r.get("protein"),
                "entity_category": None,
                "id_level": r.get("id_level"),
                "rep": r.get("rep"),
                "refusal": bool(r.get("refusal")),
                "recognized": None,
                "stop_reason": r.get("stop_reason"),
            }
    # Chemistry: open-weight identifier gradient
    for fp in sorted(SRC.glob("chem_domain_vllm*.json")):
        d = json.loads(fp.read_text())
        for r in d.get("results", []):
            yield {
                "domain": "chemistry",
                "source_experiment": "chem_domain",
                "model": d.get("model_id"),
                "model_display": d.get("display_name"),
                "deployment": managed(d.get("model_id")),
                "model_safety_tier": d.get("safety_tier"),
                "entity": r.get("substance"),
                "entity_category": r.get("category"),
                "id_level": r.get("id_level"),
                "rep": r.get("rep"),
                "refusal": bool(r.get("is_refusal")),
                "recognized": r.get("names_correctly"),
                "stop_reason": None,
            }
    # Genomic: open-weight identifier gradient
    for fp in sorted(SRC.glob("dna_domain_vllm*.json")):
        d = json.loads(fp.read_text())
        for r in d.get("results", []):
            yield {
                "domain": "dna",
                "source_experiment": "dna_domain",
                "model": d.get("model_id"),
                "model_display": d.get("display_name"),
                "deployment": managed(d.get("model_id")),
                "model_safety_tier": d.get("safety_tier"),
                "entity": r.get("gene"),
                "entity_category": r.get("category"),
                "id_level": r.get("id_level"),
                "rep": r.get("rep"),
                "refusal": bool(r.get("is_refusal")),
                "recognized": r.get("identifies_gene"),
                "stop_reason": None,
            }


def main():
    n = 0
    with open(OUT, "w") as f:
        for row in rows():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} rows -> {OUT.name}")


if __name__ == "__main__":
    main()
