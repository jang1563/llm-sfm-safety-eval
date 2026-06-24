#!/usr/bin/env python3
"""
Redact the raw evaluation corpus into a PUBLIC-SAFE, aggregate-only copy.

WHY: pilot/results/*.json hold (a) verbatim prompts that, at the sequence-identifier
levels, embed real select-agent toxin fragments, and (b) raw model responses that can
contain hazard-mechanism text — and they reveal every over/under-refusal bypass prompt
verbatim. None of that is needed to reproduce the paper's findings (which depend only on
the labels: which substance/gene, identifier level, refusal flag, recognition flag, etc.).

This tool walks each result JSON and DROPS the free-text / sequence-bearing keys,
keeping only the scalar labels and metrics, writing the stripped copy to results_public/.
Run it, commit results_public/ (raw results/ is git-ignored), and publish only that.

Usage:
    python3 pilot/redact_results_for_public.py            # results/ -> results_public/
    python3 pilot/redact_results_for_public.py --check    # verify no sensitive keys remain
"""
import json
import os
import sys
from pathlib import Path

PILOT = Path(__file__).parent
SRC = PILOT / "results"
DST = PILOT / "results_public"

# Keys that may carry verbatim prompts (incl. sequences), raw responses, or tool I/O.
DROP_KEYS = {
    "prompt", "prompts", "response_text", "response", "final_text", "text",
    "completion", "output", "output_text", "raw_response", "full_text",
    "messages", "content", "tool_calls", "fragment", "sequence", "motif",
    "fragment_500bp", "motif_50bp", "smiles", "inchi",
    "response_preview", "preview", "frame_text",
}


_HOME = os.path.expanduser("~")
_REPO_ROOTS = ("Science_FM_Safety", "Narrow_Model_Safety_Eval")


def _scrub_path(s):
    """Genericize local absolute paths: make repo-relative, else home -> '~'.

    Config keys like input_file/output_dir survive the key-drop (they are useful
    provenance), but they must not leak the local username or directory tree.
    """
    for r in _REPO_ROOTS:
        i = s.find("/" + r + "/")
        if i != -1:
            return s[i + len(r) + 2:]
    return s.replace(_HOME, "~")


def scrub(obj):
    """Recursively drop sensitive keys; keep scalar labels/metrics (paths genericized)."""
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items() if k not in DROP_KEYS}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, str):
        return _scrub_path(obj)
    return obj


def _has_sensitive(obj) -> bool:
    if isinstance(obj, dict):
        if any(k in DROP_KEYS for k in obj):
            return True
        return any(_has_sensitive(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_sensitive(v) for v in obj)
    return False


def main():
    check = "--check" in sys.argv
    if not SRC.is_dir():
        sys.exit(f"no source dir: {SRC}")
    DST.mkdir(exist_ok=True)
    files = sorted(SRC.glob("*.json"))
    n_ok = n_bad = 0
    src_bytes = dst_bytes = 0
    for f in files:
        try:
            raw = f.read_text()
            src_bytes += len(raw)
            data = json.loads(raw)
        except Exception as e:
            print(f"  SKIP {f.name}: {e}")
            n_bad += 1
            continue
        if check:
            # verify the already-written public copy is clean
            pub = DST / f.name
            if pub.exists() and _has_sensitive(json.loads(pub.read_text())):
                print(f"  ⚠ SENSITIVE KEY REMAINS in results_public/{f.name}")
                n_bad += 1
            continue
        clean = scrub(data)
        # annotate provenance + that this is the redacted public copy
        if isinstance(clean, dict):
            clean.setdefault("_redacted", "aggregate-only public copy; prompts/responses/"
                             "sequences stripped (see redact_results_for_public.py)")
        out = json.dumps(clean, indent=1)
        (DST / f.name).write_text(out)
        dst_bytes += len(out)
        assert not _has_sensitive(clean), f"sensitive key survived in {f.name}"
        n_ok += 1
    if check:
        print(f"--check: scanned {len(files)} files, {n_bad} with residual sensitive keys")
        sys.exit(1 if n_bad else 0)
    print(f"redacted {n_ok} files ({n_bad} skipped) -> {DST}")
    print(f"size: {src_bytes/1e6:.1f} MB raw -> {dst_bytes/1e6:.1f} MB public "
          f"({100*dst_bytes/max(1,src_bytes):.1f}% retained)")
    print("verify: python3 pilot/redact_results_for_public.py --check")


if __name__ == "__main__":
    main()
