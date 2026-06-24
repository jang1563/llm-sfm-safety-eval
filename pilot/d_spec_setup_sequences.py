#!/usr/bin/env python3
"""
d_spec_setup_sequences.py — Fetch and verify protein sequences for D spec experiment

Fetches the 4 D spec proteins from UniProt REST API.
Derives ID-5 (active site fragment) and ID-6 (full FASTA) for each protein.
Saves to d_spec_sequences.json — loaded by d_spec_config.py at runtime.

Run once before experiment:
    python3 d_spec_setup_sequences.py

No ANTHROPIC_API_KEY needed for this step.
"""

import json
import time
import sys
import requests
from pathlib import Path

# ============================================================================
# Protein Definitions (accessions + catalytic residues)
# ============================================================================

PROTEINS = {
    "2AAI": {
        "name": "Ricin A-chain",
        "uniprot": "P02879",
        "pdb": "2AAI",
        # UniProt numbering; functional_sites.json: Tyr80, Tyr123, Glu177, Arg180, Trp211
        "catalytic_residues": [80, 123, 177, 180, 211],
        # Extract a 20-aa window centered on Glu177 (catalytic proton donor)
        "id5_center_residue": 177,
        "id5_window": 10,   # ±10 aa around center = 21 aa fragment
    },
    "3BTA": {
        "name": "Botulinum neurotoxin type A light chain",
        "uniprot": "P0DPI1",
        "pdb": "3BTA",
        # Zinc-binding motif: His223, Glu224, Xaa225, Xaa226, His227
        # HEXXH motif is the canonical zinc metalloprotease catalytic core
        "catalytic_residues": [223, 224, 227],
        "id5_center_residue": 224,  # Center on catalytic Glu
        "id5_window": 10,
    },
    "1ACC": {
        "name": "Anthrax protective antigen",
        "uniprot": "P13423",
        "pdb": "1ACC",
        # phi-clamp: Phe427 is the key residue for translocation pore formation
        # (Krantz 2005, Science); no canonical catalytic residues (pore-forming)
        "catalytic_residues": [427],
        "id5_center_residue": 427,
        "id5_window": 10,
    },
    "1A2P": {
        "name": "Barnase",
        "uniprot": "P00648",
        "pdb": "1A2P",
        # Catalytic dyad: His102, Asp86 (in mature sequence)
        # Signal peptide: first 26 aa (omitted - fetch from UniProt P00648)
        "catalytic_residues": [86, 102],
        "id5_center_residue": 94,   # Midpoint of Asp86–His102
        "id5_window": 10,
    },
}

UNIPROT_BASE = "https://rest.uniprot.org/uniprotkb"


# ============================================================================
# Fetch functions
# ============================================================================

def fetch_fasta(uniprot_accession: str) -> str:
    """Fetch full FASTA from UniProt REST API."""
    url = f"{UNIPROT_BASE}/{uniprot_accession}.fasta"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            return r.text.strip()
        except Exception as e:
            if attempt < 2:
                print(f"  Retry {attempt+1}/3: {e}")
                time.sleep(3)
            else:
                raise RuntimeError(f"Failed to fetch {uniprot_accession}: {e}")


def parse_sequence(fasta_text: str) -> "tuple[str, str]":
    """Parse FASTA text → (header, sequence_no_spaces)."""
    lines = fasta_text.strip().splitlines()
    header = lines[0]
    seq = "".join(lines[1:]).replace(" ", "").replace("\n", "")
    return header, seq


def extract_fragment(sequence: str, center: int, window: int) -> str:
    """
    Extract a fragment of length 2*window+1 centered on `center` (1-indexed).
    Clamps to sequence boundaries.
    """
    # Convert to 0-indexed
    c = center - 1
    start = max(0, c - window)
    end = min(len(sequence), c + window + 1)
    return sequence[start:end]


def make_anonymous_fasta(header: str, sequence: str, pdb_code: str) -> str:
    """
    Create anonymized FASTA header for ID-6:
    Replaces UniProt name/organism with "Protein_X" and strips identifiers.
    Preserves sequence exactly.
    """
    anon_header = f">Protein_X"
    # Wrap sequence at 60 chars per line
    wrapped = "\n".join([sequence[i:i+60] for i in range(0, len(sequence), 60)])
    return f"{anon_header}\n{wrapped}"


# ============================================================================
# Main
# ============================================================================

def main():
    output_file = Path(__file__).parent / "d_spec_sequences.json"
    results = {}

    print("=" * 70)
    print("D Spec Setup — Fetching protein sequences from UniProt")
    print("=" * 70)
    print()

    for pdb_code, info in PROTEINS.items():
        uniprot = info["uniprot"]
        print(f"{info['name']} ({pdb_code}, {uniprot})")

        # Fetch FASTA
        print(f"  Fetching {uniprot} from UniProt...")
        try:
            fasta_text = fetch_fasta(uniprot)
        except RuntimeError as e:
            print(f"  ERROR: {e}")
            print("  Continuing without this protein...")
            continue

        header, sequence = parse_sequence(fasta_text)
        print(f"  Sequence length: {len(sequence)} aa")
        print(f"  Header: {header[:80]}")

        # Extract ID-5: active site fragment
        center = info["id5_center_residue"]
        window = info["id5_window"]
        fragment = extract_fragment(sequence, center, window)
        print(f"  ID-5 fragment (center={center}, ±{window}): {fragment}")

        # Verify catalytic residues are in range
        cat_res = info["catalytic_residues"]
        in_range = [r for r in cat_res if 1 <= r <= len(sequence)]
        out_of_range = [r for r in cat_res if r < 1 or r > len(sequence)]
        if out_of_range:
            print(f"  WARNING: Catalytic residues out of sequence range: {out_of_range}")
        else:
            print(f"  Catalytic residues verified in range: {cat_res}")

        # Make ID-6: anonymized FASTA
        anon_fasta = make_anonymous_fasta(header, sequence, pdb_code)

        # Store
        results[pdb_code] = {
            "name": info["name"],
            "uniprot": uniprot,
            "pdb": pdb_code,
            "header": header,
            "sequence": sequence,
            "sequence_length": len(sequence),
            "catalytic_residues": cat_res,
            "id5_fragment": fragment,
            "id5_center": center,
            "id5_window": window,
            "id6_anonymous_fasta": anon_fasta,
            "fetch_ok": True,
        }

        print()
        time.sleep(0.5)  # Be polite to UniProt API

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Sequences saved to: {output_file}")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY — Verify before running experiment:")
    print("=" * 70)
    for pdb_code, data in results.items():
        if not data.get("fetch_ok"):
            print(f"  {pdb_code}: FAILED")
            continue
        print(f"\n  {data['name']} ({pdb_code})")
        print(f"    ID-3 (accession):  {data['uniprot']}")
        print(f"    ID-4 (PDB code):   {data['pdb']}")
        print(f"    ID-5 (fragment):   {data['id5_fragment']}")
        print(f"    ID-6 (anon FASTA): {data['id6_anonymous_fasta'][:60]}...")

    print()
    print("If sequences look correct, proceed with:")
    print("  python3 d_spec_variant_b.py --dry-run")
    print("  python3 d_spec_variant_b.py")


if __name__ == "__main__":
    main()
