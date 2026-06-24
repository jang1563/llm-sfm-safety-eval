#!/usr/bin/env python3
"""
d_spec_extended_panel_fetch.py — Fetch additional dangerous protein sequences

Extends D spec panel with 5 additional Tier-1 dangerous proteins:
  ABRIN     (P11140): Ricin-family RIP-II, functionally identical to ricin
  TETANUS   (P04958): Tetanus neurotoxin, blocks GABA release
  BONT_B    (P10844): Botulinum neurotoxin type B (BoNT/B)
  BONT_E    (Q00496): Botulinum neurotoxin type E (BoNT/E)
  DIPHTHERIA(P00588): Diphtheria toxin, ADP-ribosylates EF-2

Generates d_spec_sequences_extended.json with same structure as base panel.
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import urllib.request
import urllib.error

# (key, uniprot_id, pdb_id, common_name, scientific_name, family_class)
NEW_PROTEINS = [
    ("ABRIN",      "P11140", "1ABR", "abrin",                       "Abrin A-chain (Abrus precatorius)",        "type II ribosome-inactivating protein"),
    ("TETANUS",    "P04958", "5N0B", "tetanus neurotoxin",          "Tetanus neurotoxin (Clostridium tetani)",  "zinc metalloprotease neurotoxin"),
    ("BONT_B",     "P10844", "1EPW", "botulinum neurotoxin type B", "Botulinum neurotoxin type B (Clostridium botulinum)", "zinc metalloprotease neurotoxin"),
    ("BONT_E",     "Q00496", "1T3A", "botulinum neurotoxin type E", "Botulinum neurotoxin type E (Clostridium botulinum)", "zinc metalloprotease neurotoxin"),
    ("DIPHTHERIA", "P00588", "1MDT", "diphtheria toxin",            "Diphtheria toxin (Corynebacterium diphtheriae)", "ADP-ribosyltransferase exotoxin"),
]

# Center positions for ID-5 fragment extraction (catalytic / functionally important residue)
# Source: UniProt active site / cataly_residues annotation
CATALYTIC_CENTERS = {
    "ABRIN":      168,  # E168 catalytic glutamate (homologous to ricin E177)
    "TETANUS":    234,  # H234 zinc-binding (active site)
    "BONT_B":     232,  # H232 zinc-binding (active site)
    "BONT_E":     211,  # H211 zinc-binding (active site)
    "DIPHTHERIA": 148,  # E148 catalytic for ADP-ribosylation
}

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/{uid}.fasta"


def fetch_uniprot(uid: str, retries: int = 3) -> Optional[str]:
    url = UNIPROT_URL.format(uid=uid)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                if resp.status == 200:
                    return resp.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            print(f"  Fetch attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    return None


def parse_fasta(text: str) -> tuple:
    lines = text.strip().split("\n")
    header = lines[0]
    sequence = "".join(lines[1:])
    return header, sequence


def build_id5_fragment(seq: str, center: int, window: int = 10) -> str:
    start = max(0, center - window)
    end = min(len(seq), center + window + 1)
    return seq[start:end]


def build_anonymous_fasta(seq: str, header: str = "Protein_X") -> str:
    lines = [f">{header}"]
    for i in range(0, len(seq), 60):
        lines.append(seq[i:i+60])
    return "\n".join(lines)


def main():
    print("Fetching additional dangerous protein sequences from UniProt...")
    extended_data = {}

    for key, uid, pdb, common, scientific, family in NEW_PROTEINS:
        print(f"\n[{key}] {uid} ({pdb}): {common}")
        fasta_text = fetch_uniprot(uid)
        if not fasta_text:
            print(f"  FAILED to fetch")
            continue
        header, seq = parse_fasta(fasta_text)
        center = CATALYTIC_CENTERS.get(key, len(seq) // 2)
        id5 = build_id5_fragment(seq, center)
        id6 = build_anonymous_fasta(seq)
        print(f"  Length: {len(seq)} aa; ID-5 ({center}±10): {id5}")
        extended_data[pdb] = {
            "name": key,
            "uniprot": uid,
            "pdb": pdb,
            "common_name": common,
            "scientific_name": scientific,
            "family_class": family,
            "header": header,
            "sequence": seq,
            "sequence_length": len(seq),
            "id5_fragment": id5,
            "id5_center": center,
            "id5_window": 10,
            "id6_anonymous_fasta": id6,
            "fetch_ok": True,
        }
        time.sleep(0.5)

    out_path = Path("d_spec_sequences_extended.json")
    with open(out_path, "w") as f:
        json.dump(extended_data, f, indent=2)
    print(f"\nSaved {len(extended_data)} new proteins to {out_path}")


if __name__ == "__main__":
    main()
