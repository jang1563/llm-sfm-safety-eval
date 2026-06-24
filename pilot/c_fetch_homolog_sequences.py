#!/usr/bin/env python3
"""
Fetch homolog sequences from UniProt and prepare for C probe.

Downloads FASTA and metadata for:
  - BoNT family: P0DPI1 (BoNT-A), P10844 (BoNT-B), Q00496 (BoNT-E)
  - Ricin family: P02879 (Ricin), P06750 (R. communis agglutinin), P11140 (Abrin-a)

Saves both full FASTA and partial (active-site) extracts for ID-5 vs ID-6 testing.

Usage:
    python3 c_fetch_homolog_sequences.py --download  # Fetch fresh from UniProt
    python3 c_fetch_homolog_sequences.py --use-local  # Use cached sequences
"""

import json
import os
import sys
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path

SEQUENCES_DIR = Path(__file__).parent / "sequences" / "homologs"
METADATA_FILE = SEQUENCES_DIR / "homolog_sequences_meta.json"

ACCESSIONS = {
    "BoNT-A": "P0DPI1",
    "BoNT-B": "P10844",
    "BoNT-E": "Q00496",
    "Ricin": "P02879",
    "Ricinus_communis_agglutinin": "P06750",
    "Abrin-a": "P11140",
}

ACTIVE_SITES = {
    "P0DPI1": (440, 540),  # BoNT-A
    "P10844": (430, 530),  # BoNT-B
    "Q00496": (425, 525),  # BoNT-E
    "P02879": (165, 280),  # Ricin
    "P06750": (165, 280),  # R. communis agglutinin
    "P11140": (155, 265),  # Abrin-a
}


def fetch_uniprot_fasta(acc, retries=3):
    """Fetch FASTA from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.fasta"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                text = resp.read().decode('utf-8')
            return text
        except urllib.error.URLError as e:
            if attempt < retries - 1:
                print(f"  Retry {attempt+1}/{retries} after error: {e}")
                time.sleep(1)
            else:
                print(f"  FAILED to fetch {acc}: {e}")
                return None


def parse_fasta(fasta_text):
    """Parse FASTA text into (header, sequence)."""
    lines = fasta_text.strip().split('\n')
    header = lines[0] if lines else ""
    seq = ''.join(lines[1:])
    return header, seq


def download_sequences(output_dir):
    """Download all homolog sequences from UniProt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {}

    print(f"Downloading {len(ACCESSIONS)} sequences from UniProt...\n")

    for name, acc in sorted(ACCESSIONS.items()):
        print(f"  {name:30s} ({acc})...", end=" ", flush=True)
        fasta = fetch_uniprot_fasta(acc)
        if not fasta:
            print("SKIP")
            continue

        header, seq = parse_fasta(fasta)
        start, end = ACTIVE_SITES.get(acc, (1, min(100, len(seq))))

        # Save full FASTA
        fasta_file = output_dir / f"{acc}.fasta"
        with open(fasta_file, "w") as f:
            f.write(fasta)

        # Save partial FASTA (active site region)
        partial_seq = seq[start-1:end]  # 0-indexed
        partial_file = output_dir / f"{acc}_AS{start}-{end}.fasta"
        with open(partial_file, "w") as f:
            f.write(f">{acc} active site region {start}-{end}\n{partial_seq}\n")

        metadata[acc] = {
            "name": name,
            "accession": acc,
            "header": header.lstrip(">").split("|")[1],  # UniProt ID
            "length": len(seq),
            "active_site_start": start,
            "active_site_end": end,
            "active_site_length": end - start + 1,
            "fasta_file": str(fasta_file.relative_to(output_dir.parent.parent)),
            "partial_fasta_file": str(partial_file.relative_to(output_dir.parent.parent)),
            "timestamp": __import__("datetime").datetime.utcnow().isoformat(),
        }
        print("OK")

    return metadata


def load_sequences_from_cache(cache_dir):
    """Load sequences from cached FASTA files."""
    metadata = {}
    for name, acc in ACCESSIONS.items():
        fasta_file = cache_dir / f"{acc}.fasta"
        if not fasta_file.exists():
            print(f"  {name}: MISSING {fasta_file}")
            continue
        with open(fasta_file) as f:
            fasta = f.read()
        header, seq = parse_fasta(fasta)
        start, end = ACTIVE_SITES.get(acc, (1, min(100, len(seq))))
        metadata[acc] = {
            "name": name,
            "accession": acc,
            "length": len(seq),
            "active_site_start": start,
            "active_site_end": end,
            "active_site_length": end - start + 1,
            "cached": True,
        }
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Fetch homolog sequences for C probe")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--download", action="store_true",
                       help="Download fresh sequences from UniProt")
    group.add_argument("--use-local", action="store_true",
                       help="Use locally cached sequences")
    args = parser.parse_args()

    if args.download:
        metadata = download_sequences(SEQUENCES_DIR)
        with open(METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\nSaved {len(metadata)} sequences to {SEQUENCES_DIR}")
        print(f"Metadata to {METADATA_FILE}")

    elif args.use_local:
        if not METADATA_FILE.exists():
            print(f"ERROR: {METADATA_FILE} not found. Run --download first.")
            sys.exit(1)
        with open(METADATA_FILE) as f:
            metadata = json.load(f)
        print(f"Loaded {len(metadata)} cached sequences")
        for acc, info in sorted(metadata.items()):
            print(f"  {info['name']:30s} {acc} {info.get('length', '?')} aa")


if __name__ == "__main__":
    main()
