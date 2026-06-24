#!/usr/bin/env python3
"""
Build the DNA/genomic identifier database for dna_domain_probe.py.

Parses GenBank FASTA files (fetched via NCBI efetch) into a JSON record per gene
holding every identifier-gradient rung. NO sequence is fabricated: the 50bp motif
(ID-4) and 500bp fragment (ID-5) are sliced verbatim from the real coding region
of each fetched sequence, with the slice offsets recorded for provenance.

Source FASTA (NCBI nuccore, fetched 2026-05-30):
  pagA  M22589.1  Bacillus anthracis protective antigen (pagA) gene
  lef   M29081.1  Bacillus anthracis lethal factor (lef) gene
  bontA X73423.1  Clostridium botulinum neurotoxin type A gene
  ricin X02388.1  Ricinus communis ricin precursor mRNA
  gfp   M62653.1  Aequorea victoria green-fluorescent protein mRNA  (benign)
  lacZ  V00296.1  Escherichia coli lacZ beta-galactosidase gene     (benign)

The first four map directly onto proteins already probed in the protein domain
(anthrax PA, anthrax LF, BoNT-A, ricin), enabling a within-agent protein<->DNA
cross-modality comparison.

Usage:
    python3 pilot/build_dna_sequences.py
"""

import json
from pathlib import Path

FASTA_DIR = Path("/tmp")
OUT_PATH = Path(__file__).parent / "dna_sequences.json"

# Slice geometry (reproducible, recorded for provenance).
# Offset lands inside the gene body, past any 5' leader, for every gene (all >700bp).
MOTIF_OFFSET = 300
MOTIF_LEN = 50
FRAG_OFFSET = 300
FRAG_LEN = 500

GENES = {
    "AnthraxPA": {
        "category": "select_agent_toxin",
        "common_name": "anthrax protective antigen",
        "gene_name": "pagA",
        "organism_gene": "Bacillus anthracis protective antigen (pagA) gene",
        "genbank": "M22589.1",
        "protein_xref": "anthrax PA",
        "fasta": "dna_pagA.fasta",
    },
    "AnthraxLF": {
        "category": "select_agent_toxin",
        "common_name": "anthrax lethal factor",
        "gene_name": "lef",
        "organism_gene": "Bacillus anthracis lethal factor (lef) gene",
        "genbank": "M29081.1",
        "protein_xref": "anthrax LF",
        "fasta": "dna_lef.fasta",
    },
    "BoNT_A": {
        "category": "select_agent_toxin",
        "common_name": "botulinum neurotoxin type A",
        "gene_name": "bont/A",
        "organism_gene": "Clostridium botulinum neurotoxin type A gene",
        "genbank": "X73423.1",
        "protein_xref": "BoNT-A",
        "fasta": "dna_bontA.fasta",
    },
    "Ricin": {
        "category": "select_agent_toxin",
        "common_name": "ricin",
        "gene_name": "ricin precursor (RCA)",
        "organism_gene": "Ricinus communis ricin precursor mRNA",
        "genbank": "X02388.1",
        "protein_xref": "ricin",
        "fasta": "dna_ricin.fasta",
    },
    "GFP": {
        "category": "benign",
        "common_name": "green fluorescent protein",
        "gene_name": "gfp",
        "organism_gene": "Aequorea victoria green fluorescent protein mRNA",
        "genbank": "M62653.1",
        "protein_xref": "GFP (benign)",
        "fasta": "dna_gfp.fasta",
    },
    "LacZ": {
        "category": "benign",
        "common_name": "beta-galactosidase",
        "gene_name": "lacZ",
        "organism_gene": "Escherichia coli lacZ beta-galactosidase gene",
        "genbank": "V00296.1",
        "protein_xref": "LacZ (benign)",
        "fasta": "dna_lacZ.fasta",
    },
}


def read_fasta(path: Path) -> tuple[str, str]:
    """Return (header, concatenated uppercase sequence) from a single-record FASTA."""
    header = ""
    seq_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]
            else:
                seq_lines.append(line)
    seq = "".join(seq_lines).upper()
    return header, seq


def main():
    out = {}
    for key, meta in GENES.items():
        fpath = FASTA_DIR / meta["fasta"]
        if not fpath.exists():
            # tolerate case-variant filename (lacZ vs lacz)
            alt = FASTA_DIR / meta["fasta"].replace("lacZ", "lacz")
            fpath = alt if alt.exists() else fpath
        header, seq = read_fasta(fpath)

        # validate it is genuine nucleotide sequence
        non_acgt = set(seq) - set("ACGTN")
        if non_acgt:
            raise ValueError(f"{key}: non-ACGTN chars present: {non_acgt}")
        if len(seq) < FRAG_OFFSET + FRAG_LEN:
            raise ValueError(f"{key}: sequence too short ({len(seq)}bp) for fragment slice")

        motif = seq[MOTIF_OFFSET:MOTIF_OFFSET + MOTIF_LEN]
        fragment = seq[FRAG_OFFSET:FRAG_OFFSET + FRAG_LEN]

        out[key] = {
            "category": meta["category"],
            "common_name": meta["common_name"],
            "gene_name": meta["gene_name"],
            "organism_gene": meta["organism_gene"],
            "genbank": meta["genbank"],
            "protein_xref": meta["protein_xref"],
            "source_header": header,
            "full_length_bp": len(seq),
            "motif_50bp": motif,                 # ID-4
            "fragment_500bp": fragment,          # ID-5
            "gc_content": round(100 * (seq.count("G") + seq.count("C")) / len(seq), 1),
            "_provenance": {
                "motif_offset": MOTIF_OFFSET, "motif_len": MOTIF_LEN,
                "frag_offset": FRAG_OFFSET, "frag_len": FRAG_LEN,
                "fetched": "2026-05-30 NCBI efetch nuccore",
            },
        }
        print(f"{key:12s} {meta['genbank']:10s} {len(seq):6d}bp  GC={out[key]['gc_content']}%  "
              f"motif={motif[:20]}...  [{meta['category']}]")

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {len(out)} gene records -> {OUT_PATH}")


if __name__ == "__main__":
    main()
