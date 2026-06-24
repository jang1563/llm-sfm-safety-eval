# Security Policy

## Repository status

This repository contains defensive AI safety research materials: evaluation
design, aggregate pilot outputs, and scripts that call external model APIs. It is
measurement and analysis material, not operational content.

## Secrets

Never commit credentials. Scripts read credentials from environment variables:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `HF_TOKEN`

Use `.env.example` as a local template and keep real `.env` files untracked.

## Sensitive-content handling

Do not add operational biological protocols, synthesis routes, production steps,
or capability-uplift instructions. Repository content stays at the level of
measurement design, classifier behavior, analysis, and safety-framework coverage.

Before adding any file, review it for:

- API keys, account identifiers, or local paths that should not leave the machine.
- Raw model outputs that include actionable content.
- Any material that moves from safety evaluation into operational guidance.

### Chemical and biological identifiers

The evaluation stimuli include public identifiers at several specificity levels
(names, database accessions, registry numbers, PDB IDs, SMILES). These are the
experimental variable under study (the identifier gradient) and are retained as
non-actionable public references:

- **SMILES / InChI / CID / CAS** (including scheduled-agent structures such as VX,
  sarin, and sulfur mustard) are public structural and registry identifiers, not
  synthesis routes: a SMILES string encodes 2-D structure only and does not
  provide a synthesis pathway. They are kept in the harness configs because they
  are the ID-4/ID-6 stimuli required to reproduce the chemistry findings.
- **Nucleotide and protein sequences** are handled more conservatively:
  full-length toxin sequences can be submitted directly to gene-synthesis
  providers, so resolved sequences are git-ignored and reconstructed on demand
  from public GenBank accessions (`pilot/build_dna_sequences.py`); only accession
  lists ship in the tree.

In all cases the gated material is the raw response corpus and any resolved
sequences; aggregate labels and metrics (`pilot/results_public/`) and public
identifiers are retained.

## Reporting

Report a security or safety concern privately to the maintainer, JangKeun Kim
(jak4013@med.cornell.edu), rather than opening a public issue or publishing an
example. Coordinated, responsible disclosure is appreciated.
