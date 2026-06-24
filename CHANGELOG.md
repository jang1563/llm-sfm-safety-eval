# Changelog

Notable milestones for the LLM x SFM safety evaluation. See [`README.md`](README.md)
for repository structure and `git log` for full detail.

## Unreleased

### Evaluation
- Cross-modal safety evaluation across three hazard domains (protein, chemistry,
  genomic) on the LLM x SFM interpretation channel.
- System-card-format supplement summarizing the cross-modal findings
  ([`System_Card_Supplement_CrossModal_SFM_Safety.md`](System_Card_Supplement_CrossModal_SFM_Safety.md)).
- Open-weight comparison panel (3.8B to 111B parameters, 7 vendors) isolating the
  safety layer as managed-access infrastructure rather than model scale.

### Findings
- **Classifier pendulum:** refusal swings discontinuously across one model update
  (Sonnet 4 to Sonnet 4.6) and is stable across three later releases.
- **Calibration gradient:** from genuine sequence-hazard recognition (protein),
  through keyword-amplified registry recognition (chemistry), to context-keyword
  over-refusal (genomic).
- **Managed-access, not scale:** open-weight panels produce no genuine refusals
  while the managed deployment does, at comparable model capability.
- **Tool-mode trigger surface:** Sonnet 4.6's tool-mode classifier is
  keyword-triggered; Opus 4.8 shifts to context-sensitive classification at the
  same aggregate refusal rate.

### Safety and hygiene
- Raw response corpus and resolved select-agent sequences are gated out of the
  shareable tree; only redacted, aggregate-only outputs ship under
  `pilot/results_public/`.
- The redactor (`pilot/redact_results_for_public.py`) genericizes absolute paths
  and drops response and prompt content keys.
