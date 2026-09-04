# Changelog

Notable milestones for the LLM x SFM safety evaluation. See [`README.md`](README.md)
for repository structure and `git log` for full detail.

## Unreleased

### Documentation
- **Scoping added to two headline findings so they can be recomputed from the
  released table** (2026-09-04). Neither result changed; both were stated without
  the basis a reader needs to reproduce them.
  - `refusal` in `data/llm_sfm_refusal_trials.jsonl` is **detector-level** and
    recomputes to 326/23,100 = 1.41% across the open-weight rows, spread over 13
    of 14 configurations at 0.00-2.95%. The "0% genuine refusals" figure is a
    classification of that text as capability disclaimer rather than safety
    refusal, and the response text is not released, so the split cannot be
    re-derived here. Both numbers are now named in the README and the supplement.
  - "Naming the hazard agents 98-100%" is measured at `ID-0`. Chemistry is
    99.8% and 100%; DNA select-agent toxins pool to 89.6% (806/900) because one
    outlier model sits at 14% while seven of nine are at exactly 100% and the
    median model is 100%. The DNA phrasing is corrected to "eight of nine models
    at 92% or above, one that does not recognize these agents by name."
  - Noted that the classifier-pendulum sequence **is not in this table**, which
    contains one managed configuration (`claude-sonnet-4-6`, protein, 1,200 rows).

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
- Documented the public disclosure scope after 30 June 2026 and replaced the
  obsolete private/authorized-review classification with explicit included and
  withheld boundaries.
- Removed audience-specific wording, session handoff notes, user-specific
  cluster paths, and local credential-file assumptions from the public tree.
- Raw response corpus and resolved select-agent sequences are gated out of the
  shareable tree; only redacted, aggregate-only outputs ship under
  `pilot/results_public/`.
- The redactor (`pilot/redact_results_for_public.py`) genericizes absolute paths
  and drops response and prompt content keys.
