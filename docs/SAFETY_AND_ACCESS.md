# Safety And Access

## Classification

This is private defensive AI safety research. It contains evaluation methodology, model-output analysis, and scientific identifiers used to measure safety-recognition boundaries.

Access should be limited to collaborators who need the material for safety evaluation, manuscript development, or authorized application/research review.

## Handling Principles

1. Keep the repository private.
2. Keep credentials outside git.
3. Keep outputs at the level of evaluation and analysis.
4. Do not add operational biological instructions.
5. Review raw model outputs before copying them into memos or public-facing artifacts.

## Release Review

Before any external sharing, check:

- Is the target audience authorized for the full context?
- Does the file include raw outputs that could be more operational than the evaluation requires?
- Are local paths, credentials, account names, or cluster details exposed unnecessarily?
- Can the same point be made with an aggregate statistic or sanitized excerpt?

## Safe Writing Standard

Prefer language like:

- "recognition boundary"
- "coverage gap"
- "safety classifier behavior"
- "measurement axis"
- "defensive evaluation"

Avoid language that turns an evaluation result into a how-to description.

## Incident Response

If a credential or unsafe artifact is committed:

1. Stop pushing or syncing.
2. Rotate the credential if applicable.
3. Remove the artifact in a new commit.
4. If it reached a remote, treat the remote as exposed and coordinate cleanup with the owner.
