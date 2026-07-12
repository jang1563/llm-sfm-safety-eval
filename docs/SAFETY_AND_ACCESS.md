# Safety and Public-Release Scope

## Release Status

This is a public defensive AI safety-evaluation repository. It is the reviewed
public subset prepared for disclosure after 30 June 2026. Public availability
does not change the repository's narrow scope: it measures safety-recognition
boundaries and does not provide operational biological guidance.

## Included Material

The public release may contain:

- measurement specifications and evaluation code;
- public scientific identifiers needed to reproduce the identifier-gradient
  experiments;
- redacted, aggregate-only outcomes and summary statistics; and
- documentation needed to interpret and reproduce those aggregates.

## Withheld Material

The following do not belong in the public repository:

- raw prompts or model responses;
- resolved nucleotide or protein sequences for controlled or high-risk agents;
- wet-lab protocols, synthesis routes, production steps, or capability-uplift
  instructions;
- credentials, account identifiers, private infrastructure paths, scheduler
  logs, or unpublished collaborator material; and
- any artifact that cannot be reviewed safely at the aggregate level.

## Handling Principles

1. Keep credentials and local runtime configuration outside git.
2. Keep published outputs aggregate-only and free of prompt or response text.
3. Do not add operational biological instructions or materialized hazardous
   sequences.
4. Review generated artifacts before moving them into the public tree.
5. Prefer the least detailed artifact that still supports the scientific claim.

## Public-Release Review

Before committing or releasing a file, check:

- Does it contain raw model text, hidden prompt fields, or materialized
  sequences?
- Does it expose credentials, account names, local paths, cluster details, or
  unpublished collaborator information?
- Can the same result be represented by a label, count, rate, confidence
  interval, or sanitized excerpt?
- Does the file remain clearly defensive and non-operational when read without
  project context?

Material that fails any check should remain outside this repository. Security or
safety concerns should be reported privately through the channel in
[`SECURITY.md`](../SECURITY.md).

## Safe Writing Standard

Prefer language such as "recognition boundary," "coverage gap," "safety
classifier behavior," "measurement axis," and "defensive evaluation." Avoid
language that turns an evaluation result into a how-to description.

## Incident Response

If a credential or unsafe artifact is committed:

1. Stop publishing new changes.
2. Rotate the credential if applicable.
3. Remove the artifact from the current tree.
4. If it reached a remote, treat it as exposed and coordinate history cleanup
   with the repository owner.
