# VersionQuery — Demo Script

> DRAFT. The plan specifies these talking points should be mentor-provided
> and pasted verbatim. That text doesn't exist yet — this is a starting
> point grounded in what's actually built and verified, to replace once
> the real one arrives.

## Opening (30s)

"VersionQuery answers developer questions about the Notion API strictly
from versioned documentation — never from general knowledge, never mixing
versions, and it says so honestly when it can't find an answer instead of
guessing."

## Demo 1 — Reference lookup, happy path

Ask: *"How do I query a data source in Notion-Version 2026-03-11?"*

Talking point: "The answer comes from the reference doc for exactly this
version, with a citation you can check yourself — document, section,
version."

## Demo 2 — The version boundary (this is the core claim)

Ask the same style of question at a version outside the reference's
coverage: *"How do I query a database in Notion-Version 2022-06-28?"*

Talking point: "This is the important one. The reference documentation
only covers 2025-09-03 and 2026-03-11 — that's a real limitation of the
dataset, not a bug. A generic RAG system would still return an answer
here, because something in the corpus is semantically similar to the
question. VersionQuery won't — the hard version filter runs *before*
scoring, so a wrong-version chunk is never even a candidate. It says 'Not
found in docs for this version' instead."

## Demo 3 — Missing version, clarification

Ask: *"How do I query a database?"*

Talking point: "No version given, and the question depends on one — so it
asks instead of assuming the latest version, which is exactly the failure
mode this system is built to avoid."

## Demo 4 — Multi-hop migration

Ask: *"How do I upgrade from 2021-08-16 to 2022-06-28?"*

Talking point: "There's no direct guide between these two versions — the
system finds 2022-02-22 sitting between them and chains both hops in
order, without ever inventing a version that doesn't exist in the
dataset."

## Closing (15s)

"Every answer is grounded, every citation is checkable, and every failure
is honest. That's the whole premise."
