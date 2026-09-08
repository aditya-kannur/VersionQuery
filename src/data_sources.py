"""
Static mapping of dataset files to their doc type.
No logic here — just paths and labels, used by chunkers/ingestion later.
"""

DATA_SOURCES = [
    {
        "path": "data/changelog/changelog.md",
        "doc_type": "changelog",
    },
    {
        "path": "data/changelog/historical-changelog.md",
        "doc_type": "changelog",
    },
    {
        "path": "data/migrations/2021-05-13.md",
        "doc_type": "migration",
        "version": "2021-05-13",
    },
    {
        "path": "data/migrations/2021-08-16.md",
        "doc_type": "migration",
        "version": "2021-08-16",
    },
    {
        "path": "data/migrations/2022-02-22.md",
        "doc_type": "migration",
        "version": "2022-02-22",
    },
    {
        "path": "data/migrations/2022-06-28.md",
        "doc_type": "migration",
        "version": "2022-06-28",
    },
    {
        "path": "data/migrations/2025-09-03.md",
        "doc_type": "migration",
        "version": "2025-09-03",
    },
    {
        "path": "data/migrations/2026-03-11.md",
        "doc_type": "migration",
        "version": "2026-03-11",
    },
    {
        "path": "data/reference/openapi.json",
        "doc_type": "reference",
    },
]
