# Maps each dataset file to its doc type, for the ingestion pipeline.
DATA_SOURCES = [
    {"path": "data/reference/openapi.json", "doc_type": "reference"},
    {"path": "data/changelog/changelog.md", "doc_type": "changelog"},
    {"path": "data/changelog/historical-changelog.md", "doc_type": "changelog"},
    {"path": "data/migrations/2021-05-13.md", "doc_type": "migration"},
    {"path": "data/migrations/2021-08-16.md", "doc_type": "migration"},
    {"path": "data/migrations/2022-02-22.md", "doc_type": "migration"},
    {"path": "data/migrations/2022-06-28.md", "doc_type": "migration"},
    {"path": "data/migrations/2025-09-03.md", "doc_type": "migration"},
    {"path": "data/migrations/2026-03-11.md", "doc_type": "migration"},
]
