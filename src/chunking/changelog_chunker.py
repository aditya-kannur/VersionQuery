import json
import re

# Matches <Update label="...">...</Update> blocks, across multiple lines.
# re.DOTALL: lets '.' match newlines too (body spans multiple lines).
# .*? (non-greedy): stops at the FIRST </Update>, not the last one in the file.
UPDATE_PATTERN = re.compile(
    r'<Update label="([^"]+)">(.*?)</Update>',
    re.DOTALL
)

# Matches a '###' header line. Same shape as Day 2's '##' pattern,
# but one level deeper. The headers are indented (they sit inside the
# <Update> block), so ^\s* actually allows that leading whitespace —
# without it the pattern never matches and every release becomes one
# giant unsplit chunk.
SUBHEADER_SPLIT_PATTERN = re.compile(r'^\s*###\s+(.+)$', re.MULTILINE)


def split_by_subheader(body: str):
    """
    Day 2's '##' splitter, reused at the '###' level.
    Returns a list of (section_title, section_text) tuples.
    If there are no '###' headers, returns [(None, whole_body)].
    """
    parts = SUBHEADER_SPLIT_PATTERN.split(body)

    if len(parts) == 1:
        # No '###' headers found — split() returns the original string
        # untouched in a list of length 1. Whole block is one chunk.
        return [(None, body.strip())]

    # parts = [intro(discard), title1, text1, title2, text2, ...]
    # Same alternating shape as Day 2 — loop in pairs, starting after index 0.
    sections = []
    for i in range(1, len(parts), 2):
        title = parts[i]
        text = parts[i + 1].strip()
        sections.append((title, text))
    return sections


def chunk_changelog(text: str, doc_type: str = "changelog"):
    """
    Shared chunker for changelog.md and historical-changelog.md.
    doc_type is passed in by the caller so both files can reuse this
    function but still tag their chunks correctly if you ever need to
    tell them apart later.

    Returns a flat list of dicts — doc_type, release_date and text all at
    the top level, matching the shape chunk_migration_file() and
    chunk_reference_file() return, so retrieval_pipeline.load_all_chunks()
    can treat every chunk uniformly regardless of which chunker produced it.
    """
    chunks = []

    for match in UPDATE_PATTERN.finditer(text):
        release_date = match.group(1)   # the label attribute
        body = match.group(2)           # everything between the tags

        for section_title, section_text in split_by_subheader(body):
            if not section_text:
                continue  # skip empty sections defensively

            chunk_text = section_text
            if section_title:
                # keep the ### title attached to its own chunk's text
                chunk_text = f"{section_title}\n\n{section_text}"

            chunks.append({
                "doc_type": doc_type,
                "release_date": release_date,
                "section": section_title,
                "text": chunk_text,
            })

    return chunks


if __name__ == "__main__":
    all_chunks = []
    for filename in ("data/changelog/changelog.md", "data/changelog/historical-changelog.md"):
        with open(filename, "r", encoding="utf-8") as f:
            all_chunks.extend(chunk_changelog(f.read(), doc_type="changelog"))

    print(f"Produced {len(all_chunks)} chunks from 2 changelog files")
    with open("data/chunks_changelog.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
