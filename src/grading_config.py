# Grading and retry configuration for src/grading.py.
#
# MAX_RETRIES caps the query-rewrite retry loop per the PRD's capped-retry
# decision: retrieval failures trigger one rewrite and one retry, never more,
# so an honest "not found" beats an uncontrolled retrieval loop.
MAX_RETRIES = 2

# Minimum score a chunk must clear to be considered relevant during grading.
# grade_chunk() currently grades true/false per dimension rather than a
# score, so this is unused by today's grading.py but is imported by it —
# kept here as the threshold to wire in if grading moves to scored output.
RELEVANCE_THRESHOLD = 0.7
