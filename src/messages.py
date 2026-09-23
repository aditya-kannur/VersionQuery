"""
Static user-facing message strings.
Used by src/router.py.
"""

CLARIFICATION_QUESTION = "Which API version are you using? (e.g. 2022-06-28)"

NOT_FOUND_MESSAGE = "Not found in docs for this version."

SERVICE_UNAVAILABLE_MESSAGE = (
    "The AI provider could not complete this request. Please try again shortly "
    "or check the configured Groq API key/model."
)

INVALID_VERSION_MESSAGE = (
    "I couldn't find API version {version} in this documentation set. "
    "Try one of: {supported_versions}."
)

LATEST_VERSION_MESSAGE = "The latest API version covered by these docs is {version}."
