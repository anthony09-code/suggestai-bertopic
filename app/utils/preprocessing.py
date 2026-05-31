import re
import unicodedata

_REPEATED_CHARS = re.compile(r"(.)\1{2,}")

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_MENTION = re.compile(r"@\w+")

_EMOJI = re.compile(
    "[\U00010000-\U0010ffff"
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "\u2600-\u26ff\u2700-\u27bf]+",
    flags=re.UNICODE,
)

_NOISE = re.compile(r"[^a-zA-ZÀ-ÿ\s']")

_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    if not text or not text.strip():
        return ""

    text = unicodedata.normalize("NFKC", text)

    text = _URL.sub(" ", text)
    text = _EMAIL.sub(" ", text)
    text = _MENTION.sub(" ", text)

    text = _EMOJI.sub(" ", text)

    text = _REPEATED_CHARS.sub(r"\1\1", text)

    text = text.lower()

    text = _NOISE.sub(" ", text)

    text = _WHITESPACE.sub(" ", text).strip()

    return text
