"""
Stopwords for BERTopic analysis of Filipino academic feedback.

Sources:
- English: sklearn's built-in English stopwords
- Filipino/Tagalog: stopwords-iso/stopwords-tl
  https://github.com/stopwords-iso/stopwords-tl
- Domain-specific: informal Tagalog, Taglish, and academic feedback noise
  tuned from observed feedback patterns
"""

import logging
import urllib.request
from functools import lru_cache

from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger(__name__)

_ISO_TAGALOG_URL = "https://raw.githubusercontent.com/stopwords-iso/stopwords-tl/master/stopwords-tl.txt"

# Informal / colloquial Tagalog and Taglish not covered by the ISO list
_COLLOQUIAL_TAGALOG = [
    # Pronouns and particles
    "yung",
    "yun",
    "yan",
    "yon",
    "nya",
    "niya",
    "kasi",
    "lang",
    "rin",
    "din",
    "nga",
    "naman",
    "daw",
    "raw",
    "po",
    "ho",
    "opo",
    "oho",
    # Discourse markers and fillers
    "talaga",
    "pala",
    "siguro",
    "medyo",
    "sobra",
    "grabe",
    "tapos",
    "kaya",
    "lagi",
    "palagi",
    "minsan",
    "madalas",
    "sige",
    "okay",
    "ok",
    "nung",
    "pag",
    "kapag",
    "tska",
    "tsaka",
    # Tagalog verb roots and affixes that leak through as tokens
    "mag",
    "nag",
    "ma",
    "na",
    "magturo",
    "turo",
    "nagturo",
    "ituro",
    "galing",
    "magaling",
    "husay",
    "mabuti",
    "naibigay",
    "godbless",
    # Laugh/reaction tokens from actual feedback
    "hehe",
    "haha",
    "hihi",
    "huhu",
    "hahaha",
    "hahahaha",
    "hahahahahahahahahahahahaha",
    "lol",
    "lols",
    # Informal English common in Taglish writing
    "gonna",
    "gotta",
    "wanna",
    "yeah",
    "yep",
    "nope",
    "hi",
    "hello",
    "miss",
    "missed",
    "share",
    "shared",
]

# Academic feedback noise — words too generic to carry topic signal
_DOMAIN_NOISE = [
    # Generic positive adjectives
    "good",
    "great",
    "nice",
    "best",
    "better",
    "well",
    "very",
    "really",
    "much",
    "more",
    "also",
    "like",
    "just",
    "even",
    "still",
    "every",
    "many",
    "lot",
    "lots",
    "bit",
    "quite",
    # Academic roles and concepts (appear in almost every feedback)
    "teacher",
    "instructor",
    "professor",
    "subject",
    "class",
    "course",
    "student",
    "students",
    "school",
    "learning",
    "teaching",
    "teach",
    "learn",
    "learned",
    "taught",
    "lessons",
    "lesson",
    # Gratitude expressions (flood every topic)
    "thank",
    "thanks",
    "thankyou",
    "thnak",
    "thanm",
    "thankyouu",
    "thankyousomuch",
    "tenkyouuusomuch",
    "appreciate",
    "appreciated",
    "appreciation",
    # Blessings
    "god",
    "bless",
    "godbless",
    # Honorifics
    "sir",
    "mam",
    "maam",
    "madam",
    "npo",
    # Noise tokens observed in actual data
    "mo",
    "ja",
    "ve",
    "om",
    "alwa",
    "youuuu",
    "youuuuuu",
    "ammmmm",
    "meee",
    "alot",
    "learnd",
    "gonna",
    "goodwork",
]


def _fetch_iso_tagalog_stopwords() -> list[str]:
    """Fetch the stopwords-iso Tagalog list. Falls back to empty list on failure."""
    try:
        with urllib.request.urlopen(_ISO_TAGALOG_URL, timeout=10) as response:
            lines = response.read().decode().splitlines()
            return [
                w.strip().lower() for w in lines if w.strip() and not w.startswith("#")
            ]
    except Exception as e:
        logger.warning(
            f"Failed to fetch ISO Tagalog stopwords, using local fallback: {e}"
        )
        return []


_ISO_TAGALOG_FALLBACK = [
    "akin",
    "aking",
    "ako",
    "alin",
    "amin",
    "aming",
    "ang",
    "ano",
    "anumang",
    "apat",
    "at",
    "atin",
    "ating",
    "ay",
    "bago",
    "bakit",
    "bawat",
    "bilang",
    "dahil",
    "dalawa",
    "dapat",
    "din",
    "dito",
    "doon",
    "gayunman",
    "ginagawa",
    "ginawa",
    "gusto",
    "habang",
    "hanggang",
    "hindi",
    "huwag",
    "iba",
    "ibaba",
    "ibabaw",
    "ibang",
    "ikaw",
    "ilagay",
    "ilalim",
    "ilan",
    "ito",
    "iyon",
    "kami",
    "kanila",
    "kanilang",
    "kanino",
    "kanya",
    "kanyang",
    "kapag",
    "kapwa",
    "katulad",
    "kaya",
    "kaysa",
    "ko",
    "komo",
    "kong",
    "kulang",
    "kung",
    "lahat",
    "lalo",
    "lamang",
    "likod",
    "lima",
    "maaari",
    "maging",
    "mahusay",
    "makita",
    "marami",
    "mataas",
    "mga",
    "minsan",
    "mismo",
    "mula",
    "muli",
    "na",
    "nabanggit",
    "naging",
    "nais",
    "namin",
    "napaka",
    "narito",
    "nasaan",
    "natin",
    "nawa",
    "ng",
    "ngayon",
    "ni",
    "nila",
    "nilang",
    "nito",
    "niya",
    "niyaang",
    "noon",
    "o",
    "pa",
    "paano",
    "pababa",
    "paggawa",
    "pakiramdam",
    "pala",
    "palagi",
    "paminsan",
    "para",
    "paraan",
    "pareho",
    "pati",
    "patuloy",
    "pero",
    "pumunta",
    "roon",
    "sa",
    "saan",
    "sama",
    "samantala",
    "sila",
    "sino",
    "siya",
    "tatlo",
    "tayo",
    "tulad",
    "tungkol",
    "una",
    "walang",
    "wala",
]


@lru_cache(maxsize=1)
def get_combined_stopwords() -> list[str]:
    """
    Returns a deduplicated sorted list of stopwords combining:
    - sklearn English stopwords
    - ISO Tagalog stopwords (fetched from stopwords-iso/stopwords-tl)
    - Colloquial Tagalog / Taglish
    - Domain-specific academic feedback noise
    """
    english = list(CountVectorizer(stop_words="english").get_stop_words())
    iso_tagalog = _fetch_iso_tagalog_stopwords() or _ISO_TAGALOG_FALLBACK

    combined = (
        set(english) | set(iso_tagalog) | set(_COLLOQUIAL_TAGALOG) | set(_DOMAIN_NOISE)
    )

    logger.info(
        f"Stopwords loaded: {len(english)} English + {len(iso_tagalog)} ISO Tagalog "
        f"+ {len(_COLLOQUIAL_TAGALOG)} colloquial + {len(_DOMAIN_NOISE)} domain "
        f"= {len(combined)} unique"
    )

    return sorted(combined)
