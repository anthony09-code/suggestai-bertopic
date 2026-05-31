import logging
import re

import requests
from keybert import KeyBERT

from app.core.config import settings
from app.prompts.topic_label_prompt import topic_label_prompt
from app.services.cache_service import cache_service

logger = logging.getLogger(__name__)

# Matches any label that looks like the model leaked reasoning
_REASONING_LEAK = re.compile(
    r"(\n|because|keyword|cluster|label|phrase|topic)", re.IGNORECASE
)


class LabelService:
    def __init__(self):
        self.api_key = settings.XAI_API_KEY
        self.base_url = "https://api.x.ai/v1/chat/completions"
        self._kw_model: KeyBERT | None = None

    @property
    def kw_model(self) -> KeyBERT:
        if self._kw_model is None:
            logger.info("Loading KeyBERT model for local label fallback...")
            self._kw_model = KeyBERT(model="all-MiniLM-L6-v2")
        return self._kw_model

    def generate_label(self, keywords: list[str]) -> str:
        keywords = [k for k in keywords if k.strip()]

        if not keywords:
            return "Uncategorized"

        cache_key = "label_" + "_".join(sorted(keywords))

        try:
            if cache_service.exists(cache_key):
                return cache_service.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")

        try:
            label = self._call_xai_api(topic_label_prompt(keywords))
            label = self._sanitize(label, keywords)
        except Exception as e:
            logger.warning(f"xAI API failed, falling back to KeyBERT: {e}")
            label = self._local_label(keywords)

        try:
            cache_service.set(cache_key, label)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

        return label

    def _call_xai_api(self, prompt: str) -> str:
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-3",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,  # lower = less creative, more predictable
                "max_tokens": 20,  # hard cap — a 5-word label needs ~10 tokens max
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    def _sanitize(self, label: str, keywords: list[str]) -> str:
        """
        Guard against the model leaking reasoning into the label.
        Falls back to local label if the response looks suspicious.
        """
        # Take only the first line in case the model added explanation after
        label = label.splitlines()[0].strip()

        # Strip surrounding quotes
        label = label.strip("\"'")

        # If it's too long or contains reasoning words, fall back
        words = label.split()
        if len(words) > 6 or _REASONING_LEAK.search(label):
            logger.warning(f"Suspicious label from xAI, falling back: {label!r}")
            return self._local_label(keywords)

        return label.title()

    def _local_label(self, keywords: list[str]) -> str:
        if not keywords:
            return "Unknown Topic"

        try:
            text = " ".join(keywords)
            results = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words="english",
                top_n=1,
            )
            if results:
                return results[0][0].title()
        except Exception as e:
            logger.warning(f"KeyBERT fallback also failed: {e}")

        return keywords[0].replace("_", " ").title()
