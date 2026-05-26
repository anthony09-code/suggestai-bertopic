import logging

import requests
from keybert import KeyBERT

from app.core.config import settings
from app.prompts.topic_label_prompt import topic_label_prompt
from app.services.cache_service import cache_service

logger = logging.getLogger(__name__)


class LabelService:
    def __init__(self):
        self.api_key = settings.XAI_API_KEY
        self.base_url = "https://api.x.ai/v1/chat/completions"
        self._kw_model = None

    @property
    def kw_model(self):
        if self._kw_model is None:
            logger.info("Loading KeyBERT model for local label fallback...")
            self._kw_model = KeyBERT(model="all-MiniLM-L6-v2")
        return self._kw_model

    def generate_label(self, keywords: list[str]) -> str:
        prompt = topic_label_prompt(keywords)

        cache_key = "_".join(sorted(keywords))

        if cache_service.exists(cache_key):
            return cache_service.get(cache_key)

        try:
            label = self.call_xai_api(prompt)
        except Exception as e:
            logger.warning(f"xAI API failed, falling back to KeyBERT: {e}")
            label = self._local_label(keywords)

        cache_service.set(cache_key, label)

        return label

    def call_xai_api(self, prompt: str) -> str:
        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-3",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"].strip()

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
