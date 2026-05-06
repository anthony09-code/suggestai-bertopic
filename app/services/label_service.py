import requests

from app.core.config import settings
from app.prompts.topic_label_prompt import topic_label_prompt
from app.services.cache_service import cache_service


class LabelService:
    def __init__(self):
        self.api_key = settings.XAI_API_KEY
        self.base_url = "https://api.x.ai/v1/chat/completions"

    def generate_label(self, keywords: list[str]) -> str:
        prompt = topic_label_prompt(keywords)

        cache_key = "_".join(sorted(keywords))

        if cache_service.exists(cache_key):
            return cache_service.get(cache_key)

        label = self.call_xai_api(prompt)

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
