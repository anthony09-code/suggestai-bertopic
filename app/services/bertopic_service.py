import hashlib
import json
import logging
from collections import defaultdict

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from app.services.cache_service import cache_service
from app.services.label_service import LabelService
from app.utils.stopwords import get_combined_stopwords
from bertopic import BERTopic

logger = logging.getLogger(__name__)


class BertopicService:
    def __init__(self):
        try:
            self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

            vectorizer = CountVectorizer(stop_words=get_combined_stopwords())

            self.model = BERTopic(
                embedding_model=self.embedding_model,
                vectorizer_model=vectorizer,
                min_topic_size=3,
                verbose=True,
            )
            self.label_service = LabelService()

        except Exception as e:
            logging.error(f"Failed to initialize BertopicService: {e}")
            raise RuntimeError(f"BertopicService initialization failed: {e}")

    def _make_cache_key(self, office_id: str, documents: list[str]) -> str:
        content = office_id + json.dumps(sorted(documents))
        return "topics_" + hashlib.md5(content.encode()).hexdigest()

    def get_enhanced_label(self, topic_id: int):
        try:
            words = self.model.get_topic(topic_id)
            if not words:
                return "Unknown Topic"

            keywords = [w[0] for w in words[:5]]
            return self.label_service.generate_label(keywords)

        except Exception as e:
            logging.error(f"Failed to get enhanced label for topic {topic_id}: {e}")
            return "Unknown Topic"

    def fit_topics(self, office_id: str, documents: list[str]):
        if not office_id or not office_id.strip():
            raise ValueError("office_id is required")
        if not documents:
            raise ValueError("Documents list cannot be empty")
        if len(documents) < 5:
            raise ValueError(f"At least 5 documents required, got {len(documents)}")
        if any(not doc.strip() for doc in documents):
            raise ValueError("Documents cannot contain empty or blank strings")

        cache_key = self._make_cache_key(office_id, documents)

        try:
            if cache_service.exists(cache_key):
                logger.info(f"[Cache HIT] Returning cached topics for {office_id}")
                return cache_service.get(cache_key)

        except Exception as e:
            logger.warning(f"Cache read failed, proceeding without cache: {e}")

        logger.info(f"[Cache MISS] Running BERTopic for {office_id}")
        try:
            topics, probs = self.model.fit_transform(documents)

        except Exception as e:
            logger.error(f"BERTopic fit_transform failed: {e}")
            raise RuntimeError("Topic modeling failed. Please try again.")

        meaningful_topics = [t for t in topics if t != -1]
        if not meaningful_topics:
            raise ValueError(
                "Could not extract meaningful topics. "
                "Try providing more diverse or longer documents."
            )

        topic_groups = defaultdict(list)
        for doc, topic_id in zip(documents, topics):
            topic_groups[topic_id].append(doc)

        results = []
        for topic_id, docs in topic_groups.items():
            if topic_id == -1:
                label = "Uncategorized"
                keywords = []
            else:
                try:
                    words = self.model.get_topic(topic_id)
                    keywords = [w[0] for w in words[:5]] if words else []
                    label = self.get_enhanced_label(topic_id)

                except Exception as e:
                    logger.warning(f"Failed to process topic {topic_id}: {e}")
                    keywords = []
                    label = "Unlabeled Topic"

            results.append(
                {
                    "topic_id": topic_id,
                    "label": label,
                    "keywords": keywords,
                    "feedback_count": len(docs),
                }
            )

        result = {
            "topics": results,
            "results": [
                {
                    "topic_id": topic_id,
                    "cleaned_text": doc,
                    "translated_text": None,
                    "summary": None,
                    "confidence_score": float(probs[i])
                    if probs is not None and i < len(probs)
                    else 0.0,
                }
                for i, (doc, topic_id) in enumerate(zip(documents, topics))
            ],
            "message": f"Successfully analyzed {len(documents)} documents for office {office_id}",
        }

        try:
            cache_service.set(cache_key, result)

        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

        return result
