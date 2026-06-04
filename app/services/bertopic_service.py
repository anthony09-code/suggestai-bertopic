import hashlib
import json
import logging
from collections import defaultdict

from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from app.services.cache_service import cache_service
from app.services.label_service import LabelService
from app.utils.preprocessing import clean_text
from app.utils.stopwords import get_combined_stopwords
from bertopic import BERTopic

logger = logging.getLogger(__name__)


class BertopicService:
    def __init__(self):
        try:
            self.embedding_model = SentenceTransformer(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )

            umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            )

            hdbscan_model = HDBSCAN(
                min_cluster_size=5,
                min_samples=2,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )

            vectorizer = CountVectorizer(
                stop_words=get_combined_stopwords(),
                token_pattern=r"(?u)\b[a-zA-Z]\w{2,}\b",
                min_df=2,
                ngram_range=(1, 2),
            )

            self.model = BERTopic(
                embedding_model=self.embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer,
                min_topic_size=5,
                nr_topics="auto",
                verbose=True,
                calculate_probabilities=True,
            )

            self.label_service = LabelService()

        except Exception as e:
            logging.error(f"Failed to initialize BertopicService: {e}")
            raise RuntimeError(f"BertopicService initialization failed: {e}")

    def _make_cache_key(
        self, office_id: str, documents: list[str], feedback_ids: list[str]
    ) -> str:
        content = (
            "v4" + office_id + json.dumps(sorted(documents)) + json.dumps(feedback_ids)
        )
        return "topics_" + hashlib.md5(content.encode()).hexdigest()

    def get_enhanced_label(self, topic_id: int) -> str:
        try:
            words = self.model.get_topic(topic_id)
            if not words:
                return "Unknown Topic"
            keywords = [w[0] for w in words[:5]]
            return self.label_service.generate_label(keywords)
        except Exception as e:
            logging.error(f"Failed to get enhanced label for topic {topic_id}: {e}")
            return "Unknown Topic"

    def fit_topics(
        self, office_id: str, documents: list[str], feedback_ids: list[str]
    ) -> dict:
        if not office_id or not office_id.strip():
            raise ValueError("office_id is required")
        if not documents:
            raise ValueError("Documents list cannot be empty")
        if len(documents) < 5:
            raise ValueError(f"At least 5 documents required, got {len(documents)}")

        pairs = [(clean_text(doc), fid) for doc, fid in zip(documents, feedback_ids)]
        pairs = [(doc, fid) for doc, fid in pairs if doc]

        if len(pairs) < 5:
            raise ValueError(
                "Too many documents were empty after cleaning. "
                "Provide at least 5 non-empty documents."
            )

        cleaned_docs, cleaned_ids = zip(*pairs)
        cleaned_docs = list(cleaned_docs)
        cleaned_ids = list(cleaned_ids)

        cache_key = self._make_cache_key(office_id, cleaned_docs, cleaned_ids)

        try:
            if cache_service.exists(cache_key):
                logger.info(f"[Cache HIT] Returning cached topics for {office_id}")
                return cache_service.get(cache_key)
        except Exception as e:
            logger.warning(f"Cache read failed, proceeding without cache: {e}")

        logger.info(
            f"[Cache MISS] Running BERTopic for {office_id} "
            f"({len(cleaned_docs)} docs after cleaning)"
        )

        try:
            topics, probs = self.model.fit_transform(cleaned_docs)
            topics = self.model.reduce_outliers(
                cleaned_docs, topics, strategy="probabilities"
            )
        except Exception as e:
            logger.error(f"BERTopic fit_transform failed: {e}")
            raise RuntimeError("Topic modeling failed. Please try again.")

        if not any(t != -1 for t in topics):
            raise ValueError(
                "Could not extract meaningful topics. "
                "Try providing more diverse or longer documents."
            )

        topic_groups: dict[int, list[str]] = defaultdict(list)
        for doc, topic_id in zip(cleaned_docs, topics):
            topic_groups[topic_id].append(doc)

        topic_summaries = []
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

            topic_summaries.append(
                {
                    "topic_id": topic_id,
                    "label": label,
                    "keywords": [k for k in keywords if k.strip()],
                    "feedback_count": len(docs),
                }
            )

        result = {
            "topics": topic_summaries,
            "results": [
                {
                    "feedback_id": cleaned_ids[i],
                    "topic_id": topic_id,
                    "cleaned_text": cleaned_docs[i],
                    "translated_text": None,
                    "summary": None,
                    "confidence_score": self._get_confidence(probs, i),
                }
                for i, topic_id in enumerate(topics)
            ],
            "message": (
                f"Successfully analyzed {len(cleaned_docs)} documents "
                f"for office {office_id}"
            ),
        }

        try:
            cache_service.set(cache_key, result)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

        return result

    def _get_confidence(self, probs, i: int) -> float:
        if probs is None or i >= len(probs):
            return 0.0
        score = probs[i]
        if hasattr(score, "__len__"):
            return float(max(score))
        return float(score)
