from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from services.label_service import LabelService


class BertopicService:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.model = BERTopic(embedding_model=self.embedding_model, verbose=True)
        self.label_service = LabelService()

    def get_enhanced_label(self, topic_id: int):
        words = self.model.get_topic(topic_id)

        keywords = [w[0] for w in words[:5]]

        return self.label_service.generate_label(keywords)
