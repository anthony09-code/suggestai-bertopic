from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TopicResultSchema(BaseModel):
    feedback_id: str
    topic_id: int
    cleaned_text: Optional[str] = None
    translated_text: Optional[str] = None
    summary: Optional[str] = None
    confidence_score: float
