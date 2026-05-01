from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TopicResultSchema(BaseModel):
    feedback_id: str
    office_id: str
    topic_id: int

    cleaned_text: Optional[str]
    translated_text: Optional[str]
    summary: Optional[str]

    confidence_score: float
    processed_at: datetime
