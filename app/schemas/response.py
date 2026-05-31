from typing import List

from pydantic import BaseModel

from app.schemas.topic import TopicSchema
from app.schemas.topic_result import TopicResultSchema


class TopicResponse(BaseModel):
    topics: List[TopicSchema]
    results: List[TopicResultSchema]
    message: str
