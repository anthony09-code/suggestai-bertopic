from typing import List

from pydantic import BaseModel
from schemas.topic import TopicSchema


class TopicResponse(BaseModel):
    topics: List[TopicSchema]
    message: str
