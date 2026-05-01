from typing import List, Optional

from pydantic import BaseModel


class TopicSchema(BaseModel):
    topic_id: int
    label: str
    keywords: List[str]
    feedback_count: int
    cluster_x: Optional[float] = None
    cluster_y: Optional[float] = None
