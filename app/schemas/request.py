from typing import List

from pydantic import BaseModel


class FeedbackRequest(BaseModel):
    office_id: str
    documents: List[str]
