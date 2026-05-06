from typing import List

from pydantic import BaseModel, field_validator


class FeedbackRequest(BaseModel):
    office_id: str
    documents: List[str]

    @field_validator("office_id")
    @classmethod
    def office_id_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("office_id cannot be empty")
        return v

    @field_validator("documents")
    @classmethod
    def documents_must_be_valid(cls, v):
        if not v:
            raise ValueError("documents list cannot be empty")
        if len(v) < 5:
            raise ValueError("At least 5 documents are required for topic modeling")
        if any(not doc.strip() for doc in v):
            raise ValueError("Documents cannot contain empty strings")
        return v
