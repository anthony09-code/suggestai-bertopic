from fastapi import APIRouter
from schemas.request import FeedbackRequest
from schemas.response import TopicResponse
from services.bertopic_service import BertopicService

router = APIRouter()
bertopic_service = BertopicService()

@router.post("/analyze", response_model=TopicResponse)
async def analyze_feedback(request: FeedbackRequest):
    topics, probs = bertopic_service.fit_topics(request.)
    return TopicResponse(topics=topics, probs=probs)
