import logging

from fastapi import APIRouter, HTTPException

from app.schemas.request import FeedbackRequest
from app.schemas.response import TopicResponse
from app.services.bertopic_service import BertopicService

logger = logging.getLogger(__name__)
router = APIRouter()
bertopic_service = BertopicService()


@router.post("/analyze", response_model=TopicResponse)
async def analyze_feedback(request: FeedbackRequest):
    try:
        result = bertopic_service.fit_topics(
            office_id=request.office_id,
            documents=request.documents,
            feedback_ids=request.feedback_ids,
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected error occurred")
