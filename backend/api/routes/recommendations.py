from fastapi import APIRouter, Query
from typing import List, Dict, Any
from backend.services.recommendation_service import get_recommendations
from backend.models.response_models import RecommendationResponse

router = APIRouter()

@router.get("/", response_model=Dict[str, Any])
def recommend_videos(
    user_id: str = Query(..., description="User ID"),
    top_k: int = Query(10, description="Number of recommendations to return")
):
    recommendations = get_recommendations(user_id=user_id, top_k=top_k)
    return recommendations