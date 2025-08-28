 # Pydantic models for outgoing API responses
from pydantic import BaseModel
from typing import List, Optional

class BaseResponse(BaseModel):
    """Base response model for API endpoints"""
    success: bool = True
    message: Optional[str] = None

class VideoRecommendation(BaseModel):
    video_id: str
    title: str
    score: float

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[VideoRecommendation]
