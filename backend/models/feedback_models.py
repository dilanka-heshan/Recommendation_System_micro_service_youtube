 # Models for user feedback storage
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class FeedbackRecord(BaseModel):
    user_id: str
    video_id: str
    rating: Optional[int] = None  # 1–5 stars
    watch_time_seconds: Optional[int] = None
    liked: Optional[bool] = None

class HighRatingVideo(BaseModel):
    """Model for high-rating video data used in recommendations"""
    video_id: str
    rating: int  # User's rating (4-5)
    embedding: Optional[List[float]] = None
    
    class Config:
        arbitrary_types_allowed = True
