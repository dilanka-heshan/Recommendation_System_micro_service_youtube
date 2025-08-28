# Models for user metadata & preferences
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class UserProfile(BaseModel):
    user_id: str
    preferences: List[str]
    last_active: Optional[str] = None
    watched_videos: List[str] = []

class UserEmbedding(BaseModel):
    """Model for user embedding data"""
    user_id: str
    embedding: List[float]  # 768-dimensional vector
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class UserPreferencesState(BaseModel):
    """Model for user preferences state in LangGraph pipeline"""
    user_id: str
    preferences: List[str] = []
    embedding: Optional[List[float]] = None
    high_rating_videos: List[Dict[str, Any]] = []
    user_metadata: Dict[str, Any] = {}
    
    class Config:
        arbitrary_types_allowed = True
