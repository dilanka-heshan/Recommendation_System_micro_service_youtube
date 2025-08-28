# Pydantic models for incoming API requests
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class RecommendationRequest(BaseModel):
    user_id:str
    query: Optional[str] = None
    top_k: int = 10
    boost_recent: bool = True