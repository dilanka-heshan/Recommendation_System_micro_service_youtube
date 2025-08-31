# Models for User Vector Update Pipeline
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel
from datetime import datetime

class UserVectorUpdateState(TypedDict, total=False):
    """
    State object for the daily user vector update pipeline
    Compatible with LangGraph's state handling
    """
    # Input parameters
    date_range: Dict[str, str]  # start_date, end_date
    
    # Pipeline data
    user_feedback_data: Dict[str, List[Dict]]  # user_id -> [{"video_id": str, "rating": int, "timestamp": str}]
    user_embedding_ids: Dict[str, str]  # user_id -> embedding_id
    current_user_vectors: Dict[str, List[float]]  # embedding_id -> current_vector
    video_embeddings: Dict[str, List[float]]  # video_id -> get from qdrant
    updated_user_vectors: Dict[str, List[float]]  # embedding_id -> new_vector  replace previous vector with new vector
    new_embedding_ids: Dict[str, str]  # user_id -> new_embedding_id (for new users)
    
    # Pipeline metadata
    pipeline_metrics: Dict[str, Any]  # execution statistics
    pipeline_step: str
    errors: List[str]  # error tracking
    execution_time: Optional[float]

class DailyFeedbackRecord(BaseModel):
    """Model for individual feedback record"""
    user_id: str
    video_id: str
    rating: int  # 1-5 scale
    timestamp: datetime
    source: str  # 'feedback' or 'newsletter_click'

class NewsletterClickRecord(BaseModel):
    """Model for newsletter click data"""
    user_id: str
    video_id: str
    clicked: bool
    newsletter_id: int
    sent_at: datetime

class UserFeedbackAggregation(BaseModel):
    """Model for aggregated user feedback"""
    user_id: str
    embedding_id: Optional[str] = None
    positive_videos: List[Dict[str, Any]] = []  # ratings 4-5 or clicks
    negative_videos: List[Dict[str, Any]] = []  # ratings 1-2 or no clicks
    neutral_videos: List[Dict[str, Any]] = []   # rating 3
    total_feedback_count: int = 0

class RocchioParameters(BaseModel):
    """Configuration for Rocchio's Algorithm"""
    alpha: float = 0.7    # Weight for original user vector
    beta: float = 0.3     # Weight for positive feedback vectors
    gamma: float = 0.1    # Weight for negative feedback vectors
    
    # Rating-based weights
    rating_weights: Dict[int, float] = {
        5: 1.0,    # Strongly positive - full weight
        4: 0.75,   # Positive - reduced weight
        3: 0.0,    # Neutral - ignored
        2: 0.75,   # Negative - reduced weight
        1: 1.0     # Strongly negative - full weight
    }
    
    # Classification for Rocchio's algorithm
    positive_ratings: List[int] = [4, 5]
    negative_ratings: List[int] = [1, 2]
    neutral_ratings: List[int] = [3]

class VectorUpdateMetrics(BaseModel):
    """Model for pipeline execution metrics"""
    execution_date: datetime
    total_users_processed: int = 0
    successful_updates: int = 0
    failed_updates: int = 0
    new_users_created: int = 0
    total_feedback_records: int = 0
    total_newsletter_clicks: int = 0
    average_vector_change: float = 0.0
    execution_time_seconds: float = 0.0
    errors: List[str] = []

class UserVectorUpdate(BaseModel):
    """Model for individual user vector update"""
    user_id: str
    embedding_id: str
    original_vector: List[float]
    updated_vector: List[float]
    feedback_count: int
    positive_feedback_count: int
    negative_feedback_count: int
    vector_change_magnitude: float
    update_timestamp: datetime
