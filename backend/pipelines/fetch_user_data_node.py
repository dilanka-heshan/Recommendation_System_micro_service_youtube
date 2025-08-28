from typing import Dict, Any, TYPE_CHECKING
import logging
from langsmith import traceable
from backend.services.user_preferences_service import user_preferences_service

if TYPE_CHECKING:
    from backend.models.pipeline_models import PipelineState

logger = logging.getLogger(__name__)

@traceable(name="fetch_user_data")
def fetch_user_data_node(state: 'PipelineState') -> 'PipelineState':
    """
    Fetch user embedding and high-rating videos
    """
    try:
        logger.info(f"Fetching user data for: {state['user_id']}")
        
        user_prefs = user_preferences_service.fetch_user_preferences_data(state["user_id"])
        
        if not user_prefs or not user_prefs.get("embedding"):
            state["is_new_user"] = True
            state["user_embedding"] = [0.0] * 768  # Zero vector for new users
            state["high_rating_videos"] = []
        else:
            state["user_embedding"] = user_prefs.get("embedding")
            state["high_rating_videos"] = user_prefs.get("high_rating_videos", [])
        
        state["pipeline_step"] = "user_data_fetched"
        
        logger.info(f"User data fetched: embedding_dim={len(state['user_embedding'])}, "
                   f"high_rating_videos={len(state['high_rating_videos'])}")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in fetch_user_data_node: {str(e)}")
        state["error"] = str(e)
        state["pipeline_step"] = "error"
        return state
