from typing import Dict, Any
import logging
from backend.services.retrieval_service import retrieval_service
from backend.models.pipeline_models import PipelineState

logger = logging.getLogger(__name__)

def vector_retrieval_node(state: PipelineState) -> PipelineState:
    """
    Vector similarity search with filtering and time decay
    """
    try:
        if not state.get("user_id"):
            logger.error("No user_id provided for retrieval")
            state["candidate_videos"] = []
            state["pipeline_step"] = "retrieval_completed"
            return state
        
        # Get candidates using vector search with all filtering built-in
        candidate_videos = retrieval_service.retrieve_videos_for_user(
            user_id=state["user_id"],
            similarity_threshold=0.6,
            limit=100,
            time_decay_days=30,
            decay_factor=0.1
        )
        
        state["candidate_videos"] = candidate_videos
        state["pipeline_step"] = "vector_retrieval_completed"
        
        logger.info(f"Vector retrieval completed: {len(candidate_videos)} candidates")
        
        # Debug: Ensure the state is properly set
        logger.info(f"DEBUG: After setting, state['candidate_videos'] length: {len(state['candidate_videos']) if state.get('candidate_videos') else 0}")
        logger.info(f"DEBUG: State type: {type(state)}")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in vector_retrieval_node: {str(e)}")
        state["error"] = str(e)
        state["pipeline_step"] = "error"
        return state
