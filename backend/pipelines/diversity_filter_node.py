from typing import Dict, Any, TYPE_CHECKING
import logging
from backend.services.retrieval_service import retrieval_service

if TYPE_CHECKING:
    from backend.models.pipeline_models import PipelineState

logger = logging.getLogger(__name__)

def diversity_filter_node(state: 'PipelineState') -> 'PipelineState':
    """
    Apply MMR diversity filtering for final recommendations
    """
    try:
        # Debug: Check what we received
        logger.info(f"DEBUG: Diversity filter received state type: {type(state)}")
        logger.info(f"DEBUG: State keys: {list(state.keys())}")
        
        videos_to_filter = state.get("candidate_videos")
        
        logger.info(f"Diversity filtering starting with {len(videos_to_filter) if videos_to_filter else 0} candidate videos")
        logger.info(f"DEBUG: videos_to_filter type: {type(videos_to_filter)}")
        
        if not videos_to_filter:
            logger.warning("No candidate videos available for diversity filtering")
            state["final_list"] = []
            state["pipeline_step"] = "diversity_filtering_completed"
            return state
        
        # Log sample video data for debugging
        if videos_to_filter:
            sample_video = videos_to_filter[0]
            logger.info(f"Sample video keys: {list(sample_video.keys())}")
            logger.info(f"Sample video has embedding: {'embedding' in sample_video}")
            logger.info(f"Sample video has final_score: {'final_score' in sample_video}")
            logger.info(f"Sample video: {sample_video}")  # Add full sample video
        
        final_list = retrieval_service.apply_mmr_diversity(
            videos=videos_to_filter,
            query_embedding=state.get("user_embedding"),
            lambda_param=0.7,  # Balance relevance vs diversity
            top_k=state["top_k"]
        )
        
        state["final_list"] = final_list
        state["pipeline_step"] = "diversity_filtering_completed"
        
        logger.info(f"Diversity filtering completed: {len(final_list)} final recommendations")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in diversity_filter_node: {str(e)}")
        state["error"] = str(e)
        state["pipeline_step"] = "error"
        return state
