from typing import Dict, Any
import logging

from langsmith import traceable
from backend.models.pipeline_models import PipelineState
from backend.services.rerank import video_reranker

logger = logging.getLogger(__name__)

# @traceable(name="Rerank Videos Node")  # Disabled due to circular reference issues
def rerank_videos_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node for two-stage video reranking:
    Stage 1: Reranker model with user embedding + feedback videos → top 50
    Stage 2: Pairwise analysis with feedback videos → final ranking
    """
    try:
        #print(state["candidate_videos"])
        state["pipeline_step"] = "reranking"
        logger.info(f"Starting two-stage reranking for user {state['user_id']}")
        
        # Debug: Check what we received
        logger.info(f"DEBUG: Received state type: {type(state)}")
        candidate_videos = state.get("candidate_videos")
        if candidate_videos:
            logger.info(f"DEBUG: candidate_videos length: {len(candidate_videos)}")
            logger.info(f"DEBUG: candidate_videos type: {type(candidate_videos)}")
        else:
            logger.error("DEBUG: State does not have candidate_videos")
        
        # Ensure we have the necessary data
        if not candidate_videos:
            logger.error(f"No candidate videos available for reranking. State keys: {list(state.keys())}")
            state["error"] = "No candidate videos available for reranking"
            return state
        
        high_rating_videos = state.get("high_rating_videos")
        if not high_rating_videos:
            logger.warning(f"No feedback videos for user {state['user_id']}, skipping reranking")
            # Keep candidate_videos as-is for diversity filtering
            return state
        
        # Convert feedback videos to format expected by reranker
        user_history = []
        for video in high_rating_videos:
            user_history.append({
                'video_id': video.get('video_id'),
                'rating': video.get('rating', video.get('feedback_rating', 5))
            })
        
        # Apply two-stage reranking - get more than top_k for diversity filtering
        rerank_pool_size = min(len(candidate_videos), max(state["top_k"] * 3, 30))
        logger.info(f"Reranking {len(candidate_videos)} videos to pool size {rerank_pool_size}")
        
        reranked_videos = video_reranker.rerank_with_user_history(
            user_history=user_history,
            candidate_videos=candidate_videos,
            top_k=rerank_pool_size,  # Get larger pool for diversity filtering
            agg="mean"  # Use mean aggregation for pairwise scores
        )
        
        # Update candidate_videos with reranked results for diversity filtering
        # Debug: Log instead of print
        logger.debug(f"Reranked videos: {len(reranked_videos)}")
        state["candidate_videos"] = reranked_videos
        
        logger.info(f"Two-stage reranking completed: {len(candidate_videos)} → {len(reranked_videos)} videos for diversity filtering")
        
        # Log sample reranked video for debugging
        if reranked_videos:
            sample_video = reranked_videos[0]
            logger.info(f"Sample reranked video keys: {list(sample_video.keys())}")
            logger.info(f"Sample reranked video has final_score: {'final_score' in sample_video}")
        
        logger.debug(f"Final candidate_videos count: {len(state['candidate_videos'])}")
        return state
        
    except Exception as e:
        logger.error(f"Error in rerank_videos_node: {str(e)}")
        state["error"] = f"Reranking failed: {str(e)}"
        # Keep original candidate videos for diversity filtering
        return state
