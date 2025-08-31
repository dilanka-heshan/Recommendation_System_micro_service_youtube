from typing import Dict, Any, TYPE_CHECKING, List, Set
import logging
from langsmith import traceable
from backend.database.supabase_client import supabase_client
from backend.database.qdrant_client import qdrant_client

if TYPE_CHECKING:
    from backend.models.user_vector_update_models import UserVectorUpdateState

logger = logging.getLogger(__name__)

# @traceable(name="retrieve_user_vectors")  # Disabled due to circular reference issues
def retrieve_user_vectors_node(state: 'UserVectorUpdateState') -> 'UserVectorUpdateState':
    """
    Retrieve current user vectors and video embeddings
    
    Functionality:
    - Extract embedding_ids from users table for active users
    - Retrieve current user preference vectors from supabase users embedding_id
    - Extract unique video_ids from feedback data
    - Batch retrieve video embeddings from Qdrant video collection
    - Create mappings: user_id -> current_vector, video_id -> embedding
    - Handle missing embeddings gracefully (new users, missing videos)
    """
    try:
        logger.info("Starting user vectors and video embeddings retrieval")
        
        # Check if we have user feedback data
        if not state.get("user_feedback_data"):
            logger.warning("No user feedback data found. Skipping vector retrieval.")
            state["pipeline_step"] = "error"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append("No user feedback data available for vector retrieval")
            return state
        
        user_feedback_data = state["user_feedback_data"]
        user_embedding_ids = state.get("user_embedding_ids", {})
        
        # Extract all unique video IDs from feedback data
        all_video_ids: Set[str] = set()
        for user_id, feedback_list in user_feedback_data.items():
            for feedback in feedback_list:
                all_video_ids.add(feedback["video_id"])
        
        logger.info(f"Retrieving embeddings for {len(all_video_ids)} unique videos")
        
        # Batch retrieve video embeddings from Qdrant
        video_embeddings = {}
        if all_video_ids:
            try:
                video_embeddings = qdrant_client.get_video_embeddings_batch(list(all_video_ids))
                logger.info(f"Retrieved {len(video_embeddings)} video embeddings from Qdrant")
            except Exception as e:
                logger.error(f"Error retrieving video embeddings: {str(e)}")
                # Continue with empty video embeddings - will be handled in calculation
        
        # Retrieve current user vectors from Supabase
        current_user_vectors = {}
        missing_user_vectors = []
        
        for user_id, embedding_id in user_embedding_ids.items():
            try:
                # Get user embedding from Supabase
                user_vector = supabase_client.get_user_embedding(user_id)
                if user_vector and len(user_vector) == 768:
                    current_user_vectors[embedding_id] = user_vector
                    logger.debug(f"Retrieved vector for user {user_id} (embedding_id: {embedding_id})")
                else:
                    missing_user_vectors.append(user_id)
                    logger.warning(f"No valid vector found for user {user_id}")
            except Exception as e:
                logger.error(f"Error retrieving vector for user {user_id}: {str(e)}")
                missing_user_vectors.append(user_id)
        
        logger.info(f"Retrieved vectors for {len(current_user_vectors)} users, "
                   f"{len(missing_user_vectors)} users missing vectors")
        
        # Handle missing video embeddings
        missing_video_ids = all_video_ids - set(video_embeddings.keys())
        if missing_video_ids:
            logger.warning(f"Missing embeddings for {len(missing_video_ids)} videos: {list(missing_video_ids)[:5]}...")
        
        # Filter out feedback for videos without embeddings
        filtered_user_feedback = {}
        for user_id, feedback_list in user_feedback_data.items():
            filtered_feedback = [
                feedback for feedback in feedback_list 
                if feedback["video_id"] in video_embeddings
            ]
            if filtered_feedback:  # Only keep users with at least some valid video feedback
                filtered_user_feedback[user_id] = filtered_feedback
        
        # Update state
        state["current_user_vectors"] = current_user_vectors
        state["video_embeddings"] = video_embeddings
        state["user_feedback_data"] = filtered_user_feedback  # Update with filtered data
        state["pipeline_step"] = "vectors_retrieved"
        
        # Update metrics
        state["pipeline_metrics"].update({
            "total_unique_videos": len(all_video_ids),
            "retrieved_video_embeddings": len(video_embeddings),
            "missing_video_embeddings": len(missing_video_ids),
            "retrieved_user_vectors": len(current_user_vectors),
            "missing_user_vectors": len(missing_user_vectors),
            "users_with_valid_feedback": len(filtered_user_feedback)
        })
        
        logger.info(f"Vector retrieval completed. Users with valid feedback: {len(filtered_user_feedback)}")
        return state
        
    except Exception as e:
        logger.error(f"Error in retrieve_user_vectors_node: {str(e)}")
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Vector retrieval error: {str(e)}")
        state["pipeline_step"] = "error"
        return state
