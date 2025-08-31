from typing import Dict, Any, TYPE_CHECKING, List
import logging
from langsmith import traceable
from backend.services.rocchio_algorithm_service import RocchioAlgorithmService

if TYPE_CHECKING:
    from backend.models.user_vector_update_models import UserVectorUpdateState

logger = logging.getLogger(__name__)

# @traceable(name="calculate_user_vectors")  # Disabled due to circular reference issues
def calculate_user_vectors_node(state: 'UserVectorUpdateState') -> 'UserVectorUpdateState':
    """
    Calculate updated user vectors using Rocchio's Algorithm
    
    Functionality:
    - Group feedback by user_id with rating-based weights
    - Calculate weighted positive feedback centroids (ratings 4-5)
    - Calculate weighted negative feedback centroids (ratings 1-2)
    - Apply rating-specific weights: rating*5 * 1.0, rating*4 * 0.75, rating*2 * 0.75, rating*1 * 1.0
    - Apply Rocchio's algorithm with current user vectors from users table
    - Handle edge cases (no feedback, only positive/negative, new users)
    """
    try:
        logger.info("Starting user vector calculation using Rocchio's Algorithm")
        
        # Validate required data
        if not state.get("user_feedback_data"):
            logger.warning("No user feedback data found for vector calculation")
            state["pipeline_step"] = "error"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append("No user feedback data available for vector calculation")
            return state
        
        user_feedback_data = state["user_feedback_data"]
        current_user_vectors = state.get("current_user_vectors", {})
        video_embeddings = state.get("video_embeddings", {})
        user_embedding_ids = state.get("user_embedding_ids", {})
        
        if not video_embeddings:
            logger.warning("No video embeddings available for vector calculation")
            state["pipeline_step"] = "error"
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append("No video embeddings available for vector calculation")
            return state
        
        # Initialize Rocchio Algorithm Service
        rocchio_service = RocchioAlgorithmService()
        
        # Calculate updated vectors for each user
        updated_user_vectors = {}
        new_embedding_ids = {}
        calculation_stats = {
            "users_processed": 0,
            "users_with_positive_feedback": 0,
            "users_with_negative_feedback": 0,
            "users_with_mixed_feedback": 0,
            "new_users": 0,
            "calculation_errors": 0
        }
        
        for user_id, feedback_list in user_feedback_data.items():
            try:
                embedding_id = user_embedding_ids.get(user_id)
                if not embedding_id:
                    logger.warning(f"No embedding_id found for user {user_id}")
                    continue
                
                # Get current user vector
                current_vector = current_user_vectors.get(embedding_id)
                
                # Separate positive and negative feedback
                positive_feedback = []
                negative_feedback = []
                
                for feedback in feedback_list:
                    video_id = feedback["video_id"]
                    rating = feedback["rating"]
                    weight = feedback.get("weight", _get_rating_weight(rating))
                    
                    # Skip if no video embedding available
                    if video_id not in video_embeddings:
                        continue
                    
                    video_embedding = video_embeddings[video_id]
                    
                    # Classify as positive (4-5) or negative (1-2) feedback
                    if rating >= 4:  # Positive feedback
                        positive_feedback.append({
                            "embedding": video_embedding,
                            "weight": weight,
                            "rating": rating
                        })
                    elif rating <= 2:  # Negative feedback
                        negative_feedback.append({
                            "embedding": video_embedding,
                            "weight": weight,
                            "rating": rating
                        })
                    # Rating 3 is ignored as per pipeline specification
                
                # Skip users with no valid feedback
                if not positive_feedback and not negative_feedback:
                    logger.debug(f"No valid feedback for user {user_id}")
                    continue
                
                # Handle new users (no current vector)
                if current_vector is None:
                    logger.info(f"New user detected: {user_id}. Creating initial vector.")
                    current_vector = [0.0] * 768  # Start with zero vector
                    calculation_stats["new_users"] += 1
                
                # Apply Rocchio's Algorithm
                # Extract embeddings and weights from feedback data
                positive_embeddings = [item["embedding"] for item in positive_feedback]
                negative_embeddings = [item["embedding"] for item in negative_feedback]
                positive_weights = [item["weight"] for item in positive_feedback]
                negative_weights = [item["weight"] for item in negative_feedback]
                
                updated_vector = rocchio_service.apply_rocchio_algorithm(
                    original_vector=current_vector,
                    positive_embeddings=positive_embeddings,
                    negative_embeddings=negative_embeddings,
                    positive_weights=positive_weights if positive_weights else None,
                    negative_weights=negative_weights if negative_weights else None
                )
                
                # Store updated vector
                updated_user_vectors[embedding_id] = updated_vector
                
                # Update statistics
                calculation_stats["users_processed"] += 1
                if positive_feedback:
                    calculation_stats["users_with_positive_feedback"] += 1
                if negative_feedback:
                    calculation_stats["users_with_negative_feedback"] += 1
                if positive_feedback and negative_feedback:
                    calculation_stats["users_with_mixed_feedback"] += 1
                
                logger.debug(f"Updated vector for user {user_id}: "
                           f"positive_items={len(positive_feedback)}, "
                           f"negative_items={len(negative_feedback)}")
                
            except Exception as e:
                logger.error(f"Error calculating vector for user {user_id}: {str(e)}")
                calculation_stats["calculation_errors"] += 1
                continue
        
        # Update state
        state["updated_user_vectors"] = updated_user_vectors
        state["new_embedding_ids"] = new_embedding_ids
        state["pipeline_step"] = "vectors_calculated"
        
        # Update metrics
        state["pipeline_metrics"].update({
            "vector_calculation_stats": calculation_stats,
            "updated_vectors_count": len(updated_user_vectors)
        })
        
        logger.info(f"Vector calculation completed. Updated vectors for {len(updated_user_vectors)} users. "
                   f"Stats: {calculation_stats}")
        return state
        
    except Exception as e:
        logger.error(f"Error in calculate_user_vectors_node: {str(e)}")
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Vector calculation error: {str(e)}")
        state["pipeline_step"] = "error"
        return state


def _get_rating_weight(rating: int) -> float:
    """
    Get weight for rating based on pipeline document specifications
    Rating weights: 5=1.0, 4=0.75, 3=0.0, 2=0.75, 1=1.0
    """
    rating_weights = {
        5: 1.0,    # Strongly positive - full weight
        4: 0.75,   # Positive - reduced weight
        3: 0.0,    # Neutral - ignored
        2: 0.75,   # Negative - reduced weight
        1: 1.0     # Strongly negative - full weight
    }
    return rating_weights.get(rating, 0.0)
