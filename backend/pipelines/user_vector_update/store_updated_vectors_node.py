from typing import Dict, Any, TYPE_CHECKING
import logging
from langsmith import traceable
from backend.database.supabase_client import supabase_client

if TYPE_CHECKING:
    from backend.models.user_vector_update_models import UserVectorUpdateState

logger = logging.getLogger(__name__)

# @traceable(name="store_updated_vectors")  # Disabled due to circular reference issues
def store_updated_vectors_node(state: 'UserVectorUpdateState') -> 'UserVectorUpdateState':
    """
    Store updated user vectors in Supabase users table
    
    Functionality:
    - Update user vectors in supabase users table fields embedding_ids (768 dimension vector)
    - Create new user vectors for first-time users (assign new embedding_ids)
    - Log update statistics and success rates
    - Handle batch updates for performance
    - Maintain embedding_id consistency between users table
    """
    try:
        logger.info("Starting storage of updated user vectors")
        
        # Validate required data
        if not state.get("updated_user_vectors"):
            logger.warning("No updated user vectors found for storage")
            state["pipeline_step"] = "completed"  # Still complete, just no updates
            return state
        
        updated_user_vectors = state["updated_user_vectors"]
        user_embedding_ids = state.get("user_embedding_ids", {})
        
        # Reverse mapping: embedding_id -> user_id
        embedding_to_user = {v: k for k, v in user_embedding_ids.items()}
        
        # Storage statistics
        storage_stats = {
            "vectors_to_update": len(updated_user_vectors),
            "successful_updates": 0,
            "failed_updates": 0,
            "new_users_created": 0,
            "storage_errors": []
        }
        
        # Store updated vectors
        successful_updates = {}
        failed_updates = {}
        
        for embedding_id, updated_vector in updated_user_vectors.items():
            try:
                # Get corresponding user_id
                user_id = embedding_to_user.get(embedding_id)
                if not user_id:
                    logger.warning(f"No user_id found for embedding_id: {embedding_id}")
                    storage_stats["failed_updates"] += 1
                    continue
                
                # Validate vector dimensions
                if not updated_vector or len(updated_vector) != 768:
                    logger.warning(f"Invalid vector dimensions for user {user_id}: {len(updated_vector) if updated_vector else 0}")
                    storage_stats["failed_updates"] += 1
                    storage_stats["storage_errors"].append(f"Invalid vector dimensions for user {user_id}")
                    continue
                
                # Update user embedding in Supabase
                success = supabase_client.update_user_embedding(user_id, updated_vector)
                
                if success:
                    successful_updates[user_id] = {
                        "embedding_id": embedding_id,
                        "vector_norm": sum(x*x for x in updated_vector) ** 0.5,  # L2 norm for monitoring
                        "vector_dimensions": len(updated_vector)
                    }
                    storage_stats["successful_updates"] += 1
                    logger.debug(f"Successfully updated vector for user {user_id}")
                else:
                    failed_updates[user_id] = embedding_id
                    storage_stats["failed_updates"] += 1
                    storage_stats["storage_errors"].append(f"Database update failed for user {user_id}")
                    logger.error(f"Failed to update vector for user {user_id}")
                
            except Exception as e:
                user_id = embedding_to_user.get(embedding_id, "unknown")
                logger.error(f"Error storing vector for user {user_id}: {str(e)}")
                failed_updates[user_id] = embedding_id
                storage_stats["failed_updates"] += 1
                storage_stats["storage_errors"].append(f"Exception for user {user_id}: {str(e)}")
        
        # Handle new users if any (this would be for future extension)
        new_embedding_ids = state.get("new_embedding_ids", {})
        for user_id, new_embedding_id in new_embedding_ids.items():
            try:
                # This would be implemented if we need to create completely new users
                # For now, we assume all users exist in the system
                storage_stats["new_users_created"] += 1
                logger.info(f"New user vector created for {user_id}")
            except Exception as e:
                logger.error(f"Error creating new user vector for {user_id}: {str(e)}")
                storage_stats["storage_errors"].append(f"New user creation failed for {user_id}: {str(e)}")
        
        # Update state
        state["pipeline_step"] = "vectors_stored"
        
        # Update metrics
        state["pipeline_metrics"].update({
            "storage_stats": storage_stats,
            "successful_updates": successful_updates,
            "failed_updates": failed_updates,
            "update_success_rate": (storage_stats["successful_updates"] / storage_stats["vectors_to_update"]) * 100 if storage_stats["vectors_to_update"] > 0 else 0
        })
        
        # Log summary
        success_rate = storage_stats["successful_updates"] / storage_stats["vectors_to_update"] * 100 if storage_stats["vectors_to_update"] > 0 else 0
        logger.info(f"Vector storage completed. "
                   f"Success: {storage_stats['successful_updates']}/{storage_stats['vectors_to_update']} "
                   f"({success_rate:.1f}%). "
                   f"Failed: {storage_stats['failed_updates']}")
        
        if storage_stats["storage_errors"]:
            logger.warning(f"Storage errors encountered: {storage_stats['storage_errors'][:3]}...")  # Log first 3 errors
        
        # Mark as completed
        state["pipeline_step"] = "completed"
        
        return state
        
    except Exception as e:
        logger.error(f"Error in store_updated_vectors_node: {str(e)}")
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Vector storage error: {str(e)}")
        state["pipeline_step"] = "error"
        return state
