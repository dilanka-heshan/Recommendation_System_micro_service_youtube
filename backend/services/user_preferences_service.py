from typing import List, Dict, Any, Optional
from backend.database.supabase_client import supabase_client
import logging

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    EMBEDDING_MODEL_AVAILABLE = False

logger = logging.getLogger(__name__)

class UserPreferencesService:
    """
    Simplified service for fetching user embeddings only
    This implements the [User Preferences Fetch] node from the LangGraph pipeline
    """
    
    def __init__(self):
        self.client = supabase_client
        self.embed_model = None
        
        # Initialize embedding model (same as reranking service)
        if EMBEDDING_MODEL_AVAILABLE:
            try:
                self.embed_model = SentenceTransformer("BAAI/bge-base-en")
                logger.info("User preferences embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load user preferences embedding model: {str(e)}")
                self.embed_model = None
    
    def fetch_user_preferences_data(self, user_id: str) -> Dict[str, Any]:
        """
        Simplified method that fetches only user embedding (768-dimensional vector)
        If embedding_id is null or empty [], creates embedding from user preferences
        
        Returns state.user_prefs for the LangGraph pipeline
        """
        try:
            # Fetch user embedding
            user_embedding = self.client.get_user_embedding(user_id)
            
            # Check if embedding is null, empty, or zero vector
            if not user_embedding or len(user_embedding) == 0 or all(x == 0.0 for x in user_embedding):
                logger.info(f"No valid embedding found for user_id: {user_id}. Creating from preferences...")
                
                # Try to create embedding from user preferences
                created_embedding = self.create_user_embedding_from_preferences(user_id)
                
                if created_embedding:
                    user_embedding = created_embedding
                    logger.info(f"Successfully created embedding from preferences for user {user_id}")
                else:
                    logger.warning(f"Failed to create embedding from preferences for user {user_id}. Using empty state.")
                    return self._create_empty_user_state(user_id)
            
            # Fetch user's high-rating videos from feedback
            high_rating_videos = self.client.get_high_rating_videos(
                user_id=user_id,
                min_rating=4,  # 4-5 star ratings
                limit=20
            )
            
            # Structure the data for LangGraph state - only embedding and high rating videos
            user_state = {
                "user_id": user_id,
                "preferences": [],  # Not used anymore
                "embedding": user_embedding,
                "high_rating_videos": high_rating_videos,
                "user_metadata": {
                    "total_high_ratings": len(high_rating_videos)
                }
            }
            
            logger.info(f"Successfully fetched user embedding for {user_id}: "
                       f"embedding dimension: {len(user_embedding) if user_embedding else 0}")
            
            return user_state
            
        except Exception as e:
            logger.error(f"Error fetching user preferences for {user_id}: {str(e)}")
            return self._create_empty_user_state(user_id)
    
    def get_user_embedding(self, user_id: str) -> Optional[List[float]]:
        """
        Fetch user embedding vector (768-dimensional) from users table
        If embedding is null or empty, create from preferences
        """
        try:
            embedding = self.client.get_user_embedding(user_id)
            
            # Check if embedding is null, empty, or zero vector
            if not embedding or len(embedding) == 0 or all(x == 0.0 for x in embedding):
                logger.info(f"No valid embedding found for user {user_id}, creating from preferences...")
                
                # Try to create embedding from preferences
                created_embedding = self.create_user_embedding_from_preferences(user_id)
                
                if created_embedding and len(created_embedding) == 768:
                    return created_embedding
                else:
                    logger.warning(f"Failed to create valid embedding for user {user_id}")
                    return [0.0] * 768  # Return zero vector as fallback
            
            if embedding and len(embedding) == 768:
                return embedding
            elif embedding:
                logger.warning(f"User {user_id} embedding has incorrect dimension: {len(embedding)}")
                
            return [0.0] * 768  # Return zero vector instead of None
            
        except Exception as e:
            logger.error(f"Error fetching user embedding for {user_id}: {str(e)}")
            return [0.0] * 768  # Return zero vector on error
    
    def create_user_embedding_from_preferences(self, user_id: str) -> Optional[List[float]]:
        """
        Create user embedding from user preferences text using SentenceTransformer
        Store the created embedding in the database
        
        Args:
            user_id: User ID to create embedding for
            
        Returns:
            List[float]: Created 768-dimensional embedding vector or None if failed
        """
        try:
            if not EMBEDDING_MODEL_AVAILABLE or not self.embed_model:
                logger.warning(f"Embedding model not available for creating user embedding for {user_id}")
                return None
                
            # Fetch user preferences from database
            preferences = self.client.get_user_preferences(user_id)
            
            if not preferences or len(preferences) == 0:
                logger.warning(f"No preferences found for user {user_id} to create embedding")
                # Return zero vector for users with no preferences
                zero_embedding = [0.0] * 768
                # Store the zero embedding in database
                if self.client.update_user_embedding(user_id, zero_embedding):
                    logger.info(f"Created and stored zero embedding for user {user_id} (no preferences)")
                    return zero_embedding
                return None
                
            # Convert preferences list to text
            preferences_text = " ".join(preferences)
            logger.info(f"Creating embedding for user {user_id} from preferences: {preferences_text[:100]}...")
            
            # Generate embedding using SentenceTransformer
            embedding = self.embed_model.encode(preferences_text, normalize_embeddings=True)
            
            # Convert numpy array to list if needed
            if hasattr(embedding, 'tolist'):
                embedding_list = embedding.tolist()
            else:
                embedding_list = list(embedding)
                
            # Ensure 768 dimensions (pad or truncate if necessary)
            if len(embedding_list) != 768:
                if len(embedding_list) < 768:
                    # Pad with zeros
                    embedding_list.extend([0.0] * (768 - len(embedding_list)))
                else:
                    # Truncate to 768
                    embedding_list = embedding_list[:768]
                    
            logger.info(f"Generated embedding for user {user_id}: dimension={len(embedding_list)}")
            
            # Store the embedding in database
            if self.client.update_user_embedding(user_id, embedding_list):
                logger.info(f"Successfully created and stored embedding for user {user_id}")
                return embedding_list
            else:
                logger.error(f"Failed to store embedding for user {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating user embedding from preferences for {user_id}: {str(e)}")
            return None

    def _create_empty_user_state(self, user_id: str) -> Dict[str, Any]:
        """
        Create empty user state for new users with 768-dimensional zero vector
        """
        return {
            "user_id": user_id,
            "preferences": [],  # Not used anymore
            "embedding": [0.0] * 768,  # 768-dimensional zero vector for new users
            "high_rating_videos": [],
            "user_metadata": {
                "total_high_ratings": 0
            }
        }

# Global service instance
user_preferences_service = UserPreferencesService()
