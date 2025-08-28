from typing import List, Dict, Any, Optional
from backend.database.supabase_client import supabase_client
import logging

logger = logging.getLogger(__name__)

class UserPreferencesService:
    """
    Simplified service for fetching user embeddings only
    This implements the [User Preferences Fetch] node from the LangGraph pipeline
    """
    
    def __init__(self):
        self.client = supabase_client
    
    def fetch_user_preferences_data(self, user_id: str) -> Dict[str, Any]:
        """
        Simplified method that fetches only user embedding (768-dimensional vector)
        
        Returns state.user_prefs for the LangGraph pipeline
        """
        try:
            # Fetch user embedding only
            user_embedding = self.client.get_user_embedding(user_id)
            
            if not user_embedding:
                logger.warning(f"No user embedding found for user_id: {user_id}")
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
        """
        try:
            embedding = self.client.get_user_embedding(user_id)
            
            if embedding and len(embedding) == 768:
                return embedding
            elif embedding:
                logger.warning(f"User {user_id} embedding has incorrect dimension: {len(embedding)}")
                
            return [0.0] * 768  # Return zero vector instead of None
            
        except Exception as e:
            logger.error(f"Error fetching user embedding for {user_id}: {str(e)}")
            return [0.0] * 768  # Return zero vector on error
    
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
