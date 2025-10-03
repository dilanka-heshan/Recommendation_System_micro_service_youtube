from typing import List, Dict, Any, Optional
from backend.database.supabase_client import supabase_client
import logging
import time

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL_AVAILABLE = True
    logger.info("Embedding model available")
except ImportError as e:
    EMBEDDING_MODEL_AVAILABLE = False
    logger.error(f"Failed to import sentence_transformers: {e}")

class UserPreferencesService:
    """
    Simplified service for fetching user embeddings only
    This implements the [User Preferences Fetch] node from the LangGraph pipeline
    """
    
    def __init__(self):
        self.client = supabase_client
        self.embed_model = None
        self._model_load_attempted = False
        self._model_load_failed = False
        self._last_load_attempt = 0
        self._load_retry_delay = 30  # 1 minute between retry attempts
    
    def _ensure_embedding_model_loaded(self) -> bool:
        """
        Lazy loading of embedding model with retry logic and comprehensive error handling
        
        Returns:
            bool: True if model is loaded and functional, False otherwise
        """
        current_time = time.time()
        
        # If model is already loaded and working, return True
        if self.embed_model is not None:
            return True
            
        # If we recently failed to load, don't retry immediately
        if self._model_load_failed and (current_time - self._last_load_attempt) < self._load_retry_delay:
            logger.debug(f"Skipping model load retry - waiting {self._load_retry_delay}s since last attempt")
            return False
            
        # Check if imports are available
        if not EMBEDDING_MODEL_AVAILABLE:
            logger.error("sentence_transformers not available - cannot load embedding model")
            self._model_load_failed = True
            self._last_load_attempt = current_time
            return False
            
        # Check system memory before loading (model needs ~1GB)
        if PSUTIL_AVAILABLE:
            try:
                available_memory_gb = psutil.virtual_memory().available / (1024**3)
                if available_memory_gb < 1.5:
                    logger.warning(f"Low memory ({available_memory_gb:.1f}GB available) - may fail to load embedding model")
            except Exception as mem_e:
                logger.warning(f"Could not check memory usage: {mem_e}")
        else:
            logger.debug("psutil not available - cannot check memory usage")
            
        # Attempt to load the model
        self._last_load_attempt = current_time
        try:
            logger.info("Loading embedding model...")
            start_time = time.time()
            
            # Try local bundled model first, fallback to HuggingFace
            try:
                self.embed_model = SentenceTransformer("/app/models/bge-base-en")
                logger.info("Loaded embedding model from bundled path")
            except Exception as local_e:
                logger.warning(f"Failed to load bundled model: {local_e}. Trying HuggingFace...")
                self.embed_model = SentenceTransformer("BAAI/bge-base-en")
                logger.info("Loaded embedding model from HuggingFace")
            
            # Validate model functionality with a test embedding
            test_embedding = self.embed_model.encode("test", normalize_embeddings=True)
            if len(test_embedding) != 768:
                raise ValueError(f"Model produced incorrect embedding dimension: {len(test_embedding)} (expected 768)")
                
            load_time = time.time() - start_time
            logger.info(f"Embedding model loaded successfully in {load_time:.2f}s")
            
            self._model_load_attempted = True
            self._model_load_failed = False
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {type(e).__name__}: {str(e)}")
            logger.error(f"Model loading attempt failed after {time.time() - start_time:.2f}s")
            
            self.embed_model = None
            self._model_load_attempted = True
            self._model_load_failed = True
            
            # Log specific error types for debugging
            if "OutOfMemoryError" in str(e) or "CUDA out of memory" in str(e):
                logger.error("Model loading failed due to insufficient memory")
            elif "ConnectTimeout" in str(e) or "ReadTimeout" in str(e):
                logger.error("Model loading failed due to network timeout - model may need to be downloaded")
            elif "404" in str(e) or "Repository" in str(e):
                logger.error("Model repository not found - check model name")
                
            return False
    
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
                    logger.error(f"Failed to create embedding from preferences for user {user_id}. Reason: {self._get_embedding_failure_reason()}")
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
            # Use lazy loading with detailed error reporting
            if not self._ensure_embedding_model_loaded():
                if not EMBEDDING_MODEL_AVAILABLE:
                    logger.error(f"Cannot create embedding for user {user_id}: sentence_transformers not installed")
                elif self._model_load_failed:
                    logger.error(f"Cannot create embedding for user {user_id}: model loading failed (retry in {self._load_retry_delay}s)")
                else:
                    logger.error(f"Cannot create embedding for user {user_id}: model not available for unknown reason")
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

    def _get_embedding_failure_reason(self) -> str:
        """
        Get detailed reason for embedding creation failure
        """
        if not EMBEDDING_MODEL_AVAILABLE:
            return "sentence_transformers not installed"
        elif self._model_load_failed:
            return "model loading failed - check logs for details"
        elif self.embed_model is None:
            return "model not loaded yet"
        else:
            return "unknown reason"
    
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
