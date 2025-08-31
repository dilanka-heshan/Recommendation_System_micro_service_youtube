import os
from supabase import create_client, Client
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SupabaseClient:
    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_ANON_KEY")
        
        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables")
        
        self.client: Client = create_client(self.url, self.key)
    
    def get_user_embedding(self, user_id: str) -> Optional[List[float]]:
        """
        Fetch user embedding vector from users table
        Returns 768-dimensional embedding vector or None if user not found
        """
        try:
            response = self.client.table("users").select("embedding_id").eq("user_id", user_id).execute()
            
            if response.data and len(response.data) > 0:
                embedding_str = response.data[0].get("embedding_id")
                if embedding_str:
                    # Parse the text string into a list of floats
                    try:
                        if isinstance(embedding_str, str):
                            # Remove any whitespace and parse the string representation of list
                            import ast
                            embedding_list = ast.literal_eval(embedding_str.strip())
                            
                            # Ensure it's a list and convert to floats
                            if isinstance(embedding_list, list):
                                return [float(x) for x in embedding_list]
                            else:
                                logger.error(f"Embedding is not a list: {type(embedding_list)}")
                                return None
                        else:
                            # If it's already a list, just ensure floats
                            return [float(x) for x in embedding_str]
                            
                    except (ValueError, SyntaxError) as parse_error:
                        logger.error(f"Failed to parse embedding string: {str(parse_error)}")
                        return None
            
            logger.warning(f"No embedding found for user_id: {user_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error fetching user embedding for {user_id}: {str(e)}")
            return None
    
    
    def get_high_rating_videos(self, user_id: str, min_rating: int = 4, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Fetch recent high-rating videos for a user from feedback table
        Returns list of video data with ratings >= min_rating, ordered by recency
        """
        try:
            # Get recent high-rating feedback with video details
            response = self.client.table("feedback").select(
                "video_id, rating, timestamp"
            ).eq("user_id", user_id).gte("rating", min_rating).order(
                "timestamp", desc=True
            ).limit(limit).execute()
            
            if response.data:
                # Enrich with video metadata if available
                video_ids = [item["video_id"] for item in response.data]
                video_details = self.get_videos_by_ids(video_ids)
                
                # Merge feedback with video details
                video_map = {v["video_id"]: v for v in video_details}
                
                enriched_data = []
                for feedback in response.data:
                    video_data = video_map.get(feedback["video_id"], {})
                    merged_data = {
                        **feedback,
                        **video_data,
                        "feedback_rating": feedback["rating"]  # Keep original rating
                    }
                    enriched_data.append(merged_data)
                
                return enriched_data
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching high-rating videos for {user_id}: {str(e)}")
            return []
    
    def get_videos_by_ids(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch video metadata by video IDs from Qdrant vector database
        """
        try:
            from backend.database.qdrant_client import qdrant_client
            return qdrant_client.get_videos_by_ids(video_ids)
            
        except Exception as e:
            logger.error(f"Error fetching videos by IDs from Qdrant: {str(e)}")
            return []
    
    def get_user_watched_videos(self, user_id: str) -> List[str]:
        """
        Fetch list of video IDs that user has previously watched from interactions table
        """
        try:
            response = self.client.table("interactions").select("video_id").eq("user_id", user_id).execute()
            
            if response.data:
                return [item["video_id"] for item in response.data]
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching watched videos for {user_id}: {str(e)}")
            return []
    
    def get_video_publish_dates(self, video_ids: List[str]) -> Dict[str, str]:
        """
        Fetch publish dates for videos from videos table
        Returns dict mapping video_id to published_at timestamp
        """
        try:
            if not video_ids:
                return {}
                
            response = self.client.table("videos").select(
                "video_id, published_at"
            ).in_("video_id", video_ids).execute()
            
            if response.data:
                return {item["video_id"]: item["published_at"] for item in response.data if item.get("published_at")}
            
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching video publish dates: {str(e)}")
            return {}

    def update_user_embedding(self, user_id: str, embedding: List[float]) -> bool:
        """
        Update user embedding in users table
        """
        try:
            # Convert embedding list to string format for storage (consistent with other methods)
            embedding_str = str(embedding)
            
            response = self.client.table("users").update({
                "embedding_id": embedding_str
                # "updated_at": datetime.utcnow().isoformat()
            }).eq("user_id", user_id).execute()
            
            return len(response.data) > 0
            
        except Exception as e:
            logger.error(f"Error updating user embedding for {user_id}: {str(e)}")
            return False

    def validate_videos_exist(self, video_ids: List[str]) -> List[str]:
        """
        Validate which video IDs exist in the videos table
        
        Args:
            video_ids: List of video IDs to check
            
        Returns:
            List of video IDs that exist in the videos table
        """
        try:
            if not video_ids:
                return []
            
            response = self.client.table("videos").select("video_id").in_("video_id", video_ids).execute()
            
            if response.data:
                existing_video_ids = [video["video_id"] for video in response.data]
                logger.info(f"Found {len(existing_video_ids)}/{len(video_ids)} videos in videos table")
                return existing_video_ids
            else:
                logger.warning(f"No videos found in videos table for {len(video_ids)} video IDs")
                return []
                
        except Exception as e:
            logger.error(f"Error validating video existence: {str(e)}")
            return []

    def create_newsletter(self, user_id: str, video_recommendations: List[Dict[str, Any]]) -> Optional[int]:
        """
        Create a new newsletter entry and store recommended videos
        
        Args:
            user_id: The user ID for the newsletter
            video_recommendations: List of recommended videos with metadata
            
        Returns:
            Newsletter ID if successful, None if failed
        """
        try:
            # Create newsletter entry
            newsletter_response = self.client.table("newsletters").insert({
                "user_id": user_id,
                "sent_at": datetime.utcnow().isoformat()
            }).execute()
            
            if not newsletter_response.data or len(newsletter_response.data) == 0:
                logger.error(f"Failed to create newsletter for user {user_id}")
                return None
            
            newsletter_id = newsletter_response.data[0]["id"]
            
            # Validate that videos exist in the videos table before inserting
            video_ids_to_check = [video.get("video_id") for video in video_recommendations if video.get("video_id")]
            existing_video_ids = self.validate_videos_exist(video_ids_to_check)
            
            if len(existing_video_ids) < len(video_ids_to_check):
                missing_videos = set(video_ids_to_check) - set(existing_video_ids)
                logger.warning(f"Skipping {len(missing_videos)} videos not found in videos table: {missing_videos}")
            
            # Store individual video recommendations (only for existing videos)
            video_entries = []
            for video in video_recommendations:
                video_id = video.get("video_id")
                if video_id in existing_video_ids:
                    video_entries.append({
                        "newsletter_id": newsletter_id,
                        "video_id": video_id,
                        "clicked": False  # Default to not clicked
                    })
                else:
                    logger.debug(f"Skipping video {video_id} - not found in videos table")
            
            if video_entries:
                videos_response = self.client.table("newsletter_videos").insert(video_entries).execute()
                
                if not videos_response.data:
                    logger.error(f"Failed to store videos for newsletter {newsletter_id}")
                    return None
                    
                logger.info(f"Created newsletter {newsletter_id} for user {user_id} with {len(video_entries)} videos")
            else:
                logger.warning(f"No valid videos to store for newsletter {newsletter_id}")
            
            return newsletter_id
            
        except Exception as e:
            logger.error(f"Error creating newsletter for user {user_id}: {str(e)}")
            return None

    # ============= USER VECTOR UPDATE PIPELINE METHODS =============
    
    def get_daily_feedback(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Fetch daily feedback data for user vector update pipeline
        Args:
            start_date: ISO format date string (e.g., '2024-08-29')
            end_date: ISO format date string (e.g., '2024-08-30')
        Returns:
            List of feedback records with user_id, video_id, rating, timestamp
        """
        try:
            response = self.client.table("feedback").select(
                "user_id, video_id, rating, timestamp"
            ).gte("timestamp", start_date).lt("timestamp", end_date).execute()
            
            logger.info(f"Retrieved {len(response.data)} feedback records from {start_date} to {end_date}")
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Error fetching daily feedback: {str(e)}")
            return []
    
    # def get_newsletter_click_data(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
    #     """
    #     Fetch newsletter click data for user vector update pipeline
    #     Args:
    #         start_date: ISO format date string
    #         end_date: ISO format date string
    #     Returns:
    #         List of newsletter video records with user_id, video_id, clicked, sent_at
    #     """
    #     try:
    #         # First get newsletters sent in the date range
    #         newsletters_response = self.client.table("newsletters").select(
    #             "id, user_id, sent_at"
    #         ).gte("sent_at", start_date).lt("sent_at", end_date).execute()
            
    #         if not newsletters_response.data:
    #             logger.info(f"No newsletters found from {start_date} to {end_date}")
    #             return []
            
    #         newsletter_ids = [n["id"] for n in newsletters_response.data]
    #         newsletter_user_map = {n["id"]: n["user_id"] for n in newsletters_response.data}
    #         newsletter_date_map = {n["id"]: n["sent_at"] for n in newsletters_response.data}
            
    #         # Get video click data for these newsletters
    #         videos_response = self.client.table("newsletter_videos").select(
    #             "newsletter_id, video_id, clicked"
    #         ).in_("newsletter_id", newsletter_ids).execute()
            
    #         # Combine data with user_id and sent_at
    #         click_data = []
    #         for video in videos_response.data if videos_response.data else []:
    #             newsletter_id = video["newsletter_id"]
    #             click_data.append({
    #                 "user_id": newsletter_user_map.get(newsletter_id),
    #                 "video_id": video["video_id"],
    #                 "clicked": video["clicked"],
    #                 "newsletter_id": newsletter_id,
    #                 "sent_at": newsletter_date_map.get(newsletter_id)
    #             })
            
    #         logger.info(f"Retrieved {len(click_data)} newsletter video records from {start_date} to {end_date}")
    #         return click_data
            
    #     except Exception as e:
    #         logger.error(f"Error fetching newsletter click data: {str(e)}")
    #         return []
    
    def get_active_users_with_embeddings(self) -> List[Dict[str, Any]]:
        """
        Get all users who have embedding_id (active users with preference vectors)
        Returns:
            List of user records with user_id and embedding_id
        """
        try:
            response = self.client.table("users").select(
                "user_id, embedding_id"
            ).neq("embedding_id", None).execute()  # Missing .execute()
            
            logger.info(f"Retrieved {len(response.data)} active users with embeddings")
            return response.data if response.data else []
            
        except Exception as e:
            logger.error(f"Error fetching active users with embeddings: {str(e)}")
            return []
    
    def get_user_vectors_batch(self, user_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Batch retrieve user embeddings for multiple users
        Args:
            user_ids: List of user IDs to fetch embeddings for
        Returns:
            Dict mapping user_id to {embedding_id: str, embedding: List[float]}
        """
        try:
            response = self.client.table("users").select(
                "user_id, embedding_id"
            ).in_("user_id", user_ids).execute()
            
            user_vector_map = {}
            for user_data in response.data if response.data else []:
                user_id = user_data["user_id"]
                embedding_str = user_data.get("embedding_id")
                
                if embedding_str:
                    try:
                        # Parse the embedding string to list of floats
                        if isinstance(embedding_str, str):
                            import ast
                            embedding_list = ast.literal_eval(embedding_str.strip())
                            if isinstance(embedding_list, list):
                                user_vector_map[user_id] = {
                                    "embedding_id": user_id,  # Using user_id as embedding_id for consistency
                                    "embedding": [float(x) for x in embedding_list]
                                }
                        else:
                            user_vector_map[user_id] = {
                                "embedding_id": user_id,
                                "embedding": [float(x) for x in embedding_str]
                            }
                    except (ValueError, SyntaxError) as parse_error:
                        logger.error(f"Failed to parse embedding for user {user_id}: {str(parse_error)}")
                        continue
            
            logger.info(f"Successfully retrieved embeddings for {len(user_vector_map)} users")
            return user_vector_map
            
        except Exception as e:
            logger.error(f"Error in batch user vector retrieval: {str(e)}")
            return {}
    
    def update_user_embeddings_batch(self, user_vector_updates: Dict[str, List[float]]) -> Dict[str, bool]:
        """
        Batch update user embeddings
        Args:
            user_vector_updates: Dict mapping user_id to new embedding vector
        Returns:
            Dict mapping user_id to success status (bool)
        """
        update_results = {}
        
        try:
            for user_id, embedding_vector in user_vector_updates.items():
                try:
                    # Convert embedding list to string format for storage
                    embedding_str = str(embedding_vector)
                    
                    response = self.client.table("users").update({
                        "embedding_id": embedding_str,
                        #"updated_at": datetime.utcnow().isoformat()  # Check updated_at time need or not
                    }).eq("user_id", user_id).execute()
                    
                    update_results[user_id] = len(response.data) > 0
                    
                except Exception as e:
                    logger.error(f"Error updating embedding for user {user_id}: {str(e)}")
                    update_results[user_id] = False
            
            successful_updates = sum(update_results.values())
            logger.info(f"Batch embedding update: {successful_updates}/{len(user_vector_updates)} successful")
            
            return update_results
            
        except Exception as e:
            logger.error(f"Error in batch embedding update: {str(e)}")
            return {user_id: False for user_id in user_vector_updates.keys()}
    
#check below I is needed
    # user table preference column need to use create initial vector
    # def create_new_user_embeddings(self, user_embeddings: Dict[str, List[float]]) -> Dict[str, bool]:
    #     """
    #     Create embeddings for new users who don't have embedding_id yet
    #     Args:
    #         user_embeddings: Dict mapping user_id to embedding vector
    #     Returns:
    #         Dict mapping user_id to success status
    #     """
    #     creation_results = {}
        
    #     try:
    #         for user_id, embedding_vector in user_embeddings.items():
    #             try:
    #                 embedding_str = str(embedding_vector)
                    
    #                 # Use upsert to handle both new and existing users
    #                 response = self.client.table("users").upsert({
    #                     "user_id": user_id,
    #                     "embedding_id": embedding_str,
    #                     "created_at": datetime.utcnow().isoformat(),
    #                     "updated_at": datetime.utcnow().isoformat()
    #                 }).execute()
                    
    #                 creation_results[user_id] = len(response.data) > 0
                    
    #             except Exception as e:
    #                 logger.error(f"Error creating embedding for new user {user_id}: {str(e)}")
    #                 creation_results[user_id] = False
            
    #         successful_creations = sum(creation_results.values())
    #         logger.info(f"New user embedding creation: {successful_creations}/{len(user_embeddings)} successful")
            
    #         return creation_results
            
    #     except Exception as e:
    #         logger.error(f"Error in new user embedding creation: {str(e)}")
    #         return {user_id: False for user_id in user_embeddings.keys()}


# Global instance
supabase_client = SupabaseClient()
