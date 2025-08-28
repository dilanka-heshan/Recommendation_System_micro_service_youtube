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
            response = self.client.table("users").upsert({
                "user_id": user_id,
                "embedding": embedding,
                "updated_at": datetime.utcnow().isoformat()
            }).execute()
            
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


# Global instance
supabase_client = SupabaseClient()
