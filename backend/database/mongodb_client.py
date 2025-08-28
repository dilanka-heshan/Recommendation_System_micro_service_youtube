import logging
import os
from typing import Dict, Any, List, Optional
try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False

logger = logging.getLogger(__name__)

class MongoDBClient:
    """
    MongoDB client for fetching video extractive summaries from videosummary.summaries collection
    """
    
    def __init__(self, connection_string: str = None):
        self.client = None
        self.db = None
        self.collection = None
        
        # Use environment variable if no connection string provided
        if not connection_string:
            connection_string = os.getenv("MONGODB_CONNECTION_STRING")
        
        if PYMONGO_AVAILABLE and connection_string:
            try:
                self.client = MongoClient(connection_string)
                self.db = self.client.videosummary
                self.collection = self.db.summaries
                logger.info("MongoDB connection established")
            except Exception as e:
                logger.error(f"Failed to connect to MongoDB: {str(e)}")
        elif not PYMONGO_AVAILABLE:
            logger.warning("PyMongo not available. Install pymongo package.")
        else:
            logger.warning("No MongoDB connection string provided in environment variables or constructor")
    
    def get_extractive_summary(self, video_id: str) -> Optional[str]:
        """
        Fetch extractive summary for a video by video_id
        
        Args:
            video_id: The video ID to fetch summary for
            
        Returns:
            Extractive summary text or None if not found
        """
        try:
            if self.collection is None:
                logger.warning("MongoDB collection not available")
                return None
            
            document = self.collection.find_one(
                {"video_id": video_id},
                {"extractive_summary": 1, "_id": 0}
            )
            
            if document and "extractive_summary" in document:
                return document["extractive_summary"]
            
            return None
            
        except PyMongoError as e:
            logger.error(f"MongoDB error fetching summary for {video_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error fetching summary for {video_id}: {str(e)}")
            return None
    
    def get_multiple_extractive_summaries(self, video_ids: List[str]) -> Dict[str, str]:
        """
        Fetch extractive summaries for multiple videos
        
        Args:
            video_ids: List of video IDs
            
        Returns:
            Dictionary mapping video_id to extractive_summary
        """
        try:
            if self.collection is None:
                logger.warning("MongoDB collection not available")
                return {}
            
            cursor = self.collection.find(
                {"video_id": {"$in": video_ids}},
                {"video_id": 1, "extractive_summary": 1, "_id": 0}
            )
            
            summaries = {}
            for doc in cursor:
                if "video_id" in doc and "extractive_summary" in doc:
                    summaries[doc["video_id"]] = doc["extractive_summary"]
            
            logger.info(f"Fetched summaries for {len(summaries)}/{len(video_ids)} videos")
            return summaries
            
        except PyMongoError as e:
            logger.error(f"MongoDB error fetching multiple summaries: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Error fetching multiple summaries: {str(e)}")
            return {}
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

# Global MongoDB client instance (automatically initialized with environment variables)
mongodb_client = MongoDBClient()

def initialize_mongodb(connection_string: str):
    """Initialize MongoDB client with connection string"""
    global mongodb_client
    mongodb_client = MongoDBClient(connection_string)
    return mongodb_client
