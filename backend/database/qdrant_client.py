import os
from typing import List, Dict, Any, Optional
import logging

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    models = None

logger = logging.getLogger(__name__)

class QdrantVectorClient:
    """
    Client for interacting with Qdrant vector database
    Handles video embeddings and similarity search
    """
    
    def __init__(self):
        self.client = None
        self.collection_name = "video_title_desc"
        
        if QDRANT_AVAILABLE:
            try:
                qdrant_host = os.getenv("QDRANT_HOST", "localhost")
                qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
                qdrant_api_key = os.getenv("QDRANT_API_KEY")
                
                # Check if host contains protocol (http:// or https://)
                if qdrant_host.startswith(("http://", "https://")):
                    # Use url parameter for full URLs
                    self.client = QdrantClient(
                        url=qdrant_host,
                        api_key=qdrant_api_key,
                    )
                else:
                    # Use host/port parameters for hostname only
                    self.client = QdrantClient(
                        host=qdrant_host,
                        port=qdrant_port,
                        api_key=qdrant_api_key,
                    )
                
                # Test connection
                collections = self.client.get_collections()
                logger.info(f"Connected to Qdrant successfully. Collections: {[c.name for c in collections.collections]}")
                
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {str(e)}")
                self.client = None
        else:
            logger.warning("Qdrant client not available. Install qdrant-client package.")
    
    def get_videos_by_ids(self, video_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch video embeddings by video IDs from Qdrant - returns only video_id and embedding
        Since video_id is not indexed, we'll scroll through points and filter client-side
        """
        try:
            if not self.client or not video_ids:
                return []
            
            videos_data = []
            video_ids_set = set(video_ids)  # For faster lookup
            
            try:
                # Use scroll without filter to get all points, then filter client-side
                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=10000,  # Get a large batch of points
                    with_payload=["video_id"],  # Only get video_id from payload
                    with_vectors=True
                )
                
                # Filter the results client-side to match our video_ids
                for point in scroll_result[0]:  # scroll_result is (points, next_page_offset)
                    point_video_id = point.payload.get("video_id")
                    if point_video_id in video_ids_set:
                        # Extract only video_id and embedding
                        video_data = {
                            "video_id": point_video_id,
                            "embedding": list(point.vector) if point.vector else None
                        }
                        videos_data.append(video_data)
                        
                        # Remove found video_id from set to avoid duplicates
                        video_ids_set.remove(point_video_id)
                        
                        # Stop if we've found all requested videos
                        if not video_ids_set:
                            break
                            
            except Exception as scroll_error:
                logger.error(f"Error during scroll operation: {str(scroll_error)}")
                return []
            
            logger.info(f"Successfully fetched {len(videos_data)} video embeddings from Qdrant")
            return videos_data
            
        except Exception as e:
            logger.error(f"Error fetching videos by IDs from Qdrant: {str(e)}")
            return []
    
    def vector_similarity_search(self, query_embedding: List[float], 
                               similarity_threshold: float = 0.7,
                               limit: int = 50) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search in Qdrant - returns only video_id, embedding, and similarity
        """
        try:
            if not self.client or not query_embedding:
                return []
            
            # Validate and convert query_embedding if it's a string
            if isinstance(query_embedding, str):
                try:
                    import ast
                    query_embedding = ast.literal_eval(query_embedding.strip())
                    logger.info("Converted string embedding to list")
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Failed to parse query embedding string: {str(e)}")
                    return []
            
            # Ensure it's a list of floats
            if not isinstance(query_embedding, list):
                logger.error(f"Query embedding must be a list, got: {type(query_embedding)}")
                return []
            
            try:
                query_embedding = [float(x) for x in query_embedding]
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert embedding elements to float: {str(e)}")
                return []
            
            # Perform vector similarity search - only get vectors and video_id from payload
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=similarity_threshold,
                with_payload=["video_id"],  # Only get video_id from payload
                with_vectors=True
            )
            
            # Convert Qdrant results to minimal format - only embedding vectors
            videos_data = []
            for result in search_results:
                video_data = {
                    "video_id": result.payload.get("video_id"),
                    "embedding": list(result.vector) if result.vector else None,
                    "similarity": float(result.score)
                }
                videos_data.append(video_data)
            
            logger.info(f"Found {len(videos_data)} similar video embeddings from Qdrant")
            return videos_data
            
        except Exception as e:
            logger.error(f"Error in Qdrant vector similarity search: {str(e)}")
            return []
    
    def search_videos_by_text(self, text_query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search videos by text using payload filtering
        """
        try:
            if not self.client or not text_query:
                return []
            
            # Split query into keywords
            keywords = [word.lower().strip() for word in text_query.split() if len(word) > 2]
            
            if not keywords:
                return []
            
            # Create filter conditions for text search
            should_conditions = []
            for keyword in keywords:
                # Search in title
                should_conditions.append(
                    models.FieldCondition(
                        key="title",
                        match=models.MatchText(text=keyword)
                    )
                )
                # Search in description
                should_conditions.append(
                    models.FieldCondition(
                        key="description", 
                        match=models.MatchText(text=keyword)
                    )
                )
                # Search in tags
                should_conditions.append(
                    models.FieldCondition(
                        key="tags",
                        match=models.MatchAny(any=[keyword])
                    )
                )
            
            # Use scroll for text-based filtering
            search_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(should=should_conditions),
                limit=limit,
                with_payload=True,
                with_vectors=True
            )
            
            videos_data = []
            for point in search_result[0]:  # search_result is (points, next_page_offset)
                video_data = {
                    "video_id": point.payload.get("video_id"),
                    "title": point.payload.get("title", ""),
                    "description": point.payload.get("description", ""),
                    "channel_name": point.payload.get("channel_name", ""),
                    "duration_seconds": point.payload.get("duration_seconds"),
                    "view_count": point.payload.get("view_count"),
                    "publish_date": point.payload.get("publish_date"),
                    "embedding": list(point.vector) if point.vector else None,
                    "tags": point.payload.get("tags", []),
                    "category": point.payload.get("category", ""),
                    "qdrant_id": str(point.id),
                    "source": "qdrant_text_search"
                }
                videos_data.append(video_data)
            
            logger.info(f"Found {len(videos_data)} videos by text search")
            return videos_data
            
        except Exception as e:
            logger.error(f"Error in text search: {str(e)}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the video collection
        """
        try:
            if not self.client:
                return {}
            
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": collection_info.config.name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
                "optimizer_status": collection_info.optimizer_status
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {}

# Global Qdrant client instance
qdrant_client = QdrantVectorClient()
