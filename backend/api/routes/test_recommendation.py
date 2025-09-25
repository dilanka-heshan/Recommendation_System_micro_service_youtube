from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import logging

from backend.database.supabase_client import supabase_client

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/random-videos")
async def get_random_videos(limit: int = 4) -> Dict[str, Any]:
    """
    Get random videos from the database for testing purposes
    
    Args:
        limit: Number of random videos to fetch (default: 4, max: 20)
    
    Returns:
        Dictionary containing list of random videos and metadata
    """
    try:
        # Validate limit parameter
        if limit < 1 or limit > 20:
            raise HTTPException(status_code=400, detail="Limit must be between 1 and 20")
        
        # Fetch random videos from database
        random_videos = supabase_client.get_random_videos(limit=limit)
        
        if not random_videos:
            raise HTTPException(status_code=404, detail="No videos found in database")
        
        return {
            "status": "success",
            "count": len(random_videos),
            "videos": random_videos,
            "message": f"Successfully retrieved {len(random_videos)} random videos"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_random_videos endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/test-database-connection")
async def test_database_connection() -> Dict[str, str]:
    """
    Test the Supabase database connection
    
    Returns:
        Connection status message
    """
    try:
        # Try to fetch one video to test connection
        response = supabase_client.client.table("videos").select("video_id").limit(1).execute()
        
        if response.data is not None:
            return {
                "status": "success",
                "message": "Database connection successful",
                "table_accessible": "videos table is accessible"
            }
        else:
            return {
                "status": "warning",
                "message": "Database connected but no data found",
                "table_accessible": "videos table is empty or inaccessible"
            }
            
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

@router.get("/debug-database")
async def debug_database() -> Dict[str, Any]:
    """
    Debug endpoint to check database status and table information
    
    Returns:
        Detailed database debug information
    """
    try:
        debug_info = {}
        
        # Test basic connection
        try:
            # Try to get count of videos
            count_response = supabase_client.client.table("videos").select("*", count="exact").execute()
            debug_info["videos_count"] = count_response.count
            debug_info["connection_status"] = "success"
        except Exception as e:
            debug_info["connection_error"] = str(e)
            debug_info["connection_status"] = "failed"
        
        # Try to get first video
        try:
            first_video = supabase_client.client.table("videos").select("*").limit(1).execute()
            if first_video.data:
                debug_info["sample_video_columns"] = list(first_video.data[0].keys()) if first_video.data else []
                debug_info["has_data"] = True
            else:
                debug_info["has_data"] = False
        except Exception as e:
            debug_info["query_error"] = str(e)
        
        # Check if table exists by trying different table names
        possible_tables = ["videos", "video", "youtube_videos", "content"]
        debug_info["table_tests"] = {}
        
        for table_name in possible_tables:
            try:
                test_resp = supabase_client.client.table(table_name).select("*").limit(1).execute()
                debug_info["table_tests"][table_name] = {
                    "exists": True,
                    "has_data": len(test_resp.data) > 0 if test_resp.data else False
                }
            except Exception as e:
                debug_info["table_tests"][table_name] = {
                    "exists": False,
                    "error": str(e)
                }
        
        return debug_info
        
    except Exception as e:
        logger.error(f"Database debug failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database debug failed: {str(e)}")
