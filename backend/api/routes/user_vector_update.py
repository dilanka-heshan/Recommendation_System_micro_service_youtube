# API Routes for User Vector Update Pipeline
from fastapi import APIRouter, Body, HTTPException, Query
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging

from backend.pipelines.user_vector_update_orchestrator import user_vector_update_orchestrator

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/run-daily-update")
def run_daily_vector_update(
    date_range: Optional[Dict[str, str]] = Body(
        None, 
        description="Optional date range with start_date and end_date in YYYY-MM-DD format. Defaults to yesterday if not provided."
    )
):
    """
    Trigger daily user vector update pipeline
    
    This endpoint runs the complete user vector update pipeline using Rocchio's algorithm
    to update user preference vectors based on feedback and newsletter interactions.
    """
    try:
        logger.info("Triggering daily user vector update pipeline via API")
        
        result = user_vector_update_orchestrator.run_daily_update(date_range)
        
        # Return the result directly without wrapping to avoid circular references
        return result
            
    except Exception as e:
        logger.error(f"Error running daily vector update via API: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run daily vector update: {str(e)}"
        )

@router.post("/run-manual-update")
def run_manual_vector_update(
    user_ids: List[str] = Body(..., description="List of user IDs to update"),
    date_range: Optional[Dict[str, str]] = Body(
        None, 
        description="Optional date range with start_date and end_date in YYYY-MM-DD format"
    )
):
    """
    Trigger manual user vector update for specific users
    """
    try:
        logger.info(f"Triggering manual vector update for {len(user_ids)} users via API")
        
        result = user_vector_update_orchestrator.run_manual_update(user_ids, date_range)
        
        # Return the result directly without wrapping to avoid circular references
        return result
            
    except Exception as e:
        logger.error(f"Error running manual vector update via API: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run manual vector update: {str(e)}"
        )

@router.get("/status")
def get_pipeline_status():
    """
    Get current status and health of the user vector update pipeline
    """
    try:
        # Basic health check - try to import the orchestrator
        from backend.pipelines.user_vector_update_orchestrator import user_vector_update_orchestrator
        
        return {
            "status": "healthy",
            "message": "User vector update pipeline is available",
            "graph_available": user_vector_update_orchestrator.graph is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error checking pipeline status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline health check failed: {str(e)}"
        )
