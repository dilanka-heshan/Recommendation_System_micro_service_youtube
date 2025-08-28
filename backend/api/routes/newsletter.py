from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
from backend.database.supabase_client import supabase_client
from backend.models.response_models import BaseResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/newsletter", tags=["newsletter"])

@router.get("/health")
async def newsletter_health():
    """
    Simple health check for newsletter functionality
    """
    return {
        "status": "healthy",
        "message": "Newsletter storage is operational"
    }
