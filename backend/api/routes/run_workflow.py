from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from backend.pipelines.orchestrator import recommendation_orchestrator

router = APIRouter()

class WorkflowRequest(BaseModel):
    user_id: str
    top_k: int = 10

class WorkflowResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, Any]]
    newsletter_id: str = None
    metadata: Dict[str, Any]

class WorkflowVideoIdsResponse(BaseModel):
    user_id: str
    video_ids: List[str]
    total_count: int

@router.post("/run-workflow", response_model=WorkflowResponse)
async def run_workflow(request: WorkflowRequest):
    """
    Trigger the YouTube-to-newsletter workflow for a user.
    """
    try:
        # Call orchestrator to run the workflow for the user
        result = recommendation_orchestrator.generate_recommendations(
            user_id=request.user_id, 
            top_k=request.top_k
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow failed: {e}")

@router.post("/run-workflow-video-ids", response_model=WorkflowVideoIdsResponse)
async def run_workflow_video_ids(request: WorkflowRequest):
    """
    Trigger the YouTube-to-newsletter workflow for a user and return only video IDs.
    """
    try:
        # Call orchestrator to run the workflow for the user
        result = recommendation_orchestrator.generate_recommendations(
            user_id=request.user_id, 
            top_k=request.top_k
        )
        
        # Extract only video IDs from the recommendations
        video_ids = []
        recommendations = result.get("final_list", []) or result.get("recommendations", [])
        
        for video in recommendations:
            if isinstance(video, dict) and video.get("video_id"):
                video_ids.append(video["video_id"])
        
        return WorkflowVideoIdsResponse(
            user_id=request.user_id,
            video_ids=video_ids,
            total_count=len(video_ids)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow failed: {e}")