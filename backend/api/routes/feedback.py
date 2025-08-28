from fastapi import APIRouter, Body
# from backend.services.feedback_service import process_feedback

router = APIRouter()

@router.post("/")
def submit_feedback(
    user_id: str = Body(...),
    video_id: str = Body(...),
    feedback: str = Body(... , description= "positive or negative")
):

    """
    Receive user feedback and update the user embedding in Supabase.
    """

    success = process_feedback(user_id, video_id, feedback)
    return {"status": success if success else "failed"}

