# from fastapi import APIRouter
# from backend.database.supabase_client import get_user_embedding

# router = APIRouter()

# @router.get("/{user_id}/embedding")
# def fetch_user_embedding(user_id :str):
#     """
#     Fetch current user embedding from supabase (for debugging or LangSmith visualization)
#     """
#     embedding = get_user_embedding(user_id)
#     return {"user_id": user_id, "embedding": embedding}
