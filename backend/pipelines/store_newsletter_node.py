from typing import Dict, Any
import logging
from backend.models.pipeline_models import PipelineState
from backend.database.supabase_client import supabase_client

logger = logging.getLogger(__name__)

def store_newsletter_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node for storing final recommendations in newsletter tables
    """
    try:
        state["pipeline_step"] = "storing_newsletter"
        logger.info(f"Starting newsletter storage for user {state['user_id']}")
        
        # Check if we have final recommendations to store
        final_list = state.get("final_list")
        if not final_list:
            logger.warning(f"No recommendations to store for user {state['user_id']}")
            state["newsletter_id"] = None
            return state
        
        # Store newsletter in Supabase
        newsletter_id = supabase_client.create_newsletter(state["user_id"], final_list)
        
        if newsletter_id:
            state["newsletter_id"] = newsletter_id
            logger.info(f"Stored newsletter {newsletter_id} for user {state['user_id']} with {len(final_list)} videos")
        else:
            logger.error(f"Failed to store newsletter for user {state['user_id']}")
            state["newsletter_id"] = None
        
        state["pipeline_step"] = "newsletter_stored"
        
        return state
        
    except Exception as e:
        logger.error(f"Error in store_newsletter_node: {str(e)}")
        state["error"] = f"Newsletter storage failed: {str(e)}"
        state["newsletter_id"] = None
        return state
