from typing import Dict, Any, TYPE_CHECKING
import logging
from datetime import datetime, timedelta
from langsmith import traceable
from backend.database.supabase_client import supabase_client
from backend.models.user_vector_update_models import DailyFeedbackRecord, NewsletterClickRecord

if TYPE_CHECKING:
    from backend.models.user_vector_update_models import UserVectorUpdateState

logger = logging.getLogger(__name__)

# @traceable(name="extract_daily_feedback")  # Disabled due to circular reference issues
def extract_daily_feedback_node(state: 'UserVectorUpdateState') -> 'UserVectorUpdateState':
    """
    Extract daily feedback data from feedback table and newsletter_videos table
    
    Functionality:
    - Query feedback table for yesterday's user ratings (1-5 scale)
    - Query newsletter_videos table for yesterday's sent newsletters
    - Query users table to get current embedding_ids for active users
    - Aggregate data by user_id with rating-based weighting
    - Filter for active users with feedback/clicks
    """
    try:
        logger.info("Starting daily feedback extraction")

        # Get date range (yesterday by default)
        date_range = state.get("date_range", {})
        
        # Check if date_range is valid and has required keys
        if not date_range or "start_date" not in date_range or "end_date" not in date_range:
            logger.info("No valid date range provided, using yesterday as default")
            yesterday = datetime.now() - timedelta(days=3)
            today = datetime.now() + timedelta(days=1)
            date_range = {
                "start_date": yesterday.strftime("%Y-%m-%d"),
                "end_date": today.strftime("%Y-%m-%d")
            }
            state["date_range"] = date_range
        
        start_date = date_range["start_date"]
        end_date = date_range["end_date"]
        logger.info(f"Extracting feedback for date range: {start_date} to {end_date}")
        
        # Extract daily feedback from feedback table
        feedback_data = supabase_client.get_daily_feedback(start_date, end_date)
        logger.info(f"Retrieved {len(feedback_data)} feedback records")
        
        # Extract newsletter click data
        #newsletter_data = supabase_client.get_newsletter_click_data(start_date, end_date)
        #logger.info(f"Retrieved {len(newsletter_data)} newsletter click records")
        
        # Get active users with embeddings
        active_users = supabase_client.get_active_users_with_embeddings()
        active_user_map = {user["user_id"]: user["embedding_id"] for user in active_users if user.get("embedding_id")}
        logger.info(f"Found {len(active_user_map)} active users with embeddings")
        
        # Aggregate feedback by user_id
        user_feedback_aggregated = {}
        user_embedding_ids = {}
        
        # Process explicit feedback (ratings 1-5)
        for feedback in feedback_data:
            user_id = feedback["user_id"]
            
            # Only process users with embeddings
            if user_id not in active_user_map:
                continue
                
            if user_id not in user_feedback_aggregated:
                user_feedback_aggregated[user_id] = []
                user_embedding_ids[user_id] = active_user_map[user_id]
            
            # Apply rating-based weights as per pipeline document
            rating = feedback["rating"]
            weight = _get_rating_weight(rating)
            
            feedback_record = {
                "video_id": feedback["video_id"],
                "rating": rating,
                "weight": weight,
                "timestamp": feedback["timestamp"],
                "source": "feedback"
            }
            user_feedback_aggregated[user_id].append(feedback_record)
        
        # Process newsletter click data (implicit feedback)
        # for newsletter in newsletter_data:
        #     user_id = newsletter["user_id"]
            
        #     # Only process users with embeddings
        #     if user_id not in active_user_map:
        #         continue
                
        #     if user_id not in user_feedback_aggregated:
        #         user_feedback_aggregated[user_id] = []
        #         user_embedding_ids[user_id] = active_user_map[user_id]
            
        #     # Treat clicks as positive feedback (rating 4), no clicks as negative (rating 2)
        #     implicit_rating = 4 if newsletter["clicked"] else 2
        #     weight = _get_rating_weight(implicit_rating)
            
        #     feedback_record = {
        #         "video_id": newsletter["video_id"],
        #         "rating": implicit_rating,
        #         "weight": weight,
        #         "timestamp": newsletter.get("sent_at", datetime.now().isoformat()),
        #         "source": "newsletter_click" if newsletter["clicked"] else "newsletter_no_click"
        #     }
        #     user_feedback_aggregated[user_id].append(feedback_record)
        
        # Filter users with actual feedback
        filtered_feedback = {
            user_id: feedback_list 
            for user_id, feedback_list in user_feedback_aggregated.items() 
            if len(feedback_list) > 0
        }
        
        # Update state
        state["user_feedback_data"] = filtered_feedback
        state["user_embedding_ids"] = {
            user_id: user_embedding_ids[user_id] 
            for user_id in filtered_feedback.keys()
        }
        state["pipeline_step"] = "feedback_extracted"
        
        # Update metrics
        if "pipeline_metrics" not in state:
            state["pipeline_metrics"] = {}
        
        state["pipeline_metrics"].update({
            "total_feedback_records": len(feedback_data),
            #"total_newsletter_records": len(newsletter_data),
            "active_users_count": len(active_user_map),
            "users_with_feedback": len(filtered_feedback),
            "explicit_feedback_users": len([u for u, f in filtered_feedback.items() 
                                          if any(r["source"] == "feedback" for r in f)]),
            # "implicit_feedback_users": len([u for u, f in filtered_feedback.items() 
            #                               if any(r["source"].startswith("newsletter") for r in f)])
        })
        
        logger.info(f"Feedback extraction completed. Users with feedback: {len(filtered_feedback)}")
        return state
        
    except Exception as e:
        logger.error(f"Error in extract_daily_feedback_node: {str(e)}")
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Feedback extraction error: {str(e)}")
        state["pipeline_step"] = "error"
        return state


def _get_rating_weight(rating: int) -> float:
    """
    Get weight for rating based on pipeline document specifications
    Rating weights: 5=1.0, 4=0.75, 3=0.0, 2=0.75, 1=1.0
    """
    rating_weights = {
        5: 1.0,    # Strongly positive - full weight
        4: 0.75,   # Positive - reduced weight
        3: 0.0,    # Neutral - ignored
        2: 0.75,   # Negative - reduced weight
        1: 1.0     # Strongly negative - full weight
    }
    return rating_weights.get(rating, 0.0)
