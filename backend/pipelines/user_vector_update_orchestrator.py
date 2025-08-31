# User Vector Update Orchestrator
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import json
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# LangSmith tracing
from langsmith import traceable

# Import models and nodes
from backend.models.user_vector_update_models import UserVectorUpdateState
from backend.pipelines.user_vector_update.extract_daily_feedback_node import extract_daily_feedback_node
from backend.pipelines.user_vector_update.retrieve_user_vectors_node import retrieve_user_vectors_node
from backend.pipelines.user_vector_update.calculate_user_vectors_node import calculate_user_vectors_node
from backend.pipelines.user_vector_update.store_updated_vectors_node import store_updated_vectors_node
from backend.pipelines.user_vector_update.monitor_update_pipeline_node import monitor_update_pipeline_node

logger = logging.getLogger(__name__)
# check this function needed
def _ensure_json_serializable(obj):
    """
    Recursively ensure an object is JSON serializable
    """
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        if isinstance(obj, dict):
            return {key: _ensure_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [_ensure_json_serializable(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        else:
            return str(obj)

class UserVectorUpdateOrchestrator:
    """
    Orchestrator for daily user vector update pipeline using Rocchio's Algorithm
    
    Pipeline Flow:
    1. Extract daily feedback data (ratings + newsletter clicks)
    2. Retrieve current user vectors and video embeddings
    3. Calculate updated user vectors using Rocchio's Algorithm
    4. Store updated vectors in Supabase
    """
    
    def __init__(self):
        self.graph = None
        self._build_graph()
    
    def _build_graph(self):
        """Build the User Vector Update LangGraph workflow"""
        try:
            workflow = StateGraph(UserVectorUpdateState)
            
            # Add pipeline nodes
            workflow.add_node("extract_feedback", extract_daily_feedback_node)
            workflow.add_node("retrieve_vectors", retrieve_user_vectors_node)
            workflow.add_node("calculate_vectors", calculate_user_vectors_node)
            workflow.add_node("store_vectors", store_updated_vectors_node)
            workflow.add_node("monitor_pipeline", monitor_update_pipeline_node)
            
            # Define the pipeline flow
            workflow.set_entry_point("extract_feedback")
            workflow.add_edge("extract_feedback", "retrieve_vectors")
            workflow.add_edge("retrieve_vectors", "calculate_vectors")
            workflow.add_edge("calculate_vectors", "store_vectors")
            workflow.add_edge("store_vectors", "monitor_pipeline")
            workflow.add_edge("monitor_pipeline", END)
            
            self.graph = workflow.compile()
            logger.info("User Vector Update LangGraph workflow compiled successfully")
            
        except Exception as e:
            logger.error(f"Error building User Vector Update LangGraph workflow: {str(e)}")
            self.graph = None
    
    # @traceable(name="user_vector_update_pipeline")  # Disabled due to circular reference issues
    def run_daily_update(self, date_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Main entry point for running daily user vector updates
        
        Args:
            date_range: Optional dict with 'start_date' and 'end_date' in YYYY-MM-DD format
                       If not provided, defaults to yesterday
        
        Returns:
            Dict containing update results and metrics
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate and set default date range if not provided
            if not date_range or not isinstance(date_range, dict):
                yesterday = datetime.now() - timedelta(days=3)
                today = datetime.now() + timedelta(days=1)
                date_range = {
                    "start_date": yesterday.strftime("%Y-%m-%d"),
                    "end_date": today.strftime("%Y-%m-%d")
                }
                logger.info(f"Using default date range: {date_range}")
            elif "start_date" not in date_range or "end_date" not in date_range:
                # If date_range is provided but missing keys, use defaults
                yesterday = datetime.now() - timedelta(days=3)
                today = datetime.now() + timedelta(days=1)
                date_range = {
                    "start_date": yesterday.strftime("%Y-%m-%d"),
                    "end_date": today.strftime("%Y-%m-%d")
                }
                logger.info(f"Invalid date range provided, using default: {date_range}")
            
            # Validate date format
            try:
                datetime.strptime(date_range["start_date"], "%Y-%m-%d")
                datetime.strptime(date_range["end_date"], "%Y-%m-%d")
            except ValueError as e:
                logger.warning(f"Invalid date format in date_range: {e}. Using defaults.")
                yesterday = datetime.now() - timedelta(days=1)
                date_range = {
                    "start_date": yesterday.strftime("%Y-%m-%d"),
                    "end_date": yesterday.strftime("%Y-%m-%d")
                }
                logger.debug(f"Using fallback date range: {date_range}")
            
            initial_state: UserVectorUpdateState = {
                "date_range": date_range,
                "user_feedback_data": {},
                "user_embedding_ids": {},
                "current_user_vectors": {},
                "video_embeddings": {},
                "updated_user_vectors": {},
                "new_embedding_ids": {},
                "pipeline_metrics": {},
                "pipeline_step": "initialized",
                "errors": [],
                "execution_time": None
            }
            
            logger.info(f"Starting user vector update pipeline for date range: {date_range}")
            
            if self.graph:
                # Run the LangGraph workflow
                result = self.graph.invoke(initial_state)
            else:
                # Sequential execution fallback when LangGraph is not available
                logger.warning("LangGraph not available, using sequential execution")
                result = initial_state
                result = extract_daily_feedback_node(result)
                if not result.get("errors"):
                    result = retrieve_user_vectors_node(result)
                if not result.get("errors"):
                    result = calculate_user_vectors_node(result)
                if not result.get("errors"):
                    result = store_updated_vectors_node(result)
                if not result.get("errors"):
                    result = monitor_update_pipeline_node(result)
            
            # Calculate total execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result["execution_time"] = execution_time
            
            # Extract final metrics and ensure they're JSON serializable
            pipeline_metrics = result.get("pipeline_metrics", {})
            errors = result.get("errors", [])
            pipeline_step = result.get("pipeline_step", "unknown")
            
            # Clean metrics to ensure JSON serialization
            clean_metrics = _ensure_json_serializable(pipeline_metrics)
            
            # Ensure errors are strings
            clean_errors = [str(error) for error in errors] if errors else []
            
            logger.info(f"User vector update pipeline completed in {execution_time:.2f}s")
            
            return {
                "status": "success" if pipeline_step in ["completed", "monitoring_completed"] else "failed",
                "execution_time": execution_time,
                "date_range": date_range,
                "pipeline_step": pipeline_step,
                "errors": clean_errors,
                "metrics": clean_metrics,
                "summary": {
                    "users_processed": clean_metrics.get("users_with_feedback", 0),
                    "vectors_updated": clean_metrics.get("updated_vectors_count", 0),
                    "success_rate": clean_metrics.get("update_success_rate", 0),
                    "total_feedback_records": clean_metrics.get("total_feedback_records", 0),
                    "total_newsletter_records": clean_metrics.get("total_newsletter_records", 0)
                }
            }
            
        except Exception as e:
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            logger.error(f"Error in user vector update pipeline: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "execution_time": execution_time,
                "date_range": date_range,
                "pipeline_step": "error"
            }
    
    # @traceable(name="manual_user_vector_update")  # Disabled due to circular reference issues
    def run_manual_update(self, user_ids: List[str], date_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Run vector update for specific users manually
        
        Args:
            user_ids: List of user IDs to update
            date_range: Optional date range for feedback data
        
        Returns:
            Dict containing update results
        """
        logger.info(f"Running manual vector update for {len(user_ids)} users")
        
        # For now, this will run the full pipeline but could be optimized to filter by user_ids
        # This is a future enhancement
        result = self.run_daily_update(date_range)
        
        # Create a clean response without circular references
        clean_result = {
            "status": result.get("status", "unknown"),
            "execution_time": result.get("execution_time", 0),
            "date_range": result.get("date_range", {}),
            "pipeline_step": result.get("pipeline_step", "unknown"),
            "errors": result.get("errors", []),
            "metrics": result.get("metrics", {}),
            "summary": result.get("summary", {}),
            "manual_update": True,
            "requested_user_ids": user_ids
        }
        
        return clean_result

# Global orchestrator instance
user_vector_update_orchestrator = UserVectorUpdateOrchestrator()
