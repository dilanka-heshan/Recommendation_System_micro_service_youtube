from typing import Dict, Any, TYPE_CHECKING
import logging
from datetime import datetime
from langsmith import traceable

if TYPE_CHECKING:
    from backend.models.user_vector_update_models import UserVectorUpdateState

logger = logging.getLogger(__name__)

# @traceable(name="monitor_update_pipeline")  # Disabled due to circular reference issues
def monitor_update_pipeline_node(state: 'UserVectorUpdateState') -> 'UserVectorUpdateState':
    """
    Monitor and log pipeline execution metrics
    
    Functionality:
    - Track pipeline execution metrics
    - Log user update counts and success rates
    - Generate daily reports
    - Alert on failures or anomalies
    """
    try:
        logger.info("Starting pipeline monitoring and reporting")
        
        # Calculate execution time
        execution_time = state.get("execution_time")
        if not execution_time:
            # This should be set by the orchestrator, but calculate if missing
            execution_time = 0.0
        
        # Gather all metrics
        pipeline_metrics = state.get("pipeline_metrics", {})
        errors = state.get("errors", [])
        pipeline_step = state.get("pipeline_step", "unknown")
        date_range = state.get("date_range", {})
        
        # Create comprehensive monitoring report
        monitoring_report = {
            "execution_timestamp": datetime.utcnow().isoformat(),
            "pipeline_status": "success" if pipeline_step == "completed" else "failed",
            "execution_time_seconds": execution_time,
            "date_range": date_range,
            "pipeline_step": pipeline_step,
            "errors": errors,
            "metrics": pipeline_metrics
        }
        
        # Extract key metrics for logging
        feedback_metrics = {
            "total_feedback_records": pipeline_metrics.get("total_feedback_records", 0),
            "total_newsletter_records": pipeline_metrics.get("total_newsletter_records", 0),
            "users_with_feedback": pipeline_metrics.get("users_with_feedback", 0),
            "explicit_feedback_users": pipeline_metrics.get("explicit_feedback_users", 0),
            "implicit_feedback_users": pipeline_metrics.get("implicit_feedback_users", 0)
        }
        
        vector_metrics = {
            "retrieved_video_embeddings": pipeline_metrics.get("retrieved_video_embeddings", 0),
            "retrieved_user_vectors": pipeline_metrics.get("retrieved_user_vectors", 0),
            "updated_vectors_count": pipeline_metrics.get("updated_vectors_count", 0)
        }
        
        storage_metrics = pipeline_metrics.get("storage_stats", {})
        update_success_rate = pipeline_metrics.get("update_success_rate", 0)
        
        # Log detailed metrics
        logger.info("=== User Vector Update Pipeline Monitoring Report ===")
        logger.info(f"Status: {monitoring_report['pipeline_status'].upper()}")
        logger.info(f"Execution Time: {execution_time:.2f} seconds")
        logger.info(f"Date Range: {date_range.get('start_date', 'N/A')} to {date_range.get('end_date', 'N/A')}")
        
        logger.info("--- Feedback Metrics ---")
        logger.info(f"Total Feedback Records: {feedback_metrics['total_feedback_records']}")
        logger.info(f"Total Newsletter Records: {feedback_metrics['total_newsletter_records']}")
        logger.info(f"Users with Feedback: {feedback_metrics['users_with_feedback']}")
        logger.info(f"Explicit Feedback Users: {feedback_metrics['explicit_feedback_users']}")
        logger.info(f"Implicit Feedback Users: {feedback_metrics['implicit_feedback_users']}")
        
        logger.info("--- Vector Processing Metrics ---")
        logger.info(f"Video Embeddings Retrieved: {vector_metrics['retrieved_video_embeddings']}")
        logger.info(f"User Vectors Retrieved: {vector_metrics['retrieved_user_vectors']}")
        logger.info(f"User Vectors Updated: {vector_metrics['updated_vectors_count']}")
        
        logger.info("--- Storage Metrics ---")
        logger.info(f"Successful Updates: {storage_metrics.get('successful_updates', 0)}")
        logger.info(f"Failed Updates: {storage_metrics.get('failed_updates', 0)}")
        logger.info(f"Update Success Rate: {update_success_rate:.1f}%")
        
        if errors:
            logger.warning("--- Errors Encountered ---")
            for i, error in enumerate(errors[:5], 1):  # Log first 5 errors
                logger.warning(f"Error {i}: {error}")
            if len(errors) > 5:
                logger.warning(f"... and {len(errors) - 5} more errors")
        
        # Performance analysis and alerts
        _analyze_pipeline_performance(monitoring_report)
        
        # Update state with simple monitoring results (avoid circular references)
        state["pipeline_metrics"]["monitoring_completed"] = True
        state["pipeline_metrics"]["monitoring_timestamp"] = datetime.utcnow().isoformat()
        state["pipeline_step"] = "monitoring_completed"
        
        logger.info("=== Pipeline Monitoring Completed ===")
        
        return state
        
    except Exception as e:
        logger.error(f"Error in monitor_update_pipeline_node: {str(e)}")
        if "errors" not in state:
            state["errors"] = []
        state["errors"].append(f"Monitoring error: {str(e)}")
        return state


def _analyze_pipeline_performance(monitoring_report: Dict[str, Any]) -> None:
    """
    Analyze pipeline performance and generate alerts if needed
    """
    try:
        metrics = monitoring_report.get("metrics", {})
        
        # Performance thresholds
        EXECUTION_TIME_THRESHOLD = 300  # 5 minutes
        SUCCESS_RATE_THRESHOLD = 90  # 90%
        MIN_USERS_THRESHOLD = 1  # At least 1 user should have feedback
        
        # Check execution time
        execution_time = monitoring_report.get("execution_time_seconds", 0)
        if execution_time > EXECUTION_TIME_THRESHOLD:
            logger.warning(f"ALERT: Pipeline execution time ({execution_time:.1f}s) exceeds threshold ({EXECUTION_TIME_THRESHOLD}s)")
        
        # Check success rate
        success_rate = metrics.get("update_success_rate", 0)
        if success_rate < SUCCESS_RATE_THRESHOLD:
            logger.warning(f"ALERT: Update success rate ({success_rate:.1f}%) below threshold ({SUCCESS_RATE_THRESHOLD}%)")
        
        # Check if we have any users processed
        users_processed = metrics.get("users_with_feedback", 0)
        if users_processed < MIN_USERS_THRESHOLD:
            logger.warning(f"ALERT: Very few users processed ({users_processed}). Check data availability.")
        
        # Check for high error count
        errors = monitoring_report.get("errors", [])
        if len(errors) > 10:
            logger.warning(f"ALERT: High error count ({len(errors)}) detected in pipeline")
        
        # Performance summary
        if (execution_time <= EXECUTION_TIME_THRESHOLD and 
            success_rate >= SUCCESS_RATE_THRESHOLD and 
            users_processed >= MIN_USERS_THRESHOLD and 
            len(errors) <= 5):
            logger.info("✓ Pipeline performance within normal parameters")
        else:
            logger.warning("⚠ Pipeline performance issues detected - review alerts above")
            
    except Exception as e:
        logger.error(f"Error in performance analysis: {str(e)}")
