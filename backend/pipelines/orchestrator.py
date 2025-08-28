from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# LangSmith tracing
from langsmith import traceable

# Import database client for storing newsletters
from backend.database.supabase_client import supabase_client

# Import pipeline models and nodes
from backend.models.pipeline_models import PipelineState
from backend.pipelines.fetch_user_data_node import fetch_user_data_node
from backend.pipelines.vector_retrieval_node import vector_retrieval_node
from backend.pipelines.diversity_filter_node import diversity_filter_node
from backend.pipelines.rerank_videos_node import rerank_videos_node
from backend.pipelines.store_newsletter_node import store_newsletter_node

logger = logging.getLogger(__name__)

class RecommendationOrchestrator:
    """
    Enhanced orchestrator for vector-only YouTube recommendation pipeline with optimized flow:
    1. Fetch user data (user profile + feedback videos + watch history)
    2. Vector retrieval (similarity search + time decay + watched filter)
    3. Two-stage reranking (cross-encoder → top 50 → pairwise analysis)
    4. Diversity filtering (MMR algorithm on reranked results)
    5. Store newsletter (save final recommendations to database)
    """
    
    def __init__(self):
        self.graph = None
        self._build_graph()
    
    def _build_graph(self):
        """Build the simplified LangGraph workflow"""
        try:
            if StateGraph is None:
                logger.warning("LangGraph not available, using fallback implementation")
                return
            
            workflow = StateGraph(PipelineState)
            
            # Add pipeline nodes from separate files
            workflow.add_node("fetch_user_data", fetch_user_data_node)
            workflow.add_node("vector_retrieval", vector_retrieval_node)
            workflow.add_node("diversity_filter", diversity_filter_node)
            workflow.add_node("rerank_videos", rerank_videos_node)
            workflow.add_node("store_newsletter", store_newsletter_node)
            
            # Define the enhanced flow: reranking before diversity filtering, then store newsletter
            workflow.set_entry_point("fetch_user_data")
            workflow.add_edge("fetch_user_data", "vector_retrieval")
            workflow.add_edge("vector_retrieval", "rerank_videos")
            workflow.add_edge("rerank_videos", "diversity_filter")
            workflow.add_edge("diversity_filter", "store_newsletter")
            workflow.add_edge("store_newsletter", END)
            
            self.graph = workflow.compile()
            logger.info("LangGraph workflow compiled successfully")
            
        except Exception as e:
            logger.error(f"Error building LangGraph workflow: {str(e)}")
            self.graph = None
    
    @traceable(name="recommendation_pipeline")
    def generate_recommendations(self, user_id: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Main entry point for generating recommendations
        """
        start_time = datetime.utcnow()
        
        try:
            initial_state: PipelineState = {
                "user_id": user_id,
                "top_k": top_k,
                "user_embedding": None,
                "high_rating_videos": None,
                "candidate_videos": None,
                "final_list": None,
                "pipeline_step": "initialized",
                "error": None,
                "is_new_user": False,
                "execution_time": None,
                "newsletter_id": None
            }
            
            if self.graph:
                result = self.graph.invoke(initial_state)
                # LangGraph returns the state dictionary directly
            else:
                # Sequential execution when LangGraph is not available
                result = initial_state
                result = fetch_user_data_node(result)
                if not result.get("error"):
                    result = vector_retrieval_node(result)
                if not result.get("error"):
                    result = rerank_videos_node(result)
                if not result.get("error"):
                    result = diversity_filter_node(result)
                if not result.get("error"):
                    result = store_newsletter_node(result)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            result["execution_time"] = execution_time
            
            logger.info(f"Pipeline completed for user {user_id} in {execution_time:.2f}s")
            
            return {
                "user_id": user_id,
                "recommendations": result.get("final_list", []),
                "newsletter_id": result.get("newsletter_id"),  # Include newsletter ID from pipeline state
                "metadata": {
                    "execution_time": execution_time,
                    "total_candidates": len(result.get("candidate_videos", [])),
                    "is_new_user": result.get("is_new_user", False),
                    "pipeline_step": result.get("pipeline_step", "completed"),
                    "newsletter_stored": result.get("newsletter_id") is not None
                }
            }
            
        except Exception as e:
            logger.error(f"Error in recommendation pipeline for user {user_id}: {str(e)}")
            return {
                "user_id": user_id,
                "recommendations": [],
                "error": str(e),
                "metadata": {
                    "execution_time": (datetime.utcnow() - start_time).total_seconds(),
                    "pipeline_step": "error"
                }
            }
    
# Global orchestrator instance
recommendation_orchestrator = RecommendationOrchestrator()
