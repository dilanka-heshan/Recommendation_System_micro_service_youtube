"""
Recommendation service that interfaces with the orchestrator
"""
from typing import Dict, Any, List
from backend.pipelines.orchestrator import recommendation_orchestrator

def get_recommendations(user_id: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Get recommendations for a user using the orchestrator
    
    Args:
        user_id: The ID of the user to get recommendations for
        top_k: Number of recommendations to return
        
    Returns:
        Dictionary containing recommendations and metadata
    """
    return recommendation_orchestrator.generate_recommendations(user_id=user_id, top_k=top_k)



