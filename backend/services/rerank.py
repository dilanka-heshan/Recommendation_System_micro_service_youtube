import logging
from typing import List, Dict, Any

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer, util
    import numpy as np
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

# LangSmith tracing
from langsmith import traceable

# Import MongoDB client for extractive summaries
from backend.database.mongodb_client import mongodb_client
from backend.database.qdrant_client import qdrant_client

logger = logging.getLogger(__name__)

class VideoReranker:
    """
    Two-stage video reranking service:
    Stage 1: User embedding + feedback videos with reranker → top 50
    Stage 2: Pairwise analysis with feedback video vectors → final ranking
    """
    
    def __init__(self):
        self.embed_model = None
        self.rerank_model = None
        
        if DEPENDENCIES_AVAILABLE:
            try:
                self.embed_model = SentenceTransformer("BAAI/bge-base-en")
                self.rerank_model = CrossEncoder("BAAI/bge-reranker-base")
                logger.info("Reranking models loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load reranking models: {str(e)}")

    def _get_video_text_representation(self, video: Dict[str, Any], use_extractive_summary: bool = True) -> str:
        """
        Get text representation of a video, preferring extractive summary if available
        
        Args:
            video: Video dictionary with metadata
            use_extractive_summary: Whether to try fetching extractive summary
            
        Returns:
            Text representation of the video
        """
        try:
            video_id = video.get('video_id', '')
            
            # Try to get extractive summary if enabled
            if use_extractive_summary and video_id:
                extractive_summary = mongodb_client.get_extractive_summary(video_id)
                if extractive_summary:
                    logger.debug(f"Using extractive summary for video {video_id}")
                    return extractive_summary.strip()
            
            # Fallback to title and description
            title = video.get('title', '')
            description = video.get('description', '')
            
            if title or description:
                combined_text = f"{title}. {description}".strip()
                return combined_text if combined_text != "." else "Unknown video"
            
            # Final fallback to video_id as readable text
            return video_id.replace('_', ' ').replace('-', ' ') if video_id else "Unknown video"
            
        except Exception as e:
            logger.error(f"Error getting text representation for video: {str(e)}")
            return video.get('title', 'Unknown video')

    # @traceable(name="two_stage_reranking")  # Disabled due to circular reference issues
    def rerank_with_user_history(self, user_history: List[Dict[str, Any]], 
                               candidate_videos: List[Dict[str, Any]], 
                               top_k: int = 10, 
                               agg: str = "mean") -> List[Dict[str, Any]]:
        """
        Two-stage reranking process:
        1. Stage 1: Reranker model with user embedding + feedback videos → top 50
        2. Stage 2: Pairwise analysis with feedback videos → final ranking
        
        Args:
            user_history: List of high-rating videos (HighRatingVideo model)
            candidate_videos: List of 100 candidate videos from vector retrieval
            top_k: Number of final videos to return
            agg: Aggregation method ("mean" or "max")
            
        Returns:
            List of ranked videos with scores
        """
        try:
            if not user_history or not candidate_videos:
                return candidate_videos[:top_k]
            
            if not DEPENDENCIES_AVAILABLE or not self.rerank_model:
                logger.warning("Reranking models not available, using fallback")
                return self._fallback_reranking(user_history, candidate_videos, top_k)
            
            # Stage 1: Reranker model to get top 50 from 100 candidates
            stage1_candidates = self._stage1_reranker_filtering(user_history, candidate_videos, top_k=50)
            logger.debug(f"Stage 1 candidates: {len(stage1_candidates)}")
            
            # Stage 2: Pairwise analysis for final ranking
            final_ranked = self._stage2_pairwise_analysis(user_history, stage1_candidates, top_k, agg)
            logger.debug(f"Final ranked: {len(final_ranked)}")
            
            logger.info(f"Two-stage reranking: {len(candidate_videos)} → {len(stage1_candidates)} → {len(final_ranked)}")
            
            
            return final_ranked
            
        except Exception as e:
            logger.error(f"Error in two-stage reranking: {str(e)}")
            return self._fallback_reranking(user_history, candidate_videos, top_k)
    
    def _stage1_reranker_filtering(self, user_history: List[Dict[str, Any]], 
                                 candidate_videos: List[Dict[str, Any]], 
                                 top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Stage 1: Use reranker model with user embedding + feedback videos to get top 50
        Enhanced with extractive summaries from MongoDB
        """
        try:
            # Create user query from feedback videos using extractive summaries
            user_query_parts = []
            
            # Get extractive summaries for user history videos
            history_video_ids = [video.get('video_id', '') for video in user_history if video.get('video_id')]
            history_summaries = mongodb_client.get_multiple_extractive_summaries(history_video_ids)
            
            for video in user_history:
                video_id = video.get('video_id', '')
                rating = video.get('rating', 5)
                
                # Try to use extractive summary, fallback to video_id
                if video_id in history_summaries:
                    video_text = history_summaries[video_id]
                else:
                    video_text = video_id.replace('_', ' ').replace('-', ' ')
                
                # Higher rating = more weight in query
                weight = int(rating) if rating >= 4 else 1
                user_query_parts.append(video_text)
            
            user_query = " ".join(user_query_parts)
            
            # Prepare reranker input pairs using extractive summaries for candidates
            reranker_input = []
            for video in candidate_videos:
                video_text = self._get_video_text_representation(video, use_extractive_summary=True)
                reranker_input.append((user_query, video_text))
            
            # Get reranker scores
            rerank_scores = self.rerank_model.predict(reranker_input)
            
            # Attach scores and get top candidates
            scored_videos = []
            for i, video in enumerate(candidate_videos):
                video_copy = video.copy()
                video_copy["stage1_score"] = float(rerank_scores[i])
                scored_videos.append(video_copy)
            
            # Sort by stage1 score and return top 50
            scored_videos.sort(key=lambda x: x["stage1_score"], reverse=True)
            
            logger.info(f"Stage 1: Used extractive summaries for {len(history_summaries)}/{len(history_video_ids)} history videos")
            
            return scored_videos[:top_k]
            
        except Exception as e:
            logger.error(f"Error in stage 1 reranking: {str(e)}")
            return candidate_videos[:top_k]
    
    # def _get_video_vectors_from_qdrant(self, video_ids: List[str]) -> Dict[str, List[float]]:
    #     """
    #     Get video vectors from Qdrant by video IDs
        
    #     Args:
    #         video_ids: List of video IDs to fetch vectors for
            
    #     Returns:
    #         Dictionary mapping video_id to embedding vector
    #     """
    #     try:
    #         from backend.database.qdrant_client import qdrant_client
            
    #         if not video_ids:
    #             return {}
            
    #         # Get videos with embeddings from Qdrant
    #         videos_data = qdrant_client.get_videos_by_ids(video_ids)
            
    #         # Create mapping of video_id to embedding
    #         video_vectors = {}
    #         for video in videos_data:
    #             video_id = video.get('video_id')
    #             embedding = video.get('embedding')
    #             if video_id and embedding:
    #                 video_vectors[video_id] = embedding
            
    #         logger.info(f"Retrieved vectors for {len(video_vectors)}/{len(video_ids)} videos from Qdrant")
    #         return video_vectors
            
    #     except Exception as e:
    #         logger.error(f"Error fetching video vectors from Qdrant: {str(e)}")
    #         return {}_get_video_vectors_from_qdrant
    
    def _stage2_pairwise_analysis(self, user_history: List[Dict[str, Any]], 
                                stage1_candidates: List[Dict[str, Any]], 
                                top_k: int, 
                                agg: str = "mean") -> List[Dict[str, Any]]:
        """
        Stage 2: Pairwise analysis between top 20 candidate video vectors and feedback video vectors
        Uses actual vectors from Qdrant for similarity computation
        """
        try:
            # Take top 20 candidates from stage 1 for vector-based analysis
            top_20_candidates = stage1_candidates[:20]
            #print(top_20_candidates)
            # Get video vectors for top 20 candidates from Qdrant
            candidate_video_ids = [video.get('video_id') for video in top_20_candidates if video.get('video_id')]
            candidate_vectors_data = qdrant_client.get_videos_by_ids(candidate_video_ids)
            
            # Convert to dictionary mapping video_id to embedding
            candidate_vectors = {}
            for video_data in candidate_vectors_data:
                video_id = video_data.get('video_id')
                embedding = video_data.get('embedding')
                if video_id and embedding:
                    candidate_vectors[video_id] = embedding
            
            # Get video vectors for user history from Qdrant
            history_video_ids = [video.get('video_id') for video in user_history if video.get('video_id')]
            history_vectors_data = qdrant_client.get_videos_by_ids(history_video_ids)
            
            # Convert to dictionary mapping video_id to embedding
            history_vectors = {}
            for video_data in history_vectors_data:
                video_id = video_data.get('video_id')
                embedding = video_data.get('embedding')
                if video_id and embedding:
                    history_vectors[video_id] = embedding
            
            logger.debug(f"Candidate vectors loaded: {len(candidate_vectors)} out of {len(candidate_video_ids)}")
            logger.debug(f"History vectors loaded: {len(history_vectors)} out of {len(history_video_ids)}")
            
            if not candidate_vectors or not history_vectors:
                logger.warning("Could not retrieve vectors from Qdrant, falling back to stage 1 results")
                return top_20_candidates[:top_k]
            
            # Calculate pairwise similarities between candidate and history vectors
            similarity_scores = {}
            
            for candidate_idx, candidate in enumerate(top_20_candidates):
                candidate_id = candidate.get('video_id')
                logger.debug(f"Processing candidate {candidate_idx}: {candidate_id}")
                
                if candidate_id not in candidate_vectors:
                    logger.debug(f"Candidate {candidate_id} not found in candidate_vectors")
                    continue
                    
                candidate_vector = candidate_vectors[candidate_id]
                similarities = []
                
                for history_video in user_history:
                    history_id = history_video.get('video_id')
                    if history_id not in history_vectors:
                        logger.debug(f"History video {history_id} not found in history_vectors")
                        continue
                        
                    history_vector = history_vectors[history_id]
                    rating = history_video.get('rating', 5)
                    
                    # Calculate cosine similarity
                    similarity = np.dot(candidate_vector, history_vector) / (
                        np.linalg.norm(candidate_vector) * np.linalg.norm(history_vector)
                    )
                    
                    # Weight by rating
                    weighted_similarity = similarity * (rating / 5.0)
                    similarities.append(weighted_similarity)
                logger.debug(f"Similarities for candidate {candidate_idx}: {len(similarities)} calculated")
                
                # Mean or get max
                if similarities:
                    if agg == 'mean':
                        similarity_scores[candidate_idx] = np.mean(similarities)
                    elif agg == 'max':
                        similarity_scores[candidate_idx] = np.max(similarities)
                    else:
                        similarity_scores[candidate_idx] = np.mean(similarities)
                else:
                    logger.debug(f"No similarities calculated for candidate {candidate_idx}")
            logger.debug(f"Final similarity_scores calculated for {len(similarity_scores)} candidates")
            # Build final results with similarity scores
            final_results = []
            for candidate_idx, score in similarity_scores.items():
                video = top_20_candidates[candidate_idx].copy()
                video["stage2_score"] = float(score)
                video["final_score"] = float(score)
                final_results.append(video)
            
            
            # Sort by final score
            final_results.sort(key=lambda x: x["final_score"], reverse=True)
            
            # Add final ranking
            for i, video in enumerate(final_results[:top_k]):
                video["final_rank"] = i + 1
            
            logger.info(f"Stage 2: Vector-based pairwise analysis with {len(candidate_vectors)} candidates and {len(history_vectors)} history videos")
            
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in stage 2 pairwise analysis: {str(e)}")
            return stage1_candidates[:top_k]

    

# Global reranker instance
video_reranker = VideoReranker()
