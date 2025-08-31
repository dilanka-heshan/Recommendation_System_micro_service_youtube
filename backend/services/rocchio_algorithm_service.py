# Rocchio's Algorithm Service for User Vector Updates
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from backend.models.user_vector_update_models import RocchioParameters, UserFeedbackAggregation

logger = logging.getLogger(__name__)

class RocchioAlgorithmService:
    """
    Service implementing Rocchio's Algorithm for user vector updates
    Formula: Updated_User_Vector = α * Original_User_Vector + β * Relevant_Docs_Centroid - γ * Non_Relevant_Docs_Centroid
    """
    
    def __init__(self, parameters: Optional[RocchioParameters] = None):
        self.params = parameters or RocchioParameters()
        logger.info(f"Initialized Rocchio Algorithm with α={self.params.alpha}, β={self.params.beta}, γ={self.params.gamma}")
    
    def calculate_weighted_centroid(self, video_embeddings: List[List[float]], weights: List[float]) -> List[float]:
        """
        Calculate weighted centroid of video embeddings
        Args:
            video_embeddings: List of video embedding vectors
            weights: List of weights for each embedding
        Returns:
            Weighted centroid vector
        """
        if not video_embeddings or not weights:
            return [0.0] * 768  # Return zero vector for empty input
        
        try:
            # Convert to numpy arrays for efficient computation
            embeddings_array = np.array(video_embeddings)
            weights_array = np.array(weights)
            
            # Calculate weighted sum
            weighted_sum = np.sum(embeddings_array * weights_array.reshape(-1, 1), axis=0)
            
            # Normalize by sum of weights
            total_weight = np.sum(weights_array)
            if total_weight > 0:
                centroid = weighted_sum / total_weight
            else:
                centroid = np.zeros(embeddings_array.shape[1])
            
            return centroid.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating weighted centroid: {str(e)}")
            return [0.0] * 768
    
    def classify_feedback_by_rating(self, feedback_records: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Classify feedback records into positive, negative, and neutral categories
        Args:
            feedback_records: List of feedback records with rating field
        Returns:
            Tuple of (positive_feedback, negative_feedback, neutral_feedback)
        """
        positive_feedback = []
        negative_feedback = []
        neutral_feedback = []
        
        for record in feedback_records:
            rating = record.get("rating", 0)
            
            if rating in self.params.positive_ratings:
                positive_feedback.append(record)
            elif rating in self.params.negative_ratings:
                negative_feedback.append(record)
            elif rating in self.params.neutral_ratings:
                neutral_feedback.append(record)
            else:
                logger.warning(f"Unknown rating value: {rating}")
        
        logger.info(f"Classified feedback: {len(positive_feedback)} positive, {len(negative_feedback)} negative, {len(neutral_feedback)} neutral")
        return positive_feedback, negative_feedback, neutral_feedback
    
    def calculate_rating_weights(self, feedback_records: List[Dict[str, Any]]) -> List[float]:
        """
        Calculate weights for feedback records based on ratings
        Args:
            feedback_records: List of feedback records with rating field
        Returns:
            List of weights corresponding to each feedback record
        """
        weights = []
        for record in feedback_records:
            rating = record.get("rating", 3)  # Default to neutral
            weight = self.params.rating_weights.get(rating, 0.0)
            weights.append(weight)
        
        return weights
    
    # def process_newsletter_clicks(self, click_records: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    #     """
    #     Process newsletter click data into positive and negative feedback
    #     Args:
    #         click_records: List of newsletter click records
    #     Returns:
    #         Tuple of (clicked_videos, unclicked_videos)
    #     """
    #     clicked_videos = []
    #     unclicked_videos = []
        
    #     for record in click_records:
    #         if record.get("clicked", False):
    #             # Treat clicks as positive feedback (rating 4)
    #             clicked_record = record.copy()
    #             clicked_record["rating"] = 4
    #             clicked_record["source"] = "newsletter_click"
    #             clicked_videos.append(clicked_record)
    #         else:
    #             # Treat non-clicks as weak negative feedback (rating 2)
    #             unclicked_record = record.copy()
    #             unclicked_record["rating"] = 2
    #             unclicked_record["source"] = "newsletter_no_click"
    #             unclicked_videos.append(unclicked_record)
        
    #     logger.info(f"Processed newsletter clicks: {len(clicked_videos)} clicked, {len(unclicked_videos)} not clicked")
    #     return clicked_videos, unclicked_videos
    
    def aggregate_user_feedback(self, 
                               feedback_records: List[Dict[str, Any]], 
                               click_records: List[Dict[str, Any]],
                               user_id: str,
                               embedding_id: Optional[str] = None) -> UserFeedbackAggregation:
        """
        Aggregate all feedback for a user into positive, negative, and neutral categories
        Args:
            feedback_records: User's rating feedback records
            click_records: User's newsletter click records
            user_id: User identifier
            embedding_id: User's embedding identifier
        Returns:
            UserFeedbackAggregation object
        """
        try:
            # Process explicit ratings
            pos_ratings, neg_ratings, neu_ratings = self.classify_feedback_by_rating(feedback_records)
            
            # # Process newsletter clicks
            # clicked_videos, unclicked_videos = self.process_newsletter_clicks(click_records)
            
            # Combine positive feedback (high ratings + clicks)
            positive_videos = pos_ratings # + clicked_videos
            
            # Combine negative feedback (low ratings + no clicks)
            negative_videos = neg_ratings # + unclicked_videos
            
            # Neutral feedback (only from explicit ratings)
            neutral_videos = neu_ratings
            
            total_feedback = len(feedback_records) # + len(click_records)
            
            aggregation = UserFeedbackAggregation(
                user_id=user_id,
                embedding_id=embedding_id,
                positive_videos=positive_videos,
                negative_videos=negative_videos,
                neutral_videos=neutral_videos,
                total_feedback_count=total_feedback
            )
            
            logger.info(f"User {user_id} feedback aggregation: {len(positive_videos)} positive, {len(negative_videos)} negative, {len(neutral_videos)} neutral")
            return aggregation
            
        except Exception as e:
            logger.error(f"Error aggregating feedback for user {user_id}: {str(e)}")
            return UserFeedbackAggregation(user_id=user_id, embedding_id=embedding_id)
    
    def apply_rocchio_algorithm(self, 
                               original_vector: List[float],
                               positive_embeddings: List[List[float]],
                               negative_embeddings: List[List[float]],
                               positive_weights: Optional[List[float]] = None,
                               negative_weights: Optional[List[float]] = None) -> List[float]:
        """
        Apply Rocchio's algorithm to update user vector
        Args:
            original_vector: User's current preference vector
            positive_embeddings: Embeddings of positively rated content
            negative_embeddings: Embeddings of negatively rated content
            positive_weights: Weights for positive embeddings
            negative_weights: Weights for negative embeddings
        Returns:
            Updated user vector
        """
        try:
            # Convert original vector to numpy array
            original_array = np.array(original_vector)
            vector_dim = len(original_vector)
            
            # Calculate positive centroid
            if positive_embeddings:
                pos_weights = positive_weights or [1.0] * len(positive_embeddings)
                positive_centroid = np.array(self.calculate_weighted_centroid(positive_embeddings, pos_weights))
            else:
                positive_centroid = np.zeros(vector_dim)
            
            # Calculate negative centroid
            if negative_embeddings:
                neg_weights = negative_weights or [1.0] * len(negative_embeddings)
                negative_centroid = np.array(self.calculate_weighted_centroid(negative_embeddings, neg_weights))
            else:
                negative_centroid = np.zeros(vector_dim)
            
            # Apply Rocchio's formula: α * original + β * positive - γ * negative
            updated_vector = (
                self.params.alpha * original_array +
                self.params.beta * positive_centroid -
                self.params.gamma * negative_centroid
            )
            
            # Normalize the vector (optional, but often helpful)
            vector_norm = np.linalg.norm(updated_vector)
            if vector_norm > 0:
                updated_vector = updated_vector / vector_norm
            
            logger.debug(f"Applied Rocchio algorithm: {len(positive_embeddings)} positive, {len(negative_embeddings)} negative vectors")
            return updated_vector.tolist()
            
        except Exception as e:
            logger.error(f"Error applying Rocchio algorithm: {str(e)}")
            return original_vector  # Return original vector on error
    
    def calculate_vector_change_magnitude(self, original_vector: List[float], updated_vector: List[float]) -> float:
        """
        Calculate the magnitude of change between original and updated vectors
        Args:
            original_vector: Original user vector
            updated_vector: Updated user vector
        Returns:
            Magnitude of change (Euclidean distance)
        """
        try:
            orig_array = np.array(original_vector)
            updated_array = np.array(updated_vector)
            
            change_magnitude = np.linalg.norm(updated_array - orig_array)
            return float(change_magnitude)
            
        except Exception as e:
            logger.error(f"Error calculating vector change magnitude: {str(e)}")
            return 0.0
    
    def should_update_vector(self, change_magnitude: float, min_change_threshold: float = 0.01) -> bool:
        """
        Determine if vector change is significant enough to warrant an update
        Args:
            change_magnitude: Magnitude of vector change
            min_change_threshold: Minimum threshold for significant change
        Returns:
            True if vector should be updated, False otherwise
        """
        return change_magnitude >= min_change_threshold


# Global service instance
rocchio_service = RocchioAlgorithmService()
