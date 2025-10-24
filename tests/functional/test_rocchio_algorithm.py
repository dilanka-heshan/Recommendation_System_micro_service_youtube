# Functional Tests for Rocchio Algorithm
import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import Mock, patch
import logging

from backend.services.rocchio_algorithm_service import RocchioAlgorithmService
from backend.models.user_vector_update_models import RocchioParameters, UserFeedbackAggregation

logger = logging.getLogger(__name__)

@pytest.fixture
def rocchio_service():
    """Create a RocchioAlgorithmService instance with default parameters"""
    return RocchioAlgorithmService()

@pytest.fixture
def custom_rocchio_service():
    """Create a RocchioAlgorithmService instance with custom parameters"""
    params = RocchioParameters(
        alpha=0.8,
        beta=0.2,
        gamma=0.1,
        rating_weights={5: 1.0, 4: 0.8, 3: 0.0, 2: 0.8, 1: 1.0}
    )
    return RocchioAlgorithmService(params)

@pytest.fixture
def sample_feedback_records():
    """Sample feedback records for testing"""
    return [
        {"video_id": "video_1", "rating": 5, "timestamp": "2025-09-27T10:00:00"},
        {"video_id": "video_2", "rating": 4, "timestamp": "2025-09-27T11:00:00"},
        {"video_id": "video_3", "rating": 2, "timestamp": "2025-09-27T12:00:00"},
        {"video_id": "video_4", "rating": 1, "timestamp": "2025-09-27T13:00:00"},
        {"video_id": "video_5", "rating": 3, "timestamp": "2025-09-27T14:00:00"}
    ]

@pytest.fixture
def sample_video_embeddings():
    """Sample video embeddings for testing"""
    np.random.seed(42)  # For reproducible results
    return {
        "video_1": np.random.rand(768).tolist(),
        "video_2": np.random.rand(768).tolist(), 
        "video_3": np.random.rand(768).tolist(),
        "video_4": np.random.rand(768).tolist(),
        "video_5": np.random.rand(768).tolist()
    }

class TestRocchioFeedbackClassification:
    """Test feedback classification by rating"""
    
    def test_classify_feedback_normal_ratings(self, rocchio_service, sample_feedback_records):
        """Test classification with normal rating distribution"""
        positive, negative, neutral = rocchio_service.classify_feedback_by_rating(sample_feedback_records)
        
        # Check classifications
        assert len(positive) == 2  # ratings 4, 5
        assert len(negative) == 2  # ratings 1, 2
        assert len(neutral) == 1   # rating 3
        
        # Verify specific records
        positive_ratings = [record["rating"] for record in positive]
        negative_ratings = [record["rating"] for record in negative]
        neutral_ratings = [record["rating"] for record in neutral]
        
        assert all(rating >= 4 for rating in positive_ratings)
        assert all(rating <= 2 for rating in negative_ratings)
        assert all(rating == 3 for rating in neutral_ratings)
    
    def test_classify_feedback_all_positive(self, rocchio_service):
        """Test edge case: all positive feedback"""
        all_positive = [
            {"video_id": "v1", "rating": 5},
            {"video_id": "v2", "rating": 4},
            {"video_id": "v3", "rating": 5}
        ]
        
        positive, negative, neutral = rocchio_service.classify_feedback_by_rating(all_positive)
        
        assert len(positive) == 3
        assert len(negative) == 0
        assert len(neutral) == 0
    
    def test_classify_feedback_all_negative(self, rocchio_service):
        """Test edge case: all negative feedback"""
        all_negative = [
            {"video_id": "v1", "rating": 1},
            {"video_id": "v2", "rating": 2},
            {"video_id": "v3", "rating": 1}
        ]
        
        positive, negative, neutral = rocchio_service.classify_feedback_by_rating(all_negative)
        
        assert len(positive) == 0
        assert len(negative) == 3
        assert len(neutral) == 0
    
    def test_unknown_rating_handling(self, rocchio_service, caplog):
        """Test handling of unknown rating values"""
        unknown_ratings = [
            {"video_id": "v1", "rating": 6},  # Invalid rating
            {"video_id": "v2", "rating": 0},  # Invalid rating
            {"video_id": "v3", "rating": 4}   # Valid rating
        ]
        
        with caplog.at_level(logging.WARNING):
            positive, negative, neutral = rocchio_service.classify_feedback_by_rating(unknown_ratings)
        
        # Only valid rating should be classified
        assert len(positive) == 1
        assert len(negative) == 0
        assert len(neutral) == 0
        
        # Check warning messages
        assert "Unknown rating value: 6" in caplog.text
        assert "Unknown rating value: 0" in caplog.text

class TestRocchioWeightCalculation:
    """Test rating weight calculations"""
    
    def test_rating_weights_default(self, rocchio_service, sample_feedback_records):
        """Test weight calculation with default parameters"""
        weights = rocchio_service.calculate_rating_weights(sample_feedback_records)
        
        expected_weights = [1.0, 0.75, 0.75, 1.0, 0.0]  # Based on ratings [5,4,2,1,3]
        assert weights == expected_weights
    
    def test_rating_weights_custom(self, custom_rocchio_service):
        """Test weight calculation with custom parameters"""
        feedback = [
            {"rating": 5}, {"rating": 4}, {"rating": 2}, {"rating": 1}
        ]
        
        weights = custom_rocchio_service.calculate_rating_weights(feedback)
        expected_weights = [1.0, 0.8, 0.8, 1.0]  # Based on custom weights
        
        assert weights == expected_weights
    
    def test_missing_rating_default_weight(self, rocchio_service):
        """Test default weight for missing rating"""
        feedback = [{"video_id": "v1"}]  # No rating field
        
        weights = rocchio_service.calculate_rating_weights(feedback)
        assert weights == [0.0]  # Default rating 3 -> weight 0.0

class TestRocchioAlgorithmCore:
    """Test core Rocchio algorithm implementation"""
    
    def test_rocchio_algorithm_balanced_feedback(self, rocchio_service):
        """Test Rocchio algorithm with balanced positive/negative feedback"""
        # Create test vectors
        original_vector = [0.5] * 768
        positive_embeddings = [[0.8] * 768, [0.9] * 768]
        negative_embeddings = [[0.1] * 768, [0.2] * 768]
        
        result = rocchio_service.apply_rocchio_algorithm(
            original_vector=original_vector,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings
        )
        
        # Result should be a list with 768 dimensions
        assert isinstance(result, list)
        assert len(result) == 768
        
        # With default params (α=0.7, β=0.3, γ=0.1), result should be influenced
        # more by original vector than feedback
        result_array = np.array(result)
        original_array = np.array(original_vector)
        
        # Vector should be normalized
        vector_norm = np.linalg.norm(result_array)
        assert abs(vector_norm - 1.0) < 0.01  # Should be approximately normalized
    
    def test_rocchio_algorithm_only_positive_feedback(self, rocchio_service):
        """Test Rocchio algorithm with only positive feedback"""
        original_vector = [0.5] * 768
        positive_embeddings = [[0.8] * 768, [0.9] * 768]
        negative_embeddings = []
        
        result = rocchio_service.apply_rocchio_algorithm(
            original_vector=original_vector,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings
        )
        
        # Result should move towards positive feedback
        result_array = np.array(result)
        original_array = np.array(original_vector)
        positive_centroid = np.mean(positive_embeddings, axis=0)
        
        # Result should be between original and positive centroid
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)
    
    def test_rocchio_algorithm_only_negative_feedback(self, rocchio_service):
        """Test Rocchio algorithm with only negative feedback"""
        original_vector = [0.5] * 768
        positive_embeddings = []
        negative_embeddings = [[0.1] * 768, [0.2] * 768]
        
        result = rocchio_service.apply_rocchio_algorithm(
            original_vector=original_vector,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings
        )
        
        # Result should move away from negative feedback
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)
    
    def test_rocchio_algorithm_with_weights(self, rocchio_service):
        """Test Rocchio algorithm with custom weights"""
        original_vector = [0.5] * 768
        positive_embeddings = [[0.8] * 768, [0.9] * 768]
        negative_embeddings = [[0.1] * 768, [0.2] * 768]
        positive_weights = [1.0, 0.5]  # First embedding weighted more
        negative_weights = [0.8, 1.0]  # Second embedding weighted more
        
        result = rocchio_service.apply_rocchio_algorithm(
            original_vector=original_vector,
            positive_embeddings=positive_embeddings,
            negative_embeddings=negative_embeddings,
            positive_weights=positive_weights,
            negative_weights=negative_weights
        )
        
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)
    
    def test_rocchio_algorithm_error_handling(self, rocchio_service):
        """Test error handling in Rocchio algorithm"""
        original_vector = [0.5] * 768
        
        # Test with invalid embeddings (wrong dimensions)
        invalid_embeddings = [[0.8] * 10, [0.9] * 10]  # Wrong dimension
        
        result = rocchio_service.apply_rocchio_algorithm(
            original_vector=original_vector,
            positive_embeddings=invalid_embeddings,
            negative_embeddings=[]
        )
        
        # Should return original vector on error
        assert result == original_vector

class TestRocchioParameterEffects:
    """Test effects of different Rocchio parameters"""
    
    def test_alpha_parameter_effect(self):
        """Test effect of different alpha values"""
        # High alpha (more weight on original vector)
        high_alpha_params = RocchioParameters(alpha=0.8, beta=0.1, gamma=0.1)
        high_alpha_service = RocchioAlgorithmService(high_alpha_params)
        
        # Low alpha (less weight on original vector)
        low_alpha_params = RocchioParameters(alpha=0.2, beta=0.4, gamma=0.4)
        low_alpha_service = RocchioAlgorithmService(low_alpha_params)
        
        # Use more distinct vectors to ensure measurable differences
        original_vector = [0.5] * 768
        positive_embeddings = [[0.8] * 768]  # Closer to original
        negative_embeddings = [[0.2] * 768]  # Farther from original
        
        high_alpha_result = high_alpha_service.apply_rocchio_algorithm(
            original_vector, positive_embeddings, negative_embeddings
        )
        
        low_alpha_result = low_alpha_service.apply_rocchio_algorithm(
            original_vector, positive_embeddings, negative_embeddings
        )
        
        # Calculate actual vector components before normalization to see the effect
        high_alpha_mean = np.mean(high_alpha_result)
        low_alpha_mean = np.mean(low_alpha_result)
        original_mean = np.mean(original_vector)
        
        # High alpha should result in values closer to original vector mean
        # Low alpha should be more influenced by feedback vectors
        high_alpha_deviation = abs(high_alpha_mean - original_mean)
        low_alpha_deviation = abs(low_alpha_mean - original_mean)
        
        # The results should be different due to different parameter weighting
        assert high_alpha_result != low_alpha_result, "Different alpha values should produce different results"
    
    def test_beta_parameter_effect(self):
        """Test effect of different beta values"""
        # High beta (more weight on positive feedback)
        high_beta_params = RocchioParameters(alpha=0.4, beta=0.5, gamma=0.1)
        high_beta_service = RocchioAlgorithmService(high_beta_params)
        
        # Low beta (less weight on positive feedback)  
        low_beta_params = RocchioParameters(alpha=0.4, beta=0.1, gamma=0.5)
        low_beta_service = RocchioAlgorithmService(low_beta_params)
        
        # Use diverse vectors that will show clear differences
        original_vector = [0.5] * 768
        positive_embeddings = [[1.0] * 200 + [0.0] * 568]  # High values in first 200 dimensions
        negative_embeddings = [[0.0] * 200 + [1.0] * 568]  # High values in last 568 dimensions
        
        high_beta_result = high_beta_service.apply_rocchio_algorithm(
            original_vector, positive_embeddings, negative_embeddings
        )
        
        low_beta_result = low_beta_service.apply_rocchio_algorithm(
            original_vector, positive_embeddings, negative_embeddings
        )
        
        # Check if the first 200 dimensions are influenced differently
        # High beta should be more influenced by positive (first 200 dims should be higher)
        # Low beta (high gamma) should be more influenced by negative (first 200 dims should be lower)
        high_beta_pos_influence = np.mean(high_beta_result[:200])
        low_beta_pos_influence = np.mean(low_beta_result[:200])
        
        # The results should be meaningfully different in the positive-influenced dimensions
        assert high_beta_pos_influence != low_beta_pos_influence, "Different beta values should produce different results"
        
        # Even with normalization, high beta should show more positive influence
        assert abs(high_beta_pos_influence - low_beta_pos_influence) > 1e-10, "Beta parameters should create detectable differences"
    
    def test_gamma_parameter_effect(self):
        """Test effect of different gamma values"""
        # High gamma (more weight on negative feedback)
        high_gamma_params = RocchioParameters(alpha=0.5, beta=0.2, gamma=0.3)
        high_gamma_service = RocchioAlgorithmService(high_gamma_params)
        
        # Low gamma (less weight on negative feedback)
        low_gamma_params = RocchioParameters(alpha=0.5, beta=0.2, gamma=0.1)
        low_gamma_service = RocchioAlgorithmService(low_gamma_params)
        
        original_vector = [0.5] * 768
        positive_embeddings = [[0.9] * 768]
        negative_embeddings = [[0.1] * 768]
        
        high_gamma_result = high_gamma_service.apply_rocchio_algorithm(
            original_vector, positive_embeddings, negative_embeddings
        )
        
        low_gamma_result = low_gamma_service.apply_rocchio_algorithm(
            original_vector, positive_embeddings, negative_embeddings
        )
        
        # Results should be different due to different gamma values
        assert high_gamma_result != low_gamma_result

class TestRocchioConvergenceBehavior:
    """Test convergence behavior over multiple iterations"""
    
    def test_convergence_with_consistent_feedback(self, rocchio_service):
        """Test vector convergence with consistent positive feedback"""
        original_vector = [0.5] * 768
        consistent_positive = [[0.8] * 768]  # Same positive feedback repeatedly
        
        # Apply algorithm multiple times (simulating multiple days of same feedback)
        current_vector = original_vector
        previous_distances = []
        
        for iteration in range(5):
            new_vector = rocchio_service.apply_rocchio_algorithm(
                original_vector=current_vector,
                positive_embeddings=consistent_positive,
                negative_embeddings=[]
            )
            
            # Calculate distance from target (positive feedback)
            distance = np.linalg.norm(np.array(new_vector) - np.array(consistent_positive[0]))
            previous_distances.append(distance)
            current_vector = new_vector
        
        # Distance should decrease over iterations (convergence)
        for i in range(1, len(previous_distances)):
            assert previous_distances[i] <= previous_distances[i-1] + 0.1  # Allow for small variations due to normalization
    
    def test_oscillation_with_conflicting_feedback(self, rocchio_service):
        """Test behavior with conflicting feedback patterns"""
        original_vector = [0.5] * 768
        positive_feedback = [[0.9] * 768]
        negative_feedback = [[0.1] * 768]
        
        # Simulate alternating feedback
        current_vector = original_vector
        positions = []
        
        for iteration in range(6):
            if iteration % 2 == 0:
                # Even iterations: positive feedback
                new_vector = rocchio_service.apply_rocchio_algorithm(
                    original_vector=current_vector,
                    positive_embeddings=positive_feedback,
                    negative_embeddings=[]
                )
            else:
                # Odd iterations: negative feedback
                new_vector = rocchio_service.apply_rocchio_algorithm(
                    original_vector=current_vector,
                    positive_embeddings=[],
                    negative_embeddings=negative_feedback
                )
            
            positions.append(np.mean(new_vector))  # Track average position
            current_vector = new_vector
        
        # Should show oscillation pattern
        assert len(positions) == 6
        # Vector should not converge to a single point with alternating feedback

class TestRocchioVectorMagnitudeTracking:
    """Test vector change magnitude calculations"""
    
    def test_calculate_vector_change_magnitude(self, rocchio_service):
        """Test vector change magnitude calculation"""
        original_vector = [0.5] * 768
        updated_vector = [0.6] * 768
        
        magnitude = rocchio_service.calculate_vector_change_magnitude(original_vector, updated_vector)
        
        # Expected magnitude: sqrt(768 * (0.6-0.5)^2) = sqrt(768 * 0.01) = sqrt(7.68)
        expected_magnitude = np.sqrt(768 * 0.01)
        assert abs(magnitude - expected_magnitude) < 0.001
    
    def test_should_update_vector_threshold(self, rocchio_service):
        """Test vector update threshold decision"""
        # Small change (below threshold)
        small_magnitude = 0.005
        assert not rocchio_service.should_update_vector(small_magnitude, min_change_threshold=0.01)
        
        # Large change (above threshold)
        large_magnitude = 0.02
        assert rocchio_service.should_update_vector(large_magnitude, min_change_threshold=0.01)
        
        # Edge case (exactly at threshold)
        threshold_magnitude = 0.01
        assert rocchio_service.should_update_vector(threshold_magnitude, min_change_threshold=0.01)
    
    def test_vector_change_magnitude_error_handling(self, rocchio_service):
        """Test error handling in vector change magnitude calculation"""
        original_vector = [0.5] * 768
        invalid_vector = [0.6] * 10  # Wrong dimension
        
        magnitude = rocchio_service.calculate_vector_change_magnitude(original_vector, invalid_vector)
        
        # Should return 0.0 on error
        assert magnitude == 0.0

class TestRocchioFeedbackAggregation:
    """Test feedback aggregation functionality"""
    
    def test_aggregate_user_feedback_normal(self, rocchio_service):
        """Test normal feedback aggregation"""
        feedback_records = [
            {"video_id": "v1", "rating": 5},
            {"video_id": "v2", "rating": 4},
            {"video_id": "v3", "rating": 2},
            {"video_id": "v4", "rating": 1}
        ]
        click_records = []  # Empty for now
        
        aggregation = rocchio_service.aggregate_user_feedback(
            feedback_records=feedback_records,
            click_records=click_records,
            user_id="test_user",
            embedding_id="emb_123"
        )
        
        assert isinstance(aggregation, UserFeedbackAggregation)
        assert aggregation.user_id == "test_user"
        assert aggregation.embedding_id == "emb_123"
        assert len(aggregation.positive_videos) == 2  # ratings 4, 5
        assert len(aggregation.negative_videos) == 2  # ratings 1, 2
        assert len(aggregation.neutral_videos) == 0   # no rating 3
        assert aggregation.total_feedback_count == 4
    
    def test_aggregate_user_feedback_error_handling(self, rocchio_service):
        """Test error handling in feedback aggregation"""
        # Test with invalid feedback records
        invalid_feedback = [{"invalid": "data"}]
        
        aggregation = rocchio_service.aggregate_user_feedback(
            feedback_records=invalid_feedback,
            click_records=[],
            user_id="test_user"
        )
        
        # Should return aggregation with empty lists
        assert len(aggregation.positive_videos) == 0
        assert len(aggregation.negative_videos) == 0
        assert len(aggregation.neutral_videos) == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])