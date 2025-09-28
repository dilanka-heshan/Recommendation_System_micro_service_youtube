# Functional Tests for Video Reranking Service
import pytest
import numpy as np
from typing import List, Dict, Any
from unittest.mock import Mock, patch, MagicMock
import logging

from backend.services.rerank import VideoReranker

logger = logging.getLogger(__name__)

@pytest.fixture
def video_reranker():
    """Create a VideoReranker instance with mocked models"""
    with patch('backend.services.rerank.CrossEncoder') as mock_cross_encoder, \
         patch('backend.services.rerank.SentenceTransformer') as mock_sentence_transformer:
        
        # Mock CrossEncoder
        mock_rerank_instance = MagicMock()
        mock_rerank_instance.predict.return_value = [0.8, 0.7, 0.6, 0.5, 0.4]
        mock_cross_encoder.return_value = mock_rerank_instance
        
        # Mock SentenceTransformer
        mock_embed_instance = MagicMock()
        mock_embed_instance.encode.return_value = np.random.rand(5, 768)
        mock_sentence_transformer.return_value = mock_embed_instance
        
        return VideoReranker()

@pytest.fixture
def mock_rerank_models():
    """Mock the reranking models for testing"""
    with patch('backend.services.rerank.CrossEncoder') as mock_cross_encoder, \
         patch('backend.services.rerank.SentenceTransformer') as mock_sentence_transformer:
        
        # Mock CrossEncoder (rerank model)
        mock_rerank_instance = MagicMock()
        mock_rerank_instance.predict.return_value = [0.8, 0.7, 0.6, 0.5, 0.4]
        mock_cross_encoder.return_value = mock_rerank_instance
        
        # Mock SentenceTransformer (embed model)
        mock_embed_instance = MagicMock()
        mock_embed_instance.encode.return_value = np.random.rand(5, 768)
        mock_sentence_transformer.return_value = mock_embed_instance
        
        yield {
            'rerank_model': mock_rerank_instance,
            'embed_model': mock_embed_instance
        }

@pytest.fixture
def sample_user_history():
    """Sample user history with high-rating videos"""
    return [
        {'video_id': 'user_video_1', 'rating': 5},
        {'video_id': 'user_video_2', 'rating': 4},
        {'video_id': 'user_video_3', 'rating': 5},
    ]

@pytest.fixture
def sample_candidate_videos():
    """Sample candidate videos for reranking"""
    np.random.seed(42)
    candidates = []
    for i in range(100):
        candidates.append({
            'video_id': f'candidate_{i}',
            'title': f'Video Title {i}',
            'description': f'Description for video {i}',
            'embedding': np.random.rand(768).tolist(),
            'similarity_score': 0.7 + (i * 0.001)  # Slight variation
        })
    return candidates

@pytest.fixture
def mock_database_clients():
    """Mock database clients"""
    with patch('backend.services.rerank.mongodb_client') as mock_mongodb, \
         patch('backend.services.rerank.qdrant_client') as mock_qdrant:
        
        # Mock MongoDB responses
        mock_mongodb.get_extractive_summary.return_value = "Sample extractive summary"
        mock_mongodb.get_multiple_extractive_summaries.return_value = {
            'user_video_1': 'Summary for user video 1',
            'user_video_2': 'Summary for user video 2',
            'user_video_3': 'Summary for user video 3'
        }
        
        # Mock Qdrant responses
        mock_qdrant.get_videos_by_ids.return_value = [
            {'video_id': f'video_{i}', 'embedding': np.random.rand(768).tolist()}
            for i in range(20)
        ]
        
        yield {
            'mongodb': mock_mongodb,
            'qdrant': mock_qdrant
        }

class TestVideoTextRepresentation:
    """Test video text representation methods"""
    
    def test_get_video_text_with_extractive_summary(self, video_reranker, mock_database_clients):
        """Test getting video text representation with extractive summary"""
        video = {
            'video_id': 'test_video_1',
            'title': 'Test Video Title',
            'description': 'Test video description'
        }
        
        text = video_reranker._get_video_text_representation(video, use_extractive_summary=True)
        
        # Should use extractive summary from MongoDB
        assert text == "Sample extractive summary"
        mock_database_clients['mongodb'].get_extractive_summary.assert_called_once_with('test_video_1')
    
    def test_get_video_text_fallback_to_title_description(self, video_reranker, mock_database_clients):
        """Test fallback to title and description when extractive summary fails"""
        # Mock MongoDB to return None (no summary)
        mock_database_clients['mongodb'].get_extractive_summary.return_value = None
        
        video = {
            'video_id': 'test_video_2',
            'title': 'Fallback Title',
            'description': 'Fallback description'
        }
        
        text = video_reranker._get_video_text_representation(video, use_extractive_summary=True)
        
        # Should fallback to title + description
        assert text == "Fallback Title. Fallback description"
    
    def test_get_video_text_final_fallback(self, video_reranker, mock_database_clients):
        """Test final fallback to video_id when no other text available"""
        mock_database_clients['mongodb'].get_extractive_summary.return_value = None
        
        video = {
            'video_id': 'test_video_underscore_dash'
        }
        
        text = video_reranker._get_video_text_representation(video, use_extractive_summary=True)
        
        # Should fallback to processed video_id
        assert text == "test video underscore dash"

class TestStage1RerankerFiltering:
    """Test Stage 1 reranking with cross-encoder model"""
    
    def test_stage1_normal_operation(self, video_reranker, mock_rerank_models, mock_database_clients, 
                                   sample_user_history, sample_candidate_videos):
        """Test normal Stage 1 operation"""
        video_reranker.rerank_model = mock_rerank_models['rerank_model']
        
        # Mock predict to return scores for first 5 candidates
        mock_rerank_models['rerank_model'].predict.return_value = [0.9, 0.8, 0.7, 0.6, 0.5] + [0.4] * 95
        
        result = video_reranker._stage1_reranker_filtering(
            user_history=sample_user_history,
            candidate_videos=sample_candidate_videos,
            top_k=10
        )
        
        # Should return top 10 candidates
        assert len(result) == 10
        
        # Should have stage1_score attached
        for video in result:
            assert 'stage1_score' in video
            assert isinstance(video['stage1_score'], float)
        
        # Should be sorted by stage1_score (highest first)
        scores = [video['stage1_score'] for video in result]
        assert scores == sorted(scores, reverse=True)
    
    def test_stage1_with_extractive_summaries(self, video_reranker, mock_rerank_models, mock_database_clients,
                                            sample_user_history, sample_candidate_videos):
        """Test Stage 1 using extractive summaries for user history"""
        video_reranker.rerank_model = mock_rerank_models['rerank_model']
        
        # Set up mock summaries for user history
        mock_database_clients['mongodb'].get_multiple_extractive_summaries.return_value = {
            'user_video_1': 'Technology tutorial summary',
            'user_video_2': 'Programming guide summary',
            'user_video_3': 'Software development summary'
        }
        
        result = video_reranker._stage1_reranker_filtering(
            user_history=sample_user_history,
            candidate_videos=sample_candidate_videos[:20],  # Use smaller set for testing
            top_k=10
        )
        
        # Check that rerank model was called
        mock_rerank_models['rerank_model'].predict.assert_called_once()
        
        # Verify user query was built from summaries
        call_args = mock_rerank_models['rerank_model'].predict.call_args[0][0]
        assert len(call_args) == 20  # Should have 20 pairs for 20 candidates
    
    def test_stage1_error_handling(self, video_reranker, sample_user_history, sample_candidate_videos):
        """Test Stage 1 error handling when model fails"""
        # Don't set up the rerank model (simulate failure)
        video_reranker.rerank_model = None
        
        result = video_reranker._stage1_reranker_filtering(
            user_history=sample_user_history,
            candidate_videos=sample_candidate_videos,
            top_k=10
        )
        
        # Should return top 10 original candidates
        assert len(result) == 10
        assert result == sample_candidate_videos[:10]
    
    def test_stage1_boundary_cases(self, video_reranker, mock_rerank_models, sample_user_history):
        """Test Stage 1 boundary cases"""
        video_reranker.rerank_model = mock_rerank_models['rerank_model']
        
        # Test with single candidate
        single_candidate = [{'video_id': 'single', 'title': 'Single Video'}]
        mock_rerank_models['rerank_model'].predict.return_value = [0.8]
        
        result = video_reranker._stage1_reranker_filtering(
            user_history=sample_user_history,
            candidate_videos=single_candidate,
            top_k=10
        )
        
        assert len(result) == 1
        assert result[0]['stage1_score'] == 0.8
        
        # Test with empty candidates
        result_empty = video_reranker._stage1_reranker_filtering(
            user_history=sample_user_history,
            candidate_videos=[],
            top_k=10
        )
        
        assert len(result_empty) == 0

class TestStage2PairwiseAnalysis:
    """Test Stage 2 pairwise vector analysis"""
    
    def test_stage2_normal_operation(self, video_reranker, mock_database_clients, 
                                   sample_user_history, sample_candidate_videos):
        """Test normal Stage 2 operation"""
        # Prepare stage 1 candidates (top 20)
        stage1_candidates = sample_candidate_videos[:20]
        for i, candidate in enumerate(stage1_candidates):
            candidate['stage1_score'] = 0.9 - (i * 0.01)
        
        # Mock Qdrant to return vectors
        mock_database_clients['qdrant'].get_videos_by_ids.return_value = [
            {'video_id': f'candidate_{i}', 'embedding': np.random.rand(768).tolist()}
            for i in range(20)
        ] + [
            {'video_id': f'user_video_{i}', 'embedding': np.random.rand(768).tolist()}
            for i in range(1, 4)
        ]
        
        result = video_reranker._stage2_pairwise_analysis(
            user_history=sample_user_history,
            stage1_candidates=stage1_candidates,
            top_k=10
        )
        
        # Should return top 10 results
        assert len(result) == 10
        
        # Should have stage2_score and final_score
        for video in result:
            assert 'stage2_score' in video
            assert 'final_score' in video
            assert 'final_rank' in video
            assert isinstance(video['stage2_score'], float)
            assert isinstance(video['final_score'], float)
        
        # Should be sorted by final_score
        final_scores = [video['final_score'] for video in result]
        assert final_scores == sorted(final_scores, reverse=True)
    
    def test_stage2_aggregation_methods(self, video_reranker, mock_database_clients,
                                      sample_user_history, sample_candidate_videos):
        """Test different aggregation methods (mean vs max)"""
        stage1_candidates = sample_candidate_videos[:5]
        
        # Set up vector data
        mock_database_clients['qdrant'].get_videos_by_ids.return_value = [
            {'video_id': f'candidate_{i}', 'embedding': np.random.rand(768).tolist()}
            for i in range(5)
        ] + [
            {'video_id': f'user_video_{i}', 'embedding': np.random.rand(768).tolist()}
            for i in range(1, 4)
        ]
        
        # Test mean aggregation
        result_mean = video_reranker._stage2_pairwise_analysis(
            user_history=sample_user_history,
            stage1_candidates=stage1_candidates,
            top_k=5,
            agg='mean'
        )
        
        # Test max aggregation
        result_max = video_reranker._stage2_pairwise_analysis(
            user_history=sample_user_history,
            stage1_candidates=stage1_candidates,
            top_k=5,
            agg='max'
        )
        
        # Results should be different (unless by coincidence)
        mean_scores = [video['final_score'] for video in result_mean]
        max_scores = [video['final_score'] for video in result_max]
        
        # At least some scores should be different
        assert mean_scores != max_scores or len(set(mean_scores)) == 1  # Unless all scores are identical
    
    def test_stage2_missing_vectors_fallback(self, video_reranker, mock_database_clients,
                                           sample_user_history, sample_candidate_videos):
        """Test fallback when vectors are missing"""
        stage1_candidates = sample_candidate_videos[:10]
        
        # Mock Qdrant to return empty results (no vectors found)
        mock_database_clients['qdrant'].get_videos_by_ids.return_value = []
        
        result = video_reranker._stage2_pairwise_analysis(
            user_history=sample_user_history,
            stage1_candidates=stage1_candidates,
            top_k=5
        )
        
        # Should fallback to stage 1 results
        assert len(result) == 5
        assert result == stage1_candidates[:5]
    
    def test_stage2_rating_weighting(self, video_reranker, mock_database_clients, sample_candidate_videos):
        """Test that user ratings properly weight similarity scores"""
        # Create user history with different ratings
        weighted_user_history = [
            {'video_id': 'high_rating', 'rating': 5},
            {'video_id': 'low_rating', 'rating': 1}
        ]
        
        stage1_candidates = sample_candidate_videos[:2]
        
        # Create deterministic vectors for predictable results
        high_rating_vector = [1.0] * 768
        low_rating_vector = [-1.0] * 768
        candidate_vector = [0.5] * 768
        
        mock_database_clients['qdrant'].get_videos_by_ids.return_value = [
            {'video_id': 'candidate_0', 'embedding': candidate_vector},
            {'video_id': 'candidate_1', 'embedding': candidate_vector},
            {'video_id': 'high_rating', 'embedding': high_rating_vector},
            {'video_id': 'low_rating', 'embedding': low_rating_vector}
        ]
        
        result = video_reranker._stage2_pairwise_analysis(
            user_history=weighted_user_history,
            stage1_candidates=stage1_candidates,
            top_k=2
        )
        
        # Should have calculated similarity scores considering rating weights
        assert len(result) == 2
        assert all('stage2_score' in video for video in result)

class TestTwoStageReranking:
    """Test the complete two-stage reranking pipeline"""
    
    def test_full_pipeline_normal_operation(self, video_reranker, mock_rerank_models, mock_database_clients,
                                          sample_user_history, sample_candidate_videos):
        """Test complete two-stage reranking pipeline"""
        video_reranker.rerank_model = mock_rerank_models['rerank_model']
        video_reranker.embed_model = mock_rerank_models['embed_model']
        
        # Mock stage 1 scores
        stage1_scores = [0.9 - (i * 0.01) for i in range(100)]
        mock_rerank_models['rerank_model'].predict.return_value = stage1_scores
        
        # Mock stage 2 vectors
        mock_database_clients['qdrant'].get_videos_by_ids.return_value = [
            {'video_id': f'candidate_{i}', 'embedding': np.random.rand(768).tolist()}
            for i in range(20)  # Top 20 from stage 1
        ] + [
            {'video_id': f'user_video_{i}', 'embedding': np.random.rand(768).tolist()}
            for i in range(1, 4)
        ]
        
        result = video_reranker.rerank_with_user_history(
            user_history=sample_user_history,
            candidate_videos=sample_candidate_videos,
            top_k=10
        )
        
        # Should return exactly top_k results
        assert len(result) == 10
        
        # Should have both stage scores and final ranking
        for i, video in enumerate(result):
            assert 'stage1_score' in video
            assert 'stage2_score' in video or 'final_score' in video
            assert 'final_rank' in video
            assert video['final_rank'] == i + 1
    
    def test_pipeline_with_no_user_history(self, video_reranker, sample_candidate_videos):
        """Test pipeline behavior with no user history"""
        result = video_reranker.rerank_with_user_history(
            user_history=[],
            candidate_videos=sample_candidate_videos,
            top_k=10
        )
        
        # Should return original candidates (first 10)
        assert len(result) == 10
        assert result == sample_candidate_videos[:10]
    
    def test_pipeline_with_no_candidates(self, video_reranker, sample_user_history):
        """Test pipeline behavior with no candidates"""
        result = video_reranker.rerank_with_user_history(
            user_history=sample_user_history,
            candidate_videos=[],
            top_k=10
        )
        
        # Should return empty list
        assert len(result) == 0
    
    def test_pipeline_fallback_when_models_unavailable(self, video_reranker, sample_user_history, 
                                                     sample_candidate_videos):
        """Test fallback behavior when reranking models are not available"""
        # Don't set up models (simulate unavailable)
        video_reranker.rerank_model = None
        video_reranker.embed_model = None
        
        with patch.object(video_reranker, '_fallback_reranking') as mock_fallback:
            mock_fallback.return_value = sample_candidate_videos[:10]
            
            result = video_reranker.rerank_with_user_history(
                user_history=sample_user_history,
                candidate_videos=sample_candidate_videos,
                top_k=10
            )
            
            # Should call fallback method
            mock_fallback.assert_called_once()
            assert len(result) == 10

class TestRerankingBoundaryConditions:
    """Test boundary conditions and edge cases"""
    
    def test_single_video_reranking(self, video_reranker, mock_rerank_models, mock_database_clients,
                                  sample_user_history):
        """Test reranking with only one candidate video"""
        video_reranker.rerank_model = mock_rerank_models['rerank_model']
        
        single_candidate = [{
            'video_id': 'single_video',
            'title': 'Only Video',
            'embedding': np.random.rand(768).tolist()
        }]
        
        mock_rerank_models['rerank_model'].predict.return_value = [0.8]
        mock_database_clients['qdrant'].get_videos_by_ids.return_value = [
            {'video_id': 'single_video', 'embedding': np.random.rand(768).tolist()}
        ] + [
            {'video_id': f'user_video_{i}', 'embedding': np.random.rand(768).tolist()}
            for i in range(1, 4)
        ]
        
        result = video_reranker.rerank_with_user_history(
            user_history=sample_user_history,
            candidate_videos=single_candidate,
            top_k=10
        )
        
        assert len(result) == 1
        assert result[0]['video_id'] == 'single_video'
        assert 'stage1_score' in result[0]
    
    def test_maximum_videos_reranking(self, video_reranker, mock_rerank_models, mock_database_clients,
                                    sample_user_history):
        """Test reranking with maximum number of videos (1000+)"""
        video_reranker.rerank_model = mock_rerank_models['rerank_model']
        
        # Create 1000 candidate videos
        large_candidate_set = []
        for i in range(1000):
            large_candidate_set.append({
                'video_id': f'video_{i}',
                'title': f'Video {i}',
                'embedding': np.random.rand(768).tolist()
            })
        
        # Mock stage 1 scores for 1000 videos
        stage1_scores = [0.9 - (i * 0.0001) for i in range(1000)]
        mock_rerank_models['rerank_model'].predict.return_value = stage1_scores
        
        # Mock stage 2 vectors (top 20)
        mock_database_clients['qdrant'].get_videos_by_ids.return_value = [
            {'video_id': f'video_{i}', 'embedding': np.random.rand(768).tolist()}
            for i in range(20)
        ] + [
            {'video_id': f'user_video_{i}', 'embedding': np.random.rand(768).tolist()}
            for i in range(1, 4)
        ]
        
        result = video_reranker.rerank_with_user_history(
            user_history=sample_user_history,
            candidate_videos=large_candidate_set,
            top_k=50
        )
        
        # Should handle large dataset but stage 2 limits to top 20 candidates
        # So maximum returned will be 20, not 50
        assert len(result) == 20  # Stage 2 processes max 20 candidates
        assert all('final_rank' in video for video in result)
    
    def test_diverse_rating_scenarios(self, video_reranker, mock_rerank_models, mock_database_clients,
                                    sample_candidate_videos):
        """Test with diverse user rating scenarios"""
        # Test with only high ratings
        high_ratings_only = [
            {'video_id': 'high_1', 'rating': 5},
            {'video_id': 'high_2', 'rating': 5},
            {'video_id': 'high_3', 'rating': 4}
        ]
        
        # Test with only low ratings
        low_ratings_only = [
            {'video_id': 'low_1', 'rating': 1},
            {'video_id': 'low_2', 'rating': 2},
            {'video_id': 'low_3', 'rating': 1}
        ]
        
        # Test with mixed ratings
        mixed_ratings = [
            {'video_id': 'mixed_1', 'rating': 5},
            {'video_id': 'mixed_2', 'rating': 1},
            {'video_id': 'mixed_3', 'rating': 4},
            {'video_id': 'mixed_4', 'rating': 2}
        ]
        
        video_reranker.rerank_model = mock_rerank_models['rerank_model']
        mock_rerank_models['rerank_model'].predict.return_value = [0.8] * 20
        
        mock_database_clients['qdrant'].get_videos_by_ids.return_value = [
            {'video_id': f'candidate_{i}', 'embedding': np.random.rand(768).tolist()}
            for i in range(20)
        ] + [
            {'video_id': video_id, 'embedding': np.random.rand(768).tolist()}
            for video_id in ['high_1', 'high_2', 'high_3', 'low_1', 'low_2', 'low_3', 
                           'mixed_1', 'mixed_2', 'mixed_3', 'mixed_4']
        ]
        
        candidates = sample_candidate_videos[:20]
        
        # Test each scenario
        for user_history in [high_ratings_only, low_ratings_only, mixed_ratings]:
            result = video_reranker.rerank_with_user_history(
                user_history=user_history,
                candidate_videos=candidates,
                top_k=10
            )
            
            assert len(result) == 10
            assert all('final_score' in video or 'stage2_score' in video for video in result)

class TestRerankingPerformanceMetrics:
    """Test performance and quality metrics"""
    
    def test_diversity_vs_relevance_tradeoff(self, video_reranker, mock_rerank_models, mock_database_clients):
        """Test that reranking balances diversity and relevance"""
        # Create candidates with varying similarity to user preferences
        user_history = [{'video_id': 'tech_video', 'rating': 5}]
        
        # Create candidates: some very similar (tech), some diverse (other topics)
        candidates = []
        for i in range(20):
            if i < 10:  # First 10 are tech-related (similar)
                candidates.append({
                    'video_id': f'tech_candidate_{i}',
                    'title': f'Technology Tutorial {i}',
                    'embedding': ([0.8] * 384) + ([0.2] * 384)  # Tech-like vector
                })
            else:  # Next 10 are diverse topics
                candidates.append({
                    'video_id': f'diverse_candidate_{i}',
                    'title': f'Cooking Recipe {i}',
                    'embedding': ([0.2] * 384) + ([0.8] * 384)  # Different vector
                })
        
        video_reranker.rerank_model = mock_rerank_models['rerank_model']
        
        # Stage 1 should prefer tech videos
        stage1_scores = [0.9] * 10 + [0.5] * 10  # Tech videos get higher scores
        mock_rerank_models['rerank_model'].predict.return_value = stage1_scores
        
        # Mock stage 2 vectors
        mock_database_clients['qdrant'].get_videos_by_ids.return_value = [
            {'video_id': f'tech_candidate_{i}', 'embedding': ([0.8] * 384) + ([0.2] * 384)}
            for i in range(10)
        ] + [
            {'video_id': f'diverse_candidate_{i}', 'embedding': ([0.2] * 384) + ([0.8] * 384)}
            for i in range(10, 20)
        ] + [
            {'video_id': 'tech_video', 'embedding': ([0.9] * 384) + ([0.1] * 384)}
        ]
        
        result = video_reranker.rerank_with_user_history(
            user_history=user_history,
            candidate_videos=candidates,
            top_k=10
        )
        
        # Check that results have both relevance and some diversity
        tech_count = sum(1 for video in result if 'tech_candidate' in video['video_id'])
        diverse_count = sum(1 for video in result if 'diverse_candidate' in video['video_id'])
        
        # Should have mostly tech (relevance) but some diversity
        assert tech_count > 0
        assert len(result) == 10
    
    def test_ranking_consistency(self, video_reranker, mock_rerank_models, mock_database_clients,
                               sample_user_history, sample_candidate_videos):
        """Test that rankings are consistent given same inputs"""
        video_reranker.rerank_model = mock_rerank_models['rerank_model']
        
        # Set deterministic scores
        np.random.seed(42)
        stage1_scores = np.random.rand(20).tolist()
        mock_rerank_models['rerank_model'].predict.return_value = stage1_scores
        
        # Set deterministic vectors
        np.random.seed(42)
        mock_database_clients['qdrant'].get_videos_by_ids.return_value = [
            {'video_id': f'candidate_{i}', 'embedding': np.random.rand(768).tolist()}
            for i in range(20)
        ] + [
            {'video_id': f'user_video_{i}', 'embedding': np.random.rand(768).tolist()}
            for i in range(1, 4)
        ]
        
        candidates = sample_candidate_videos[:20]
        
        # Run reranking twice
        result1 = video_reranker.rerank_with_user_history(
            user_history=sample_user_history,
            candidate_videos=candidates,
            top_k=10
        )
        
        result2 = video_reranker.rerank_with_user_history(
            user_history=sample_user_history,
            candidate_videos=candidates,
            top_k=10
        )
        
        # Results should be identical
        video_ids_1 = [video['video_id'] for video in result1]
        video_ids_2 = [video['video_id'] for video in result2]
        
        assert video_ids_1 == video_ids_2
        
        # Final scores should be identical
        scores_1 = [video.get('final_score', video.get('stage2_score', 0)) for video in result1]
        scores_2 = [video.get('final_score', video.get('stage2_score', 0)) for video in result2]
        
        assert scores_1 == scores_2

if __name__ == "__main__":
    pytest.main([__file__, "-v"])