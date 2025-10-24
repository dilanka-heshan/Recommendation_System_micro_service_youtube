"""
Data Consistency Validation Tests
Tests for synchronized data consistency across the multi-database system
"""
import pytest
import logging
from typing import Dict, Any, List, Set
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@pytest.mark.database_integrity
@pytest.mark.data_consistency
class TestDataConsistencyValidation:
    """Test data consistency across databases and tables"""

    def test_user_embedding_vector_consistency(self, database_clients, sample_user_ids, integrity_test_helpers, integrity_reporter):
        """Test consistency of user embedding vectors"""
        test_name = "user_embedding_vector_consistency"
        logger.info(f"Running {test_name}")
        
        try:
            # Get users with embeddings
            users_response = database_clients["supabase"].client.table("users").select(
                "user_id, embedding_id"
            ).neq("embedding_id", None).limit(200).execute()
            
            users_with_embeddings = users_response.data if users_response.data else []
            
            if not users_with_embeddings:
                logger.warning("No users with embeddings found")
                integrity_reporter.add_test_result(test_name, "SKIPPED", {"reason": "No users with embeddings"})
                return
            
            consistency_issues = []
            valid_embeddings = 0
            
            for user in users_with_embeddings:
                user_id = user["user_id"]
                embedding_id = user["embedding_id"]
                
                issues_for_user = []
                
                # Validate embedding format and dimensions
                try:
                    if isinstance(embedding_id, str) and embedding_id.startswith('['):
                        embedding_data = eval(embedding_id)  # Note: Use json.loads in production
                        
                        # Check dimensions
                        if not integrity_test_helpers.validate_embedding_dimensions(embedding_data, 768):
                            issues_for_user.append(f"Invalid dimensions: {len(embedding_data)}")
                        
                        # Check for NaN or infinite values
                        if any(not isinstance(x, (int, float)) or x != x for x in embedding_data):  # NaN check
                            issues_for_user.append("Contains NaN values")
                        
                        # Check for reasonable value ranges (embeddings should be normalized-ish)
                        if any(abs(x) > 10 for x in embedding_data):
                            issues_for_user.append("Contains extreme values (>10)")
                        
                        # Check for zero vector (might indicate uninitialized)
                        if all(x == 0.0 for x in embedding_data):
                            issues_for_user.append("Zero vector detected")
                        
                        if not issues_for_user:
                            valid_embeddings += 1
                        else:
                            consistency_issues.append({
                                "user_id": user_id,
                                "issues": issues_for_user
                            })
                    else:
                        consistency_issues.append({
                            "user_id": user_id,
                            "issues": ["Invalid embedding format"]
                        })
                        
                except Exception as e:
                    consistency_issues.append({
                        "user_id": user_id,
                        "issues": [f"Parse error: {str(e)}"]
                    })
            
            total_users = len(users_with_embeddings)
            consistency_score = integrity_test_helpers.calculate_consistency_score(total_users, valid_embeddings)
            
            details = {
                "total_users_tested": total_users,
                "valid_embeddings": valid_embeddings,
                "consistency_issues": len(consistency_issues),
                "consistency_percentage": consistency_score,
                "sample_issues": consistency_issues[:10]
            }
            
            status = "PASSED" if consistency_score >= 95.0 else "FAILED"
            integrity_reporter.add_test_result(test_name, status, details)
            
            assert consistency_score >= 85.0, f"Embedding consistency too low: {consistency_score}%"
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            integrity_reporter.add_test_result(test_name, "ERROR", {"error": str(e)})
            pytest.fail(f"Test failed with error: {e}")

    def test_feedback_rating_consistency(self, database_clients, integrity_test_helpers, integrity_reporter):
        """Test consistency of feedback ratings across the system"""
        test_name = "feedback_rating_consistency"
        logger.info(f"Running {test_name}")
        
        try:
            # Get feedback data
            feedback_response = database_clients["supabase"].client.table("feedback").select(
                "user_id, video_id, rating, timestamp"
            ).limit(1000).order("timestamp", desc=True).execute()
            
            feedback_data = feedback_response.data if feedback_response.data else []
            
            if not feedback_data:
                logger.warning("No feedback data found")
                integrity_reporter.add_test_result(test_name, "SKIPPED", {"reason": "No feedback data"})
                return
            
            consistency_issues = []
            valid_ratings = 0
            rating_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            user_rating_patterns = {}
            
            for feedback in feedback_data:
                user_id = feedback["user_id"]
                video_id = feedback["video_id"]
                rating = feedback["rating"]
                timestamp = feedback["timestamp"]
                
                issues_for_feedback = []
                
                # Validate rating range
                if not integrity_test_helpers.validate_rating_range(rating, 1, 5):
                    issues_for_feedback.append(f"Invalid rating: {rating}")
                else:
                    rating_distribution[rating] += 1
                    
                    # Track user rating patterns for consistency analysis
                    if user_id not in user_rating_patterns:
                        user_rating_patterns[user_id] = []
                    user_rating_patterns[user_id].append(rating)
                
                # Validate timestamp format
                try:
                    datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    issues_for_feedback.append("Invalid timestamp format")
                
                # Check for duplicate feedback (same user, same video)
                # This would need more complex logic to track across the dataset
                
                if not issues_for_feedback:
                    valid_ratings += 1
                else:
                    consistency_issues.append({
                        "user_id": user_id,
                        "video_id": video_id,
                        "rating": rating,
                        "issues": issues_for_feedback
                    })
            
            # Analyze rating distribution for anomalies
            total_ratings = sum(rating_distribution.values())
            distribution_percentages = {k: (v/total_ratings)*100 for k, v in rating_distribution.items()}
            
            # Check for suspicious patterns (e.g., too many 5s or 1s)
            distribution_warnings = []
            if distribution_percentages[5] > 60:
                distribution_warnings.append("Suspiciously high 5-star ratings")
            if distribution_percentages[1] > 30:
                distribution_warnings.append("Suspiciously high 1-star ratings")
            
            total_feedback = len(feedback_data)
            consistency_score = integrity_test_helpers.calculate_consistency_score(total_feedback, valid_ratings)
            
            details = {
                "total_feedback_records": total_feedback,
                "valid_ratings": valid_ratings,
                "consistency_issues": len(consistency_issues),
                "consistency_percentage": consistency_score,
                "rating_distribution": rating_distribution,
                "distribution_percentages": distribution_percentages,
                "distribution_warnings": distribution_warnings,
                "sample_issues": consistency_issues[:10]
            }
            
            status = "PASSED" if consistency_score >= 98.0 and len(distribution_warnings) == 0 else "FAILED"
            integrity_reporter.add_test_result(test_name, status, details)
            
            assert consistency_score >= 98.0, f"Feedback rating consistency too low: {consistency_score}%"
            if distribution_warnings:
                logger.warning(f"Rating distribution warnings: {distribution_warnings}")
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            integrity_reporter.add_test_result(test_name, "ERROR", {"error": str(e)})
            pytest.fail(f"Test failed with error: {e}")

    def test_newsletter_generation_consistency(self, database_clients, date_range_for_testing, integrity_test_helpers, integrity_reporter):
        """Test consistency of newsletter generation and video associations"""
        test_name = "newsletter_generation_consistency"
        logger.info(f"Running {test_name}")
        
        try:
            # Get recent newsletters
            newsletters_response = database_clients["supabase"].client.table("newsletters").select(
                "id, user_id, sent_at"
            ).order("sent_at", desc=True).limit(500).execute()
            
            newsletters_data = newsletters_response.data if newsletters_response.data else []
            
            if not newsletters_data:
                logger.warning("No newsletters found")
                integrity_reporter.add_test_result(test_name, "SKIPPED", {"reason": "No newsletters found"})
                return
            
            consistency_issues = []
            valid_newsletters = 0
            
            for newsletter in newsletters_data:
                newsletter_id = newsletter["id"]
                user_id = newsletter["user_id"]
                sent_at = newsletter["sent_at"]
                
                issues_for_newsletter = []
                
                # Check if newsletter has associated videos
                newsletter_videos_response = database_clients["supabase"].client.table("newsletter_videos").select(
                    "video_id"
                ).eq("newsletter_id", newsletter_id).execute()
                
                newsletter_videos = newsletter_videos_response.data if newsletter_videos_response.data else []
                
                # Validate newsletter structure
                if len(newsletter_videos) == 0:
                    issues_for_newsletter.append("No videos associated with newsletter")
                elif len(newsletter_videos) > 20:  # Assuming max 20 videos per newsletter
                    issues_for_newsletter.append(f"Too many videos: {len(newsletter_videos)}")
                
                # Check timestamp consistency
                if not integrity_test_helpers.calculate_data_freshness(sent_at, max_age_days=90):
                    issues_for_newsletter.append("Newsletter too old (>90 days)")
                
                # Verify all associated videos exist
                video_ids = [nv["video_id"] for nv in newsletter_videos]
                if video_ids:
                    existing_videos_response = database_clients["supabase"].client.table("videos").select(
                        "video_id"
                    ).in_("video_id", video_ids).execute()
                    
                    existing_video_ids = set([v["video_id"] for v in existing_videos_response.data]) if existing_videos_response.data else set()
                    missing_videos = set(video_ids) - existing_video_ids
                    
                    if missing_videos:
                        issues_for_newsletter.append(f"Missing videos: {len(missing_videos)}")
                
                if not issues_for_newsletter:
                    valid_newsletters += 1
                else:
                    consistency_issues.append({
                        "newsletter_id": newsletter_id,
                        "user_id": user_id,
                        "video_count": len(newsletter_videos),
                        "issues": issues_for_newsletter
                    })
            
            total_newsletters = len(newsletters_data)
            consistency_score = integrity_test_helpers.calculate_consistency_score(total_newsletters, valid_newsletters)
            
            details = {
                "total_newsletters": total_newsletters,
                "valid_newsletters": valid_newsletters,
                "consistency_issues": len(consistency_issues),
                "consistency_percentage": consistency_score,
                "sample_issues": consistency_issues[:10]
            }
            
            status = "PASSED" if consistency_score >= 95.0 else "FAILED"
            integrity_reporter.add_test_result(test_name, status, details)
            
            assert consistency_score >= 80.0, f"Newsletter consistency too low: {consistency_score}%"
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            integrity_reporter.add_test_result(test_name, "ERROR", {"error": str(e)})
            pytest.fail(f"Test failed with error: {e}")

    def test_user_vector_update_consistency(self, database_clients, integrity_test_helpers, integrity_reporter):
        """Test consistency of user vector updates with feedback data"""
        test_name = "user_vector_update_consistency"
        logger.info(f"Running {test_name}")
        
        try:
            # Get users with both embeddings and recent feedback
            users_with_embeddings_response = database_clients["supabase"].client.table("users").select(
                "user_id, embedding_id"
            ).neq("embedding_id", None).limit(100).execute()
            
            users_with_embeddings = users_with_embeddings_response.data if users_with_embeddings_response.data else []
            
            if not users_with_embeddings:
                logger.warning("No users with embeddings found")
                integrity_reporter.add_test_result(test_name, "SKIPPED", {"reason": "No users with embeddings"})
                return
            
            consistency_issues = []
            valid_user_states = 0
            
            for user in users_with_embeddings:
                user_id = user["user_id"]
                embedding_id = user["embedding_id"]
                
                issues_for_user = []
                
                # Get user's recent feedback
                feedback_response = database_clients["supabase"].client.table("feedback").select(
                    "video_id, rating, timestamp"
                ).eq("user_id", user_id).order("timestamp", desc=True).limit(50).execute()
                
                user_feedback = feedback_response.data if feedback_response.data else []
                
                # Validate consistency between embedding and feedback
                if len(user_feedback) > 0:
                    # User has feedback, should have a meaningful embedding
                    try:
                        if isinstance(embedding_id, str) and embedding_id.startswith('['):
                            embedding_data = eval(embedding_id)
                            
                            # Check if it's not a zero vector (should be updated based on feedback)
                            if all(x == 0.0 for x in embedding_data) and len(user_feedback) > 5:
                                issues_for_user.append("Zero embedding despite significant feedback")
                            
                            # Check for recent feedback vs embedding freshness
                            latest_feedback = max([f["timestamp"] for f in user_feedback])
                            if not integrity_test_helpers.calculate_data_freshness(latest_feedback, max_age_days=30):
                                # Old feedback is fine, but if there's very recent feedback,
                                # embedding should reflect recent preferences
                                recent_feedback = [f for f in user_feedback if integrity_test_helpers.calculate_data_freshness(f["timestamp"], max_age_days=7)]
                                if len(recent_feedback) > 0:
                                    # This would need more sophisticated analysis to determine if embedding reflects recent feedback
                                    pass
                    except:
                        issues_for_user.append("Cannot parse embedding for consistency check")
                else:
                    # User has embedding but no feedback - could be valid for new users
                    pass
                
                if not issues_for_user:
                    valid_user_states += 1
                else:
                    consistency_issues.append({
                        "user_id": user_id,
                        "feedback_count": len(user_feedback),
                        "issues": issues_for_user
                    })
            
            total_users = len(users_with_embeddings)
            consistency_score = integrity_test_helpers.calculate_consistency_score(total_users, valid_user_states)
            
            details = {
                "total_users_tested": total_users,
                "valid_user_states": valid_user_states,
                "consistency_issues": len(consistency_issues),
                "consistency_percentage": consistency_score,
                "sample_issues": consistency_issues[:10]
            }
            
            status = "PASSED" if consistency_score >= 90.0 else "FAILED"
            integrity_reporter.add_test_result(test_name, status, details)
            
            assert consistency_score >= 90.0, f"User vector update consistency too low: {consistency_score}%"
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            integrity_reporter.add_test_result(test_name, "ERROR", {"error": str(e)})
            pytest.fail(f"Test failed with error: {e}")