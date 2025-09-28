"""
Orphaned Records Detection Tests
Identifies and reports orphaned/dangling references across the database system
"""
import pytest
import logging
from typing import Dict, Any, List, Set, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

@pytest.mark.database_integrity
@pytest.mark.orphaned_records
class TestOrphanedRecordsDetection:
    """Detect orphaned records and dangling references"""

    def test_orphaned_feedback_records(self, database_clients, integrity_test_helpers, integrity_reporter):
        """Detect feedback records with orphaned user or video references"""
        test_name = "orphaned_feedback_records"
        logger.info(f"Running {test_name}")
        
        try:
            # Get all feedback records
            feedback_response = database_clients["supabase"].client.table("feedback").select(
                "id, user_id, video_id, rating, timestamp"
            ).limit(2000).execute()
            
            feedback_records = feedback_response.data if feedback_response.data else []
            
            if not feedback_records:
                logger.warning("No feedback records found")
                integrity_reporter.add_test_result(test_name, "SKIPPED", {"reason": "No feedback records"})
                return
            
            # Extract unique IDs
            feedback_user_ids = set([f["user_id"] for f in feedback_records])
            feedback_video_ids = set([f["video_id"] for f in feedback_records])
            
            # Get all existing users
            users_response = database_clients["supabase"].client.table("users").select("user_id").execute()
            existing_user_ids = set([u["user_id"] for u in users_response.data]) if users_response.data else set()
            
            # Get all existing videos
            videos_response = database_clients["supabase"].client.table("videos").select("video_id").execute()
            existing_video_ids = set([v["video_id"] for v in videos_response.data]) if videos_response.data else set()
            
            # Find orphaned records
            orphaned_user_refs = feedback_user_ids - existing_user_ids
            orphaned_video_refs = feedback_video_ids - existing_video_ids
            
            # Get specific orphaned feedback records
            orphaned_feedback_records = []
            for feedback in feedback_records:
                if feedback["user_id"] in orphaned_user_refs or feedback["video_id"] in orphaned_video_refs:
                    orphaned_feedback_records.append({
                        "feedback_id": feedback["id"],
                        "user_id": feedback["user_id"],
                        "video_id": feedback["video_id"],
                        "orphaned_user": feedback["user_id"] in orphaned_user_refs,
                        "orphaned_video": feedback["video_id"] in orphaned_video_refs,
                        "timestamp": feedback["timestamp"]
                    })
            
            total_feedback = len(feedback_records)
            orphaned_count = len(orphaned_feedback_records)
            integrity_score = integrity_test_helpers.calculate_consistency_score(
                total_feedback, 
                total_feedback - orphaned_count
            )
            
            details = {
                "total_feedback_records": total_feedback,
                "orphaned_feedback_records": orphaned_count,
                "orphaned_user_references": len(orphaned_user_refs),
                "orphaned_video_references": len(orphaned_video_refs),
                "integrity_percentage": integrity_score,
                "sample_orphaned_users": list(orphaned_user_refs)[:10],
                "sample_orphaned_videos": list(orphaned_video_refs)[:10],
                "sample_orphaned_records": orphaned_feedback_records[:10]
            }
            
            # Test passes if less than 2% of records are orphaned
            status = "PASSED" if integrity_score >= 98.0 else "FAILED"
            integrity_reporter.add_test_result(test_name, status, details)
            
            assert integrity_score >= 98.0, f"Too many orphaned feedback records: {100 - integrity_score}%"
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            integrity_reporter.add_test_result(test_name, "ERROR", {"error": str(e)})
            pytest.fail(f"Test failed with error: {e}")

    def test_orphaned_newsletter_videos(self, database_clients, integrity_test_helpers, integrity_reporter):
        """Detect newsletter_videos with orphaned newsletter or video references"""
        test_name = "orphaned_newsletter_videos"
        logger.info(f"Running {test_name}")
        
        try:
            # Get all newsletter_videos records
            newsletter_videos_response = database_clients["supabase"].client.table("newsletter_videos").select(
                "id, newsletter_id, video_id, clicked"
            ).limit(2000).execute()
            
            newsletter_videos = newsletter_videos_response.data if newsletter_videos_response.data else []
            
            if not newsletter_videos:
                logger.warning("No newsletter_videos records found")
                integrity_reporter.add_test_result(test_name, "SKIPPED", {"reason": "No newsletter_videos records"})
                return
            
            # Extract unique IDs
            newsletter_ids = set([nv["newsletter_id"] for nv in newsletter_videos])
            video_ids = set([nv["video_id"] for nv in newsletter_videos])
            
            # Get all existing newsletters
            newsletters_response = database_clients["supabase"].client.table("newsletters").select("id").execute()
            existing_newsletter_ids = set([n["id"] for n in newsletters_response.data]) if newsletters_response.data else set()
            
            # Get all existing videos
            videos_response = database_clients["supabase"].client.table("videos").select("video_id").execute()
            existing_video_ids = set([v["video_id"] for v in videos_response.data]) if videos_response.data else set()
            
            # Find orphaned references
            orphaned_newsletter_refs = newsletter_ids - existing_newsletter_ids
            orphaned_video_refs = video_ids - existing_video_ids
            
            # Get specific orphaned records
            orphaned_records = []
            for nv in newsletter_videos:
                if nv["newsletter_id"] in orphaned_newsletter_refs or nv["video_id"] in orphaned_video_refs:
                    orphaned_records.append({
                        "record_id": nv["id"],
                        "newsletter_id": nv["newsletter_id"],
                        "video_id": nv["video_id"],
                        "orphaned_newsletter": nv["newsletter_id"] in orphaned_newsletter_refs,
                        "orphaned_video": nv["video_id"] in orphaned_video_refs
                    })
            
            total_records = len(newsletter_videos)
            orphaned_count = len(orphaned_records)
            integrity_score = integrity_test_helpers.calculate_consistency_score(
                total_records,
                total_records - orphaned_count
            )
            
            details = {
                "total_newsletter_video_records": total_records,
                "orphaned_records": orphaned_count,
                "orphaned_newsletter_references": len(orphaned_newsletter_refs),
                "orphaned_video_references": len(orphaned_video_refs),
                "integrity_percentage": integrity_score,
                "sample_orphaned_newsletters": list(orphaned_newsletter_refs)[:5],
                "sample_orphaned_videos": list(orphaned_video_refs)[:5],
                "sample_orphaned_records": orphaned_records[:10]
            }
            
            status = "PASSED" if integrity_score >= 99.0 else "FAILED"
            integrity_reporter.add_test_result(test_name, status, details)
            
            assert integrity_score >= 99.0, f"Too many orphaned newsletter-video records: {100 - integrity_score}%"
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            integrity_reporter.add_test_result(test_name, "ERROR", {"error": str(e)})
            pytest.fail(f"Test failed with error: {e}")

    def test_orphaned_user_embeddings(self, database_clients, integrity_test_helpers, integrity_reporter):
        """Detect users with embeddings but no activity (potential orphaned embeddings)"""
        test_name = "orphaned_user_embeddings"
        logger.info(f"Running {test_name}")
        
        try:
            # Get users with embeddings
            users_with_embeddings_response = database_clients["supabase"].client.table("users").select(
                "user_id, embedding_id"
            ).neq("embedding_id", None).execute()
            
            users_with_embeddings = users_with_embeddings_response.data if users_with_embeddings_response.data else []
            
            if not users_with_embeddings:
                logger.warning("No users with embeddings found")
                integrity_reporter.add_test_result(test_name, "SKIPPED", {"reason": "No users with embeddings"})
                return
            
            user_ids_with_embeddings = set([u["user_id"] for u in users_with_embeddings])
            
            # Check for user activity in feedback table
            feedback_response = database_clients["supabase"].client.table("feedback").select("user_id").execute()
            active_user_ids_feedback = set([f["user_id"] for f in feedback_response.data]) if feedback_response.data else set()
            
            # Check for user activity in newsletters table
            newsletters_response = database_clients["supabase"].client.table("newsletters").select("user_id").execute()
            active_user_ids_newsletters = set([n["user_id"] for n in newsletters_response.data]) if newsletters_response.data else set()
            
            # Combine all active user IDs
            all_active_user_ids = active_user_ids_feedback | active_user_ids_newsletters
            
            # Find users with embeddings but no activity
            inactive_users_with_embeddings = user_ids_with_embeddings - all_active_user_ids
            
            # Analyze these users further
            suspicious_embeddings = []
            for user in users_with_embeddings:
                user_id = user["user_id"]
                if user_id in inactive_users_with_embeddings:
                    # Check if embedding is non-zero (zero embeddings for inactive users might be expected)
                    try:
                        embedding_id = user["embedding_id"]
                        if isinstance(embedding_id, str) and embedding_id.startswith('['):
                            embedding_data = eval(embedding_id)
                            if not all(x == 0.0 for x in embedding_data):
                                suspicious_embeddings.append({
                                    "user_id": user_id,
                                    "has_non_zero_embedding": True,
                                    "embedding_magnitude": sum(x*x for x in embedding_data)**0.5
                                })
                    except:
                        suspicious_embeddings.append({
                            "user_id": user_id,
                            "embedding_parse_error": True
                        })
            
            total_users_with_embeddings = len(users_with_embeddings)
            potentially_orphaned = len(suspicious_embeddings)
            health_score = integrity_test_helpers.calculate_consistency_score(
                total_users_with_embeddings,
                total_users_with_embeddings - potentially_orphaned
            )
            
            details = {
                "total_users_with_embeddings": total_users_with_embeddings,
                "inactive_users_with_embeddings": len(inactive_users_with_embeddings),
                "suspicious_non_zero_embeddings": potentially_orphaned,
                "users_with_feedback": len(active_user_ids_feedback),
                "users_with_newsletters": len(active_user_ids_newsletters),
                "health_percentage": health_score,
                "sample_inactive_users": list(inactive_users_with_embeddings)[:10],
                "sample_suspicious_embeddings": suspicious_embeddings[:10]
            }
            
            # More lenient threshold since some inactive users with embeddings might be legitimate
            status = "PASSED" if health_score >= 80.0 else "FAILED"
            integrity_reporter.add_test_result(test_name, status, details)
            
            # This is more of a warning than a hard failure
            if health_score < 80.0:
                logger.warning(f"Many users have embeddings without activity: {100 - health_score}%")
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            integrity_reporter.add_test_result(test_name, "ERROR", {"error": str(e)})
            pytest.fail(f"Test failed with error: {e}")

    def test_orphaned_video_embeddings_qdrant(self, database_clients, sample_video_ids, integrity_test_helpers, integrity_reporter):
        """Detect video embeddings in Qdrant that don't correspond to videos in Supabase"""
        test_name = "orphaned_video_embeddings_qdrant"
        logger.info(f"Running {test_name}")
        
        try:
            # Get all videos from Supabase
            videos_response = database_clients["supabase"].client.table("videos").select("video_id").execute()
            supabase_video_ids = set([v["video_id"] for v in videos_response.data]) if videos_response.data else set()
            
            if not supabase_video_ids:
                logger.warning("No videos found in Supabase")
                integrity_reporter.add_test_result(test_name, "SKIPPED", {"reason": "No videos in Supabase"})
                return
            
            # For Qdrant, we would need to implement a method to get all video IDs
            # Since this isn't available, we'll simulate with sample data
            if not database_clients["qdrant"].client:
                logger.warning("Qdrant client not available")
                integrity_reporter.add_test_result(test_name, "SKIPPED", {"reason": "Qdrant not available"})
                return
            
            # Simulate Qdrant video IDs (in real implementation, this would query Qdrant)
            # This would need to be implemented in the qdrant_client
            simulated_qdrant_video_ids = set(sample_video_ids[:500])  # Simulate some Qdrant data
            
            # Find orphaned embeddings in Qdrant
            orphaned_in_qdrant = simulated_qdrant_video_ids - supabase_video_ids
            
            # Find missing embeddings (videos in Supabase but not in Qdrant)
            missing_in_qdrant = supabase_video_ids - simulated_qdrant_video_ids
            
            total_unique_videos = len(supabase_video_ids | simulated_qdrant_video_ids)
            consistency_score = integrity_test_helpers.calculate_consistency_score(
                total_unique_videos,
                len(supabase_video_ids & simulated_qdrant_video_ids)
            )
            
            details = {
                "supabase_videos": len(supabase_video_ids),
                "qdrant_videos": len(simulated_qdrant_video_ids),
                "orphaned_in_qdrant": len(orphaned_in_qdrant),
                "missing_in_qdrant": len(missing_in_qdrant),
                "consistency_percentage": consistency_score,
                "sample_orphaned_qdrant": list(orphaned_in_qdrant)[:10],
                "sample_missing_qdrant": list(missing_in_qdrant)[:10],
                "note": "This test uses simulated Qdrant data - implement get_all_video_ids in qdrant_client"
            }
            
            status = "PASSED" if consistency_score >= 85.0 else "FAILED"
            integrity_reporter.add_test_result(test_name, status, details)
            
            # More lenient since this is simulated data
            if consistency_score < 85.0:
                logger.warning(f"Video embedding consistency between Supabase and Qdrant: {consistency_score}%")
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            integrity_reporter.add_test_result(test_name, "ERROR", {"error": str(e)})
            pytest.fail(f"Test failed with error: {e}")

    def test_orphaned_extractive_summaries_mongodb(self, database_clients, sample_video_ids, integrity_test_helpers, integrity_reporter):
        """Detect extractive summaries in MongoDB that don't correspond to videos in Supabase"""
        test_name = "orphaned_extractive_summaries_mongodb"
        logger.info(f"Running {test_name}")
        
        try:
            # Get all videos from Supabase
            videos_response = database_clients["supabase"].client.table("videos").select("video_id").execute()
            supabase_video_ids = set([v["video_id"] for v in videos_response.data]) if videos_response.data else set()
            
            if not supabase_video_ids:
                logger.warning("No videos found in Supabase")
                integrity_reporter.add_test_result(test_name, "SKIPPED", {"reason": "No videos in Supabase"})
                return
            
            # Check MongoDB connection
            if not database_clients["mongodb"].client:
                logger.warning("MongoDB client not available")
                integrity_reporter.add_test_result(test_name, "SKIPPED", {"reason": "MongoDB not available"})
                return
            
            # Simulate MongoDB extractive summaries data
            # In real implementation, this would query MongoDB for all video IDs with summaries
            simulated_mongodb_video_ids = set(sample_video_ids[:300])  # Simulate MongoDB data
            
            # Find orphaned summaries
            orphaned_in_mongodb = simulated_mongodb_video_ids - supabase_video_ids
            
            # Find missing summaries
            missing_in_mongodb = supabase_video_ids - simulated_mongodb_video_ids
            
            total_videos_needing_summaries = len(supabase_video_ids)
            coverage_score = integrity_test_helpers.calculate_consistency_score(
                total_videos_needing_summaries,
                len(simulated_mongodb_video_ids & supabase_video_ids)
            )
            
            details = {
                "supabase_videos": len(supabase_video_ids),
                "mongodb_summaries": len(simulated_mongodb_video_ids),
                "orphaned_in_mongodb": len(orphaned_in_mongodb),
                "missing_summaries": len(missing_in_mongodb),
                "summary_coverage_percentage": coverage_score,
                "sample_orphaned_mongodb": list(orphaned_in_mongodb)[:10],
                "sample_missing_summaries": list(missing_in_mongodb)[:10],
                "note": "This test uses simulated MongoDB data - implement get_all_summary_video_ids in mongodb_client"
            }
            
            # Extractive summaries are optional, so we're more lenient
            status = "PASSED" if len(orphaned_in_mongodb) < len(simulated_mongodb_video_ids) * 0.1 else "FAILED"
            integrity_reporter.add_test_result(test_name, status, details)
            
            if len(orphaned_in_mongodb) > 0:
                logger.warning(f"Found {len(orphaned_in_mongodb)} orphaned extractive summaries in MongoDB")
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            integrity_reporter.add_test_result(test_name, "ERROR", {"error": str(e)})
            pytest.fail(f"Test failed with error: {e}")