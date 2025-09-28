"""
Constraint Validation Tests
Tests for data format constraints and business rule enforcement
"""
import pytest
import logging
import re
from typing import Dict, Any, List, Set, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@pytest.mark.database_integrity
@pytest.mark.constraint_validation
class TestConstraintValidation:
    """Test data format constraints and business rules"""

    def test_video_id_format_validation(self, database_clients, integrity_test_helpers, integrity_reporter):
        """Validate that all video IDs follow YouTube video ID format"""
        test_name = "video_id_format_validation"
        logger.info(f"Running {test_name}")
        
        try:
            # Get all video IDs from different sources
            sources_to_check = [
                ("videos", "video_id"),
                ("feedback", "video_id"),
                ("newsletter_videos", "video_id")
            ]
            
            all_violations = []
            total_video_ids_checked = 0
            
            youtube_video_id_pattern = re.compile(r'^[a-zA-Z0-9_-]{11}$')
            
            for table_name, column_name in sources_to_check:
                try:
                    response = database_clients["supabase"].client.table(table_name).select(column_name).execute()
                    records = response.data if response.data else []
                    
                    for record in records:
                        video_id = record[column_name]
                        total_video_ids_checked += 1
                        
                        # Validate format
                        if not youtube_video_id_pattern.match(video_id):
                            all_violations.append({
                                "source_table": table_name,
                                "video_id": video_id,
                                "violation_type": "invalid_format",
                                "expected_pattern": "11 characters, alphanumeric with _ and -"
                            })
                        
                        # Check for common issues
                        if len(video_id) != 11:
                            all_violations.append({
                                "source_table": table_name,
                                "video_id": video_id,
                                "violation_type": "invalid_length",
                                "length": len(video_id)
                            })
                        
                        if any(char in video_id for char in [' ', '\t', '\n']):
                            all_violations.append({
                                "source_table": table_name,
                                "video_id": video_id,
                                "violation_type": "contains_whitespace"
                            })
                
                except Exception as e:
                    logger.warning(f"Could not check {table_name}: {e}")
            
            valid_video_ids = total_video_ids_checked - len(all_violations)
            compliance_score = integrity_test_helpers.calculate_consistency_score(
                total_video_ids_checked,
                valid_video_ids
            )
            
            # Group violations by type
            violations_by_type = {}
            for violation in all_violations:
                violation_type = violation["violation_type"]
                if violation_type not in violations_by_type:
                    violations_by_type[violation_type] = 0
                violations_by_type[violation_type] += 1
            
            details = {
                "total_video_ids_checked": total_video_ids_checked,
                "valid_video_ids": valid_video_ids,
                "total_violations": len(all_violations),
                "compliance_percentage": compliance_score,
                "violations_by_type": violations_by_type,
                "sample_violations": all_violations[:20]
            }
            
            status = "PASSED" if compliance_score >= 99.0 else "FAILED"
            integrity_reporter.add_test_result(test_name, status, details)
            
            assert compliance_score >= 99.0, f"Video ID format compliance too low: {compliance_score}%"
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            integrity_reporter.add_test_result(test_name, "ERROR", {"error": str(e)})
            pytest.fail(f"Test failed with error: {e}")

    def test_user_id_format_validation(self, database_clients, integrity_test_helpers, integrity_reporter):
        """Validate user ID format consistency across tables"""
        test_name = "user_id_format_validation"
        logger.info(f"Running {test_name}")
        
        try:
            # Get user IDs from different sources
            sources_to_check = [
                ("users", "user_id"),
                ("feedback", "user_id"),
                ("newsletters", "user_id")
            ]
            
            all_violations = []
            total_user_ids_checked = 0
            
            # Define acceptable user ID patterns (adjust based on your system)
            valid_patterns = [
                re.compile(r'^[a-zA-Z0-9_-]+$'),  # Alphanumeric with underscore and dash
                re.compile(r'^[a-f0-9-]{36}$'),   # UUID format
                re.compile(r'^[a-zA-Z0-9]{20,}$') # Long alphanumeric
            ]
            
            for table_name, column_name in sources_to_check:
                try:
                    response = database_clients["supabase"].client.table(table_name).select(column_name).execute()
                    records = response.data if response.data else []
                    
                    for record in records:
                        user_id = record[column_name]
                        total_user_ids_checked += 1
                        
                        # Check if user_id matches any valid pattern
                        if not any(pattern.match(user_id) for pattern in valid_patterns):
                            all_violations.append({
                                "source_table": table_name,
                                "user_id": user_id[:50],  # Truncate for privacy
                                "violation_type": "invalid_format",
                                "length": len(user_id)
                            })
                        
                        # Check for common issues
                        if len(user_id) < 3:
                            all_violations.append({
                                "source_table": table_name,
                                "user_id": user_id,
                                "violation_type": "too_short"
                            })
                        
                        if any(char in user_id for char in [' ', '\t', '\n', '\r']):
                            all_violations.append({
                                "source_table": table_name,
                                "user_id": user_id[:20],
                                "violation_type": "contains_whitespace"
                            })
                        
                        # Check for SQL injection patterns (basic check)
                        suspicious_patterns = ["'", '"', ';', '--', '/*', '*/', 'DROP', 'SELECT']
                        if any(pattern.lower() in user_id.lower() for pattern in suspicious_patterns):
                            all_violations.append({
                                "source_table": table_name,
                                "user_id": user_id[:20],
                                "violation_type": "suspicious_content"
                            })
                
                except Exception as e:
                    logger.warning(f"Could not check {table_name}: {e}")
            
            valid_user_ids = total_user_ids_checked - len(all_violations)
            compliance_score = integrity_test_helpers.calculate_consistency_score(
                total_user_ids_checked,
                valid_user_ids
            )
            
            # Group violations by type
            violations_by_type = {}
            for violation in all_violations:
                violation_type = violation["violation_type"]
                if violation_type not in violations_by_type:
                    violations_by_type[violation_type] = 0
                violations_by_type[violation_type] += 1
            
            details = {
                "total_user_ids_checked": total_user_ids_checked,
                "valid_user_ids": valid_user_ids,
                "total_violations": len(all_violations),
                "compliance_percentage": compliance_score,
                "violations_by_type": violations_by_type,
                "sample_violations": all_violations[:10]  # Limited for privacy
            }
            
            status = "PASSED" if compliance_score >= 98.0 else "FAILED"
            integrity_reporter.add_test_result(test_name, status, details)
            
            assert compliance_score >= 98.0, f"User ID format compliance too low: {compliance_score}%"
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            integrity_reporter.add_test_result(test_name, "ERROR", {"error": str(e)})
            pytest.fail(f"Test failed with error: {e}")

    def test_rating_constraint_validation(self, database_clients, integrity_test_helpers, integrity_reporter):
        """Validate that all ratings are within the valid range (1-5)"""
        test_name = "rating_constraint_validation"
        logger.info(f"Running {test_name}")
        
        try:
            # Get all ratings from feedback table
            feedback_response = database_clients["supabase"].client.table("feedback").select(
                "id, user_id, video_id, rating"
            ).execute()
            
            feedback_records = feedback_response.data if feedback_response.data else []
            
            if not feedback_records:
                logger.warning("No feedback records found")
                integrity_reporter.add_test_result(test_name, "SKIPPED", {"reason": "No feedback records"})
                return
            
            constraint_violations = []
            valid_ratings = 0
            rating_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "invalid": 0}
            
            for feedback in feedback_records:
                rating = feedback["rating"]
                feedback_id = feedback["id"]
                
                # Check if rating is within valid range
                if isinstance(rating, (int, float)) and 1 <= rating <= 5:
                    valid_ratings += 1
                    rating_distribution[int(rating)] += 1
                else:
                    rating_distribution["invalid"] += 1
                    constraint_violations.append({
                        "feedback_id": feedback_id,
                        "user_id": feedback["user_id"],
                        "video_id": feedback["video_id"],
                        "invalid_rating": rating,
                        "violation_type": "out_of_range" if isinstance(rating, (int, float)) else "invalid_type",
                        "rating_type": str(type(rating).__name__)
                    })
            
            total_ratings = len(feedback_records)
            compliance_score = integrity_test_helpers.calculate_consistency_score(total_ratings, valid_ratings)
            
            # Check for suspicious rating distributions
            distribution_warnings = []
            total_valid_ratings = sum([rating_distribution[i] for i in range(1, 6)])
            if total_valid_ratings > 0:
                five_star_percentage = (rating_distribution[5] / total_valid_ratings) * 100
                one_star_percentage = (rating_distribution[1] / total_valid_ratings) * 100
                
                if five_star_percentage > 70:
                    distribution_warnings.append(f"Suspiciously high 5-star ratings: {five_star_percentage:.1f}%")
                if one_star_percentage > 40:
                    distribution_warnings.append(f"Suspiciously high 1-star ratings: {one_star_percentage:.1f}%")
                
                # Check for missing ratings in middle range
                middle_ratings = rating_distribution[2] + rating_distribution[3] + rating_distribution[4]
                middle_percentage = (middle_ratings / total_valid_ratings) * 100
                if middle_percentage < 10:
                    distribution_warnings.append(f"Very few middle-range ratings: {middle_percentage:.1f}%")
            
            details = {
                "total_feedback_records": total_ratings,
                "valid_ratings": valid_ratings,
                "constraint_violations": len(constraint_violations),
                "compliance_percentage": compliance_score,
                "rating_distribution": rating_distribution,
                "distribution_warnings": distribution_warnings,
                "sample_violations": constraint_violations[:10]
            }
            
            status = "PASSED" if compliance_score >= 99.5 else "FAILED"
            integrity_reporter.add_test_result(test_name, status, details)
            
            assert compliance_score >= 99.5, f"Rating constraint compliance too low: {compliance_score}%"
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            integrity_reporter.add_test_result(test_name, "ERROR", {"error": str(e)})
            pytest.fail(f"Test failed with error: {e}")

    def test_timestamp_format_validation(self, database_clients, integrity_test_helpers, integrity_reporter):
        """Validate timestamp formats across tables"""
        test_name = "timestamp_format_validation"
        logger.info(f"Running {test_name}")
        
        try:
            # Tables and columns with timestamps to check
            timestamp_sources = [
                ("feedback", "timestamp"),
                ("newsletters", "sent_at"),
                ("users", "created_at") if self._column_exists(database_clients["supabase"], "users", "created_at") else None,
                ("users", "updated_at") if self._column_exists(database_clients["supabase"], "users", "updated_at") else None
            ]
            
            # Filter out None values
            timestamp_sources = [ts for ts in timestamp_sources if ts is not None]
            
            all_violations = []
            total_timestamps_checked = 0
            
            for table_name, column_name in timestamp_sources:
                try:
                    response = database_clients["supabase"].client.table(table_name).select(
                        f"id, {column_name}"
                    ).limit(500).execute()
                    
                    records = response.data if response.data else []
                    
                    for record in records:
                        timestamp_value = record[column_name]
                        record_id = record.get("id", "unknown")
                        total_timestamps_checked += 1
                        
                        if timestamp_value is None:
                            # Null timestamps might be acceptable for some columns
                            continue
                        
                        # Try to parse timestamp
                        try:
                            if isinstance(timestamp_value, str):
                                # Handle different timestamp formats
                                parsed_timestamp = None
                                
                                # ISO format with timezone
                                if timestamp_value.endswith('Z') or '+' in timestamp_value or timestamp_value.endswith('+00:00'):
                                    parsed_timestamp = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                                else:
                                    # ISO format without timezone
                                    parsed_timestamp = datetime.fromisoformat(timestamp_value)
                                
                                # Check if timestamp is reasonable (not too far in future/past)
                                now = datetime.now()
                                if parsed_timestamp > now + timedelta(days=1):
                                    all_violations.append({
                                        "table": table_name,
                                        "column": column_name,
                                        "record_id": record_id,
                                        "timestamp": timestamp_value,
                                        "violation_type": "future_timestamp"
                                    })
                                elif parsed_timestamp < now - timedelta(days=365*5):  # 5 years ago
                                    all_violations.append({
                                        "table": table_name,
                                        "column": column_name,
                                        "record_id": record_id,
                                        "timestamp": timestamp_value,
                                        "violation_type": "very_old_timestamp"
                                    })
                            else:
                                all_violations.append({
                                    "table": table_name,
                                    "column": column_name,
                                    "record_id": record_id,
                                    "timestamp": str(timestamp_value),
                                    "violation_type": "non_string_timestamp"
                                })
                        
                        except (ValueError, TypeError) as e:
                            all_violations.append({
                                "table": table_name,
                                "column": column_name,
                                "record_id": record_id,
                                "timestamp": str(timestamp_value)[:50],
                                "violation_type": "unparseable_timestamp",
                                "error": str(e)[:100]
                            })
                
                except Exception as e:
                    logger.warning(f"Could not check timestamps in {table_name}.{column_name}: {e}")
            
            valid_timestamps = total_timestamps_checked - len(all_violations)
            compliance_score = integrity_test_helpers.calculate_consistency_score(
                total_timestamps_checked,
                valid_timestamps
            )
            
            # Group violations by type
            violations_by_type = {}
            for violation in all_violations:
                violation_type = violation["violation_type"]
                if violation_type not in violations_by_type:
                    violations_by_type[violation_type] = 0
                violations_by_type[violation_type] += 1
            
            details = {
                "total_timestamps_checked": total_timestamps_checked,
                "valid_timestamps": valid_timestamps,
                "total_violations": len(all_violations),
                "compliance_percentage": compliance_score,
                "violations_by_type": violations_by_type,
                "sample_violations": all_violations[:15]
            }
            
            status = "PASSED" if compliance_score >= 95.0 else "FAILED"
            integrity_reporter.add_test_result(test_name, status, details)
            
            assert compliance_score >= 95.0, f"Timestamp format compliance too low: {compliance_score}%"
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            integrity_reporter.add_test_result(test_name, "ERROR", {"error": str(e)})
            pytest.fail(f"Test failed with error: {e}")

    def test_embedding_dimension_constraints(self, database_clients, integrity_test_helpers, integrity_reporter):
        """Validate that all embeddings have the correct dimensions (768)"""
        test_name = "embedding_dimension_constraints"
        logger.info(f"Running {test_name}")
        
        try:
            # Get users with embeddings
            users_response = database_clients["supabase"].client.table("users").select(
                "user_id, embedding_id"
            ).neq("embedding_id", None).limit(300).execute()
            
            users_with_embeddings = users_response.data if users_response.data else []
            
            if not users_with_embeddings:
                logger.warning("No users with embeddings found")
                integrity_reporter.add_test_result(test_name, "SKIPPED", {"reason": "No users with embeddings"})
                return
            
            dimension_violations = []
            valid_embeddings = 0
            expected_dimension = 768
            
            dimension_counts = {}  # Track different dimensions found
            
            for user in users_with_embeddings:
                user_id = user["user_id"]
                embedding_id = user["embedding_id"]
                
                try:
                    if isinstance(embedding_id, str) and embedding_id.startswith('['):
                        embedding_data = eval(embedding_id)  # Use json.loads in production
                        
                        if isinstance(embedding_data, list):
                            actual_dimension = len(embedding_data)
                            
                            # Track dimension distribution
                            if actual_dimension not in dimension_counts:
                                dimension_counts[actual_dimension] = 0
                            dimension_counts[actual_dimension] += 1
                            
                            if actual_dimension == expected_dimension:
                                valid_embeddings += 1
                            else:
                                dimension_violations.append({
                                    "user_id": user_id,
                                    "expected_dimension": expected_dimension,
                                    "actual_dimension": actual_dimension,
                                    "violation_type": "wrong_dimension"
                                })
                        else:
                            dimension_violations.append({
                                "user_id": user_id,
                                "embedding_data_type": str(type(embedding_data).__name__),
                                "violation_type": "not_a_list"
                            })
                    else:
                        dimension_violations.append({
                            "user_id": user_id,
                            "embedding_format": str(embedding_id)[:50],
                            "violation_type": "invalid_format"
                        })
                
                except Exception as e:
                    dimension_violations.append({
                        "user_id": user_id,
                        "error": str(e)[:100],
                        "violation_type": "parse_error"
                    })
            
            total_embeddings = len(users_with_embeddings)
            compliance_score = integrity_test_helpers.calculate_consistency_score(
                total_embeddings,
                valid_embeddings
            )
            
            # Group violations by type
            violations_by_type = {}
            for violation in dimension_violations:
                violation_type = violation["violation_type"]
                if violation_type not in violations_by_type:
                    violations_by_type[violation_type] = 0
                violations_by_type[violation_type] += 1
            
            details = {
                "total_embeddings_checked": total_embeddings,
                "valid_embeddings": valid_embeddings,
                "expected_dimension": expected_dimension,
                "dimension_violations": len(dimension_violations),
                "compliance_percentage": compliance_score,
                "dimension_distribution": dimension_counts,
                "violations_by_type": violations_by_type,
                "sample_violations": dimension_violations[:10]
            }
            
            status = "PASSED" if compliance_score >= 85.0 else "FAILED"
            integrity_reporter.add_test_result(test_name, status, details)
            
            assert compliance_score >= 85.0, f"Embedding dimension compliance too low: {compliance_score}%"
            
        except Exception as e:
            logger.error(f"Error in {test_name}: {e}")
            integrity_reporter.add_test_result(test_name, "ERROR", {"error": str(e)})
            pytest.fail(f"Test failed with error: {e}")

    def _column_exists(self, supabase_client, table_name: str, column_name: str) -> bool:
        """Helper method to check if a column exists in a table"""
        try:
            response = supabase_client.client.table(table_name).select(column_name).limit(1).execute()
            return True
        except:
            return False