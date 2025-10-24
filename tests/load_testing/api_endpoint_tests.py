"""
API Endpoint Load Tests for YouTube Recommendation System

This module contains Locust-based load tests for all API endpoints including:
- Recommendations endpoint
- Workflow execution
- Newsletter generation
- User vector updates
- Health checks

Usage:
    locust -f api_endpoint_tests.py --host=http://localhost:8080
"""

import json
import random
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from locust import HttpUser, task, between

# Import with fallback for missing modules
try:
    from tests.load_testing.base_load_test import BaseLoadTestUser, TestDataGenerator, performance_metrics
except ImportError:
    try:
        from base_load_test import BaseLoadTestUser, TestDataGenerator, performance_metrics
    except ImportError:
        # Fallback implementations
        class BaseLoadTestUser(HttpUser):
            wait_time = between(1, 5)
            def on_start(self):
                self.user_id = f"user_{random.randint(1, 1000)}"
        
        class TestDataGenerator:
            @staticmethod
            def generate_workflow_request():
                return {"user_id": f"user_{random.randint(1, 1000)}", "preferences": {}}
            
            @staticmethod
            def generate_video_feedback():
                return {"video_id": f"video_{random.randint(1, 1000)}", "rating": random.randint(1, 5)}
        
        class MockMetrics:
            def add_response_time(self, endpoint, time): pass
        
        performance_metrics = MockMetrics()

class RecommendationAPIUser(BaseLoadTestUser):
    """Load test user for recommendation API endpoints"""
    
    weight = 3  # Higher weight = more instances of this user type
    
    @task(3)
    def get_recommendations(self):
        """Test the main recommendations endpoint"""
        with self.client.get(
            "/recommendations/",
            name="get_recommendations",
            catch_response=True
        ) as response:
            start_time = time.time()
            try:
                if response.status_code == 200:
                    data = response.json()
                    if "recommendations" in data:
                        response.success()
                    else:
                        response.failure("No recommendations in response")
                else:
                    response.failure(f"Status code: {response.status_code}")
            except json.JSONDecodeError:
                response.failure("Invalid JSON response")
            finally:
                response_time = (time.time() - start_time) * 1000
                performance_metrics.add_response_time("recommendations", response_time)
    
    @task(2)
    def run_workflow(self):
        """Test workflow execution endpoint"""
        request_data = TestDataGenerator.generate_workflow_request()
        
        with self.client.post(
            "/run-workflow/run-workflow",
            json=request_data,
            name="run_workflow",
            catch_response=True
        ) as response:
            start_time = time.time()
            try:
                if response.status_code == 200:
                    data = response.json()
                    if "status" in data:
                        response.success()
                    else:
                        response.failure("No status in workflow response")
                else:
                    response.failure(f"Workflow failed with status: {response.status_code}")
            except json.JSONDecodeError:
                response.failure("Invalid JSON in workflow response")
            except Exception as e:
                response.failure(f"Workflow error: {str(e)}")
            finally:
                response_time = (time.time() - start_time) * 1000
                performance_metrics.add_response_time("run_workflow", response_time)
    
    @task(2)
    def run_workflow_video_ids(self):
        """Test workflow execution with video IDs endpoint"""
        request_data = {
            "user_id": self.user_id,
            "limit": random.randint(5, 20)
        }
        
        with self.client.post(
            "/run-workflow/run-workflow-video-ids",
            json=request_data,
            name="run_workflow_video_ids",
            catch_response=True
        ) as response:
            start_time = time.time()
            try:
                if response.status_code == 200:
                    data = response.json()
                    if "video_ids" in data:
                        response.success()
                    else:
                        response.failure("No video_ids in response")
                else:
                    response.failure(f"Status code: {response.status_code}")
            except json.JSONDecodeError:
                response.failure("Invalid JSON response")
            finally:
                response_time = (time.time() - start_time) * 1000
                performance_metrics.add_response_time("run_workflow_video_ids", response_time)

class UserVectorUpdateUser(BaseLoadTestUser):
    """Load test user for user vector update operations"""
    
    weight = 2
    
    @task(1)
    def run_daily_update(self):
        """Test daily user vector update"""
        request_data = {
            "user_id": self.user_id,
            "update_type": "daily"
        }
        
        with self.client.post(
            "/user-vector-update/run-daily-update",
            json=request_data,
            name="daily_vector_update",
            catch_response=True
        ) as response:
            start_time = time.time()
            try:
                if response.status_code in [200, 202]:  # Accept both sync and async responses
                    response.success()
                else:
                    response.failure(f"Update failed with status: {response.status_code}")
            finally:
                response_time = (time.time() - start_time) * 1000
                performance_metrics.add_response_time("daily_vector_update", response_time)
    
    @task(1)
    def run_manual_update(self):
        """Test manual user vector update"""
        feedback_data = TestDataGenerator.generate_video_feedback()
        request_data = {
            "user_id": self.user_id,
            "feedback": feedback_data
        }
        
        with self.client.post(
            "/user-vector-update/run-manual-update",
            json=request_data,
            name="manual_vector_update",
            catch_response=True
        ) as response:
            start_time = time.time()
            try:
                if response.status_code in [200, 202]:
                    response.success()
                else:
                    response.failure(f"Manual update failed: {response.status_code}")
            finally:
                response_time = (time.time() - start_time) * 1000
                performance_metrics.add_response_time("manual_vector_update", response_time)
    
    @task(2)
    def check_update_status(self):
        """Test vector update status endpoint"""
        with self.client.get(
            f"/user-vector-update/status?user_id={self.user_id}",
            name="vector_update_status",
            catch_response=True
        ) as response:
            start_time = time.time()
            try:
                if response.status_code == 200:
                    data = response.json()
                    if "status" in data:
                        response.success()
                    else:
                        response.failure("No status in response")
                else:
                    response.failure(f"Status check failed: {response.status_code}")
            except json.JSONDecodeError:
                response.failure("Invalid JSON in status response")
            finally:
                response_time = (time.time() - start_time) * 1000
                performance_metrics.add_response_time("vector_update_status", response_time)

class NewsletterUser(BaseLoadTestUser):
    """Load test user for newsletter generation"""
    
    weight = 1  # Lower weight as newsletters are generated less frequently
    
    @task(1)
    def check_newsletter_health(self):
        """Test newsletter service health"""
        with self.client.get(
            "/newsletter/health",
            name="newsletter_health",
            catch_response=True
        ) as response:
            start_time = time.time()
            try:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"Newsletter health check failed: {response.status_code}")
            finally:
                response_time = (time.time() - start_time) * 1000
                performance_metrics.add_response_time("newsletter_health", response_time)

class TestAPIUser(BaseLoadTestUser):
    """Load test user for test/debug endpoints"""
    
    weight = 1
    
    @task(2)
    def get_random_videos(self):
        """Test random videos endpoint"""
        with self.client.get(
            "/test/random-videos",
            name="random_videos",
            catch_response=True
        ) as response:
            start_time = time.time()
            try:
                if response.status_code == 200:
                    data = response.json()
                    if "videos" in data:
                        response.success()
                    else:
                        response.failure("No videos in response")
                else:
                    response.failure(f"Random videos failed: {response.status_code}")
            except json.JSONDecodeError:
                response.failure("Invalid JSON response")
            finally:
                response_time = (time.time() - start_time) * 1000
                performance_metrics.add_response_time("random_videos", response_time)
    
    @task(1)
    def test_database_connection(self):
        """Test database connection endpoint"""
        with self.client.get(
            "/test/test-database-connection",
            name="test_db_connection",
            catch_response=True
        ) as response:
            start_time = time.time()
            try:
                if response.status_code == 200:
                    response.success()
                else:
                    response.failure(f"DB connection test failed: {response.status_code}")
            finally:
                response_time = (time.time() - start_time) * 1000
                performance_metrics.add_response_time("test_db_connection", response_time)

class HealthCheckUser(BaseLoadTestUser):
    """Dedicated user for health check monitoring"""
    
    weight = 1
    
    @task(1)
    def health_check(self):
        """Test system health endpoint"""
        with self.client.get(
            "/health",
            name="health_check",
            catch_response=True
        ) as response:
            start_time = time.time()
            try:
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "healthy":
                        response.success()
                    else:
                        response.failure("System not healthy")
                else:
                    response.failure(f"Health check failed: {response.status_code}")
            except json.JSONDecodeError:
                response.failure("Invalid JSON in health response")
            finally:
                response_time = (time.time() - start_time) * 1000
                performance_metrics.add_response_time("health_check", response_time)

# Define user classes for Locust to use
user_classes = [
    RecommendationAPIUser,
    UserVectorUpdateUser, 
    NewsletterUser,
    TestAPIUser,
    HealthCheckUser
]