"""
Simple API-only Load Test (No Database Dependencies)

This version avoids the gevent/threading conflicts by not importing database clients.
Perfect for API performance testing without database load testing.
"""

import json
import random
import time
from locust import HttpUser, task, between

class SimpleAPIUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        self.user_id = f"user_{random.randint(1, 1000)}"
    
    @task(3)
    def health_check(self):
        """Test health endpoint"""
        response = self.client.get("/health", name="health_check")
        if response.status_code != 200:
            print(f"Health check failed: {response.status_code} - {response.text}")
    
    @task(2)
    def get_recommendations(self):
        """Test recommendations endpoint with proper parameters"""
        # Use correct parameter names based on the API definition
        params = {
            "user_id": self.user_id,
            "top_k": 10
        }
        response = self.client.get("/recommendations/", params=params, name="recommendations")
        if response.status_code != 200:
            print(f"Recommendations failed: {response.status_code} - {response.text}")
    
    @task(1)
    def workflow_test(self):
        """Test workflow endpoint with correct payload structure"""
        payload = {
            "user_id": self.user_id,
            "top_k": 10
        }
        
        response = self.client.post(
            "/run-workflow/run-workflow-video-ids",
            json=payload,
            name="workflow_video_ids"
        )
        if response.status_code not in [200, 202]:
            print(f"Workflow failed: {response.status_code} - {response.text}")

# For direct Locust usage
if __name__ == "__main__":
    import os
    os.system("locust -f simple_api_test.py --host http://localhost:8080")