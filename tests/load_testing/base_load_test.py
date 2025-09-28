"""
Load Testing Configuration and Base Classes for Recommendation System

This module provides the foundation for load testing the YouTube Recommendation
System using Locust framework. It includes base classes, configuration management,
and utilities for comprehensive performance testing.

Author: AI Assistant
Date: September 2025
"""

import os
import random
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import requests
try:
    from locust import HttpUser, task, between
except ImportError:
    print("Warning: Locust not available. Install with: pip install locust")
    # Create mock classes for basic functionality
    class HttpUser:
        wait_time = None
        client = None
    class task:
        def __init__(self, weight=1): self.weight = weight
        def __call__(self, func): return func
    def between(min_wait, max_wait): return None

try:
    from faker import Faker
    fake = Faker()
except ImportError:
    print("Warning: Faker not available. Using basic test data generation")
    class MockFaker:
        def uuid4(self): return f"test-{random.randint(1000, 9999)}"
        def company(self): return f"Company{random.randint(1, 100)}"
        def city(self): return f"City{random.randint(1, 100)}"
        def date_time_between(self, **kwargs): return datetime.now()
    fake = MockFaker()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestConfig:
    """Configuration class for load testing parameters"""
    base_url: str = "http://localhost:8080"
    min_wait_time: int = 1
    max_wait_time: int = 5
    users_per_second: int = 10
    max_users: int = 100
    test_duration: str = "300s"  # 5 minutes
    
    # Test data configuration
    user_ids: List[str] = None
    video_ids: List[str] = None
    
    def __post_init__(self):
        if self.user_ids is None:
            self.user_ids = [f"user_{i}" for i in range(1, 101)]
        if self.video_ids is None:
            self.video_ids = [f"video_{i}" for i in range(1, 1001)]

class BaseLoadTestUser(HttpUser):
    """Base class for all load test users with common functionality"""
    
    abstract = True  # Mark as abstract to prevent direct instantiation
    wait_time = between(1, 5)
    
    def on_start(self):
        """Called when a user starts testing"""
        self.user_id = random.choice(LoadTestConfig().user_ids)
        self.session_id = fake.uuid4()
        logger.info(f"Starting load test for user: {self.user_id}")
        
        # Test if the service is healthy
        self.client.get("/health")
    
    def on_stop(self):
        """Called when a user stops testing"""
        logger.info(f"Stopping load test for user: {self.user_id}")

class TestDataGenerator:
    """Generates realistic test data for load testing"""
    
    @staticmethod
    def generate_user_preferences() -> Dict[str, Any]:
        """Generate realistic user preferences"""
        categories = [
            "technology", "science", "education", "entertainment",
            "music", "sports", "gaming", "cooking", "travel", "fitness"
        ]
        
        return {
            "user_id": fake.uuid4(),
            "preferences": random.sample(categories, k=random.randint(2, 5)),
            "watch_time_preference": random.choice(["short", "medium", "long"]),
            "preferred_channels": [fake.company() for _ in range(random.randint(1, 3))],
            "language": random.choice(["en", "es", "fr", "de", "it"]),
            "age_group": random.choice(["18-25", "26-35", "36-45", "46-60", "60+"]),
            "created_at": fake.date_time_between(start_date='-1y', end_date='now').isoformat()
        }
    
    @staticmethod
    def generate_video_feedback() -> Dict[str, Any]:
        """Generate realistic video feedback data"""
        return {
            "video_id": fake.uuid4(),
            "user_id": fake.uuid4(),
            "rating": random.randint(1, 5),
            "watch_duration": random.randint(30, 3600),  # 30 seconds to 1 hour
            "liked": random.choice([True, False]),
            "shared": random.choice([True, False, False, False]),  # Less likely
            "saved": random.choice([True, False, False]),  # Somewhat likely
            "feedback_type": random.choice(["positive", "negative", "neutral"]),
            "timestamp": fake.date_time_between(start_date='-30d', end_date='now').isoformat()
        }
    
    @staticmethod
    def generate_workflow_request() -> Dict[str, Any]:
        """Generate workflow execution request"""
        return {
            "user_id": fake.uuid4(),
            "preferences": TestDataGenerator.generate_user_preferences(),
            "context": {
                "device": random.choice(["mobile", "desktop", "tablet"]),
                "location": fake.city(),
                "time_of_day": random.choice(["morning", "afternoon", "evening", "night"]),
                "session_type": random.choice(["casual_browsing", "focused_learning", "entertainment"])
            }
        }

class PerformanceMetrics:
    """Collects and manages performance metrics during load testing"""
    
    def __init__(self):
        self.response_times = []
        self.error_rates = {}
        self.throughput_data = []
        self.resource_usage = []
        
    def add_response_time(self, endpoint: str, response_time: float):
        """Add response time measurement"""
        self.response_times.append({
            "endpoint": endpoint,
            "response_time": response_time,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_error(self, endpoint: str, error_type: str):
        """Record an error occurrence"""
        key = f"{endpoint}_{error_type}"
        self.error_rates[key] = self.error_rates.get(key, 0) + 1
    
    def calculate_percentiles(self) -> Dict[str, float]:
        """Calculate response time percentiles"""
        if not self.response_times:
            return {}
            
        times = [r["response_time"] for r in self.response_times]
        times.sort()
        
        return {
            "p50": self._percentile(times, 50),
            "p90": self._percentile(times, 90),
            "p95": self._percentile(times, 95),
            "p99": self._percentile(times, 99)
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        if not data:
            return 0.0
        index = int(len(data) * percentile / 100)
        return data[min(index, len(data) - 1)]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        return {
            "summary": {
                "total_requests": len(self.response_times),
                "total_errors": sum(self.error_rates.values()),
                "error_rate": sum(self.error_rates.values()) / max(len(self.response_times), 1) * 100
            },
            "response_times": self.calculate_percentiles(),
            "error_breakdown": self.error_rates,
            "timestamp": datetime.now().isoformat()
        }

# Global metrics instance
performance_metrics = PerformanceMetrics()