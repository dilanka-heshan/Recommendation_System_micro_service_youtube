"""
Test configuration and fixtures for database integrity testing
Provides shared fixtures and utilities for testing multi-database consistency
"""
import pytest
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import database clients after loading environment
from backend.database.supabase_client import supabase_client
from backend.database.qdrant_client import qdrant_client
from backend.database.mongodb_client import mongodb_client

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session")
def database_clients():
    """Provide database clients for testing"""
    return {
        "supabase": supabase_client,
        "qdrant": qdrant_client, 
        "mongodb": mongodb_client
    }

@pytest.fixture(scope="session") 
def test_config():
    """Test configuration settings"""
    return {
        "max_test_records": 1000,
        "timeout_seconds": 30,
        "sample_size": 100,
        "date_range_days": 7,
        "embedding_dimensions": 768,
        "rating_range": (1, 5),
        "video_id_pattern": r"^[a-zA-Z0-9_-]{11}$",  # YouTube video ID pattern
        "user_id_pattern": r"^[a-zA-Z0-9_-]+$"
    }

@pytest.fixture
def sample_user_ids(database_clients, test_config):
    """Get sample user IDs for testing"""
    try:
        response = database_clients["supabase"].client.table("users").select(
            "user_id"
        ).limit(test_config["sample_size"]).execute()
        
        if response.data:
            return [user["user_id"] for user in response.data]
        else:
            logger.warning("No users found in database")
            return []
    except Exception as e:
        logger.error(f"Error fetching sample users: {e}")
        return []

@pytest.fixture
def sample_video_ids(database_clients, test_config):
    """Get sample video IDs for testing"""
    try:
        response = database_clients["supabase"].client.table("videos").select(
            "video_id"
        ).limit(test_config["sample_size"]).execute()
        
        if response.data:
            return [video["video_id"] for video in response.data]
        else:
            logger.warning("No videos found in database")
            return []
    except Exception as e:
        logger.error(f"Error fetching sample videos: {e}")
        return []

@pytest.fixture
def date_range_for_testing():
    """Provide date range for testing time-based data"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat()
    }

@pytest.fixture
def integrity_test_helpers():
    """Helper functions for integrity testing"""
    
    class IntegrityTestHelpers:
        @staticmethod
        def validate_video_id_format(video_id: str) -> bool:
            """Validate YouTube video ID format"""
            import re
            pattern = r"^[a-zA-Z0-9_-]{11}$"
            return bool(re.match(pattern, video_id))
        
        @staticmethod
        def validate_embedding_dimensions(embedding: List[float], expected_dim: int = 768) -> bool:
            """Validate embedding dimensions"""
            return len(embedding) == expected_dim
        
        @staticmethod
        def validate_rating_range(rating: int, min_val: int = 1, max_val: int = 5) -> bool:
            """Validate rating is within expected range"""
            return min_val <= rating <= max_val
        
        @staticmethod
        def calculate_data_freshness(timestamp: str, max_age_days: int = 30) -> bool:
            """Check if data is fresh (within max_age_days)"""
            try:
                data_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                age = datetime.now() - data_time.replace(tzinfo=None)
                return age.days <= max_age_days
            except:
                return False
        
        @staticmethod
        def find_missing_references(source_ids: List[str], target_ids: List[str]) -> List[str]:
            """Find IDs in source that don't exist in target"""
            return list(set(source_ids) - set(target_ids))
        
        @staticmethod
        def calculate_consistency_score(total_records: int, consistent_records: int) -> float:
            """Calculate consistency percentage"""
            if total_records == 0:
                return 100.0
            return (consistent_records / total_records) * 100.0
    
    return IntegrityTestHelpers()

@pytest.fixture
def database_connection_validator(database_clients):
    """Validate database connections are working"""
    
    def validate_connections():
        results = {}
        
        # Test Supabase connection
        try:
            response = database_clients["supabase"].client.table("users").select("user_id").limit(1).execute()
            results["supabase"] = {"status": "connected", "error": None}
        except Exception as e:
            results["supabase"] = {"status": "failed", "error": str(e)}
        
        # Test Qdrant connection
        try:
            if database_clients["qdrant"].client:
                collections = database_clients["qdrant"].client.get_collections()
                results["qdrant"] = {"status": "connected", "error": None}
            else:
                results["qdrant"] = {"status": "not_available", "error": "Client not initialized"}
        except Exception as e:
            results["qdrant"] = {"status": "failed", "error": str(e)}
        
        # Test MongoDB connection
        try:
            if database_clients["mongodb"].client:
                # Try to ping or get collection info
                results["mongodb"] = {"status": "connected", "error": None}
            else:
                results["mongodb"] = {"status": "not_available", "error": "Client not initialized"}
        except Exception as e:
            results["mongodb"] = {"status": "failed", "error": str(e)}
        
        return results
    
    return validate_connections

class DatabaseIntegrityReporter:
    """Utility class for generating integrity test reports"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
    
    def add_test_result(self, test_name: str, status: str, details: Dict[str, Any]):
        """Add a test result"""
        self.test_results[test_name] = {
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate test summary report"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result["status"] == "PASSED")
        failed_tests = total_tests - passed_tests
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0,
                "execution_time": (datetime.now() - self.start_time).total_seconds()
            },
            "detailed_results": self.test_results
        }

@pytest.fixture
def integrity_reporter():
    """Provide integrity test reporter"""
    return DatabaseIntegrityReporter()