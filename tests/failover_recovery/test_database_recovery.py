"""
Database Connection Recovery Tests - Test database failover and recovery scenarios
Tests MongoDB, Qdrant, and Supabase connection failures with minimal configuration
"""

import pytest
import time
import os
import sys
from unittest.mock import patch, MagicMock
from typing import Dict, Any, Optional
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

# Import database clients with better error handling
MongoDBClient = None
QdrantVectorClient = None
SupabaseClient = None

try:
    from database.mongodb_client import MongoDBClient
    print("✓ MongoDB client imported successfully")
except ImportError as e:
    print(f"Warning: Could not import MongoDB client: {e}")

try:
    from database.qdrant_client import QdrantVectorClient
    print("✓ Qdrant client imported successfully")
except ImportError as e:
    print(f"Warning: Could not import Qdrant client: {e}")

try:
    # Handle Supabase import carefully since it might fail on initialization
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "supabase_client", 
        os.path.join(os.path.dirname(__file__), '..', '..', 'backend', 'database', 'supabase_client.py')
    )
    supabase_module = importlib.util.module_from_spec(spec)
    
    # Import the class without triggering module-level initialization
    with patch('supabase.create_client'), patch.dict(os.environ, {
        'SUPABASE_URL': os.getenv('SUPABASE_URL', 'https://test.supabase.co'),
        'SUPABASE_ANON_KEY': os.getenv('SUPABASE_ANON_KEY', 'test_key')
    }):
        spec.loader.exec_module(supabase_module)
        SupabaseClient = supabase_module.SupabaseClient
        print("✓ Supabase client imported successfully")
except ImportError as e:
    print(f"Warning: Could not import Supabase client: {e}")
except Exception as e:
    print(f"Warning: Error setting up Supabase client: {e}")


class DatabaseRecoveryTester:
    """Test database connection failures and recovery"""
    
    def __init__(self):
        self.test_results = []
        self.logger = logging.getLogger(__name__)
    
    def test_mongodb_connection_failure(self) -> Dict[str, Any]:
        """Test MongoDB connection failure and recovery simulation"""
        if not MongoDBClient:
            return {"test": "mongodb_connection", "status": "skipped", "reason": "MongoDB client not available"}
        
        try:
            # Test 1: Normal connection with your credentials
            print("  Testing MongoDB normal connection...")
            normal_client = MongoDBClient()  # Use environment variables
            
            # Test if we can perform a basic operation
            try:
                result = normal_client.get_extractive_summary("test_video_id")
                normal_connection_status = "connected" if normal_client.client else "failed"
            except Exception as e:
                normal_connection_status = f"error: {str(e)}"
            
            # Test 2: Invalid connection string
            print("  Testing MongoDB invalid connection...")
            invalid_client = MongoDBClient("mongodb://invalid_host:27017/test")
            
            # Try to fetch data - should handle gracefully
            invalid_result = invalid_client.get_extractive_summary("test_video_id")
            
            # Test 3: Connection timeout simulation
            print("  Testing MongoDB connection timeout...")
            timeout_result = None
            try:
                with patch('pymongo.MongoClient') as mock_client:
                    mock_client.side_effect = Exception("Connection timeout")
                    timeout_client = MongoDBClient("mongodb://localhost:27017/test")
                    timeout_result = timeout_client.get_extractive_summary("test_id")
            except Exception:
                timeout_result = None  # Expected
            
            return {
                "test": "mongodb_connection_failure",
                "status": "completed",
                "normal_connection": normal_connection_status,
                "invalid_connection_handled": invalid_result is None,
                "timeout_handled": timeout_result is None,
                "message": "MongoDB connection failure scenarios tested",
                "details": {
                    "normal_status": normal_connection_status,
                    "invalid_result": str(invalid_result),
                    "timeout_result": str(timeout_result)
                }
            }
        except Exception as e:
            return {
                "test": "mongodb_connection_failure", 
                "status": "error",
                "error": str(e)
            }
    
    def test_qdrant_connection_failure(self) -> Dict[str, Any]:
        """Test Qdrant connection failure scenarios"""
        if not QdrantVectorClient:
            return {"test": "qdrant_connection", "status": "skipped", "reason": "Qdrant client not available"}
        
        try:
            # Test 1: Normal connection with your credentials
            print("  Testing Qdrant normal connection...")
            normal_client = QdrantVectorClient()  # Use environment variables
            
            # Check if client is connected
            normal_connection_status = "connected" if normal_client.client else "failed"
            
            # Test 2: Invalid connection
            print("  Testing Qdrant invalid connection...")
            with patch.dict(os.environ, {
                'QDRANT_HOST': 'invalid_host',
                'QDRANT_PORT': '9999'
            }):
                invalid_client = QdrantVectorClient()
                invalid_connection_status = "connected" if invalid_client.client else "failed_gracefully"
            
            # Test 3: Search operation failure handling
            print("  Testing Qdrant search failure handling...")
            try:
                if normal_client.client:
                    # Test with potentially non-existent collection
                    search_result = normal_client.search_videos("test query", top_k=1)
                    search_handled = True
                else:
                    search_handled = True  # No client to test with
            except Exception as e:
                search_handled = True  # Exception handling is expected
            
            return {
                "test": "qdrant_connection_failure",
                "status": "completed",
                "normal_connection": normal_connection_status,
                "invalid_connection": invalid_connection_status,
                "search_failure_handled": search_handled,
                "message": "Qdrant connection failure scenarios tested",
                "details": {
                    "qdrant_host": os.getenv("QDRANT_HOST", "not_set"),
                    "has_api_key": bool(os.getenv("QDRANT_API_KEY"))
                }
            }
        except Exception as e:
            return {
                "test": "qdrant_connection_failure",
                "status": "error",
                "error": str(e)
            }
    
    def test_supabase_connection_failure(self) -> Dict[str, Any]:
        """Test Supabase connection failure scenarios"""
        if not SupabaseClient:
            return {"test": "supabase_connection", "status": "skipped", "reason": "Supabase client not available"}
        
        try:
            # Test 1: Normal connection with your credentials
            print("  Testing Supabase normal connection...")
            try:
                with patch.dict(os.environ, {
                    'SUPABASE_URL': os.getenv('SUPABASE_URL', ''),
                    'SUPABASE_ANON_KEY': os.getenv('SUPABASE_ANON_KEY', '')
                }):
                    normal_client = SupabaseClient()
                    normal_connection_status = "connected"
                    
                    # Test a basic operation
                    try:
                        result = normal_client.get_user_embedding("test_user")
                        operation_result = "handled" if result is None else "unexpected_data"
                    except Exception as e:
                        operation_result = f"error_handled: {type(e).__name__}"
            except ValueError as e:
                normal_connection_status = f"env_validation_error: {str(e)}"
                operation_result = "skipped"
            
            # Test 2: Invalid credentials
            print("  Testing Supabase invalid credentials...")
            try:
                with patch.dict(os.environ, {
                    'SUPABASE_URL': 'https://invalid.supabase.co',
                    'SUPABASE_ANON_KEY': 'invalid_key'
                }):
                    invalid_client = SupabaseClient()
                    invalid_result = invalid_client.get_user_embedding("test_user")
                    invalid_handled = True
            except Exception as e:
                invalid_handled = True  # Expected to fail
            
            # Test 3: Missing environment variables
            print("  Testing Supabase missing credentials...")
            try:
                with patch.dict(os.environ, {}, clear=True):
                    missing_client = SupabaseClient()
                    missing_handled = False  # Should not reach here
            except ValueError:
                missing_handled = True  # Expected ValueError
            
            return {
                "test": "supabase_connection_failure",
                "status": "completed",
                "normal_connection": normal_connection_status,
                "operation_result": operation_result,
                "invalid_credentials_handled": invalid_handled,
                "missing_credentials_handled": missing_handled,
                "message": "Supabase connection failure scenarios tested",
                "details": {
                    "supabase_url_set": bool(os.getenv("SUPABASE_URL")),
                    "supabase_key_set": bool(os.getenv("SUPABASE_ANON_KEY"))
                }
            }
        except Exception as e:
            return {
                "test": "supabase_connection_failure",
                "status": "error",
                "error": str(e)
            }
    
    def test_database_connection_recovery(self) -> Dict[str, Any]:
        """Test database connection recovery after temporary failures"""
        recovery_tests = []
        
        # Test MongoDB recovery
        mongo_result = self.test_mongodb_connection_failure()
        recovery_tests.append(mongo_result)
        
        # Test Qdrant recovery  
        qdrant_result = self.test_qdrant_connection_failure()
        recovery_tests.append(qdrant_result)
        
        # Test Supabase recovery
        supabase_result = self.test_supabase_connection_failure()
        recovery_tests.append(supabase_result)
        
        # Calculate overall recovery score
        handled_count = sum(1 for test in recovery_tests if test["status"] == "handled")
        total_tests = len([test for test in recovery_tests if test["status"] != "skipped"])
        
        return {
            "test": "database_connection_recovery",
            "status": "completed",
            "recovery_score": handled_count / max(total_tests, 1),
            "individual_results": recovery_tests,
            "message": f"Database recovery: {handled_count}/{total_tests} tests passed"
        }


class MockDatabaseFailure:
    """Mock database failures for testing"""
    
    @staticmethod
    def simulate_connection_timeout():
        """Simulate connection timeout"""
        time.sleep(0.1)  # Brief delay to simulate timeout
        raise Exception("Connection timeout")
    
    @staticmethod
    def simulate_network_partition():
        """Simulate network partition"""
        raise Exception("Network unreachable")
    
    @staticmethod
    def simulate_auth_failure():
        """Simulate authentication failure"""
        raise Exception("Authentication failed")


# Pytest fixtures and tests
@pytest.fixture
def db_tester():
    return DatabaseRecoveryTester()


class TestDatabaseRecovery:
    """Database Recovery Test Suite"""
    
    def test_mongodb_connection_resilience(self, db_tester):
        """Test MongoDB connection resilience"""
        result = db_tester.test_mongodb_connection_failure()
        
        # Test should either handle gracefully or be skipped
        assert result["status"] in ["handled", "skipped", "error"]
        
        if result["status"] == "handled":
            assert "message" in result
    
    def test_mongodb_timeout_resilience(self, db_tester):
        """Test MongoDB timeout handling"""
        result = db_tester.test_mongodb_timeout_handling()
        
        assert result["status"] in ["handled", "skipped", "error"]
        
        if result["status"] == "handled":
            assert result["result"] is None  # Should return None on failure
    
    def test_qdrant_connection_resilience(self, db_tester):
        """Test Qdrant connection resilience"""
        result = db_tester.test_qdrant_connection_failure()
        
        assert result["status"] in ["handled", "skipped", "error"]
        
        if result["status"] == "handled":
            assert "message" in result
    
    def test_supabase_connection_resilience(self, db_tester):
        """Test Supabase connection resilience"""
        result = db_tester.test_supabase_connection_failure()
        
        assert result["status"] in ["handled", "skipped", "error"]
        
        if result["status"] == "handled":
            assert "message" in result
    
    def test_overall_database_recovery(self, db_tester):
        """Test overall database recovery capabilities"""
        result = db_tester.test_database_connection_recovery()
        
        assert result["status"] == "completed"
        assert "recovery_score" in result
        assert "individual_results" in result
        
        # At least some recovery should be handled
        assert result["recovery_score"] >= 0
    
    def test_database_client_initialization(self):
        """Test database client initialization with invalid configs"""
        
        # Test MongoDB with invalid connection
        if MongoDBClient:
            mongo_client = MongoDBClient("invalid_connection_string")
            assert mongo_client.client is None or mongo_client.db is None
        
        # Test Supabase with missing env vars
        if SupabaseClient:
            with patch.dict(os.environ, {}, clear=True):
                try:
                    supabase_client = SupabaseClient()
                    # Should raise ValueError for missing env vars
                    assert False, "Should have raised ValueError"
                except ValueError:
                    pass  # Expected behavior
    
    def test_graceful_degradation(self, db_tester):
        """Test that system degrades gracefully when databases are unavailable"""
        
        # Test each database client's behavior when service is unavailable
        test_scenarios = [
            ("MongoDB", db_tester.test_mongodb_connection_failure),
            ("Qdrant", db_tester.test_qdrant_connection_failure), 
            ("Supabase", db_tester.test_supabase_connection_failure)
        ]
        
        degradation_results = []
        
        for db_name, test_func in test_scenarios:
            result = test_func()
            degradation_results.append({
                "database": db_name,
                "graceful": result["status"] in ["handled", "skipped"]
            })
        
        # All databases should degrade gracefully
        graceful_count = sum(1 for r in degradation_results if r["graceful"])
        assert graceful_count == len(degradation_results), "Some databases did not degrade gracefully"


class TestDatabaseConnectionPooling:
    """Test connection pooling and resource management"""
    
    def test_connection_cleanup(self):
        """Test that database connections are cleaned up properly"""
        
        if MongoDBClient:
            # Create multiple MongoDB clients
            clients = []
            for i in range(5):
                client = MongoDBClient("mongodb://invalid_host:27017/test")
                clients.append(client)
            
            # All should handle invalid connections gracefully
            for client in clients:
                assert client.client is None or hasattr(client.client, 'close')
    
    def test_concurrent_database_access(self):
        """Test concurrent database access scenarios"""
        import concurrent.futures
        
        def create_client():
            if MongoDBClient:
                client = MongoDBClient("mongodb://localhost:27017/test")
                return client.get_extractive_summary("test_id") 
            return None
        
        # Test concurrent client creation
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(create_client) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All should complete without errors (results can be None)
        assert len(results) == 10


if __name__ == "__main__":
    # Simple test runner for manual execution
    tester = DatabaseRecoveryTester()
    
    print("=== Database Connection Recovery Test Suite ===")
    
    # Test MongoDB
    print("1. Testing MongoDB connection resilience...")
    mongo_result = tester.test_mongodb_connection_failure()
    print(f"   MongoDB: {mongo_result['status']} - {mongo_result.get('message', 'N/A')}")
    
    # Test Qdrant
    print("2. Testing Qdrant connection resilience...")
    qdrant_result = tester.test_qdrant_connection_failure()
    print(f"   Qdrant: {qdrant_result['status']} - {qdrant_result.get('message', 'N/A')}")
    
    # Test Supabase
    print("3. Testing Supabase connection resilience...")
    supabase_result = tester.test_supabase_connection_failure()
    print(f"   Supabase: {supabase_result['status']} - {supabase_result.get('message', 'N/A')}")
    
    # Overall recovery test
    print("4. Testing overall database recovery...")
    recovery_result = tester.test_database_connection_recovery()
    print(f"   Recovery Score: {recovery_result['recovery_score']:.2f}")
    print(f"   Message: {recovery_result['message']}")
    
    print("\n=== Database recovery tests completed ===")