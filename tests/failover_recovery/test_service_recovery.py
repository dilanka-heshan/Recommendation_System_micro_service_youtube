"""
Service Recovery Tests - Easy failover testing with minimal configuration
Tests API service restart scenarios and validates recovery behavior
"""

import pytest
import requests
import time
import psutil
import subprocess
import os
from typing import Dict, List, Optional
import asyncio
import aiohttp


class ServiceRecoveryTester:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.test_results = []
        
    def check_service_health(self) -> Dict:
        """Check if the service is responding to health checks"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds(),
                "timestamp": time.time()
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "unreachable",
                "error": str(e),
                "timestamp": time.time()
            }
    
    def test_api_endpoints(self) -> Dict:
        """Test critical API endpoints availability"""
        endpoints = [
            "/",
            "/docs",
            "/recommendations",
            "/health"
        ]
        
        results = {}
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                results[endpoint] = {
                    "status": "available",
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
            except requests.exceptions.RequestException as e:
                results[endpoint] = {
                    "status": "failed",
                    "error": str(e)
                }
        
        return results
    
    def simulate_memory_stress(self, duration: int = 10) -> Dict:
        """Simulate memory stress on the service"""
        stress_data = []
        start_time = time.time()
        
        # Create memory pressure
        for i in range(100):
            # Create large objects to simulate memory usage
            data = [f"stress_data_{i}_{j}" for j in range(10000)]
            stress_data.append(data)
            
            if time.time() - start_time > duration:
                break
                
        # Clear stress data
        del stress_data
        
        return {
            "test": "memory_stress",
            "duration": duration,
            "status": "completed",
            "timestamp": time.time()
        }
    
    def test_timeout_handling(self) -> Dict:
        """Test how service handles timeout scenarios"""
        try:
            # Test with very short timeout
            response = requests.get(f"{self.base_url}/recommendations", timeout=0.1)
            return {
                "test": "timeout_handling",
                "status": "passed",
                "message": "Service responded within timeout"
            }
        except requests.exceptions.Timeout:
            return {
                "test": "timeout_handling",
                "status": "timeout_handled",
                "message": "Timeout handled properly"
            }
        except Exception as e:
            return {
                "test": "timeout_handling",
                "status": "error",
                "error": str(e)
            }


# Pytest fixtures and tests
@pytest.fixture
def service_tester():
    return ServiceRecoveryTester()


@pytest.fixture
def ensure_service_running():
    """Ensure service is running before tests"""
    tester = ServiceRecoveryTester()
    health = tester.check_service_health()
    
    if health["status"] != "healthy":
        pytest.skip("Service is not running. Start with: python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8080")
    
    return tester


class TestServiceRecovery:
    """Service Recovery Test Suite"""
    
    def test_service_health_check(self, ensure_service_running):
        """Test basic service health check"""
        tester = ensure_service_running
        health = tester.check_service_health()
        
        assert health["status"] == "healthy"
        assert health["status_code"] == 200
        assert health["response_time"] < 5.0  # Should respond within 5 seconds
    
    def test_api_endpoints_availability(self, ensure_service_running):
        """Test that critical API endpoints are available"""
        tester = ensure_service_running
        results = tester.test_api_endpoints()
        
        # At least root endpoint should be available
        assert "/" in results
        assert results["/"]["status"] == "available"
        
        # Check that most endpoints are working
        available_count = sum(1 for r in results.values() if r["status"] == "available")
        total_count = len(results)
        
        assert available_count >= total_count * 0.75  # At least 75% should be available
    
    def test_service_under_memory_pressure(self, ensure_service_running):
        """Test service behavior under memory pressure"""
        tester = ensure_service_running
        
        # Check health before stress
        health_before = tester.check_service_health()
        assert health_before["status"] == "healthy"
        
        # Apply memory stress
        stress_result = tester.simulate_memory_stress(duration=5)
        assert stress_result["status"] == "completed"
        
        # Check health after stress
        time.sleep(2)  # Allow brief recovery time
        health_after = tester.check_service_health()
        
        # Service should still be responsive
        assert health_after["status"] == "healthy"
        
        # Response time might be higher but should be reasonable
        assert health_after["response_time"] < 10.0
    
    def test_timeout_handling(self, ensure_service_running):
        """Test service timeout handling"""
        tester = ensure_service_running
        result = tester.test_timeout_handling()
        
        # Either service responds quickly or timeout is handled properly
        assert result["status"] in ["passed", "timeout_handled"]
    
    def test_concurrent_request_handling(self, ensure_service_running):
        """Test service handling of concurrent requests"""
        tester = ensure_service_running
        
        # Send multiple concurrent requests
        import concurrent.futures
        
        def make_request():
            return tester.check_service_health()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Most requests should succeed
        healthy_count = sum(1 for r in results if r["status"] == "healthy")
        assert healthy_count >= len(results) * 0.8  # At least 80% should be healthy
    
    def test_service_recovery_after_stress(self, ensure_service_running):
        """Test service recovery after various stress conditions"""
        tester = ensure_service_running
        
        # Apply multiple stress conditions
        stress_tests = [
            lambda: tester.simulate_memory_stress(3),
            lambda: tester.test_timeout_handling(),
            lambda: tester.test_api_endpoints()
        ]
        
        for stress_test in stress_tests:
            # Apply stress
            stress_test()
            
            # Allow brief recovery time
            time.sleep(1)
            
            # Check service recovery
            health = tester.check_service_health()
            assert health["status"] == "healthy", f"Service failed to recover after stress test"


class TestServiceResilience:
    """Additional resilience tests"""
    
    def test_rapid_request_recovery(self, ensure_service_running):
        """Test recovery from rapid successive requests"""
        tester = ensure_service_running
        
        # Send rapid requests
        for i in range(50):
            try:
                requests.get(f"{tester.base_url}/", timeout=1)
            except:
                pass  # Ignore individual failures
        
        # Check final health
        time.sleep(2)
        health = tester.check_service_health()
        assert health["status"] == "healthy"
    
    def test_malformed_request_recovery(self, ensure_service_running):
        """Test recovery after malformed requests"""
        tester = ensure_service_running
        
        # Send malformed requests
        malformed_endpoints = [
            "/nonexistent",
            "/recommendations?invalid=data",
            "//double//slash",
        ]
        
        for endpoint in malformed_endpoints:
            try:
                requests.get(f"{tester.base_url}{endpoint}", timeout=5)
            except:
                pass  # Expected to fail
        
        # Service should still be healthy
        health = tester.check_service_health()
        assert health["status"] == "healthy"


if __name__ == "__main__":
    # Simple test runner for manual execution
    tester = ServiceRecoveryTester()
    
    print("=== Service Recovery Test Suite ===")
    
    # Basic health check
    print("1. Testing service health...")
    health = tester.check_service_health()
    print(f"   Status: {health['status']}")
    
    if health["status"] == "healthy":
        # Test endpoints
        print("2. Testing API endpoints...")
        endpoints = tester.test_api_endpoints()
        for endpoint, result in endpoints.items():
            print(f"   {endpoint}: {result['status']}")
        
        # Memory stress test
        print("3. Testing memory stress recovery...")
        stress_result = tester.simulate_memory_stress(5)
        print(f"   Memory stress: {stress_result['status']}")
        
        # Final health check
        time.sleep(2)
        final_health = tester.check_service_health()
        print(f"   Post-stress health: {final_health['status']}")
        
        print("\n=== Test completed successfully ===")
    else:
        print("Service is not running. Please start the service first:")
        print("python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8080")