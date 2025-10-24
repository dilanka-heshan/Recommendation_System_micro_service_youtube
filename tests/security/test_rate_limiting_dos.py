"""
Rate Limiting and DoS Protection Security Tests

Tests for rate limiting, denial of service protection, and resource exhaustion.
"""

import pytest
import requests
import time
import threading
import json
import sys
import os
from typing import Dict, List, Any

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
tests_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(tests_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, tests_dir)

try:
    from tests.security.security_test_config import SecurityTestConfig
except ImportError:
    from security_test_config import SecurityTestConfig


class TestRateLimiting:
    """Test rate limiting and throttling mechanisms"""
    
    def test_request_rate_limiting(self):
        """Test if API implements rate limiting"""
        config = SecurityTestConfig()
        
        endpoint = f"{config.BASE_URL}/recommendations/?user_id=rate_test_user&top_k=5"
        
        # Make rapid requests to test rate limiting
        responses = []
        start_time = time.time()
        
        for i in range(20):  # 20 rapid requests
            try:
                response = requests.get(endpoint, timeout=5)
                responses.append({
                    "request_num": i + 1,
                    "status_code": response.status_code,
                    "response_time": time.time() - start_time,
                    "headers": dict(response.headers)
                })
                
                # Small delay to avoid overwhelming the server too much
                time.sleep(0.1)
                
            except requests.exceptions.RequestException as e:
                responses.append({
                    "request_num": i + 1,
                    "status_code": "error",
                    "response_time": time.time() - start_time,
                    "error": str(e)
                })
        
        # Analyze rate limiting
        status_codes = [r["status_code"] for r in responses]
        rate_limited_requests = [r for r in responses if r["status_code"] in [429, 503]]
        successful_requests = [r for r in responses if r["status_code"] == 200]
        
        print(f"\\n🔍 Rate Limiting Test Results:")
        print(f"   Total requests: {len(responses)}")
        print(f"   Successful (200): {len(successful_requests)}")
        print(f"   Rate limited (429/503): {len(rate_limited_requests)}")
        
        if rate_limited_requests:
            print(f"   ✅ Rate limiting detected after {rate_limited_requests[0]['request_num']} requests")
            
            # Check for rate limiting headers
            rate_limit_headers = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After", "X-Rate-Limit"]
            for req in rate_limited_requests[:1]:  # Check first rate limited request
                found_headers = [h for h in rate_limit_headers if h in req.get("headers", {})]
                if found_headers:
                    print(f"   Rate limiting headers found: {found_headers}")
        else:
            print(f"   🚨 NO RATE LIMITING DETECTED - All {len(successful_requests)} requests succeeded")
            print(f"   This is a security vulnerability allowing DoS attacks")
    
    def test_user_specific_rate_limiting(self):
        """Test if rate limiting is applied per user"""
        config = SecurityTestConfig()
        
        users = ["user_1", "user_2", "user_3"]
        user_results = {}
        
        for user in users:
            endpoint = f"{config.BASE_URL}/recommendations/?user_id={user}&top_k=5"
            responses = []
            
            # Make 10 requests per user
            for i in range(10):
                try:
                    response = requests.get(endpoint, timeout=5)
                    responses.append(response.status_code)
                    time.sleep(0.2)  # Small delay
                except:
                    responses.append("error")
            
            user_results[user] = {
                "successful": responses.count(200),
                "rate_limited": responses.count(429) + responses.count(503),
                "total": len(responses)
            }
        
        print(f"\\n🔍 Per-User Rate Limiting Test:")
        for user, results in user_results.items():
            print(f"   User {user}: {results['successful']}/{results['total']} successful, {results['rate_limited']} rate limited")
        
        # Check if all users are treated equally
        successful_counts = [r["successful"] for r in user_results.values()]
        if len(set(successful_counts)) == 1:
            print(f"   ✅ Consistent rate limiting across users")
        else:
            print(f"   ⚠️  Inconsistent rate limiting - may indicate per-user limits or no limits")


class TestDenialOfService:
    """Test denial of service protection"""
    
    def test_large_payload_handling(self):
        """Test how system handles large payloads"""
        config = SecurityTestConfig()
        
        # Test different payload sizes
        payload_sizes = [
            (1024, "1KB"),           # 1KB
            (10240, "10KB"),         # 10KB  
            (102400, "100KB"),       # 100KB
            (1048576, "1MB"),        # 1MB
            (10485760, "10MB")       # 10MB
        ]
        
        large_payload_results = []
        
        for size, size_name in payload_sizes:
            try:
                # Create large payload
                large_data = {
                    "user_id": "test_user",
                    "top_k": 5,
                    "large_field": "A" * size
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{config.BASE_URL}/run-workflow/run-workflow",
                    json=large_data,
                    timeout=30
                )
                end_time = time.time()
                
                large_payload_results.append({
                    "size": size_name,
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "handled": response.status_code not in [413, 500, 502, 503, 504]
                })
                
            except requests.exceptions.Timeout:
                large_payload_results.append({
                    "size": size_name,
                    "status_code": "timeout",
                    "response_time": 30.0,
                    "handled": False
                })
            except requests.exceptions.RequestException as e:
                large_payload_results.append({
                    "size": size_name,
                    "status_code": "error",
                    "response_time": 0,
                    "handled": False
                })
        
        print(f"\\n🔍 Large Payload Handling Test:")
        for result in large_payload_results:
            status = "✅" if result["handled"] else "❌"
            print(f"   {status} {result['size']}: Status {result['status_code']}, Time: {result['response_time']:.2f}s")
        
        # Check if system has payload limits
        failed_payloads = [r for r in large_payload_results if not r["handled"]]
        if failed_payloads:
            print(f"   ✅ System rejects large payloads starting at {failed_payloads[0]['size']}")
        else:
            print(f"   🚨 System accepts all payload sizes - potential DoS vulnerability")
    
    def test_concurrent_request_handling(self):
        """Test system behavior under concurrent load"""
        config = SecurityTestConfig()
        
        import threading
        
        results = []
        num_threads = 10
        requests_per_thread = 5
        
        def make_concurrent_requests(thread_id):
            thread_results = []
            for i in range(requests_per_thread):
                try:
                    start_time = time.time()
                    response = requests.get(
                        f"{config.BASE_URL}/recommendations/?user_id=concurrent_{thread_id}&top_k=5",
                        timeout=15
                    )
                    end_time = time.time()
                    
                    thread_results.append({
                        "thread_id": thread_id,
                        "request_id": i,
                        "status_code": response.status_code,
                        "response_time": end_time - start_time
                    })
                    
                except requests.exceptions.RequestException as e:
                    thread_results.append({
                        "thread_id": thread_id,
                        "request_id": i,
                        "status_code": "error",
                        "response_time": 0,
                        "error": str(e)
                    })
            
            results.extend(thread_results)
        
        # Launch concurrent threads
        threads = []
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=make_concurrent_requests, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["status_code"] == 200]
        failed_requests = [r for r in results if r["status_code"] != 200]
        
        if successful_requests:
            avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
            max_response_time = max(r["response_time"] for r in successful_requests)
        else:
            avg_response_time = 0
            max_response_time = 0
        
        print(f"\\n🔍 Concurrent Load Test Results:")
        print(f"   Total requests: {len(results)}")
        print(f"   Successful: {len(successful_requests)}")
        print(f"   Failed: {len(failed_requests)}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average response time: {avg_response_time:.2f}s")
        print(f"   Max response time: {max_response_time:.2f}s")
        
        # Evaluate performance
        if len(failed_requests) == 0:
            print(f"   ✅ System handled all concurrent requests successfully")
        elif len(failed_requests) < len(results) * 0.1:  # Less than 10% failure
            print(f"   ⚠️  System handled most requests but had some failures")
        else:
            print(f"   🚨 High failure rate under concurrent load - potential DoS vulnerability")
    
    def test_resource_exhaustion_attacks(self):
        """Test various resource exhaustion attack vectors"""
        config = SecurityTestConfig()
        
        # Test different resource exhaustion techniques
        exhaustion_tests = [
            {
                "name": "Very high top_k parameter",
                "endpoint": "/recommendations/",
                "params": {"user_id": "test_user", "top_k": 999999},
                "method": "GET"
            },
            {
                "name": "Large user_ids list",
                "endpoint": "/user-vector-update/run-manual-update",
                "payload": {"user_ids": ["user_" + str(i) for i in range(1000)]},
                "method": "POST"
            },
            {
                "name": "Deep JSON nesting",
                "endpoint": "/run-workflow/run-workflow",
                "payload": {"user_id": "test", "nested": {"level1": {"level2": {"level3": {"level4": {"level5": "deep"}}}}}},
                "method": "POST"
            }
        ]
        
        exhaustion_results = []
        
        for test in exhaustion_tests:
            try:
                start_time = time.time()
                
                if test["method"] == "GET":
                    response = requests.get(
                        f"{config.BASE_URL}{test['endpoint']}",
                        params=test["params"],
                        timeout=20
                    )
                else:
                    response = requests.post(
                        f"{config.BASE_URL}{test['endpoint']}",
                        json=test["payload"],
                        timeout=20
                    )
                
                end_time = time.time()
                
                exhaustion_results.append({
                    "test": test["name"],
                    "status_code": response.status_code,
                    "response_time": end_time - start_time,
                    "handled_gracefully": response.status_code not in [500, 502, 503, 504]
                })
                
            except requests.exceptions.Timeout:
                exhaustion_results.append({
                    "test": test["name"],
                    "status_code": "timeout",
                    "response_time": 20.0,
                    "handled_gracefully": False
                })
            except requests.exceptions.RequestException:
                exhaustion_results.append({
                    "test": test["name"],
                    "status_code": "error",
                    "response_time": 0,
                    "handled_gracefully": False
                })
        
        print(f"\\n🔍 Resource Exhaustion Test Results:")
        for result in exhaustion_results:
            status = "✅" if result["handled_gracefully"] else "❌"
            print(f"   {status} {result['test']}")
            print(f"      Status: {result['status_code']}, Time: {result['response_time']:.2f}s")
        
        graceful_count = sum(1 for r in exhaustion_results if r["handled_gracefully"])
        print(f"\\n   Summary: {graceful_count}/{len(exhaustion_results)} tests handled gracefully")


class TestSlowLorisAttack:
    """Test slow HTTP attacks"""
    
    def test_slow_request_handling(self):
        """Test how system handles deliberately slow requests"""
        config = SecurityTestConfig()
        
        # This test simulates slow requests but is kept simple for testing
        print(f"\\n🔍 Slow Request Handling Test:")
        
        try:
            # Test with a reasonable timeout to see baseline
            start_time = time.time()
            response = requests.get(
                f"{config.BASE_URL}/recommendations/?user_id=slow_test&top_k=5",
                timeout=10
            )
            baseline_time = time.time() - start_time
            
            print(f"   Baseline request time: {baseline_time:.2f}s")
            print(f"   System appears to handle normal requests properly")
            
            # Note: Real slow loris attacks would require custom socket programming
            # and are potentially harmful, so we don't implement them here
            print(f"   ℹ️  Full slow loris testing requires specialized tools")
            
        except requests.exceptions.Timeout:
            print(f"   ⚠️  System has request timeout protection")
        except requests.exceptions.RequestException as e:
            print(f"   ⚠️  Request failed: {str(e)}")


class TestApplicationLayerDDoS:
    """Test application-layer DDoS protection"""
    
    def test_expensive_operation_abuse(self):
        """Test abuse of computationally expensive operations"""
        config = SecurityTestConfig()
        
        # Test operations that might be computationally expensive
        expensive_operations = [
            {
                "name": "Large recommendation request",
                "endpoint": "/recommendations/",
                "params": {"user_id": "test_user", "top_k": 100}
            },
            {
                "name": "Complex workflow",
                "endpoint": "/run-workflow/run-workflow",
                "payload": {"user_id": "test_user", "top_k": 50}
            }
        ]
        
        expensive_test_results = []
        
        for operation in expensive_operations:
            times = []
            for i in range(3):  # Test each operation 3 times
                try:
                    start_time = time.time()
                    
                    if "params" in operation:
                        response = requests.get(
                            f"{config.BASE_URL}{operation['endpoint']}",
                            params=operation["params"],
                            timeout=15
                        )
                    else:
                        response = requests.post(
                            f"{config.BASE_URL}{operation['endpoint']}",
                            json=operation["payload"],
                            timeout=15
                        )
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
                    
                except requests.exceptions.RequestException:
                    times.append(None)
            
            valid_times = [t for t in times if t is not None]
            if valid_times:
                avg_time = sum(valid_times) / len(valid_times)
                expensive_test_results.append({
                    "operation": operation["name"],
                    "avg_time": avg_time,
                    "successful_requests": len(valid_times)
                })
        
        print(f"\\n🔍 Expensive Operation Abuse Test:")
        for result in expensive_test_results:
            print(f"   {result['operation']}: {result['avg_time']:.2f}s avg ({result['successful_requests']}/3 successful)")
        
        # Check for potential abuse
        slow_operations = [r for r in expensive_test_results if r["avg_time"] > 5.0]
        if slow_operations:
            print(f"   ⚠️  Found {len(slow_operations)} potentially expensive operations")
            print(f"   Consider implementing timeouts or caching for these operations")
        else:
            print(f"   ✅ All operations complete within reasonable time")