"""
Background Service Specific Security Tests

Security tests relevant to background/microservice architecture - focus on 
service availability, resource management, and operational security.
"""

import pytest
import requests
import json
import time
import threading
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


class TestServiceAvailability:
    """Test service availability and resilience"""
    
    def test_service_health_monitoring(self):
        """Test if service health endpoints work properly"""
        config = SecurityTestConfig()
        
        health_endpoints = [
            "/health",
            "/newsletter/health"
        ]
        
        health_results = []
        
        for endpoint in health_endpoints:
            try:
                start_time = time.time()
                response = requests.get(f"{config.BASE_URL}{endpoint}", timeout=10)
                response_time = time.time() - start_time
                
                health_results.append({
                    "endpoint": endpoint,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "is_healthy": response.status_code == 200 and response_time < 5.0
                })
                
            except requests.exceptions.RequestException as e:
                health_results.append({
                    "endpoint": endpoint,
                    "status_code": "error",
                    "response_time": 0,
                    "is_healthy": False,
                    "error": str(e)
                })
        
        print(f"\\n💚 Service Health Check Results:")
        healthy_count = sum(1 for r in health_results if r["is_healthy"])
        print(f"   {healthy_count}/{len(health_results)} health endpoints responding properly:")
        
        for result in health_results:
            status = "✅" if result["is_healthy"] else "❌"
            print(f"   {status} {result['endpoint']}: {result['status_code']} ({result['response_time']:.2f}s)")
    
    def test_service_dependency_handling(self):
        """Test how service handles external dependency failures"""
        config = SecurityTestConfig()
        
        # Test endpoints that likely depend on external services (databases, etc.)
        dependency_endpoints = [
            ("/recommendations/?user_id=test&top_k=5", "Database dependency"),
            ("/test/test-database-connection", "Database connection"),
            ("/user-vector-update/status", "Vector service dependency")
        ]
        
        dependency_results = []
        
        for endpoint, dependency_type in dependency_endpoints:
            try:
                response = requests.get(f"{config.BASE_URL}{endpoint}", timeout=15)
                
                # Check if service degrades gracefully
                graceful_degradation = response.status_code in [200, 503, 502] and len(response.text) > 0
                
                dependency_results.append({
                    "endpoint": endpoint,
                    "dependency": dependency_type,
                    "status_code": response.status_code,
                    "graceful_degradation": graceful_degradation,
                    "response_length": len(response.text)
                })
                
            except requests.exceptions.Timeout:
                dependency_results.append({
                    "endpoint": endpoint,
                    "dependency": dependency_type,
                    "status_code": "timeout",
                    "graceful_degradation": False,
                    "response_length": 0
                })
            except requests.exceptions.RequestException:
                dependency_results.append({
                    "endpoint": endpoint,
                    "dependency": dependency_type,
                    "status_code": "error",
                    "graceful_degradation": False,
                    "response_length": 0
                })
        
        print(f"\\n🔗 Service Dependency Handling Test:")
        graceful_count = sum(1 for r in dependency_results if r["graceful_degradation"])
        print(f"   {graceful_count}/{len(dependency_results)} endpoints handle dependencies gracefully:")
        
        for result in dependency_results:
            status = "✅" if result["graceful_degradation"] else "⚠️"
            print(f"   {status} {result['dependency']}: Status {result['status_code']}")


class TestResourceManagement:
    """Test resource management and limits"""
    
    def test_memory_exhaustion_protection(self):
        """Test protection against memory exhaustion attacks"""
        config = SecurityTestConfig()
        
        # Test with increasingly large payloads
        payload_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB
        
        memory_test_results = []
        
        for size in payload_sizes:
            try:
                large_payload = {
                    "user_id": "memory_test",
                    "top_k": 5,
                    "large_data": "x" * size
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{config.BASE_URL}/run-workflow/run-workflow",
                    json=large_payload,
                    timeout=20
                )
                response_time = time.time() - start_time
                
                memory_test_results.append({
                    "payload_size": f"{size//1024}KB",
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "handled_gracefully": response.status_code in [200, 400, 413]  # 413 = Payload Too Large
                })
                
            except requests.exceptions.Timeout:
                memory_test_results.append({
                    "payload_size": f"{size//1024}KB",
                    "status_code": "timeout",
                    "response_time": 20.0,
                    "handled_gracefully": False
                })
            except requests.exceptions.RequestException:
                memory_test_results.append({
                    "payload_size": f"{size//1024}KB",
                    "status_code": "error",
                    "response_time": 0,
                    "handled_gracefully": False
                })
        
        print(f"\\n💾 Memory Exhaustion Protection Test:")
        graceful_count = sum(1 for r in memory_test_results if r["handled_gracefully"])
        print(f"   {graceful_count}/{len(memory_test_results)} large payloads handled gracefully:")
        
        for result in memory_test_results:
            status = "✅" if result["handled_gracefully"] else "❌"
            print(f"   {status} {result['payload_size']}: {result['status_code']} ({result['response_time']:.2f}s)")
    
    def test_concurrent_processing_limits(self):
        """Test service behavior under concurrent processing load"""
        config = SecurityTestConfig()
        
        results = []
        num_threads = 8  # Reasonable concurrent load
        
        def make_request(thread_id):
            try:
                start_time = time.time()
                response = requests.get(
                    f"{config.BASE_URL}/recommendations/?user_id=concurrent_{thread_id}&top_k=5",
                    timeout=15
                )
                response_time = time.time() - start_time
                
                results.append({
                    "thread_id": thread_id,
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "success": response.status_code == 200
                })
                
            except requests.exceptions.RequestException as e:
                results.append({
                    "thread_id": thread_id,
                    "status_code": "error",
                    "response_time": 0,
                    "success": False,
                    "error": str(e)
                })
        
        # Launch concurrent requests
        threads = []
        start_time = time.time()
        
        for i in range(num_threads):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        if successful_requests:
            avg_response_time = sum(r["response_time"] for r in successful_requests) / len(successful_requests)
        else:
            avg_response_time = 0
        
        print(f"\\n⚡ Concurrent Processing Test:")
        print(f"   Successful: {len(successful_requests)}/{len(results)}")
        print(f"   Average response time: {avg_response_time:.2f}s")
        print(f"   Total processing time: {total_time:.2f}s")
        
        # Evaluate service performance
        if len(successful_requests) >= len(results) * 0.9:  # 90% success rate
            print(f"   ✅ Service handles concurrent load well")
        elif len(successful_requests) >= len(results) * 0.7:  # 70% success rate
            print(f"   ⚠️  Service shows some strain under concurrent load")
        else:
            print(f"   ❌ Service struggles with concurrent processing")


class TestOperationalSecurity:
    """Test operational security concerns"""
    
    def test_error_information_disclosure(self):
        """Test that errors don't disclose sensitive operational information"""
        config = SecurityTestConfig()
        
        # Trigger various error conditions
        error_triggers = [
            ("/recommendations/?user_id=&top_k=invalid", "Invalid parameter types"),
            ("/nonexistent-endpoint", "Non-existent endpoint"),
            ("/run-workflow/run-workflow", "Missing POST payload")
        ]
        
        information_disclosure_issues = []
        
        for trigger, description in error_triggers:
            try:
                if trigger.startswith("/run-workflow"):
                    response = requests.post(f"{config.BASE_URL}{trigger}", json={}, timeout=10)
                else:
                    response = requests.get(f"{config.BASE_URL}{trigger}", timeout=10)
                
                response_text = response.text.lower()
                
                # Check for sensitive information in error responses
                sensitive_patterns = [
                    'traceback', 'file "/', 'line \\d+', 'exception:',
                    'mongodb://', 'postgresql://', 'mysql://', 'connection string',
                    '/home/', 'c:\\\\', 'secret', 'password', 'key',
                    'internal server error', 'debug mode'
                ]
                
                found_sensitive = []
                for pattern in sensitive_patterns:
                    if pattern in response_text:
                        found_sensitive.append(pattern)
                
                if found_sensitive:
                    information_disclosure_issues.append({
                        "trigger": trigger,
                        "description": description,
                        "sensitive_info": found_sensitive[:3],  # First 3 matches
                        "status_code": response.status_code
                    })
                    
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n🔍 Error Information Disclosure Test:")
        if information_disclosure_issues:
            print(f"   ⚠️  Found {len(information_disclosure_issues)} error responses with sensitive info:")
            for issue in information_disclosure_issues:
                print(f"   - {issue['description']}: {', '.join(issue['sensitive_info'])}")
        else:
            print("   ✅ Error responses don't expose sensitive operational information")
    
    def test_service_configuration_exposure(self):
        """Test for exposed configuration endpoints"""
        config = SecurityTestConfig()
        
        # Test potential configuration/status endpoints
        config_endpoints = [
            "/config",
            "/status",
            "/info",
            "/env",
            "/settings",
            "/version",
            "/.env",
            "/health/detailed"
        ]
        
        exposed_config = []
        
        for endpoint in config_endpoints:
            try:
                response = requests.get(f"{config.BASE_URL}{endpoint}", timeout=5)
                
                if response.status_code == 200 and len(response.text) > 50:  # Substantial response
                    response_text = response.text.lower()
                    
                    # Check if response contains configuration information
                    config_indicators = [
                        'database', 'connection', 'host', 'port', 'url',
                        'version', 'environment', 'config', 'setting'
                    ]
                    
                    if any(indicator in response_text for indicator in config_indicators):
                        exposed_config.append({
                            "endpoint": endpoint,
                            "status_code": response.status_code,
                            "response_length": len(response.text)
                        })
                        
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n⚙️ Service Configuration Exposure Test:")
        if exposed_config:
            print(f"   ⚠️  Found {len(exposed_config)} endpoints exposing configuration:")
            for config_ep in exposed_config:
                print(f"   - {config_ep['endpoint']}: {config_ep['status_code']} ({config_ep['response_length']} chars)")
        else:
            print("   ✅ No configuration endpoints exposing sensitive information")