"""
Data Protection and Privacy Security Tests

Tests for sensitive data exposure, data leakage, and privacy violations.
"""

import pytest
import requests
import json
import re
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


class TestDataExposure:
    """Test for sensitive data exposure vulnerabilities"""
    
    def test_sensitive_data_in_error_messages(self):
        """Test if error messages expose sensitive information"""
        config = SecurityTestConfig()
        
        # Trigger errors and check for sensitive data exposure
        error_triggers = [
            "/recommendations/?user_id=&top_k=abc",  # Invalid parameters
            "/recommendations/?user_id=nonexistent_user_12345&top_k=999",  # Non-existent user
            "/run-workflow/run-workflow",  # Missing required payload
            "/user-vector-update/run-manual-update",  # Missing payload
        ]
        
        sensitive_patterns = [
            r'password[:\s]*[\\"\']?\\w+',
            r'secret[:\s]*[\\"\']?\\w+',
            r'key[:\s]*[\\"\']?[\\w\\-]+',
            r'token[:\s]*[\\"\']?[\\w\\-\\.]+',
            r'api[_\\-]?key[:\s]*[\\"\']?[\\w\\-]+',
            r'database[:\s]*[\\"\']?[\\w\\-\\./@:]+',
            r'mongodb://[\\w\\-\\./@:]+',
            r'postgresql://[\\w\\-\\./@:]+',
            r'mysql://[\\w\\-\\./@:]+',
            r'\\b\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\b',  # IP addresses
            r'[\\w\\-\\.]+@[\\w\\-\\.]+\\.[a-z]{2,}',  # Email addresses
            r'/home/[\\w/\\-\\.]+',  # Unix paths
            r'C:\\\\[\\w\\\\\\-\\.]+',  # Windows paths
        ]
        
        exposed_data = []
        
        for trigger in error_triggers:
            try:
                if trigger.startswith("/run-workflow") or trigger.startswith("/user-vector"):
                    response = requests.post(f"{config.BASE_URL}{trigger}", json={}, timeout=10)
                else:
                    response = requests.get(f"{config.BASE_URL}{trigger}", timeout=10)
                
                response_text = response.text
                
                # Check for sensitive patterns
                for pattern in sensitive_patterns:
                    matches = re.findall(pattern, response_text, re.IGNORECASE)
                    if matches:
                        exposed_data.append({
                            "trigger": trigger,
                            "pattern": pattern,
                            "matches": matches[:3],  # First 3 matches
                            "status_code": response.status_code
                        })
                        
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n🔍 Sensitive Data Exposure Test:")
        if exposed_data:
            print(f"   🚨 Found {len(exposed_data)} potential data exposures:")
            for exposure in exposed_data:
                print(f"   - Trigger: {exposure['trigger']}")
                print(f"     Pattern: {exposure['pattern']}")
                print(f"     Matches: {exposure['matches']}")
        else:
            print("   ✅ No obvious sensitive data exposure in error messages")
    
    def test_user_data_leakage_between_requests(self):
        """Test if user data leaks between different user requests"""
        config = SecurityTestConfig()
        
        # Make requests for different users and check for data leakage
        test_users = ["user_a", "user_b", "user_c", "admin", "test"]
        user_responses = {}
        
        # Collect responses for different users
        for user in test_users:
            try:
                response = requests.get(
                    f"{config.BASE_URL}/recommendations/?user_id={user}&top_k=10",
                    timeout=10
                )
                if response.status_code == 200:
                    user_responses[user] = response.json()
            except:
                continue
        
        # Check for data leakage
        leakage_issues = []
        
        for user_a, data_a in user_responses.items():
            for user_b, data_b in user_responses.items():
                if user_a != user_b and data_a and data_b:
                    # Check if user_a's data appears in user_b's response
                    data_a_str = json.dumps(data_a, sort_keys=True)
                    data_b_str = json.dumps(data_b, sort_keys=True)
                    
                    # Simple similarity check
                    if data_a_str == data_b_str:
                        leakage_issues.append({
                            "user_a": user_a,
                            "user_b": user_b,
                            "issue": "Identical responses for different users"
                        })
                    
                    # Check for user_a mentioned in user_b's response
                    if user_a in data_b_str and user_a != user_b:
                        leakage_issues.append({
                            "user_a": user_a,
                            "user_b": user_b,
                            "issue": f"User {user_a} mentioned in {user_b}'s response"
                        })
        
        print(f"\\n🔍 User Data Leakage Test:")
        if leakage_issues:
            print(f"   ⚠️  Found {len(leakage_issues)} potential data leakage issues:")
            for issue in leakage_issues[:5]:  # Show first 5
                print(f"   - {issue['issue']}")
        else:
            print(f"   ✅ No obvious data leakage between {len(user_responses)} user responses")
    
    def test_debug_information_exposure(self):
        """Test for debug information exposure"""
        config = SecurityTestConfig()
        
        # Test endpoints that might expose debug info
        debug_endpoints = [
            "/test/debug-database",
            "/test/test-database-connection",
            "/debug/",
            "/trace/",
            "/error/",
            "/stack/",
        ]
        
        debug_patterns = [
            r'traceback',
            r'stack trace',
            r'exception',
            r'debug',
            r'line \\d+',
            r'file ".*\\.py"',
            r'\\w+Error:',
            r'at \\w+\\.py:\\d+',
            r'mongodb://',
            r'postgresql://',
            r'mysql://',
            r'connection string',
            r'environment',
            r'config',
        ]
        
        debug_exposures = []
        
        for endpoint in debug_endpoints:
            try:
                response = requests.get(f"{config.BASE_URL}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    response_text = response.text.lower()
                    
                    for pattern in debug_patterns:
                        if re.search(pattern, response_text, re.IGNORECASE):
                            debug_exposures.append({
                                "endpoint": endpoint,
                                "pattern": pattern,
                                "status_code": response.status_code,
                                "response_length": len(response.text)
                            })
                            break  # Only report first match per endpoint
                            
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n🔍 Debug Information Exposure Test:")
        if debug_exposures:
            print(f"   ⚠️  Found {len(debug_exposures)} endpoints exposing debug info:")
            for exposure in debug_exposures:
                print(f"   - {exposure['endpoint']}: {exposure['pattern']}")
        else:
            print("   ✅ No obvious debug information exposure")


class TestPrivacyViolations:
    """Test for privacy violations and data protection issues"""
    
    def test_user_enumeration_via_timing_attacks(self):
        """Test if user existence can be determined via timing differences"""
        config = SecurityTestConfig()
        
        import time
        
        # Test with likely existing vs non-existing users
        existing_users = ["admin", "test", "user", "demo"]
        nonexisting_users = ["nonexistent_user_xyz", "fake_user_123", "invalid_user_999"]
        
        timing_results = {"existing": [], "nonexisting": []}
        
        for user_list, category in [(existing_users, "existing"), (nonexisting_users, "nonexisting")]:
            for user in user_list:
                try:
                    start_time = time.time()
                    response = requests.get(
                        f"{config.BASE_URL}/recommendations/?user_id={user}&top_k=5",
                        timeout=10
                    )
                    end_time = time.time()
                    
                    timing_results[category].append({
                        "user": user,
                        "response_time": end_time - start_time,
                        "status_code": response.status_code
                    })
                    
                except requests.exceptions.RequestException:
                    continue
        
        # Analyze timing differences
        if timing_results["existing"] and timing_results["nonexisting"]:
            avg_existing = sum(r["response_time"] for r in timing_results["existing"]) / len(timing_results["existing"])
            avg_nonexisting = sum(r["response_time"] for r in timing_results["nonexisting"]) / len(timing_results["nonexisting"])
            
            time_diff = abs(avg_existing - avg_nonexisting)
            
            print(f"\\n🔍 Timing Attack Analysis:")
            print(f"   Average response time for 'existing' users: {avg_existing:.3f}s")
            print(f"   Average response time for 'nonexisting' users: {avg_nonexisting:.3f}s")
            print(f"   Time difference: {time_diff:.3f}s")
            
            if time_diff > 0.1:  # Significant timing difference
                print(f"   ⚠️  Significant timing difference detected - potential user enumeration")
            else:
                print(f"   ✅ No significant timing differences")
        else:
            print(f"\\n🔍 Timing Attack Analysis: Insufficient data")
    
    def test_information_disclosure_via_response_differences(self):
        """Test information disclosure via different response patterns"""
        config = SecurityTestConfig()
        
        # Test various user IDs to see response patterns
        test_cases = [
            ("valid_user_pattern", "test_user"),
            ("admin_user", "admin"),
            ("system_user", "system"), 
            ("empty_user", ""),
            ("null_user", "null"),
            ("very_long_user", "a" * 1000),
            ("special_chars_user", "user@domain.com"),
            ("path_traversal", "../admin"),
            ("sql_pattern", "' OR '1'='1"),
        ]
        
        response_patterns = []
        
        for case_name, user_id in test_cases:
            try:
                response = requests.get(
                    f"{config.BASE_URL}/recommendations/?user_id={user_id}&top_k=5",
                    timeout=10
                )
                
                response_patterns.append({
                    "case": case_name,
                    "user_id": user_id[:50],  # Truncate very long user IDs
                    "status_code": response.status_code,
                    "response_length": len(response.text),
                    "content_type": response.headers.get("content-type", ""),
                    "has_json": "application/json" in response.headers.get("content-type", "")
                })
                
            except requests.exceptions.RequestException:
                response_patterns.append({
                    "case": case_name,
                    "user_id": user_id[:50],
                    "status_code": "error",
                    "response_length": 0,
                    "content_type": "",
                    "has_json": False
                })
        
        # Analyze response patterns
        unique_patterns = {}
        for pattern in response_patterns:
            key = (pattern["status_code"], pattern["response_length"], pattern["has_json"])
            if key not in unique_patterns:
                unique_patterns[key] = []
            unique_patterns[key].append(pattern)
        
        print(f"\\n🔍 Information Disclosure Analysis:")
        print(f"   Found {len(unique_patterns)} unique response patterns:")
        
        for (status, length, has_json), cases in unique_patterns.items():
            case_names = [c["case"] for c in cases]
            print(f"   - Status: {status}, Length: {length}, JSON: {has_json}")
            print(f"     Cases: {', '.join(case_names)}")
        
        if len(unique_patterns) > 3:
            print(f"   ⚠️  Multiple response patterns may leak information about user existence")
        else:
            print(f"   ✅ Response patterns seem consistent")


class TestDataIntegrity:
    """Test data integrity and consistency issues"""
    
    def test_concurrent_user_data_access(self):
        """Test concurrent access to user data for race conditions"""
        config = SecurityTestConfig()
        
        import threading
        import time
        
        results = []
        user_id = "concurrent_test_user"
        
        def make_request(thread_id):
            try:
                response = requests.get(
                    f"{config.BASE_URL}/recommendations/?user_id={user_id}&top_k=5",
                    timeout=10
                )
                results.append({
                    "thread_id": thread_id,
                    "status_code": response.status_code,
                    "response_time": time.time(),
                    "response_hash": hash(response.text) if response.text else None
                })
            except requests.exceptions.RequestException as e:
                results.append({
                    "thread_id": thread_id,
                    "status_code": "error",
                    "response_time": time.time(),
                    "error": str(e)
                })
        
        # Launch concurrent requests
        threads = []
        start_time = time.time()
        
        for i in range(5):  # 5 concurrent requests
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Analyze results
        successful_results = [r for r in results if r["status_code"] == 200]
        error_results = [r for r in results if r["status_code"] != 200]
        
        print(f"\\n🔍 Concurrent Access Test:")
        print(f"   Successful requests: {len(successful_results)}/{len(results)}")
        print(f"   Error requests: {len(error_results)}")
        
        if successful_results:
            # Check for response consistency
            response_hashes = [r["response_hash"] for r in successful_results if r["response_hash"]]
            unique_hashes = set(response_hashes)
            
            if len(unique_hashes) == 1:
                print(f"   ✅ All responses consistent")
            else:
                print(f"   ⚠️  Found {len(unique_hashes)} different responses - potential race condition")
        
        if error_results:
            print(f"   ⚠️  {len(error_results)} requests failed during concurrent access")
            for error in error_results[:3]:  # Show first 3 errors
                print(f"      - Thread {error['thread_id']}: {error.get('error', error['status_code'])}")
    
    def test_data_consistency_across_endpoints(self):
        """Test data consistency between different endpoints"""
        config = SecurityTestConfig()
        
        user_id = "consistency_test_user"
        
        # Get data from different endpoints for the same user
        endpoints_data = {}
        
        try:
            # Recommendations endpoint
            response = requests.get(
                f"{config.BASE_URL}/recommendations/?user_id={user_id}&top_k=10",
                timeout=10
            )
            if response.status_code == 200:
                endpoints_data["recommendations"] = response.json()
        except:
            pass
        
        try:
            # Workflow endpoint
            response = requests.post(
                f"{config.BASE_URL}/run-workflow/run-workflow",
                json={"user_id": user_id, "top_k": 10},
                timeout=10
            )
            if response.status_code == 200:
                endpoints_data["workflow"] = response.json()
        except:
            pass
        
        # Analyze consistency
        print(f"\\n🔍 Data Consistency Test:")
        if len(endpoints_data) >= 2:
            print(f"   Retrieved data from {len(endpoints_data)} endpoints")
            
            # Compare data structures
            endpoint_names = list(endpoints_data.keys())
            data_structures = {}
            
            for name, data in endpoints_data.items():
                if isinstance(data, dict):
                    data_structures[name] = set(data.keys())
                
            if len(data_structures) >= 2:
                common_keys = set.intersection(*data_structures.values())
                print(f"   Common data keys: {len(common_keys)}")
                
                if common_keys:
                    print(f"   ✅ Found common structure between endpoints")
                else:
                    print(f"   ⚠️  No common data structure - inconsistent responses")
        else:
            print(f"   ⚠️  Could only retrieve data from {len(endpoints_data)} endpoints")