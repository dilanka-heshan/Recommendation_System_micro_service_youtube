"""
Background Service Security Tests

Security tests focused on background service vulnerabilities - no authentication/authorization 
testing since this is a single-service system without multiple users or authentication.
"""

import pytest
import requests
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


class TestBackgroundServiceSecurity:
    """Test security for background service endpoints"""
    
    def test_service_endpoints_accessibility(self):
        """
        Test that service endpoints are accessible for legitimate requests
        This is expected behavior for a background service
        """
        config = SecurityTestConfig()
        
        # Test core service endpoints
        accessible_endpoints = []
        
        test_endpoints = [
            ("health", "/health", "GET", None),
            ("recommendations", "/recommendations/?user_id=test_user&top_k=5", "GET", None),
            ("newsletter_health", "/newsletter/health", "GET", None),
            ("user_vector_status", "/user-vector-update/status", "GET", None)
        ]
        
        for name, endpoint, method, payload in test_endpoints:
            try:
                if method == "GET":
                    response = requests.get(f"{config.BASE_URL}{endpoint}", timeout=10)
                else:
                    response = requests.post(f"{config.BASE_URL}{endpoint}", json=payload, timeout=10)
                
                if response.status_code in [200, 404]:  # 404 is fine, means endpoint exists
                    accessible_endpoints.append({
                        "endpoint": name,
                        "path": endpoint,
                        "status_code": response.status_code,
                        "method": method
                    })
                    
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n✅ Service Endpoints Accessibility Test:")
        print(f"   {len(accessible_endpoints)} endpoints accessible as expected:")
        for endpoint in accessible_endpoints:
            print(f"  - {endpoint['method']} {endpoint['path']} (Status: {endpoint['status_code']})")
    
    def test_admin_debugging_endpoints(self):
        """Test if debugging/admin endpoints are properly secured or disabled"""
        config = SecurityTestConfig()
        
        # Test potentially sensitive debugging endpoints
        debug_endpoints = [
            "/test/debug-database",
            "/test/test-database-connection", 
            "/admin/",
            "/debug/",
            "/status/detailed",
            "/metrics/",
            "/internal/"
        ]
        
        exposed_debug_endpoints = []
        
        for endpoint in debug_endpoints:
            try:
                response = requests.get(f"{config.BASE_URL}{endpoint}", timeout=5)
                
                # Check if endpoint exposes sensitive information
                if response.status_code == 200:
                    response_text = response.text.lower()
                    sensitive_indicators = [
                        'password', 'secret', 'key', 'token', 'database',
                        'connection string', 'mongodb://', 'postgresql://',
                        'traceback', 'error', 'exception'
                    ]
                    
                    has_sensitive_info = any(indicator in response_text for indicator in sensitive_indicators)
                    
                    exposed_debug_endpoints.append({
                        "endpoint": endpoint,
                        "status_code": response.status_code,
                        "has_sensitive_info": has_sensitive_info,
                        "response_length": len(response.text)
                    })
                    
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n🔍 Debug Endpoints Security Test:")
        if exposed_debug_endpoints:
            print(f"   ⚠️  Found {len(exposed_debug_endpoints)} accessible debug endpoints:")
            for endpoint in exposed_debug_endpoints:
                sensitive_flag = "🚨 SENSITIVE" if endpoint['has_sensitive_info'] else "📝 INFO"
                print(f"   - {endpoint['endpoint']} ({sensitive_flag}) Status: {endpoint['status_code']}")
        else:
            print("   ✅ No debug endpoints exposing sensitive information")
    
    def test_user_data_enumeration(self):
        """Test if user data can be enumerated by trying different user IDs"""
        config = SecurityTestConfig()
        
        # Test common user ID patterns
        test_user_ids = [
            "1", "2", "3", "admin", "root", "test", "user",
            "guest", "demo", "sample", "default", "system",
            "service", "api", "bot", "crawler"
        ]
        
        accessible_users = []
        
        for user_id in test_user_ids:
            try:
                response = requests.get(
                    f"{config.BASE_URL}/recommendations/?user_id={user_id}&top_k=1",
                    timeout=5
                )
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if data and 'recommendations' in data:
                            accessible_users.append({
                                "user_id": user_id,
                                "has_data": len(data['recommendations']) > 0
                            })
                    except:
                        accessible_users.append({
                            "user_id": user_id,
                            "has_data": False
                        })
                        
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n📊 User Data Enumeration Results:")
        print(f"   Found {len(accessible_users)} accessible user IDs:")
        for user in accessible_users[:10]:  # Show first 10
            status = "WITH DATA" if user['has_data'] else "NO DATA"
            print(f"   - {user['user_id']}: {status}")
        
        if len(accessible_users) > 10:
            print(f"   ... and {len(accessible_users) - 10} more")
    
    def test_admin_endpoint_access(self):
        """Test access to potential administrative endpoints"""
        config = SecurityTestConfig()
        
        # Test administrative endpoints
        admin_endpoints = [
            "/admin/",
            "/api/admin/",
            "/admin/users",
            "/admin/dashboard",
            "/admin/config",
            "/management/",
            "/debug/",
            "/status/",
            "/metrics/",
            "/health/detailed",
            "/internal/",
            "/system/",
            "/config/",
            "/logs/"
        ]
        
        accessible_admin_endpoints = []
        
        for endpoint in admin_endpoints:
            try:
                response = requests.get(f"{config.BASE_URL}{endpoint}", timeout=5)
                
                if response.status_code not in [404, 405]:  # Not just "not found"
                    accessible_admin_endpoints.append({
                        "endpoint": endpoint,
                        "status_code": response.status_code,
                        "content_length": len(response.content)
                    })
                    
            except requests.exceptions.RequestException:
                continue
        
        if accessible_admin_endpoints:
            print(f"\\n⚠️  Potentially accessible admin endpoints:")
            for endpoint in accessible_admin_endpoints:
                print(f"   - {endpoint['endpoint']} (Status: {endpoint['status_code']})")
        else:
            print(f"\\n✅ No obvious admin endpoints accessible")


class TestSessionManagement:
    """Test session management and token security"""
    
    def test_session_fixation(self):
        """Test for session fixation vulnerabilities"""
        # Since there's no session management currently, this documents the gap
        print("\\n📝 Session Management Analysis:")
        print("   - No session management implemented")
        print("   - No token-based authentication")
        print("   - No session fixation protection needed (no sessions)")
        print("   - RECOMMENDATION: Implement proper session management")
    
    def test_csrf_protection(self):
        """Test for CSRF protection on state-changing operations"""
        config = SecurityTestConfig()
        
        # Test POST endpoints without CSRF tokens
        csrf_vulnerable_endpoints = []
        
        post_endpoints = [
            ("/user-vector-update/run-daily-update", {}),
            ("/user-vector-update/run-manual-update", {"user_ids": ["test_user"]}),
            ("/run-workflow/run-workflow", {"user_id": "test_user", "top_k": 5})
        ]
        
        for endpoint, payload in post_endpoints:
            try:
                # Test without any CSRF protection
                response = requests.post(
                    f"{config.BASE_URL}{endpoint}",
                    json=payload,
                    headers={"Origin": "http://malicious-site.com"},
                    timeout=10
                )
                
                if response.status_code not in [403, 400]:  # Not blocked for CSRF
                    csrf_vulnerable_endpoints.append({
                        "endpoint": endpoint,
                        "status_code": response.status_code
                    })
                    
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n🚨 CSRF Vulnerability Analysis:")
        print(f"   {len(csrf_vulnerable_endpoints)} POST endpoints lack CSRF protection:")
        for endpoint in csrf_vulnerable_endpoints:
            print(f"   - {endpoint['endpoint']} (Status: {endpoint['status_code']})")