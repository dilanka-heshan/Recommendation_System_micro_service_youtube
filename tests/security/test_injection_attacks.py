"""
Input Validation and Injection Attack Security Tests for Background Service

Tests for service robustness against malicious inputs - focused on preventing 
service crashes and ensuring stability rather than data protection.
"""

import pytest
import requests
import json
import urllib.parse
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


class TestServiceStabilityInjection:
    """Test service stability against injection attacks"""
    
    def test_service_sql_injection_protection(self):
        """Test service stability against SQL injection payloads"""
        config = SecurityTestConfig()
        
        service_issues = []
        
        for payload in config.SQL_INJECTION_PAYLOADS:
            try:
                # Test in user_id parameter - focus on service stability
                encoded_payload = urllib.parse.quote(payload)
                response = requests.get(
                    f"{config.BASE_URL}/recommendations/?user_id={encoded_payload}&top_k=5",
                    timeout=10
                )
                
                # Check for service crashes or instability
                response_text = response.text.lower()
                service_error_indicators = [
                    'internal server error', '500', 'crashed', 'exception',
                    'traceback', 'error occurred', 'failed to process'
                ]
                
                if any(indicator in response_text for indicator in service_error_indicators):
                    service_issues.append({
                        "payload": payload,
                        "status_code": response.status_code,
                        "issue_type": "service_instability"
                    })
                elif response.status_code == 500:
                    # Server error indicates service problem
                    service_issues.append({
                        "payload": payload,
                        "status_code": response.status_code,
                        "issue_type": "server_error"
                    })
                    
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n🔍 Service SQL Injection Protection Test:")
        if service_issues:
            print(f"   ⚠️  Found {len(service_issues)} payloads causing service issues:")
            for result in service_issues[:3]:  # Show first 3
                print(f"   - Payload: {result['payload'][:50]}... Status: {result['status_code']}")
        else:
            print("   ✅ Service handles malicious SQL payloads gracefully")
    
    def test_top_k_parameter_injection(self):
        """Test injection in top_k parameter"""
        config = SecurityTestConfig()
        
        # Test various injection payloads in numeric parameter
        injection_payloads = [
            "1'; DROP TABLE users; --",
            "1 OR 1=1",
            "1 UNION SELECT 1,2,3",
            "999999999",
            "-1",
            "0",
            "NULL",
            "undefined"
        ]
        
        vulnerable_responses = []
        
        for payload in injection_payloads:
            try:
                encoded_payload = urllib.parse.quote(str(payload))
                response = requests.get(
                    f"{config.BASE_URL}/recommendations/?user_id=test_user&top_k={encoded_payload}",
                    timeout=10
                )
                
                if response.status_code == 500 or 'error' in response.text.lower():
                    vulnerable_responses.append({
                        "payload": payload,
                        "status_code": response.status_code,
                        "response_length": len(response.text)
                    })
                    
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n🔍 Parameter Injection Test (top_k):")
        if vulnerable_responses:
            print(f"   ⚠️  Found {len(vulnerable_responses)} error responses:")
            for result in vulnerable_responses:
                print(f"   - Payload: {result['payload']} → Status: {result['status_code']}")
        else:
            print("   ✅ No obvious parameter injection vulnerabilities found")


class TestNoSQLInjection:
    """Test NoSQL injection vulnerabilities (MongoDB, etc.)"""
    
    def test_nosql_injection_in_json_payloads(self):
        """Test NoSQL injection in JSON POST payloads"""
        config = SecurityTestConfig()
        
        vulnerable_responses = []
        
        # Test NoSQL injection in user vector update endpoints
        for payload in config.NOSQL_INJECTION_PAYLOADS[:5]:  # Test first 5 payloads
            try:
                # Test in manual update endpoint
                malicious_payload = {
                    "user_ids": [payload] if isinstance(payload, str) else payload,
                    "date_range": payload if not isinstance(payload, str) else None
                }
                
                response = requests.post(
                    f"{config.BASE_URL}/user-vector-update/run-manual-update",
                    json=malicious_payload,
                    timeout=10
                )
                
                if response.status_code in [200, 500]:
                    try:
                        response_data = response.json()
                        # Check if injection might have succeeded
                        if isinstance(response_data, dict) and len(str(response_data)) > 1000:
                            vulnerable_responses.append({
                                "payload": str(payload)[:50],
                                "status_code": response.status_code,
                                "response_size": len(str(response_data))
                            })
                    except:
                        if response.status_code == 500:
                            vulnerable_responses.append({
                                "payload": str(payload)[:50],
                                "status_code": response.status_code,
                                "response_size": len(response.text)
                            })
                            
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n🔍 NoSQL Injection Test Results:")
        if vulnerable_responses:
            print(f"   ⚠️  Found {len(vulnerable_responses)} suspicious responses:")
            for result in vulnerable_responses:
                print(f"   - Payload: {result['payload']}... → Status: {result['status_code']}")
        else:
            print("   ✅ No obvious NoSQL injection vulnerabilities found")


class TestXSSVulnerabilities:
    """Test Cross-Site Scripting (XSS) vulnerabilities"""
    
    def test_reflected_xss_in_parameters(self):
        """Test reflected XSS in URL parameters"""
        config = SecurityTestConfig()
        
        xss_vulnerable_responses = []
        
        for payload in config.XSS_PAYLOADS:
            try:
                # Test XSS in user_id parameter
                response = requests.get(
                    f"{config.BASE_URL}/recommendations/?user_id={urllib.parse.quote(payload)}&top_k=5",
                    timeout=10
                )
                
                # Check if XSS payload is reflected in response
                if payload in response.text or payload.replace('"', "'") in response.text:
                    xss_vulnerable_responses.append({
                        "payload": payload,
                        "status_code": response.status_code,
                        "reflected": True
                    })
                    
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n🔍 XSS Vulnerability Test Results:")
        if xss_vulnerable_responses:
            print(f"   🚨 Found {len(xss_vulnerable_responses)} reflected XSS vulnerabilities:")
            for result in xss_vulnerable_responses:
                print(f"   - Payload reflected: {result['payload'][:40]}...")
        else:
            print("   ✅ No reflected XSS vulnerabilities found")
    
    def test_stored_xss_in_json_payloads(self):
        """Test stored XSS in JSON payloads"""
        config = SecurityTestConfig()
        
        # Test XSS in POST endpoints that might store data
        test_endpoints = [
            ("/run-workflow/run-workflow", {"user_id": "<script>alert('XSS')</script>", "top_k": 5}),
            ("/user-vector-update/run-manual-update", {"user_ids": ["<img src=x onerror=alert('XSS')>"]})
        ]
        
        stored_xss_results = []
        
        for endpoint, payload in test_endpoints:
            try:
                response = requests.post(
                    f"{config.BASE_URL}{endpoint}",
                    json=payload,
                    timeout=10
                )
                
                # Check if XSS payload is in response
                xss_patterns = ["<script", "<img", "javascript:", "onerror="]
                if any(pattern in response.text for pattern in xss_patterns):
                    stored_xss_results.append({
                        "endpoint": endpoint,
                        "status_code": response.status_code,
                        "contains_xss": True
                    })
                    
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n🔍 Stored XSS Test Results:")
        if stored_xss_results:
            print(f"   ⚠️  Found {len(stored_xss_results)} potential stored XSS issues:")
            for result in stored_xss_results:
                print(f"   - Endpoint: {result['endpoint']}")
        else:
            print("   ✅ No obvious stored XSS vulnerabilities found")


class TestCommandInjection:
    """Test command injection vulnerabilities"""
    
    def test_command_injection_in_parameters(self):
        """Test command injection in various parameters"""
        config = SecurityTestConfig()
        
        command_injection_results = []
        
        for payload in config.COMMAND_INJECTION_PAYLOADS:
            try:
                # Test command injection in user_id
                response = requests.get(
                    f"{config.BASE_URL}/recommendations/?user_id={urllib.parse.quote(payload)}&top_k=5",
                    timeout=15  # Longer timeout for potential command execution
                )
                
                # Check for command injection indicators
                command_indicators = [
                    'root:', 'bin/bash', 'system32', 'cmd.exe',
                    'permission denied', 'command not found',
                    'total ', 'drwx', '-rwx', 'administrator'
                ]
                
                response_text = response.text.lower()
                if any(indicator in response_text for indicator in command_indicators):
                    command_injection_results.append({
                        "payload": payload,
                        "status_code": response.status_code,
                        "indicates_command_injection": True
                    })
                elif response.status_code == 500:
                    command_injection_results.append({
                        "payload": payload,
                        "status_code": response.status_code,
                        "indicates_command_injection": "possible"
                    })
                    
            except requests.exceptions.Timeout:
                # Timeout might indicate command execution
                command_injection_results.append({
                    "payload": payload,
                    "status_code": "timeout",
                    "indicates_command_injection": "possible"
                })
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n🔍 Command Injection Test Results:")
        if command_injection_results:
            print(f"   ⚠️  Found {len(command_injection_results)} suspicious responses:")
            for result in command_injection_results:
                print(f"   - Payload: {result['payload'][:30]}... → {result['status_code']}")
        else:
            print("   ✅ No obvious command injection vulnerabilities found")


class TestInputValidation:
    """Test input validation bypass techniques"""
    
    def test_malformed_json_handling(self):
        """Test how system handles malformed JSON"""
        config = SecurityTestConfig()
        
        malformed_json_tests = [
            '{"user_id": "test"',  # Unclosed JSON
            '{"user_id": "test",}',  # Trailing comma
            '{"user_id": }',  # Missing value
            '{user_id: "test"}',  # Unquoted key
            '{"user_id": "test"" extra"}',  # Extra quotes
            '{"user_id": null}',  # Null value
            '{"user_id": undefined}',  # Undefined value
            '{\\x00"user_id": "test"}',  # Null byte
            '{"user_id": "' + 'A' * 10000 + '"}',  # Very long value
        ]
        
        malformed_handling_results = []
        
        for malformed_json in malformed_json_tests[:6]:  # Test first 6
            try:
                response = requests.post(
                    f"{config.BASE_URL}/run-workflow/run-workflow",
                    data=malformed_json,
                    headers={'Content-Type': 'application/json'},
                    timeout=10
                )
                
                malformed_handling_results.append({
                    "test": malformed_json[:30] + "..." if len(malformed_json) > 30 else malformed_json,
                    "status_code": response.status_code,
                    "response_length": len(response.text)
                })
                
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n🔍 Malformed JSON Handling Test:")
        print(f"   Tested {len(malformed_handling_results)} malformed JSON inputs:")
        for result in malformed_handling_results:
            print(f"   - Input: {result['test']} → Status: {result['status_code']}")
    
    def test_boundary_value_testing(self):
        """Test boundary values in numeric parameters"""
        config = SecurityTestConfig()
        
        boundary_values = [
            -2147483648,  # INT_MIN
            2147483647,   # INT_MAX
            0,
            -1,
            999999999,
            -999999999,
            "0.0",
            "1.7976931348623157e+308",  # Max float
            "∞",
            "NaN",
            "null",
            "undefined"
        ]
        
        boundary_test_results = []
        
        for value in boundary_values:
            try:
                response = requests.get(
                    f"{config.BASE_URL}/recommendations/?user_id=test_user&top_k={value}",
                    timeout=10
                )
                
                boundary_test_results.append({
                    "value": str(value),
                    "status_code": response.status_code,
                    "handled_gracefully": response.status_code not in [500, 502, 503, 504]
                })
                
            except requests.exceptions.RequestException:
                boundary_test_results.append({
                    "value": str(value),
                    "status_code": "error",
                    "handled_gracefully": False
                })
        
        print(f"\\n🔍 Boundary Value Test Results:")
        graceful_count = sum(1 for r in boundary_test_results if r['handled_gracefully'])
        print(f"   {graceful_count}/{len(boundary_test_results)} boundary values handled gracefully")
        
        for result in boundary_test_results:
            status = "✅" if result['handled_gracefully'] else "❌"
            print(f"   {status} Value: {result['value'][:20]} → Status: {result['status_code']}")
    
    def test_unicode_and_encoding_attacks(self):
        """Test Unicode and encoding-based attacks"""
        config = SecurityTestConfig()
        
        unicode_payloads = [
            "admin\\u0000",  # Null byte
            "admin\\u202E",  # Right-to-left override
            "admin\\uFEFF",  # Zero width no-break space
            "admin\\u0009",  # Tab character
            "admin\\u000A",  # Line feed
            "admin\\u000D",  # Carriage return
            "测试用户",         # Chinese characters
            "🚨admin🚨",      # Emoji
            "admin%00",       # URL encoded null
            "admin%2e%2e%2f", # URL encoded ../
        ]
        
        unicode_test_results = []
        
        for payload in unicode_payloads:
            try:
                response = requests.get(
                    f"{config.BASE_URL}/recommendations/?user_id={urllib.parse.quote(payload)}&top_k=5",
                    timeout=10
                )
                
                unicode_test_results.append({
                    "payload": payload,
                    "status_code": response.status_code,
                    "response_length": len(response.text)
                })
                
            except requests.exceptions.RequestException:
                continue
        
        print(f"\\n🔍 Unicode/Encoding Attack Test:")
        print(f"   Tested {len(unicode_test_results)} Unicode payloads:")
        for result in unicode_test_results[:5]:  # Show first 5
            print(f"   - Payload: {result['payload'][:20]}... → Status: {result['status_code']}")