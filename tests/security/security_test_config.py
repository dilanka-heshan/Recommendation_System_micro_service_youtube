"""
Security and Access Control Testing for YouTube Recommendation System

This module contains comprehensive security tests for the recommendation system,
including authentication, authorization, input validation, and injection attacks.
"""

import pytest
import requests
import json
import asyncio
from typing import Dict, List, Any
import time
import random
import string


class SecurityTestConfig:
    """Configuration for security tests"""
    
    BASE_URL = "http://localhost:8080"
    
    # Test endpoints
    ENDPOINTS = {
        "recommendations": "/recommendations/",
        "run_workflow": "/run-workflow/run-workflow",
        "run_workflow_video_ids": "/run-workflow/run-workflow-video-ids", 
        "user_vector_daily_update": "/user-vector-update/run-daily-update",
        "user_vector_manual_update": "/user-vector-update/run-manual-update",
        "user_vector_status": "/user-vector-update/status",
        "newsletter_health": "/newsletter/health",
        "test_random_videos": "/test/random-videos",
        "test_db_connection": "/test/test-database-connection",
        "test_debug_db": "/test/debug-database",
        "health": "/health"
    }
    
    # SQL injection payloads
    SQL_INJECTION_PAYLOADS = [
        "'; DROP TABLE users; --",
        "' OR '1'='1' --",
        "' UNION SELECT * FROM users --",
        "admin'--",
        "admin'#",
        "' OR 1=1#",
        "' OR 'x'='x",
        "') OR ('1'='1'--",
        "'; EXEC xp_cmdshell('dir')--"
    ]
    
    # NoSQL injection payloads
    NOSQL_INJECTION_PAYLOADS = [
        {"$ne": None},
        {"$gt": ""},
        {"$regex": ".*"},
        {"$where": "function(){return true}"},
        {"$exists": True},
        {"$in": ["admin", "user"]},
        '{"$ne": null}',
        '{"$gt": ""}',
        '{"$where": "1==1"}'
    ]
    
    # XSS payloads
    XSS_PAYLOADS = [
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "<img src=x onerror=alert('XSS')>",
        "'>alert('XSS')<",
        '"><script>alert("XSS")</script>',
        "<svg onload=alert('XSS')>",
        "';alert('XSS');//"
    ]
    
    # Command injection payloads
    COMMAND_INJECTION_PAYLOADS = [
        "; ls -la",
        "| whoami",
        "&& dir",
        "; cat /etc/passwd",
        "| net user",
        "; id",
        "&& echo vulnerable"
    ]
    
    # Test user data
    TEST_USERS = [
        "test_user_1",
        "test_user_2", 
        "admin_user",
        "guest_user"
    ]
    
    # Malicious user IDs for testing
    MALICIOUS_USER_IDS = [
        "../../../etc/passwd",
        "..\\..\\windows\\system32\\config\\sam",
        "NULL",
        "undefined",
        "0",
        "-1",
        "999999999",
        " ",
        "",
        "admin",
        "root",
        "system"
    ]


def generate_random_string(length: int = 10) -> str:
    """Generate random string for testing"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def is_server_running(base_url: str) -> bool:
    """Check if the server is running"""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


@pytest.fixture(scope="session", autouse=True)
def check_server():
    """Ensure server is running before tests"""
    if not is_server_running(SecurityTestConfig.BASE_URL):
        pytest.skip("Server is not running. Start the server with: python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8080")