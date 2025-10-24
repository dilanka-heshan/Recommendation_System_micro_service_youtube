"""
Configuration Testing Suite
Tests configuration validation, environment variables, and service setup
"""
import pytest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment for testing
load_dotenv()

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing"""
    return {
        "SUPABASE_URL": "https://test.supabase.co",
        "SUPABASE_ANON_KEY": "test_key_123",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "QDRANT_API_KEY": "test_qdrant_key",
        "MONGODB_CONNECTION_STRING": "mongodb://localhost:27017/test_db",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
        "LANGCHAIN_API_KEY": "test_langchain_key",
        "LANGCHAIN_PROJECT": "test_project"
    }

@pytest.fixture
def clean_env():
    """Clean environment for testing"""
    original_env = os.environ.copy()
    # Clear all test-related env vars
    test_vars = [
        "SUPABASE_URL", "SUPABASE_ANON_KEY", "QDRANT_HOST", "QDRANT_PORT", 
        "QDRANT_API_KEY", "MONGODB_CONNECTION_STRING", "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_ENDPOINT", "LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"
    ]
    for var in test_vars:
        os.environ.pop(var, None)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture
def temp_config_file():
    """Create temporary configuration file for testing"""
    config_data = {
        "database": {
            "mongodb": {"timeout": 30, "max_pool_size": 10},
            "qdrant": {"timeout": 30, "vector_size": 768},
            "supabase": {"timeout": 30, "max_connections": 10}
        },
        "api": {
            "host": "0.0.0.0",
            "port": 8080,
            "workers": 1,
            "timeout": 60
        },
        "pipeline": {
            "max_recommendations": 20,
            "similarity_threshold": 0.7,
            "diversity_weight": 0.3
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    os.unlink(temp_file)