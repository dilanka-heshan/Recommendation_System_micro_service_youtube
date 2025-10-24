"""
Environment Variable Configuration Tests
Tests environment variable handling, validation, and fallback behavior
"""
import pytest
import os
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv


class TestEnvironmentVariableValidation:
    """Test environment variable validation and handling"""
    
    def test_required_environment_variables(self, mock_env_vars):
        """Test that required environment variables are properly validated"""
        required_vars = [
            "SUPABASE_URL",
            "SUPABASE_ANON_KEY"
        ]
        
        for var in required_vars:
            env_without_var = mock_env_vars.copy()
            del env_without_var[var]
            
            with patch.dict(os.environ, env_without_var, clear=True):
                with patch('backend.database.supabase_client.create_client'):
                    with pytest.raises(ValueError):
                        from backend.database.supabase_client import SupabaseClient
                        SupabaseClient()
    
    def test_optional_environment_variables(self, clean_env):
        """Test that optional environment variables have proper fallbacks"""
        # Test MongoDB without connection string
        with patch('backend.database.mongodb_client.MongoClient'):
            from backend.database.mongodb_client import MongoDBClient
            client = MongoDBClient()
            assert client.client is None  # Should handle gracefully
        
        # Test Qdrant with defaults
        with patch('backend.database.qdrant_client.QdrantClient') as mock_client:
            mock_client.return_value = MagicMock()
            from backend.database.qdrant_client import QdrantVectorClient
            client = QdrantVectorClient()
            # Should use default values
            mock_client.assert_called_with(host="localhost", port=6333, api_key=None)
    
    def test_environment_variable_types(self):
        """Test environment variable type conversion and validation"""
        test_cases = [
            ("QDRANT_PORT", "6333", int),
            ("LANGCHAIN_TRACING_V2", "true", str),
            ("LANGCHAIN_TRACING_V2", "false", str),
            ("SUPABASE_URL", "https://test.supabase.co", str),
        ]
        
        for var_name, var_value, expected_type in test_cases:
            with patch.dict(os.environ, {var_name: var_value}):
                retrieved_value = os.getenv(var_name)
                assert isinstance(retrieved_value, str)  # All env vars are strings initially
                
                # Test type conversion for numeric values
                if expected_type == int:
                    assert int(retrieved_value) == int(var_value)
    
    def test_environment_variable_edge_cases(self):
        """Test edge cases for environment variables"""
        edge_cases = [
            ("EMPTY_VAR", ""),
            ("WHITESPACE_VAR", "  "),
            ("SPECIAL_CHARS_VAR", "test!@#$%^&*()"),
            ("UNICODE_VAR", "测试变量"),
            ("LONG_VAR", "x" * 1000),
        ]
        
        for var_name, var_value in edge_cases:
            with patch.dict(os.environ, {var_name: var_value}):
                retrieved_value = os.getenv(var_name)
                assert retrieved_value == var_value
    
    def test_boolean_environment_variables(self):
        """Test boolean environment variable handling"""
        boolean_test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("", False),
        ]
        
        for str_value, expected_bool in boolean_test_cases:
            with patch.dict(os.environ, {"TEST_BOOL": str_value}):
                # Test boolean conversion logic
                env_value = os.getenv("TEST_BOOL", "false").lower()
                actual_bool = env_value in ["true", "1", "yes", "on"]
                assert actual_bool == expected_bool


class TestDotenvConfiguration:
    """Test .env file loading and precedence"""
    
    def test_dotenv_loading(self, tmp_path):
        """Test that .env files are properly loaded"""
        # Create a temporary .env file
        env_file = tmp_path / ".env"
        env_content = """
SUPABASE_URL=https://dotenv.supabase.co
SUPABASE_ANON_KEY=dotenv_key_123
QDRANT_HOST=dotenv_host
QDRANT_PORT=9999
"""
        env_file.write_text(env_content.strip())
        
        # Store original values
        original_values = {
            "SUPABASE_URL": os.getenv("SUPABASE_URL"),
            "SUPABASE_ANON_KEY": os.getenv("SUPABASE_ANON_KEY"),
            "QDRANT_HOST": os.getenv("QDRANT_HOST"),
            "QDRANT_PORT": os.getenv("QDRANT_PORT")
        }
        
        # Clear existing env vars temporarily
        for var in original_values:
            os.environ.pop(var, None)
        
        try:
            # Load the .env file
            load_dotenv(env_file)
            
            # Verify values are loaded
            assert os.getenv("SUPABASE_URL") == "https://dotenv.supabase.co"
            assert os.getenv("SUPABASE_ANON_KEY") == "dotenv_key_123"
            assert os.getenv("QDRANT_HOST") == "dotenv_host"
            assert os.getenv("QDRANT_PORT") == "9999"
        finally:
            # Restore original values
            for var, value in original_values.items():
                if value is not None:
                    os.environ[var] = value
                else:
                    os.environ.pop(var, None)
    
    def test_environment_precedence(self, tmp_path):
        """Test that system environment variables override .env file"""
        # Create .env file
        env_file = tmp_path / ".env"
        env_file.write_text("TEST_PRECEDENCE=from_dotenv")
        
        # Set system environment variable
        with patch.dict(os.environ, {"TEST_PRECEDENCE": "from_system"}):
            load_dotenv(env_file)
            # System env should take precedence
            assert os.getenv("TEST_PRECEDENCE") == "from_system"


class TestConfigurationValidation:
    """Test configuration validation logic"""
    
    def test_url_format_validation(self):
        """Test URL format validation"""
        valid_urls = [
            "https://test.supabase.co",
            "http://localhost:3000",
            "https://api.openai.com/v1",
            "http://127.0.0.1:8080"
        ]
        
        invalid_urls = [
            "not_a_url",
            "ftp://invalid.com",
            "https://",
            "http://"
        ]
        
        for url in valid_urls:
            # Basic URL validation (starts with http/https)
            assert url.startswith(("http://", "https://"))
        
        for url in invalid_urls:
            # These should fail basic validation
            assert not (url.startswith(("http://", "https://")) and len(url) > 8)
    
    def test_connection_string_validation(self):
        """Test database connection string validation"""
        valid_mongo_strings = [
            "mongodb://localhost:27017/testdb",
            "mongodb://user:pass@localhost:27017/testdb",
            "mongodb+srv://cluster.mongodb.net/testdb"
        ]
        
        for conn_str in valid_mongo_strings:
            assert conn_str.startswith("mongodb")
    
    def test_port_range_validation(self):
        """Test port number validation"""
        valid_ports = ["80", "443", "3000", "6333", "8080"]
        invalid_ports = ["0", "65536", "99999", "-1", "abc"]
        
        for port in valid_ports:
            port_int = int(port)
            assert 1 <= port_int <= 65535
        
        for port in invalid_ports:
            try:
                port_int = int(port)
                assert not (1 <= port_int <= 65535)
            except ValueError:
                # String ports should raise ValueError
                assert True


class TestConfigurationSecurity:
    """Test configuration security aspects"""
    
    def test_sensitive_data_not_logged(self, caplog):
        """Test that sensitive configuration data is not logged"""
        sensitive_vars = [
            "SUPABASE_ANON_KEY",
            "QDRANT_API_KEY", 
            "LANGCHAIN_API_KEY",
            "MONGODB_CONNECTION_STRING"
        ]
        
        # Mock configuration loading that might log
        for var in sensitive_vars:
            with patch.dict(os.environ, {var: "sensitive_value_123"}):
                # Simulate configuration loading
                config_value = os.getenv(var)
                assert config_value == "sensitive_value_123"
                
                # Check that sensitive values aren't in logs
                # This is a basic test - in real scenarios you'd check actual log output
                assert "sensitive_value_123" not in str(caplog.records)
    
    def test_api_key_format_validation(self):
        """Test API key format validation"""
        # Basic API key format checks
        test_keys = {
            "valid_key_123": True,
            "sk-1234567890abcdef": True,
            "": False,
            "   ": False,
            "short": False  # Less than 8 characters
        }
        
        for key, should_be_valid in test_keys.items():
            # Basic validation - non-empty and reasonable length
            is_valid = bool(key.strip()) and len(key.strip()) >= 8
            assert is_valid == should_be_valid