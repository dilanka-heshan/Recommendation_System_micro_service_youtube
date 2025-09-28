"""
Database Configuration Tests
Tests database client configurations and connection handling
"""
import pytest
import os
from unittest.mock import patch, MagicMock
from backend.database.mongodb_client import MongoDBClient
from backend.database.qdrant_client import QdrantVectorClient
from backend.database.supabase_client import SupabaseClient


class TestMongoDBConfiguration:
    """Test MongoDB client configuration scenarios"""
    
    def test_mongodb_with_valid_connection_string(self, mock_env_vars):
        """Test MongoDB client with valid connection string"""
        with patch.dict(os.environ, mock_env_vars):
            with patch('backend.database.mongodb_client.MongoClient') as mock_client:
                mock_client.return_value = MagicMock()
                client = MongoDBClient()
                assert client.client is not None
                mock_client.assert_called_once()
    
    def test_mongodb_with_missing_connection_string(self, clean_env):
        """Test MongoDB client behavior with missing connection string"""
        client = MongoDBClient()
        assert client.client is None
    
    def test_mongodb_with_invalid_connection_string(self, clean_env):
        """Test MongoDB client with malformed connection string"""
        with patch.dict(os.environ, {"MONGODB_CONNECTION_STRING": "invalid://connection"}):
            with patch('backend.database.mongodb_client.MongoClient') as mock_client:
                mock_client.side_effect = Exception("Connection failed")
                client = MongoDBClient()
                assert client.client is None
    
    def test_mongodb_connection_string_variations(self):
        """Test different MongoDB connection string formats"""
        valid_formats = [
            "mongodb://localhost:27017/testdb",
            "mongodb://user:pass@localhost:27017/testdb",
            "mongodb+srv://cluster.mongodb.net/testdb",
            "mongodb://localhost:27017,localhost:27018/testdb"
        ]
        
        for conn_str in valid_formats:
            with patch('backend.database.mongodb_client.MongoClient') as mock_client:
                mock_client.return_value = MagicMock()
                client = MongoDBClient(conn_str)
                assert client.client is not None


class TestQdrantConfiguration:
    """Test Qdrant vector database configuration scenarios"""
    
    def test_qdrant_with_default_config(self, mock_env_vars):
        """Test Qdrant client with default configuration"""
        with patch.dict(os.environ, mock_env_vars):
            with patch('backend.database.qdrant_client.QdrantClient') as mock_client:
                mock_client.return_value = MagicMock()
                client = QdrantVectorClient()
                assert client.client is not None
    
    def test_qdrant_with_missing_host(self, clean_env):
        """Test Qdrant client with missing host configuration"""
        # Should use default localhost
        with patch('backend.database.qdrant_client.QdrantClient') as mock_client:
            mock_client.return_value = MagicMock()
            client = QdrantVectorClient()
            # Verify it uses default host
            mock_client.assert_called_with(host="localhost", port=6333, api_key=None)
    
    def test_qdrant_with_api_key(self, mock_env_vars):
        """Test Qdrant client with API key authentication"""
        with patch.dict(os.environ, mock_env_vars):
            with patch('backend.database.qdrant_client.QdrantClient') as mock_client:
                mock_client.return_value = MagicMock()
                client = QdrantVectorClient()
                mock_client.assert_called_with(
                    host="localhost", 
                    port=6333, 
                    api_key="test_qdrant_key"
                )
    
    def test_qdrant_port_variations(self):
        """Test different Qdrant port configurations"""
        test_ports = ["6333", "6334", "8080"]
        
        for port in test_ports:
            # Clear any existing QDRANT_API_KEY to ensure api_key=None
            env_vars = {"QDRANT_HOST": "localhost", "QDRANT_PORT": port}
            with patch.dict(os.environ, env_vars, clear=False):
                # Explicitly remove API key if it exists
                os.environ.pop("QDRANT_API_KEY", None)
                
                with patch('backend.database.qdrant_client.QdrantClient') as mock_client:
                    mock_client.return_value = MagicMock()
                    mock_client.return_value.get_collections.return_value = MagicMock(collections=[])
                    client = QdrantVectorClient()
                    mock_client.assert_called_with(
                        host="localhost", 
                        port=int(port), 
                        api_key=None
                    )
    
    def test_qdrant_invalid_port(self):
        """Test Qdrant client with invalid port configuration"""
        # Clear any existing API key
        env_vars = {"QDRANT_HOST": "localhost", "QDRANT_PORT": "invalid_port"}
        with patch.dict(os.environ, env_vars, clear=False):
            os.environ.pop("QDRANT_API_KEY", None)
            
            # The actual implementation catches ValueError and sets client=None
            # rather than re-raising, so test for that behavior
            client = QdrantVectorClient()
            assert client.client is None


class TestSupabaseConfiguration:
    """Test Supabase client configuration scenarios"""
    
    def test_supabase_with_valid_config(self, mock_env_vars):
        """Test Supabase client with valid configuration"""
        with patch.dict(os.environ, mock_env_vars):
            with patch('backend.database.supabase_client.create_client') as mock_create:
                mock_create.return_value = MagicMock()
                client = SupabaseClient()
                assert client.url == "https://test.supabase.co"
                assert client.key == "test_key_123"
                mock_create.assert_called_once_with(
                    "https://test.supabase.co", 
                    "test_key_123"
                )
    
    def test_supabase_with_missing_url(self, clean_env):
        """Test Supabase client with missing URL"""
        with patch.dict(os.environ, {"SUPABASE_ANON_KEY": "test_key"}):
            with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_ANON_KEY must be set"):
                SupabaseClient()
    
    def test_supabase_with_missing_key(self, clean_env):
        """Test Supabase client with missing API key"""
        with patch.dict(os.environ, {"SUPABASE_URL": "https://test.supabase.co"}):
            with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_ANON_KEY must be set"):
                SupabaseClient()
    
    def test_supabase_url_formats(self):
        """Test different Supabase URL formats"""
        valid_urls = [
            "https://project.supabase.co",
            "https://project.supabase.com",
            "https://project-staging.supabase.co",
            "http://localhost:54321"  # Local Supabase
        ]
        
        for url in valid_urls:
            env_vars = {"SUPABASE_URL": url, "SUPABASE_ANON_KEY": "test_key"}
            with patch.dict(os.environ, env_vars):
                with patch('backend.database.supabase_client.create_client') as mock_create:
                    mock_create.return_value = MagicMock()
                    client = SupabaseClient()
                    assert client.url == url
    
    def test_supabase_connection_error_handling(self, mock_env_vars):
        """Test Supabase client connection error handling"""
        with patch.dict(os.environ, mock_env_vars):
            with patch('backend.database.supabase_client.create_client') as mock_create:
                mock_create.side_effect = Exception("Connection failed")
                with pytest.raises(Exception):
                    SupabaseClient()


class TestDatabaseIntegrationConfiguration:
    """Test multi-database configuration scenarios"""
    
    def test_all_databases_configured(self, mock_env_vars):
        """Test scenario where all databases are properly configured"""
        with patch.dict(os.environ, mock_env_vars):
            with patch('backend.database.mongodb_client.MongoClient'), \
                 patch('backend.database.qdrant_client.QdrantClient'), \
                 patch('backend.database.supabase_client.create_client'):
                
                mongodb = MongoDBClient()
                qdrant = QdrantVectorClient()
                supabase = SupabaseClient()
                
                assert mongodb.client is not None
                assert qdrant.client is not None
                assert supabase.client is not None
    
    def test_partial_database_configuration(self, clean_env):
        """Test scenario with partial database configuration"""
        # Only configure Supabase
        partial_env = {
            "SUPABASE_URL": "https://test.supabase.co",
            "SUPABASE_ANON_KEY": "test_key"
        }
        
        with patch.dict(os.environ, partial_env):
            with patch('backend.database.supabase_client.create_client'):
                # MongoDB should fail gracefully
                mongodb = MongoDBClient()
                assert mongodb.client is None
                
                # Qdrant should use defaults
                with patch('backend.database.qdrant_client.QdrantClient'):
                    qdrant = QdrantVectorClient()
                    assert qdrant.client is not None
                
                # Supabase should work
                supabase = SupabaseClient()
                assert supabase.client is not None