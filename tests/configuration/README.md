# Configuration Testing Suite

This directory contains comprehensive configuration tests for the YouTube Recommendation System.

## Overview

Configuration testing validates that the system behaves correctly with different configuration settings, environment variables, and deployment configurations.

## Test Categories

### 1. Database Configuration Tests (`test_database_config.py`)

- **MongoDB Configuration**: Connection strings, authentication, error handling
- **Qdrant Vector DB**: Host/port configurations, API keys, defaults
- **Supabase Configuration**: URL validation, key authentication, error handling
- **Multi-database Integration**: Testing all databases together

### 2. Environment Variable Tests (`test_environment_config.py`)

- **Required Variables**: Validation of mandatory environment variables
- **Optional Variables**: Fallback behavior for optional configurations
- **Type Validation**: Proper type conversion and validation
- **Edge Cases**: Empty values, special characters, long strings
- **Security**: Sensitive data handling and validation

### 3. Service Configuration Tests (`test_service_config.py`)

- **FastAPI Configuration**: Application setup, health checks, routers
- **Docker Configuration**: Container settings, port mapping, health checks
- **Pipeline Configuration**: Recommendation settings, vector configurations
- **Error Handling**: Logging, retry mechanisms, error responses

### 4. Integration Configuration Tests (`test_integration_config.py`)

- **Full System Scenarios**: Complete configuration testing
- **Failure Scenarios**: Graceful degradation and recovery
- **Performance Impact**: Timeout and connection pool testing
- **Security Scenarios**: API key rotation, access control
- **Monitoring**: Configuration change detection and validation

## Running Tests

### Run All Configuration Tests

```bash
python tests/configuration/run_config_tests.py
```

### Run Quick Tests (Basic Validation)

```bash
python tests/configuration/run_config_tests.py --quick
```

### Run Specific Category

```bash
python tests/configuration/run_config_tests.py --category database
python tests/configuration/run_config_tests.py --category environment
python tests/configuration/run_config_tests.py --category service
python tests/configuration/run_config_tests.py --category integration
```

### Run Individual Test Files

```bash
# Database configuration tests
pytest tests/configuration/test_database_config.py -v

# Environment variable tests
pytest tests/configuration/test_environment_config.py -v

# Service configuration tests
pytest tests/configuration/test_service_config.py -v

# Integration tests
pytest tests/configuration/test_integration_config.py -v
```

## Test Configuration

### Environment Variables Tested

- `SUPABASE_URL` - Supabase database URL (required)
- `SUPABASE_ANON_KEY` - Supabase anonymous key (required)
- `QDRANT_HOST` - Qdrant vector database host (optional, default: localhost)
- `QDRANT_PORT` - Qdrant port (optional, default: 6333)
- `QDRANT_API_KEY` - Qdrant API key (optional)
- `MONGODB_CONNECTION_STRING` - MongoDB connection string (optional)
- `LANGCHAIN_TRACING_V2` - LangChain tracing (optional)
- `LANGCHAIN_ENDPOINT` - LangChain endpoint (optional)
- `LANGCHAIN_API_KEY` - LangChain API key (optional)
- `LANGCHAIN_PROJECT` - LangChain project (optional)

### Test Scenarios Covered

#### ✅ Positive Test Cases

- Valid configurations work correctly
- Default values are applied properly
- All supported configuration combinations
- Proper type conversion and validation

#### ❌ Negative Test Cases

- Missing required configurations
- Invalid configuration formats
- Malformed connection strings
- Out-of-range values

#### 🔍 Edge Cases

- Empty configuration values
- Extremely long configuration strings
- Special characters in configurations
- Unicode characters in values

#### 🚨 Error Scenarios

- Database connection failures
- Service unavailability
- Configuration conflicts
- Partial service degradation

## Test Features

### Mocking and Isolation

- Database connections are mocked to avoid external dependencies
- Environment variables are isolated per test
- Clean test environment setup and teardown

### Comprehensive Coverage

- Unit-level configuration validation
- Integration-level system testing
- Performance impact testing
- Security configuration validation

### Realistic Scenarios

- Docker container configurations
- Multi-environment testing (dev, staging, prod)
- Service degradation and recovery
- Configuration change impact

## Integration with CI/CD

These tests are designed to be integrated into your CI/CD pipeline:

1. **Pre-deployment**: Validate configurations before deployment
2. **Environment Testing**: Test configurations across different environments
3. **Regression Testing**: Ensure configuration changes don't break functionality
4. **Security Validation**: Check for configuration security issues

## Best Practices Validated

- ✅ Configuration as Code
- ✅ Environment Variable Usage
- ✅ Secure Secret Management
- ✅ Graceful Error Handling
- ✅ Default Value Provision
- ✅ Configuration Validation
- ✅ Service Isolation
- ✅ Monitoring and Alerting

## Expected Test Results

When all tests pass, you should see:

```
🎉 All configuration tests passed!
Overall: 4/4 test categories passed
```

This indicates that your system's configuration handling is robust and production-ready.
