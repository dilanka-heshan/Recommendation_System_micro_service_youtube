# Database Integrity Testing Guide

This guide explains how to run the comprehensive database integrity tests for the LangGraph YouTube recommendation system.

## Prerequisites

1. **Python Environment**: Ensure you have Python 3.8+ installed
2. **Dependencies**: Install required packages
3. **Database Access**: Ensure connectivity to Supabase, Qdrant, and MongoDB
4. **Environment Variables**: Configure database connection settings

## Installation

```bash
# Install pytest and required dependencies
pip install pytest pytest-html pytest-cov python-dotenv

# Install your project dependencies
pip install -r requirements.txt
```

## Environment Setup

Create a `.env` file in your project root with database connection details:

```env
# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key

# Qdrant Configuration
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# MongoDB Configuration
MONGODB_URI=your_mongodb_connection_string
MONGODB_DATABASE=your_database_name
```

## Running Tests

### 1. Run All Database Integrity Tests

```bash
# Run all tests in the database_integrity directory
pytest tests/database_integrity/ -v

# Run with more detailed output
pytest tests/database_integrity/ -v --tb=short

# Run with coverage reporting
pytest tests/database_integrity/ --cov=backend --cov-report=html
```

### 2. Run Specific Test Categories

```bash
# Cross-database integrity tests
pytest -m cross_database -v

# Data consistency tests
pytest -m data_consistency -v

# Orphaned records detection
pytest -m orphaned_records -v

# Constraint validation tests
pytest -m constraint_validation -v

# Data freshness tests
pytest -m data_freshness -v

# Health monitoring tests
pytest -m health_monitoring -v

# Referential integrity tests
pytest -m referential_integrity -v

# General database integrity tests
pytest -m database_integrity -v
```

### 3. Run Specific Test Files

```bash
# Cross-database integrity
pytest tests/database_integrity/test_cross_database_integrity.py -v

# Data consistency
pytest tests/database_integrity/test_data_consistency.py -v

# Orphaned records
pytest tests/database_integrity/test_orphaned_records.py -v

# Constraint validation
pytest tests/database_integrity/test_constraint_validation.py -v

# Data freshness
pytest tests/database_integrity/test_data_freshness.py -v

# Health monitoring
pytest tests/database_integrity/test_health_monitoring.py -v
```

### 4. Run Specific Test Methods

```bash
# Run a specific test method
pytest tests/database_integrity/test_cross_database_integrity.py::TestCrossDatabaseIntegrity::test_video_id_consistency -v

# Run multiple specific tests
pytest tests/database_integrity/test_data_consistency.py::TestDataConsistency::test_user_embedding_consistency tests/database_integrity/test_orphaned_records.py::TestOrphanedRecords::test_orphaned_feedback_records -v
```

## Advanced Testing Options

### 1. Generate HTML Report

```bash
# Generate detailed HTML report
pytest tests/database_integrity/ --html=reports/database_integrity_report.html --self-contained-html
```

### 2. Run Tests with Custom Configuration

```bash
# Run tests with custom pytest configuration
pytest tests/database_integrity/ -c pytest.ini -v

# Run tests and stop on first failure
pytest tests/database_integrity/ -x

# Run tests with maximum verbosity
pytest tests/database_integrity/ -vvv
```

### 3. Parallel Test Execution

```bash
# Install pytest-xdist for parallel execution
pip install pytest-xdist

# Run tests in parallel (4 processes)
pytest tests/database_integrity/ -n 4
```

### 4. Continuous Integration

```bash
# Run tests with JUnit XML output (for CI/CD)
pytest tests/database_integrity/ --junitxml=reports/junit.xml

# Run tests with coverage and XML output
pytest tests/database_integrity/ --cov=backend --cov-report=xml --junitxml=reports/junit.xml
```

## Test Output Understanding

### Success Output Example:

```
tests/database_integrity/test_cross_database_integrity.py::TestCrossDatabaseIntegrity::test_video_id_consistency PASSED [25%]
tests/database_integrity/test_data_consistency.py::TestDataConsistency::test_user_embedding_consistency PASSED [50%]
...
================================ 20 passed in 45.67s ================================
```

### Failure Output Example:

```
tests/database_integrity/test_cross_database_integrity.py::TestCrossDatabaseIntegrity::test_video_id_consistency FAILED [25%]

FAILURES:
================================ test_video_id_consistency ================================
>       assert consistency_score >= 95.0, f"Video ID consistency too low: {consistency_score}%"
E       AssertionError: Video ID consistency too low: 87.5%
```

## Interpreting Test Results

### Test Status Meanings:

- **PASSED**: Test completed successfully, all assertions passed
- **FAILED**: Test completed but assertions failed (data integrity issues found)
- **SKIPPED**: Test was skipped (usually due to missing data or dependencies)
- **ERROR**: Test encountered an error during execution

### Health Monitoring Reports:

The health monitoring tests generate detailed reports including:

- Overall system status (healthy/warning/critical)
- Individual metric scores
- Specific recommendations for issues found
- Historical trend data

## Troubleshooting

### Common Issues:

1. **Database Connection Errors**:

   ```bash
   # Test database connectivity
   python -c "from backend.database.supabase_client import get_supabase_client; print('Supabase OK')"
   python -c "from backend.database.qdrant_client import get_qdrant_client; print('Qdrant OK')"
   python -c "from backend.database.mongodb_client import get_mongodb_client; print('MongoDB OK')"
   ```

2. **Missing Dependencies**:

   ```bash
   # Install missing packages
   pip install supabase qdrant-client pymongo sentence-transformers
   ```

3. **Environment Variables Not Set**:

   ```bash
   # Check environment variables
   python -c "import os; print('SUPABASE_URL:', os.getenv('SUPABASE_URL'))"
   ```

4. **Permission Issues**:
   - Ensure database users have read permissions
   - Check API key validity and permissions

## Continuous Monitoring

### Schedule Regular Tests:

```bash
# Create a script to run tests daily
#!/bin/bash
cd /path/to/your/project
pytest tests/database_integrity/ --html=reports/daily_$(date +%Y%m%d).html
```

### Integration with CI/CD:

Add to your GitHub Actions or similar CI/CD pipeline:

```yaml
- name: Run Database Integrity Tests
  run: |
    pytest tests/database_integrity/ --junitxml=reports/junit.xml --html=reports/integrity_report.html
```

## Best Practices

1. **Run tests regularly** (daily/weekly) to catch issues early
2. **Monitor test execution time** - slow tests may indicate database performance issues
3. **Review failure patterns** - recurring failures may indicate systemic issues
4. **Keep test data limits reasonable** to avoid long execution times
5. **Use markers effectively** to run targeted test subsets
6. **Document any test skips** and their reasons
7. **Archive test reports** for historical analysis

## Support

If you encounter issues:

1. Check the test logs for specific error messages
2. Verify database connectivity and permissions
3. Ensure all dependencies are installed
4. Review the test configuration in `pytest.ini` and `conftest.py`
5. Check if test data meets minimum requirements (some tests skip if insufficient data)
