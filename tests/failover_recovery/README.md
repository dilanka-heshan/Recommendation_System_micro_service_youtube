# Failover and Recovery Testing

## Overview

This module contains tests for validating system recovery capabilities with minimal configuration requirements.

## Test Categories

### 1. Service Recovery Tests

- API service restart validation
- Background process recovery
- Health check endpoint validation

### 2. Database Connection Recovery Tests

- MongoDB connection failure/recovery
- Qdrant vector database recovery
- Supabase connection pool recovery

### 3. Resource Stress Tests

- Memory pressure testing
- CPU load recovery
- Network timeout handling

### 4. Simple Container Tests

- Docker service restart
- Container health validation

## Running Tests

```bash
# Run all failover recovery tests
python -m pytest tests/failover_recovery/ -v

# Run specific test category
python -m pytest tests/failover_recovery/test_service_recovery.py -v

# Run with HTML report
python tests/failover_recovery/run_failover_tests.py
```

## Test Configuration

All tests are designed to work with minimal configuration and use your existing infrastructure.

## Reports

- HTML reports generated in root directory as `failover_recovery_report.html`
- JSON results available for automated analysis
- Integration with existing test reporting structure
