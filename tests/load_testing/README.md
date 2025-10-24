# Load Testing Guide for YouTube Recommendation System

## Overview

This comprehensive load testing suite is designed to evaluate the performance, scalability, and reliability of the YouTube Recommendation System under various load conditions. The testing framework includes API endpoint testing, database performance evaluation, real-time monitoring, and detailed reporting.

## Table of Contents

1. [Architecture](#architecture)
2. [Installation & Setup](#installation--setup)
3. [Quick Start](#quick-start)
4. [Load Testing Scenarios](#load-testing-scenarios)
5. [Monitoring & Metrics](#monitoring--metrics)
6. [Results Interpretation](#results-interpretation)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Architecture

### System Under Test

The YouTube Recommendation System consists of:

- **FastAPI Backend** (Port 8080)

  - `/recommendations` - Main recommendation endpoint
  - `/run-workflow` - Complete recommendation pipeline
  - `/newsletter` - Newsletter generation
  - `/user-vector-update` - User preference updates
  - `/test` - Debug and testing endpoints

- **Databases**
  - **MongoDB** - User feedback and interaction data
  - **Qdrant** - Vector embeddings for similarity search
  - **Supabase** - User profiles and preferences

### Load Testing Components

```
📁 tests/load_testing/
├── 📄 base_load_test.py          # Base classes and utilities
├── 📄 api_endpoint_tests.py      # Locust-based API tests
├── 📄 database_load_tests.py     # Database performance tests
├── 📄 performance_monitor.py     # Real-time monitoring
├── 📄 load_test_scenarios.py     # Test scenarios and runner
└── 📄 load_testing_requirements.txt # Dependencies
```

## Installation & Setup

### 1. Install Dependencies

```powershell
# Install load testing dependencies
pip install -r tests/load_testing_requirements.txt

# Core dependencies include:
# - locust (load testing framework)
# - psutil (system monitoring)
# - matplotlib, seaborn (visualization)
# - faker (test data generation)
```

### 2. Verify System is Running

Ensure your recommendation system is running on `http://localhost:8080`:

```powershell
# Start the system
uvicorn backend.api.main:app --host 0.0.0.0 --port 8080

# Verify health endpoint
curl http://localhost:8080/health
```

### 3. Configure Test Environment

Set environment variables for database connections:

```powershell
# Set environment variables (adjust for your setup)
$env:SUPABASE_URL = "your-supabase-url"
$env:SUPABASE_KEY = "your-supabase-key"
$env:QDRANT_URL = "http://localhost:6333"
$env:MONGODB_CONNECTION = "mongodb://localhost:27017"
```

## Quick Start

### Run Individual Scenario

```powershell
# Navigate to project root
cd "C:\Users\User\Documents\AI\LangGraph"

# Run a quick health check
python tests/load_testing/load_test_scenarios.py --scenario health_check

# Run daily traffic simulation
python tests/load_testing/load_test_scenarios.py --scenario daily_traffic

# Run stress test
python tests/load_testing/load_test_scenarios.py --scenario stress_test
```

### Run Test Suite

```powershell
# Run quick test suite (5-10 minutes)
python tests/load_testing/load_test_scenarios.py --suite quick

# Run comprehensive test suite (1-2 hours)
python tests/load_testing/load_test_scenarios.py --suite comprehensive
```

### Custom Parameters

```powershell
# Override default parameters
python tests/load_testing/load_test_scenarios.py --scenario peak_usage --users 500 --duration 30
```

## Load Testing Scenarios

### Available Scenarios

| Scenario                  | Description                          | Duration | Max Users | Use Case                      |
| ------------------------- | ------------------------------------ | -------- | --------- | ----------------------------- |
| `health_check`            | Quick system responsiveness test     | 2 min    | 10        | Pre-deployment verification   |
| `daily_traffic`           | Normal daily usage patterns          | 15 min   | 100       | Baseline performance          |
| `peak_usage`              | High traffic periods (viral content) | 20 min   | 300       | Capacity planning             |
| `stress_test`             | Beyond normal capacity               | 30 min   | 500       | Breaking point identification |
| `spike_test`              | Sudden traffic increases             | 10 min   | 200       | Auto-scaling validation       |
| `endurance_test`          | Sustained load over time             | 60 min   | 150       | Memory leak detection         |
| `database_focused`        | Database performance testing         | 20 min   | 100       | Database optimization         |
| `recommendation_pipeline` | Full workflow testing                | 25 min   | 200       | Pipeline performance          |
| `user_feedback_heavy`     | High feedback volume                 | 15 min   | 150       | Feedback system testing       |
| `newsletter_generation`   | Newsletter system load               | 10 min   | 50        | Newsletter performance        |

### Test Suites

| Suite           | Scenarios                                                 | Duration   | Purpose              |
| --------------- | --------------------------------------------------------- | ---------- | -------------------- |
| `quick`         | health_check, daily_traffic                               | ~20 min    | Fast validation      |
| `standard`      | health_check, daily_traffic, peak_usage, database_focused | ~60 min    | Regular testing      |
| `comprehensive` | All main scenarios                                        | ~2 hours   | Full evaluation      |
| `performance`   | daily_traffic, peak_usage, stress_test, endurance_test    | ~2.5 hours | Performance analysis |
| `stability`     | endurance_test, spike_test, user_feedback_heavy           | ~1.5 hours | Stability testing    |

## Monitoring & Metrics

### Real-Time Monitoring

The system automatically monitors:

- **System Metrics**

  - CPU usage (%)
  - Memory usage (%)
  - Network I/O
  - Disk I/O

- **Application Metrics**

  - Response times (p50, p90, p95, p99)
  - Request throughput
  - Error rates by endpoint
  - Concurrent user counts

- **Database Metrics**
  - Connection pool usage
  - Query response times
  - Operations per second
  - Error rates by database

### Alert Thresholds

Default alert thresholds (configurable):

```python
alert_thresholds = {
    "cpu_percent": 80.0,
    "memory_percent": 85.0,
    "response_time_p95": 5000.0,  # 5 seconds
    "error_rate": 10.0  # 10%
}
```

### Visualization

Automatic generation of:

- System metrics over time
- Response time trends
- Error rate analysis
- Concurrent user patterns

Charts saved to `load_test_charts/` directory.

## Results Interpretation

### Performance Metrics

#### Response Times

- **Good**: p95 < 1000ms, p99 < 2000ms
- **Acceptable**: p95 < 2000ms, p99 < 5000ms
- **Poor**: p95 > 5000ms

#### Error Rates

- **Excellent**: < 0.1%
- **Good**: < 1%
- **Acceptable**: < 5%
- **Poor**: > 10%

#### System Resources

- **CPU**: Should stay < 80% under normal load
- **Memory**: Should stay < 85% to avoid OOM
- **Growth**: Watch for memory leaks in endurance tests

### Database Performance

#### MongoDB

- **Insert/Update**: < 100ms p95
- **Queries**: < 50ms p95
- **Aggregations**: < 500ms p95

#### Qdrant (Vector DB)

- **Vector Search**: < 200ms p95
- **Batch Operations**: < 1000ms p95

#### Supabase

- **Simple Queries**: < 100ms p95
- **Complex Queries**: < 500ms p95

### Example Good Results

```json
{
  "system_metrics": {
    "cpu": { "avg": 45.2, "max": 68.7 },
    "memory": { "avg": 62.1, "max": 78.3 }
  },
  "application_metrics": {
    "total_requests": 15420,
    "overall_error_rate": 0.12,
    "endpoints": {
      "/recommendations/": {
        "response_times": {
          "avg": 245,
          "p95": 890,
          "p99": 1540
        }
      }
    }
  }
}
```

## Best Practices

### Before Running Tests

1. **Environment Consistency**

   ```powershell
   # Use same environment for all tests
   # Document system specs (CPU, RAM, network)
   # Ensure databases are properly sized
   ```

2. **Baseline Establishment**

   ```powershell
   # Run health_check first
   # Establish baseline with daily_traffic scenario
   # Document baseline performance metrics
   ```

3. **System Preparation**
   ```powershell
   # Clear logs and temporary files
   # Restart services to clear memory
   # Ensure adequate disk space for results
   ```

### During Tests

1. **Monitor System Health**

   - Watch for alerts
   - Check system resource usage
   - Monitor database connections

2. **Don't Interfere**
   - Avoid other resource-intensive operations
   - Don't restart services during tests
   - Let tests complete naturally

### After Tests

1. **Results Analysis**

   ```powershell
   # Review HTML reports
   # Analyze performance charts
   # Check alert logs
   # Compare against baselines
   ```

2. **Action Items**
   - Document performance issues
   - Create optimization tasks
   - Plan capacity upgrades if needed
   - Update monitoring thresholds

## Troubleshooting

### Common Issues

#### "Connection Refused" Errors

```powershell
# Check if system is running
curl http://localhost:8080/health

# Check port availability
netstat -an | findstr :8080
```

#### High Error Rates

1. Check application logs
2. Verify database connections
3. Check for rate limiting
4. Review system resources

#### Tests Timing Out

1. Reduce user count
2. Increase timeout values
3. Check system resources
4. Verify network connectivity

#### Memory Issues During Tests

1. Monitor with Task Manager
2. Check for memory leaks
3. Reduce concurrent users
4. Increase available memory

### Performance Optimization Tips

#### API Endpoints

- Implement caching for frequently accessed data
- Use database connection pooling
- Add request/response compression
- Implement rate limiting

#### Database Optimization

- Add appropriate indexes
- Use database connection pooling
- Monitor slow queries
- Implement query caching

#### System Optimization

- Use asynchronous processing where possible
- Implement horizontal scaling
- Use load balancers
- Monitor and optimize resource usage

### Advanced Configuration

#### Custom Scenarios

```python
# Create custom scenario in load_test_scenarios.py
custom_scenario = LoadTestScenario(
    name="Custom Test",
    description="Your custom test description",
    duration_minutes=15,
    max_users=200,
    spawn_rate=10.0,
    locust_file="api_endpoint_tests.py"
)
```

#### Environment-Specific Settings

```python
# Modify base_load_test.py
@dataclass
class LoadTestConfig:
    base_url: str = "http://your-environment:8080"
    # Adjust other parameters as needed
```

## Continuous Integration

### Automated Testing

Add to your CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run Load Tests
  run: |
    python tests/load_testing/load_test_scenarios.py --suite quick

- name: Upload Results
  uses: actions/upload-artifact@v2
  with:
    name: load-test-results
    path: load_test_results/
```

### Performance Regression Detection

Compare results over time to detect performance regressions:

```python
# Example regression check
if current_p95 > baseline_p95 * 1.2:  # 20% degradation
    raise Exception("Performance regression detected!")
```

## Support

For issues or questions:

1. Check this documentation
2. Review error logs in `load_test_results/`
3. Examine system metrics and alerts
4. Consider reducing load parameters for debugging

## Appendix

### File Structure

```
tests/load_testing/
├── base_load_test.py              # Framework foundation
├── api_endpoint_tests.py          # API testing scenarios
├── database_load_tests.py         # Database performance tests
├── performance_monitor.py         # Real-time monitoring
├── load_test_scenarios.py         # Test runner and scenarios
├── load_testing_requirements.txt  # Dependencies
└── README.md                      # This documentation

Generated during tests:
├── load_test_results/             # Test results and reports
├── load_test_charts/              # Performance visualizations
└── *.json                         # Raw data files
```

### Dependencies

- **locust**: Web-based load testing framework
- **psutil**: System and process monitoring
- **matplotlib/seaborn**: Chart generation
- **faker**: Realistic test data generation
- **requests**: HTTP client for testing

### Recommended System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: 4 cores minimum for high-load tests
- **Disk**: 10GB free space for results
- **Network**: Stable connection to target system

---

_This load testing framework provides comprehensive performance evaluation for the YouTube Recommendation System. Regular testing helps ensure system reliability and performance under various load conditions._
