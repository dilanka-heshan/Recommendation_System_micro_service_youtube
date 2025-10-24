"""
Performance Monitoring and Metrics Collection for Load Testing

This module provides comprehensive monitoring capabilities during load testing:
- Real-time performance metrics collection
- Resource utilization monitoring (CPU, memory, network)
- Response time analysis and alerting
- Custom metrics dashboard generation
- Integration with existing monitoring tools

Usage:
    from performance_monitor import PerformanceMonitor
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    # Run your load tests
    results = monitor.stop_monitoring_and_generate_report()
"""

import time
import psutil
import threading
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring"""
    collection_interval: float = 1.0  # seconds
    max_data_points: int = 10000
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_percent": 80.0,
        "memory_percent": 97.0,  # Raised for high-memory systems
        "response_time_p95": 5000.0,  # 5 seconds
        "error_rate": 10.0  # 10%
    })
    enable_alerts: bool = True
    save_raw_data: bool = True

class SystemMetrics:
    """Collects system-level performance metrics"""
    
    def __init__(self):
        self.cpu_usage = deque(maxlen=1000)
        self.memory_usage = deque(maxlen=1000)
        self.network_io = deque(maxlen=1000)
        self.disk_io = deque(maxlen=1000)
        self.timestamps = deque(maxlen=1000)
    
    def collect_metrics(self):
        """Collect current system metrics"""
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Network I/O
        net_io = psutil.net_io_counters()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        
        # Store metrics
        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(memory.percent)
        self.timestamps.append(timestamp)
        
        if net_io:
            self.network_io.append({
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            })
        
        if disk_io:
            self.disk_io.append({
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes
            })
        
        return {
            'timestamp': timestamp,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'network_io': net_io._asdict() if net_io else None,
            'disk_io': disk_io._asdict() if disk_io else None
        }

class ApplicationMetrics:
    """Collects application-specific performance metrics"""
    
    def __init__(self):
        self.response_times = defaultdict(deque)
        self.request_counts = defaultdict(int)
        self.error_counts = defaultdict(int)
        self.concurrent_users = deque(maxlen=1000)
        self.throughput = deque(maxlen=1000)
        self.custom_metrics = defaultdict(deque)
    
    def add_response_time(self, endpoint: str, response_time: float):
        """Add response time measurement"""
        self.response_times[endpoint].append({
            'time': response_time,
            'timestamp': time.time()
        })
        if len(self.response_times[endpoint]) > 1000:
            self.response_times[endpoint].popleft()
    
    def increment_request_count(self, endpoint: str):
        """Increment request counter"""
        self.request_counts[endpoint] += 1
    
    def increment_error_count(self, endpoint: str):
        """Increment error counter"""
        self.error_counts[endpoint] += 1
    
    def update_concurrent_users(self, count: int):
        """Update concurrent user count"""
        self.concurrent_users.append({
            'count': count,
            'timestamp': time.time()
        })
    
    def add_custom_metric(self, name: str, value: float):
        """Add custom application metric"""
        self.custom_metrics[name].append({
            'value': value,
            'timestamp': time.time()
        })
        if len(self.custom_metrics[name]) > 1000:
            self.custom_metrics[name].popleft()

class AlertManager:
    """Manages alerting based on performance thresholds"""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.alerts = []
        self.alert_callbacks = []
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback function to be called when alert is triggered"""
        self.alert_callbacks.append(callback)
    
    def check_thresholds(self, metrics: Dict[str, Any]):
        """Check if any metrics exceed defined thresholds"""
        if not self.config.enable_alerts:
            return
        
        alerts_triggered = []
        
        # Check CPU threshold
        if metrics.get('cpu_percent', 0) > self.config.alert_thresholds['cpu_percent']:
            alert = {
                'type': 'cpu_high',
                'value': metrics['cpu_percent'],
                'threshold': self.config.alert_thresholds['cpu_percent'],
                'timestamp': time.time(),
                'message': f"High CPU usage: {metrics['cpu_percent']:.1f}%"
            }
            alerts_triggered.append(alert)
        
        # Check memory threshold
        if metrics.get('memory_percent', 0) > self.config.alert_thresholds['memory_percent']:
            alert = {
                'type': 'memory_high',
                'value': metrics['memory_percent'],
                'threshold': self.config.alert_thresholds['memory_percent'],
                'timestamp': time.time(),
                'message': f"High memory usage: {metrics['memory_percent']:.1f}%"
            }
            alerts_triggered.append(alert)
        
        # Store and trigger callbacks for alerts
        for alert in alerts_triggered:
            self.alerts.append(alert)
            logger.warning(f"ALERT: {alert['message']}")
            
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

class PerformanceMonitor:
    """Main performance monitoring class"""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.system_metrics = SystemMetrics()
        self.app_metrics = ApplicationMetrics()
        self.alert_manager = AlertManager(self.config)
        
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        self.end_time = None
    
    def start_monitoring(self):
        """Start performance monitoring in background thread"""
        if self.monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.monitoring = True
        self.start_time = time.time()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.monitoring:
            logger.warning("Monitoring is not running")
            return
        
        self.monitoring = False
        self.end_time = time.time()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in background thread"""
        while self.monitoring:
            try:
                # Collect system metrics
                system_data = self.system_metrics.collect_metrics()
                
                # Check alert thresholds
                self.alert_manager.check_thresholds(system_data)
                
                # Sleep until next collection
                time.sleep(self.config.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.collection_interval)
    
    def add_response_time(self, endpoint: str, response_time: float):
        """Add response time measurement from load tests"""
        self.app_metrics.add_response_time(endpoint, response_time)
    
    def increment_request_count(self, endpoint: str):
        """Increment request counter"""
        self.app_metrics.increment_request_count(endpoint)
    
    def increment_error_count(self, endpoint: str):
        """Increment error counter"""
        self.app_metrics.increment_error_count(endpoint)
    
    def update_concurrent_users(self, count: int):
        """Update concurrent user count"""
        self.app_metrics.update_concurrent_users(count)
    
    def add_custom_metric(self, name: str, value: float):
        """Add custom application metric"""
        self.app_metrics.add_custom_metric(name, value)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if self.monitoring:
            self.stop_monitoring()
        
        duration = (self.end_time - self.start_time) if self.start_time and self.end_time else 0
        
        # Calculate system metrics summary
        system_summary = self._calculate_system_summary()
        
        # Calculate application metrics summary
        app_summary = self._calculate_app_summary()
        
        # Generate visualizations
        chart_paths = self._generate_charts()
        
        report = {
            'test_duration_seconds': duration,
            'timestamp': datetime.now().isoformat(),
            'system_metrics': system_summary,
            'application_metrics': app_summary,
            'alerts': self.alert_manager.alerts,
            'charts': chart_paths,
            'recommendations': self._generate_recommendations()
        }
        
        # Save raw data if enabled
        if self.config.save_raw_data:
            self._save_raw_data(report)
        
        return report
    
    def _calculate_system_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for system metrics"""
        if not self.system_metrics.cpu_usage:
            return {"status": "no_data"}
        
        return {
            'cpu': {
                'avg': sum(self.system_metrics.cpu_usage) / len(self.system_metrics.cpu_usage),
                'max': max(self.system_metrics.cpu_usage),
                'min': min(self.system_metrics.cpu_usage)
            },
            'memory': {
                'avg': sum(self.system_metrics.memory_usage) / len(self.system_metrics.memory_usage),
                'max': max(self.system_metrics.memory_usage),
                'min': min(self.system_metrics.memory_usage)
            },
            'data_points': len(self.system_metrics.cpu_usage)
        }
    
    def _calculate_app_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for application metrics"""
        endpoint_summaries = {}
        
        for endpoint, response_times in self.app_metrics.response_times.items():
            if response_times:
                times = [rt['time'] for rt in response_times]
                times.sort()
                
                endpoint_summaries[endpoint] = {
                    'total_requests': self.app_metrics.request_counts.get(endpoint, 0),
                    'total_errors': self.app_metrics.error_counts.get(endpoint, 0),
                    'error_rate': (self.app_metrics.error_counts.get(endpoint, 0) / 
                                 max(self.app_metrics.request_counts.get(endpoint, 1), 1)) * 100,
                    'response_times': {
                        'avg': sum(times) / len(times),
                        'median': times[len(times)//2],
                        'p95': times[int(len(times) * 0.95)] if len(times) > 20 else max(times),
                        'p99': times[int(len(times) * 0.99)] if len(times) > 100 else max(times),
                        'min': min(times),
                        'max': max(times)
                    }
                }
        
        return {
            'endpoints': endpoint_summaries,
            'total_requests': sum(self.app_metrics.request_counts.values()),
            'total_errors': sum(self.app_metrics.error_counts.values()),
            'overall_error_rate': (sum(self.app_metrics.error_counts.values()) / 
                                 max(sum(self.app_metrics.request_counts.values()), 1)) * 100
        }
    
    def _generate_charts(self) -> List[str]:
        """Generate performance charts and return file paths"""
        charts_dir = Path("load_test_charts")
        charts_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_paths = []
        
        # System metrics chart
        if self.system_metrics.cpu_usage:
            try:
                plt.figure(figsize=(12, 8))
                
                # CPU usage
                plt.subplot(2, 2, 1)
                plt.plot(list(self.system_metrics.cpu_usage), label='CPU %')
                plt.title('CPU Usage Over Time')
                plt.ylabel('CPU %')
                plt.legend()
                
                # Memory usage
                plt.subplot(2, 2, 2)
                plt.plot(list(self.system_metrics.memory_usage), label='Memory %', color='red')
                plt.title('Memory Usage Over Time')
                plt.ylabel('Memory %')
                plt.legend()
                
                # Response times by endpoint
                plt.subplot(2, 2, 3)
                for endpoint, response_times in list(self.app_metrics.response_times.items())[:5]:  # Top 5 endpoints
                    times = [rt['time'] for rt in list(response_times)[-100:]]  # Last 100 measurements
                    plt.plot(times, label=endpoint[:20], alpha=0.7)
                plt.title('Response Times by Endpoint')
                plt.ylabel('Response Time (ms)')
                plt.legend()
                
                # Concurrent users
                plt.subplot(2, 2, 4)
                if self.app_metrics.concurrent_users:
                    user_counts = [u['count'] for u in self.app_metrics.concurrent_users]
                    plt.plot(user_counts, label='Concurrent Users', color='green')
                plt.title('Concurrent Users Over Time')
                plt.ylabel('User Count')
                plt.legend()
                
                plt.tight_layout()
                
                chart_path = charts_dir / f"performance_metrics_{timestamp}.png"
                plt.savefig(chart_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                chart_paths.append(str(chart_path))
                
            except Exception as e:
                logger.error(f"Error generating system metrics chart: {e}")
        
        return chart_paths
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations based on metrics"""
        recommendations = []
        
        # Check system metrics
        if self.system_metrics.cpu_usage:
            avg_cpu = sum(self.system_metrics.cpu_usage) / len(self.system_metrics.cpu_usage)
            if avg_cpu > 70:
                recommendations.append("High CPU usage detected. Consider scaling horizontally or optimizing CPU-intensive operations.")
        
        if self.system_metrics.memory_usage:
            avg_memory = sum(self.system_metrics.memory_usage) / len(self.system_metrics.memory_usage)
            if avg_memory > 80:
                recommendations.append("High memory usage detected. Check for memory leaks or consider increasing available memory.")
        
        # Check application metrics
        for endpoint, response_times in self.app_metrics.response_times.items():
            if response_times:
                times = [rt['time'] for rt in response_times]
                avg_time = sum(times) / len(times)
                if avg_time > 2000:  # 2 seconds
                    recommendations.append(f"Endpoint {endpoint} has slow response times (avg: {avg_time:.0f}ms). Consider optimization.")
        
        # Check error rates
        for endpoint in self.app_metrics.error_counts:
            total_requests = self.app_metrics.request_counts.get(endpoint, 0)
            error_count = self.app_metrics.error_counts[endpoint]
            if total_requests > 0:
                error_rate = (error_count / total_requests) * 100
                if error_rate > 5:  # 5% error rate
                    recommendations.append(f"Endpoint {endpoint} has high error rate ({error_rate:.1f}%). Investigate error causes.")
        
        if not recommendations:
            recommendations.append("System performance looks good! All metrics are within acceptable ranges.")
        
        return recommendations
    
    def _save_raw_data(self, report: Dict[str, Any]):
        """Save raw monitoring data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"load_test_raw_data_{timestamp}.json"
        
        raw_data = {
            'report': report,
            'raw_metrics': {
                'system_cpu': list(self.system_metrics.cpu_usage),
                'system_memory': list(self.system_metrics.memory_usage),
                'system_timestamps': list(self.system_metrics.timestamps),
                'app_response_times': {k: list(v) for k, v in self.app_metrics.response_times.items()},
                'concurrent_users': list(self.app_metrics.concurrent_users),
                'custom_metrics': {k: list(v) for k, v in self.app_metrics.custom_metrics.items()}
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(raw_data, f, indent=2, default=str)
        
        logger.info(f"Raw monitoring data saved to {filename}")

# Example usage and integration
def example_alert_callback(alert: Dict[str, Any]):
    """Example alert callback function"""
    print(f"🚨 ALERT: {alert['message']}")
    # In production, this could send emails, Slack messages, etc.

# Global monitor instance for easy integration
global_monitor = PerformanceMonitor()

def start_load_test_monitoring():
    """Convenience function to start monitoring"""
    global_monitor.add_alert_callback(example_alert_callback)
    global_monitor.start_monitoring()

def stop_load_test_monitoring():
    """Convenience function to stop monitoring and get report"""
    return global_monitor.generate_report()