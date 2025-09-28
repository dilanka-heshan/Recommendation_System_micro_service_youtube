# Performance Profiling Infrastructure for Recommendation System
import time
import psutil
import numpy as np
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import json
from pathlib import Path
import contextlib
import tracemalloc
import threading
import concurrent.futures

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance measurement results"""
    test_name: str
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    throughput_ops_per_sec: Optional[float] = None
    latency_percentiles: Optional[Dict[str, float]] = None
    error_rate: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    num_iterations: int = 100
    warm_up_iterations: int = 10
    measure_memory: bool = True
    measure_cpu: bool = True
    timeout_seconds: int = 300
    concurrent_users: int = 1
    target_ops_per_second: Optional[float] = None

class PerformanceProfiler:
    """Main performance profiling utility"""
    
    def __init__(self, results_dir: str = "performance_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.current_metrics = []
        
    def profile_function(self, func: Callable, config: BenchmarkConfig, 
                        test_name: str, *args, **kwargs) -> PerformanceMetrics:
        """
        Profile a function's performance
        
        Args:
            func: Function to profile
            config: Benchmark configuration
            test_name: Name for this test
            *args, **kwargs: Arguments for the function
            
        Returns:
            PerformanceMetrics with results
        """
        # Warm-up runs
        logger.info(f"Starting warm-up for {test_name} ({config.warm_up_iterations} iterations)")
        for _ in range(config.warm_up_iterations):
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Warm-up iteration failed: {e}")
        
        # Start memory tracing
        if config.measure_memory:
            tracemalloc.start()
        
        # Performance measurement
        execution_times = []
        memory_usage = []
        errors = 0
        
        logger.info(f"Starting performance measurement for {test_name} ({config.num_iterations} iterations)")
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # CPU monitoring in separate thread
        cpu_usage = []
        stop_cpu_monitoring = threading.Event()
        
        def monitor_cpu():
            while not stop_cpu_monitoring.is_set():
                cpu_usage.append(psutil.cpu_percent(interval=0.1))
        
        cpu_thread = None
        if config.measure_cpu:
            cpu_thread = threading.Thread(target=monitor_cpu)
            cpu_thread.start()
        
        peak_memory = initial_memory
        
        try:
            for i in range(config.num_iterations):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time
                    execution_times.append(execution_time)
                    
                    # Memory measurement
                    if config.measure_memory:
                        current_memory = process.memory_info().rss / 1024 / 1024
                        memory_usage.append(current_memory)
                        peak_memory = max(peak_memory, current_memory)
                    
                except Exception as e:
                    errors += 1
                    execution_time = time.time() - start_time
                    execution_times.append(execution_time)  # Count failed attempts too
                    logger.warning(f"Iteration {i+1} failed: {e}")
                
                # Progress logging
                if (i + 1) % max(1, config.num_iterations // 10) == 0:
                    progress = (i + 1) / config.num_iterations * 100
                    logger.info(f"Progress: {progress:.1f}% ({i+1}/{config.num_iterations})")
        
        finally:
            # Stop monitoring
            if config.measure_cpu and cpu_thread:
                stop_cpu_monitoring.set()
                cpu_thread.join()
            
            if config.measure_memory:
                tracemalloc.stop()
        
        # Calculate metrics
        avg_execution_time = np.mean(execution_times)
        avg_memory = np.mean(memory_usage) if memory_usage else initial_memory
        avg_cpu = np.mean(cpu_usage) if cpu_usage else 0.0
        
        # Throughput calculation
        total_time = sum(execution_times)
        successful_ops = config.num_iterations - errors
        throughput = successful_ops / total_time if total_time > 0 else 0
        
        # Latency percentiles
        latency_percentiles = {
            'p50': np.percentile(execution_times, 50),
            'p90': np.percentile(execution_times, 90),
            'p95': np.percentile(execution_times, 95),
            'p99': np.percentile(execution_times, 99),
            'min': np.min(execution_times),
            'max': np.max(execution_times)
        }
        
        error_rate = errors / config.num_iterations
        
        metrics = PerformanceMetrics(
            test_name=test_name,
            execution_time=avg_execution_time,
            memory_usage_mb=avg_memory,
            peak_memory_mb=peak_memory,
            cpu_usage_percent=avg_cpu,
            throughput_ops_per_sec=throughput,
            latency_percentiles=latency_percentiles,
            error_rate=error_rate
        )
        
        self.current_metrics.append(metrics)
        return metrics
    
    def profile_concurrent(self, func: Callable, config: BenchmarkConfig, 
                         test_name: str, *args, **kwargs) -> PerformanceMetrics:
        """
        Profile function performance with concurrent users
        
        Args:
            func: Function to profile
            config: Benchmark configuration  
            test_name: Name for this test
            *args, **kwargs: Arguments for the function
            
        Returns:
            PerformanceMetrics with concurrent results
        """
        logger.info(f"Starting concurrent performance test: {test_name} "
                   f"({config.concurrent_users} users, {config.num_iterations} iterations each)")
        
        execution_times = []
        errors = 0
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            # Submit all tasks
            futures = []
            for user in range(config.concurrent_users):
                for iteration in range(config.num_iterations):
                    future = executor.submit(self._timed_execution, func, *args, **kwargs)
                    futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures, timeout=config.timeout_seconds):
                try:
                    exec_time, error = future.result()
                    execution_times.append(exec_time)
                    if error:
                        errors += 1
                except Exception as e:
                    errors += 1
                    logger.warning(f"Concurrent execution failed: {e}")
        
        total_time = time.time() - start_time
        total_operations = config.concurrent_users * config.num_iterations
        successful_ops = total_operations - errors
        
        # Calculate metrics
        avg_execution_time = np.mean(execution_times) if execution_times else 0
        throughput = successful_ops / total_time if total_time > 0 else 0
        error_rate = errors / total_operations
        
        latency_percentiles = {
            'p50': np.percentile(execution_times, 50) if execution_times else 0,
            'p90': np.percentile(execution_times, 90) if execution_times else 0,
            'p95': np.percentile(execution_times, 95) if execution_times else 0,
            'p99': np.percentile(execution_times, 99) if execution_times else 0,
            'min': np.min(execution_times) if execution_times else 0,
            'max': np.max(execution_times) if execution_times else 0
        }
        
        # System resource usage during concurrent test
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024
        cpu_percent = psutil.cpu_percent()
        
        metrics = PerformanceMetrics(
            test_name=f"{test_name}_concurrent_{config.concurrent_users}users",
            execution_time=avg_execution_time,
            memory_usage_mb=current_memory,
            peak_memory_mb=current_memory,  # Approximation for concurrent
            cpu_usage_percent=cpu_percent,
            throughput_ops_per_sec=throughput,
            latency_percentiles=latency_percentiles,
            error_rate=error_rate
        )
        
        self.current_metrics.append(metrics)
        return metrics
    
    def _timed_execution(self, func: Callable, *args, **kwargs) -> tuple:
        """Execute function with timing, return (execution_time, error_occurred)"""
        start_time = time.time()
        error_occurred = False
        
        try:
            func(*args, **kwargs)
        except Exception as e:
            error_occurred = True
            logger.debug(f"Execution error: {e}")
        
        execution_time = time.time() - start_time
        return execution_time, error_occurred
    
    def save_results(self, filename: Optional[str] = None):
        """Save current metrics to JSON file"""
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_results_{timestamp}.json"
        
        filepath = self.results_dir / filename
        results_data = {
            'test_run_timestamp': datetime.utcnow().isoformat(),
            'system_info': self._get_system_info(),
            'metrics': [metric.to_dict() for metric in self.current_metrics]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Performance results saved to {filepath}")
        return filepath
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for context"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'python_version': psutil.__version__,
            'platform': psutil.WINDOWS if psutil.WINDOWS else 'unix'
        }
    
    def compare_results(self, baseline_file: str, current_file: str) -> Dict[str, Any]:
        """Compare two performance result files"""
        baseline_data = self._load_results(baseline_file)
        current_data = self._load_results(current_file)
        
        comparison = {
            'baseline_timestamp': baseline_data['test_run_timestamp'],
            'current_timestamp': current_data['test_run_timestamp'],
            'comparisons': []
        }
        
        # Group by test name
        baseline_metrics = {m['test_name']: m for m in baseline_data['metrics']}
        current_metrics = {m['test_name']: m for m in current_data['metrics']}
        
        for test_name in baseline_metrics:
            if test_name in current_metrics:
                baseline = baseline_metrics[test_name]
                current = current_metrics[test_name]
                
                comparison['comparisons'].append({
                    'test_name': test_name,
                    'execution_time_change': self._calculate_change(
                        baseline['execution_time'], current['execution_time']
                    ),
                    'memory_change': self._calculate_change(
                        baseline['memory_usage_mb'], current['memory_usage_mb']
                    ),
                    'throughput_change': self._calculate_change(
                        baseline.get('throughput_ops_per_sec', 0), 
                        current.get('throughput_ops_per_sec', 0)
                    ),
                    'error_rate_change': self._calculate_change(
                        baseline.get('error_rate', 0), 
                        current.get('error_rate', 0)
                    )
                })
        
        return comparison
    
    def _load_results(self, filename: str) -> Dict[str, Any]:
        """Load performance results from JSON file"""
        filepath = self.results_dir / filename
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _calculate_change(self, baseline: float, current: float) -> Dict[str, Any]:
        """Calculate percentage change between baseline and current"""
        if baseline == 0 and current == 0:
            return {'percent_change': 0, 'absolute_change': 0, 'status': 'unchanged'}
        
        if baseline == 0:
            return {'percent_change': float('inf'), 'absolute_change': current, 'status': 'new'}
        
        percent_change = ((current - baseline) / baseline) * 100
        absolute_change = current - baseline
        
        if percent_change > 5:
            status = 'regression'
        elif percent_change < -5:
            status = 'improvement'
        else:
            status = 'stable'
        
        return {
            'percent_change': round(percent_change, 2),
            'absolute_change': round(absolute_change, 4),
            'status': status
        }
    
    def generate_report(self) -> str:
        """Generate a human-readable performance report"""
        if not self.current_metrics:
            return "No performance metrics available."
        
        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE TEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append(f"Total tests: {len(self.current_metrics)}")
        report.append("")
        
        for metric in self.current_metrics:
            report.append(f"Test: {metric.test_name}")
            report.append("-" * 40)
            report.append(f"  Avg Execution Time: {metric.execution_time:.4f}s")
            report.append(f"  Memory Usage: {metric.memory_usage_mb:.2f} MB")
            report.append(f"  Peak Memory: {metric.peak_memory_mb:.2f} MB")
            report.append(f"  CPU Usage: {metric.cpu_usage_percent:.1f}%")
            
            if metric.throughput_ops_per_sec:
                report.append(f"  Throughput: {metric.throughput_ops_per_sec:.2f} ops/sec")
            
            if metric.error_rate is not None:
                report.append(f"  Error Rate: {metric.error_rate:.2%}")
            
            if metric.latency_percentiles:
                report.append("  Latency Percentiles:")
                for percentile, value in metric.latency_percentiles.items():
                    report.append(f"    {percentile}: {value:.4f}s")
            
            report.append("")
        
        return "\n".join(report)

@contextlib.contextmanager
def performance_context(profiler: PerformanceProfiler, test_name: str):
    """Context manager for simple performance measurement"""
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024
    
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    tracemalloc.start()
    
    try:
        yield
    finally:
        execution_time = time.time() - start_time
        current_memory = process.memory_info().rss / 1024 / 1024
        current_trace, peak_trace = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_memory_mb = peak_trace / 1024 / 1024
        cpu_percent = psutil.cpu_percent()
        
        metrics = PerformanceMetrics(
            test_name=test_name,
            execution_time=execution_time,
            memory_usage_mb=current_memory,
            peak_memory_mb=peak_memory_mb,
            cpu_usage_percent=cpu_percent
        )
        
        profiler.current_metrics.append(metrics)
        logger.info(f"Performance measured for {test_name}: {execution_time:.4f}s, "
                   f"{current_memory:.2f}MB memory, {cpu_percent:.1f}% CPU")

class RecommendationSystemBenchmark:
    """Specific benchmark suite for the recommendation system"""
    
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
    
    def benchmark_rocchio_algorithm(self, rocchio_service, test_data: Dict[str, Any], 
                                   config: BenchmarkConfig = None) -> PerformanceMetrics:
        """Benchmark Rocchio algorithm performance"""
        if config is None:
            config = BenchmarkConfig(num_iterations=1000)
        
        def rocchio_test():
            return rocchio_service.apply_rocchio_algorithm(
                original_vector=test_data['original_vector'],
                positive_embeddings=test_data['positive_embeddings'],
                negative_embeddings=test_data['negative_embeddings']
            )
        
        return self.profiler.profile_function(
            rocchio_test, config, "rocchio_algorithm_performance"
        )
    
    def benchmark_video_reranking(self, reranker, user_history: List[Dict], 
                                 candidates: List[Dict], config: BenchmarkConfig = None) -> PerformanceMetrics:
        """Benchmark video reranking performance"""
        if config is None:
            config = BenchmarkConfig(num_iterations=100)
        
        def reranking_test():
            return reranker.rerank_with_user_history(
                user_history=user_history,
                candidate_videos=candidates,
                top_k=10
            )
        
        return self.profiler.profile_function(
            reranking_test, config, "video_reranking_performance"
        )
    
    def benchmark_full_pipeline(self, orchestrator, user_id: str, 
                              config: BenchmarkConfig = None) -> PerformanceMetrics:
        """Benchmark complete recommendation pipeline"""
        if config is None:
            config = BenchmarkConfig(num_iterations=50)
        
        def pipeline_test():
            return orchestrator.generate_recommendations(user_id=user_id, top_k=10)
        
        return self.profiler.profile_function(
            pipeline_test, config, "full_pipeline_performance"
        )
    
    def benchmark_concurrent_users(self, orchestrator, user_ids: List[str],
                                 config: BenchmarkConfig = None) -> PerformanceMetrics:
        """Benchmark system with concurrent users"""
        if config is None:
            config = BenchmarkConfig(num_iterations=10, concurrent_users=10)
        
        def concurrent_test():
            # Pick random user for this test
            import random
            user_id = random.choice(user_ids)
            return orchestrator.generate_recommendations(user_id=user_id, top_k=10)
        
        return self.profiler.profile_concurrent(
            concurrent_test, config, "concurrent_users_performance"
        )
    
    def run_comprehensive_benchmark(self, test_components: Dict[str, Any]) -> Dict[str, PerformanceMetrics]:
        """Run comprehensive performance benchmark"""
        results = {}
        
        logger.info("Starting comprehensive recommendation system benchmark")
        
        # Benchmark individual components
        if 'rocchio_service' in test_components and 'rocchio_test_data' in test_components:
            results['rocchio'] = self.benchmark_rocchio_algorithm(
                test_components['rocchio_service'],
                test_components['rocchio_test_data']
            )
        
        if 'reranker' in test_components and 'reranking_test_data' in test_components:
            results['reranking'] = self.benchmark_video_reranking(
                test_components['reranker'],
                test_components['reranking_test_data']['user_history'],
                test_components['reranking_test_data']['candidates']
            )
        
        if 'orchestrator' in test_components:
            results['pipeline'] = self.benchmark_full_pipeline(
                test_components['orchestrator'],
                test_components.get('test_user_id', 'test_user')
            )
            
            # Concurrent users test
            user_ids = test_components.get('test_user_ids', ['user1', 'user2', 'user3'])
            results['concurrent'] = self.benchmark_concurrent_users(
                test_components['orchestrator'],
                user_ids
            )
        
        logger.info("Comprehensive benchmark completed")
        return results

if __name__ == "__main__":
    # Example usage
    profiler = PerformanceProfiler()
    
    # Example function to profile
    def example_function(n: int):
        return sum(i**2 for i in range(n))
    
    # Profile the function
    config = BenchmarkConfig(num_iterations=100, warm_up_iterations=10)
    metrics = profiler.profile_function(example_function, config, "example_test", 10000)
    
    # Generate and print report
    report = profiler.generate_report()
    print(report)
    
    # Save results
    profiler.save_results()