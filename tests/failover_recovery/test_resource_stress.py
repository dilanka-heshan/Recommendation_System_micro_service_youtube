"""
Resource Stress and Recovery Tests - Test system behavior under resource pressure
Tests memory, CPU, and I/O stress scenarios with minimal configuration
"""

import pytest
import time
import psutil
import threading
import gc
import os
import tempfile
from typing import Dict, Any, List
import concurrent.futures
import requests


class ResourceStressTester:
    """Test system resource stress and recovery scenarios"""
    
    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url
        self.test_results = []
        self.initial_memory = psutil.virtual_memory().used
        self.initial_cpu = psutil.cpu_percent()
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource metrics"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        return {
            "memory_used_mb": memory.used / (1024 * 1024),
            "memory_percent": memory.percent,
            "memory_available_mb": memory.available / (1024 * 1024),
            "cpu_percent": cpu_percent,
            "disk_used_gb": disk.used / (1024 * 1024 * 1024),
            "disk_free_gb": disk.free / (1024 * 1024 * 1024),
            "disk_percent": (disk.used / disk.total) * 100
        }
    
    def stress_test_memory(self, duration: int = 10, target_mb: int = 100) -> Dict[str, Any]:
        """Apply memory stress and monitor system behavior"""
        start_time = time.time()
        start_metrics = self.get_system_metrics()
        
        # Create memory stress
        stress_data = []
        try:
            while time.time() - start_time < duration:
                # Allocate memory in chunks
                chunk = [i for i in range(10000)]  # ~80KB per chunk
                stress_data.append(chunk)
                
                # Check if we've reached target memory usage
                current_usage = len(stress_data) * 80 / 1024  # MB
                if current_usage >= target_mb:
                    break
                
                time.sleep(0.1)  # Brief pause
            
            peak_metrics = self.get_system_metrics()
            
        finally:
            # Cleanup
            del stress_data
            gc.collect()
        
        end_metrics = self.get_system_metrics()
        recovery_time = time.time() - start_time
        
        return {
            "test": "memory_stress",
            "duration": duration,
            "target_mb": target_mb,
            "start_memory_mb": start_metrics["memory_used_mb"],
            "peak_memory_mb": peak_metrics["memory_used_mb"],
            "end_memory_mb": end_metrics["memory_used_mb"],
            "memory_increase_mb": peak_metrics["memory_used_mb"] - start_metrics["memory_used_mb"],
            "recovery_time": recovery_time,
            "status": "completed"
        }
    
    def stress_test_cpu(self, duration: int = 10, num_threads: int = 2) -> Dict[str, Any]:
        """Apply CPU stress and monitor system behavior"""
        start_time = time.time()
        start_metrics = self.get_system_metrics()
        
        def cpu_intensive_task():
            """CPU-intensive calculation"""
            end_time = time.time() + duration
            count = 0
            while time.time() < end_time:
                # CPU-intensive calculation
                count += sum(i * i for i in range(1000))
            return count
        
        # Start CPU stress threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(cpu_intensive_task) for _ in range(num_threads)]
            
            # Monitor CPU usage during stress
            peak_cpu = 0
            monitoring_start = time.time()
            while time.time() - monitoring_start < duration:
                current_cpu = psutil.cpu_percent(interval=0.5)
                peak_cpu = max(peak_cpu, current_cpu)
            
            # Wait for all tasks to complete
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_metrics = self.get_system_metrics()
        
        return {
            "test": "cpu_stress",
            "duration": duration,
            "num_threads": num_threads,
            "start_cpu_percent": start_metrics["cpu_percent"],
            "peak_cpu_percent": peak_cpu,
            "end_cpu_percent": end_metrics["cpu_percent"],
            "cpu_increase": peak_cpu - start_metrics["cpu_percent"],
            "thread_results": len(results),
            "status": "completed"
        }
    
    def stress_test_disk_io(self, duration: int = 5, file_size_mb: int = 10) -> Dict[str, Any]:
        """Apply disk I/O stress and monitor system behavior"""
        start_time = time.time()
        start_metrics = self.get_system_metrics()
        
        temp_files = []
        try:
            while time.time() - start_time < duration:
                # Create temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_files.append(temp_file.name)
                
                # Write data to file
                data = b'x' * (1024 * 1024)  # 1MB chunk
                for _ in range(file_size_mb):
                    temp_file.write(data)
                temp_file.close()
                
                # Read data back
                with open(temp_file.name, 'rb') as f:
                    _ = f.read()
                
                time.sleep(0.1)
        
        finally:
            # Cleanup temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except OSError:
                    pass  # File might already be deleted
        
        end_metrics = self.get_system_metrics()
        
        return {
            "test": "disk_io_stress",
            "duration": duration,
            "files_created": len(temp_files),
            "file_size_mb": file_size_mb,
            "start_disk_free_gb": start_metrics["disk_free_gb"],
            "end_disk_free_gb": end_metrics["disk_free_gb"],
            "status": "completed"
        }
    
    def test_api_under_stress(self, stress_type: str = "memory", duration: int = 5) -> Dict[str, Any]:
        """Test API responsiveness during resource stress"""
        
        def make_api_request():
            """Make API request and measure response time"""
            try:
                start_time = time.time()
                response = requests.get(f"{self.api_url}/health", timeout=10)
                response_time = time.time() - start_time
                
                return {
                    "status_code": response.status_code,
                    "response_time": response_time,
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {
                    "status_code": 0,
                    "response_time": 10.0,  # Timeout
                    "success": False,
                    "error": str(e)
                }
        
        # Get baseline API performance
        baseline_response = make_api_request()
        
        # Apply stress and test API simultaneously
        api_results = []
        
        if stress_type == "memory":
            stress_thread = threading.Thread(target=lambda: self.stress_test_memory(duration, 50))
        elif stress_type == "cpu":
            stress_thread = threading.Thread(target=lambda: self.stress_test_cpu(duration, 2))
        else:
            stress_thread = threading.Thread(target=lambda: self.stress_test_disk_io(duration))
        
        stress_thread.start()
        
        # Test API during stress
        start_time = time.time()
        while time.time() - start_time < duration:
            result = make_api_request()
            api_results.append(result)
            time.sleep(1)
        
        stress_thread.join()
        
        # Calculate API performance metrics
        successful_requests = [r for r in api_results if r["success"]]
        avg_response_time = sum(r["response_time"] for r in successful_requests) / max(len(successful_requests), 1)
        success_rate = len(successful_requests) / max(len(api_results), 1)
        
        return {
            "test": f"api_under_{stress_type}_stress",
            "stress_duration": duration,
            "baseline_response_time": baseline_response["response_time"],
            "stress_avg_response_time": avg_response_time,
            "success_rate": success_rate,
            "total_requests": len(api_results),
            "successful_requests": len(successful_requests),
            "performance_degradation": avg_response_time / max(baseline_response["response_time"], 0.001),
            "status": "completed"
        }
    
    def test_recovery_after_stress(self, recovery_wait: int = 5) -> Dict[str, Any]:
        """Test system recovery after various stress conditions"""
        
        # Apply multiple stress conditions
        stress_results = []
        
        # Memory stress
        memory_stress = self.stress_test_memory(duration=3, target_mb=30)
        stress_results.append(memory_stress)
        time.sleep(1)  # Brief pause between stress tests
        
        # CPU stress  
        cpu_stress = self.stress_test_cpu(duration=3, num_threads=2)
        stress_results.append(cpu_stress)
        time.sleep(1)
        
        # Disk I/O stress
        disk_stress = self.stress_test_disk_io(duration=3, file_size_mb=5)
        stress_results.append(disk_stress)
        
        # Wait for recovery
        time.sleep(recovery_wait)
        
        # Check final system state
        final_metrics = self.get_system_metrics()
        
        # Test API after recovery
        api_test = requests.get(f"{self.api_url}/health", timeout=5)
        api_healthy = api_test.status_code == 200
        
        return {
            "test": "recovery_after_stress",
            "stress_tests_completed": len(stress_results),
            "recovery_wait_time": recovery_wait,
            "final_memory_mb": final_metrics["memory_used_mb"],
            "final_cpu_percent": final_metrics["cpu_percent"],
            "final_disk_free_gb": final_metrics["disk_free_gb"],
            "api_healthy_after_recovery": api_healthy,
            "stress_results": stress_results,
            "status": "completed"
        }


# Pytest fixtures and tests
@pytest.fixture
def resource_tester():
    return ResourceStressTester()


@pytest.fixture
def check_api_available():
    """Check if API is available before running tests"""
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        if response.status_code != 200:
            pytest.skip("API not available. Start with: python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8080")
    except requests.exceptions.RequestException:
        pytest.skip("API not reachable. Start with: python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8080")


class TestResourceStress:
    """Resource Stress Test Suite"""
    
    def test_memory_stress_recovery(self, resource_tester):
        """Test system recovery after memory stress"""
        result = resource_tester.stress_test_memory(duration=5, target_mb=50)
        
        assert result["status"] == "completed"
        assert result["memory_increase_mb"] > 0  # Memory should have increased
        assert result["recovery_time"] > 0
        
        # Memory should be mostly recovered (within reasonable margin)
        memory_growth = result["end_memory_mb"] - result["start_memory_mb"]
        assert memory_growth < result["memory_increase_mb"]  # Some recovery occurred
    
    def test_cpu_stress_recovery(self, resource_tester):
        """Test system recovery after CPU stress"""
        result = resource_tester.stress_test_cpu(duration=5, num_threads=2)
        
        assert result["status"] == "completed"
        assert result["thread_results"] == 2  # Both threads completed
        assert result["peak_cpu_percent"] > result["start_cpu_percent"]  # CPU usage increased
        
        # CPU should recover (end CPU should be lower than peak)
        assert result["end_cpu_percent"] < result["peak_cpu_percent"]
    
    def test_disk_io_stress_recovery(self, resource_tester):
        """Test system recovery after disk I/O stress"""
        result = resource_tester.stress_test_disk_io(duration=3, file_size_mb=5)
        
        assert result["status"] == "completed"
        assert result["files_created"] > 0  # Files were created and cleaned up
        
        # Disk space should be recovered (files cleaned up)
        disk_change = result["start_disk_free_gb"] - result["end_disk_free_gb"]
        assert abs(disk_change) < 0.1  # Minimal permanent disk usage change
    
    def test_api_resilience_under_memory_stress(self, resource_tester, check_api_available):
        """Test API resilience under memory stress"""
        result = resource_tester.test_api_under_stress("memory", duration=5)
        
        assert result["status"] == "completed"
        assert result["success_rate"] > 0.5  # At least 50% of requests should succeed
        assert result["performance_degradation"] < 10.0  # Response time shouldn't degrade too much
    
    def test_api_resilience_under_cpu_stress(self, resource_tester, check_api_available):
        """Test API resilience under CPU stress"""
        result = resource_tester.test_api_under_stress("cpu", duration=5)
        
        assert result["status"] == "completed"
        assert result["success_rate"] > 0.5  # At least 50% of requests should succeed
        assert result["total_requests"] > 0
    
    def test_comprehensive_recovery(self, resource_tester, check_api_available):
        """Test comprehensive system recovery after multiple stress conditions"""
        result = resource_tester.test_recovery_after_stress(recovery_wait=3)
        
        assert result["status"] == "completed"
        assert result["stress_tests_completed"] == 3  # All stress tests completed
        assert result["api_healthy_after_recovery"] is True  # API should be healthy after recovery
        
        # System metrics should be reasonable after recovery
        assert result["final_memory_mb"] > 0
        assert result["final_cpu_percent"] >= 0
        assert result["final_disk_free_gb"] > 0
    
    def test_concurrent_stress_scenarios(self, resource_tester):
        """Test system behavior under concurrent stress scenarios"""
        
        def run_stress_test(stress_type):
            if stress_type == "memory":
                return resource_tester.stress_test_memory(duration=3, target_mb=20)
            elif stress_type == "cpu":
                return resource_tester.stress_test_cpu(duration=3, num_threads=1)
            else:
                return resource_tester.stress_test_disk_io(duration=3, file_size_mb=3)
        
        # Run concurrent stress tests
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            stress_types = ["memory", "cpu", "disk"]
            futures = [executor.submit(run_stress_test, stress_type) for stress_type in stress_types]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All stress tests should complete
        assert len(results) == 3
        for result in results:
            assert result["status"] == "completed"
    
    def test_resource_monitoring_accuracy(self, resource_tester):
        """Test accuracy of resource monitoring"""
        initial_metrics = resource_tester.get_system_metrics()
        
        # Verify all expected metrics are present
        expected_metrics = [
            "memory_used_mb", "memory_percent", "memory_available_mb",
            "cpu_percent", "disk_used_gb", "disk_free_gb", "disk_percent"
        ]
        
        for metric in expected_metrics:
            assert metric in initial_metrics
            assert isinstance(initial_metrics[metric], (int, float))
            assert initial_metrics[metric] >= 0


class TestResourceLimits:
    """Test resource limit scenarios"""
    
    def test_memory_limit_handling(self, resource_tester):
        """Test handling of memory limit scenarios"""
        # Try to allocate memory and see how system handles it
        try:
            result = resource_tester.stress_test_memory(duration=5, target_mb=200)  # Larger allocation
            
            # System should handle this gracefully
            assert result["status"] == "completed"
            assert result["memory_increase_mb"] >= 0
            
        except MemoryError:
            # Memory error is acceptable - system is protecting itself
            assert True
    
    def test_system_resource_thresholds(self, resource_tester):
        """Test system behavior at resource thresholds"""
        current_metrics = resource_tester.get_system_metrics()
        
        # System should not be at critical resource levels during testing
        assert current_metrics["memory_percent"] < 95  # Less than 95% memory usage
        assert current_metrics["disk_percent"] < 95    # Less than 95% disk usage
        
        # If CPU is high, that's expected during testing
        assert current_metrics["cpu_percent"] >= 0


if __name__ == "__main__":
    # Simple test runner for manual execution
    tester = ResourceStressTester()
    
    print("=== Resource Stress and Recovery Test Suite ===")
    
    # Test system metrics
    print("1. Getting baseline system metrics...")
    metrics = tester.get_system_metrics()
    print(f"   Memory: {metrics['memory_used_mb']:.1f} MB ({metrics['memory_percent']:.1f}%)")
    print(f"   CPU: {metrics['cpu_percent']:.1f}%")
    print(f"   Disk: {metrics['disk_free_gb']:.1f} GB free")
    
    # Memory stress test
    print("2. Testing memory stress recovery...")
    memory_result = tester.stress_test_memory(duration=5, target_mb=30)
    print(f"   Memory stress: {memory_result['status']} - Increased by {memory_result['memory_increase_mb']:.1f} MB")
    
    # CPU stress test
    print("3. Testing CPU stress recovery...")
    cpu_result = tester.stress_test_cpu(duration=5, num_threads=2)
    print(f"   CPU stress: {cpu_result['status']} - Peak CPU: {cpu_result['peak_cpu_percent']:.1f}%")
    
    # Disk I/O stress test
    print("4. Testing disk I/O stress recovery...")
    disk_result = tester.stress_test_disk_io(duration=3, file_size_mb=5)
    print(f"   Disk I/O stress: {disk_result['status']} - Created {disk_result['files_created']} files")
    
    # API under stress (if available)
    print("5. Testing API under stress...")
    try:
        api_result = tester.test_api_under_stress("memory", duration=5)
        print(f"   API under stress: {api_result['status']} - Success rate: {api_result['success_rate']:.2f}")
    except Exception as e:
        print(f"   API test skipped: {str(e)}")
    
    # Recovery test
    print("6. Testing comprehensive recovery...")
    recovery_result = tester.test_recovery_after_stress(recovery_wait=3)
    print(f"   Recovery test: {recovery_result['status']} - API healthy: {recovery_result['api_healthy_after_recovery']}")
    
    print("\n=== Resource stress tests completed ===")