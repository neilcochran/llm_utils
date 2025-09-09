"""Hardware performance monitoring during model evaluation - data only."""

import psutil
import threading
import time
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import sys
from pathlib import Path

# Add src to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from cuda_info.cuda_status import CudaStatusChecker

logger = logging.getLogger(__name__)


@dataclass
class ResourceSnapshot:
    """Single point-in-time resource usage snapshot."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    gpu_utilization_percent: Optional[float] = None
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None
    gpu_temperature_c: Optional[float] = None
    gpu_power_draw_w: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ResourceMetrics:
    """Aggregated resource usage metrics over time."""
    duration_seconds: float
    cpu_percent_avg: float
    cpu_percent_max: float
    memory_percent_avg: float
    memory_percent_max: float
    memory_used_mb_avg: float
    memory_used_mb_max: float
    memory_total_mb: Optional[float] = None  # System total memory
    gpu_utilization_avg: Optional[float] = None
    gpu_utilization_max: Optional[float] = None
    gpu_memory_used_mb_avg: Optional[float] = None
    gpu_memory_used_mb_max: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None  # GPU total memory
    gpu_temperature_avg: Optional[float] = None
    gpu_temperature_max: Optional[float] = None
    gpu_power_avg: Optional[float] = None
    gpu_power_max: Optional[float] = None
    sample_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PerformanceMonitor:
    """Monitors system resource usage during model inference."""
    
    def __init__(self, sample_interval_seconds: float = 0.1):
        """Initialize performance monitor.
        
        Args:
            sample_interval_seconds: How often to sample resource usage
        """
        self.sample_interval = sample_interval_seconds
        self.cuda_checker = CudaStatusChecker()
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.snapshots: List[ResourceSnapshot] = []
        self.start_time: Optional[float] = None
    
    def _sample_resources(self) -> ResourceSnapshot:
        """Take a single resource usage snapshot."""
        timestamp = time.perf_counter()
        
        # Get CPU and memory info
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Get GPU info if available
        gpu_util = None
        gpu_memory_used = None
        gpu_memory_total = None
        gpu_temp = None
        gpu_power = None
        
        try:
            cuda_status = self.cuda_checker.get_cuda_status()
            if cuda_status.cuda_available and cuda_status.gpus:
                # Use first GPU for monitoring
                gpu = cuda_status.gpus[0]
                gpu_util = float(gpu['gpu_utilization_percent']) if gpu.get('gpu_utilization_percent') and gpu['gpu_utilization_percent'] != 'N/A' else None
                gpu_memory_used = float(gpu['used_memory_mb']) if gpu.get('used_memory_mb') and gpu['used_memory_mb'] != 'N/A' else None
                gpu_memory_total = float(gpu['total_memory_mb']) if gpu.get('total_memory_mb') and gpu['total_memory_mb'] != 'N/A' else None
                gpu_temp = float(gpu['temperature_c']) if gpu.get('temperature_c') and gpu['temperature_c'] != 'N/A' else None
                gpu_power = float(gpu['power_draw_w']) if gpu.get('power_draw_w') and gpu['power_draw_w'] != 'N/A' else None
        except Exception as e:
            logger.debug(f"Failed to get GPU metrics: {e}")
        
        return ResourceSnapshot(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            gpu_utilization_percent=gpu_util,
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            gpu_temperature_c=gpu_temp,
            gpu_power_draw_w=gpu_power
        )
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop that runs in separate thread."""
        while self.monitoring:
            try:
                snapshot = self._sample_resources()
                self.snapshots.append(snapshot)
            except Exception as e:
                logger.error(f"Error sampling resources: {e}")
            
            time.sleep(self.sample_interval)
    
    def start_monitoring(self) -> None:
        """Start resource monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.snapshots = []
        self.start_time = time.perf_counter()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
    
    def get_current_snapshot(self) -> ResourceSnapshot:
        """Get a single current resource snapshot without starting monitoring."""
        return self._sample_resources()
    
    def calculate_metrics(self) -> ResourceMetrics:
        """Calculate aggregated metrics from collected snapshots."""
        if not self.snapshots:
            raise ValueError("No resource snapshots available")
        
        if not self.start_time:
            raise ValueError("Monitoring was not properly started")
        
        duration = time.perf_counter() - self.start_time
        
        # Calculate CPU metrics
        cpu_values = [s.cpu_percent for s in self.snapshots]
        cpu_avg = sum(cpu_values) / len(cpu_values)
        cpu_max = max(cpu_values)
        
        # Calculate memory metrics  
        memory_percent_values = [s.memory_percent for s in self.snapshots]
        memory_percent_avg = sum(memory_percent_values) / len(memory_percent_values)
        memory_percent_max = max(memory_percent_values)
        
        memory_used_values = [s.memory_used_mb for s in self.snapshots]
        memory_used_avg = sum(memory_used_values) / len(memory_used_values)
        memory_used_max = max(memory_used_values)
        
        # Calculate total memory (used + available)
        memory_total_values = [s.memory_used_mb + s.memory_available_mb for s in self.snapshots]
        memory_total_mb = sum(memory_total_values) / len(memory_total_values) if memory_total_values else None
        
        # Calculate GPU metrics if available
        gpu_util_values = [s.gpu_utilization_percent for s in self.snapshots if s.gpu_utilization_percent is not None]
        gpu_util_avg = sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else None
        gpu_util_max = max(gpu_util_values) if gpu_util_values else None
        
        gpu_memory_values = [s.gpu_memory_used_mb for s in self.snapshots if s.gpu_memory_used_mb is not None]
        gpu_memory_avg = sum(gpu_memory_values) / len(gpu_memory_values) if gpu_memory_values else None
        gpu_memory_max = max(gpu_memory_values) if gpu_memory_values else None
        
        # Get GPU total memory (should be consistent across all snapshots)
        gpu_memory_total_values = [s.gpu_memory_total_mb for s in self.snapshots if s.gpu_memory_total_mb is not None]
        gpu_memory_total_mb = gpu_memory_total_values[0] if gpu_memory_total_values else None
        
        gpu_temp_values = [s.gpu_temperature_c for s in self.snapshots if s.gpu_temperature_c is not None]
        gpu_temp_avg = sum(gpu_temp_values) / len(gpu_temp_values) if gpu_temp_values else None
        gpu_temp_max = max(gpu_temp_values) if gpu_temp_values else None
        
        gpu_power_values = [s.gpu_power_draw_w for s in self.snapshots if s.gpu_power_draw_w is not None]
        gpu_power_avg = sum(gpu_power_values) / len(gpu_power_values) if gpu_power_values else None
        gpu_power_max = max(gpu_power_values) if gpu_power_values else None
        
        return ResourceMetrics(
            duration_seconds=duration,
            cpu_percent_avg=cpu_avg,
            cpu_percent_max=cpu_max,
            memory_percent_avg=memory_percent_avg,
            memory_percent_max=memory_percent_max,
            memory_used_mb_avg=memory_used_avg,
            memory_used_mb_max=memory_used_max,
            memory_total_mb=memory_total_mb,
            gpu_utilization_avg=gpu_util_avg,
            gpu_utilization_max=gpu_util_max,
            gpu_memory_used_mb_avg=gpu_memory_avg,
            gpu_memory_used_mb_max=gpu_memory_max,
            gpu_memory_total_mb=gpu_memory_total_mb,
            gpu_temperature_avg=gpu_temp_avg,
            gpu_temperature_max=gpu_temp_max,
            gpu_power_avg=gpu_power_avg,
            gpu_power_max=gpu_power_max,
            sample_count=len(self.snapshots)
        )
    
    def get_snapshots(self) -> List[ResourceSnapshot]:
        """Get all collected resource snapshots."""
        return self.snapshots.copy()