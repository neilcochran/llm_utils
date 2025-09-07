"""CUDA/nvidia-smi status checker - data collection only."""

import logging
import subprocess
import re
import platform
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class CudaInfo:
    """CUDA system information."""
    cuda_available: bool
    cuda_version: Optional[str]
    driver_version: Optional[str]
    cuda_runtime_version: Optional[str]
    gpus: List[Dict[str, Any]]
    total_memory_gb: float
    pytorch_compatible: bool
    recommended_pytorch_version: Optional[str]
    compatibility_notes: List[str]
    system_info: Dict[str, str]
    pytorch_installed: bool
    pytorch_version: Optional[str]
    pytorch_cuda_version: Optional[str]
    pytorch_compatibility_status: str
    pytorch_install_suggestion: Optional[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class CudaStatusChecker:
    """Check CUDA status and PyTorch compatibility - data only."""
    
    # PyTorch CUDA compatibility matrix (as of January 2025)
    PYTORCH_CUDA_COMPATIBILITY = {
        "11.7": ["1.13.0", "2.0.0"],
        "11.8": ["2.0.0", "2.1.0", "2.2.0", "2.5.0", "2.7.0"],
        "12.1": ["2.1.0", "2.2.0", "2.3.0", "2.5.0"],
        "12.4": ["2.3.0", "2.4.0", "2.5.0"],
        "12.6": ["2.4.0", "2.5.0", "2.7.0"],
        "12.8": ["2.7.0", "2.7.1"]
    }
    
    def check_nvidia_smi(self) -> Tuple[bool, Optional[Dict]]:
        """Check if nvidia-smi is available and get GPU information.
        
        Returns:
            Tuple of (success, gpu_data) where gpu_data contains GPU info and total memory.
            
        Example:
            >>> checker = CudaStatusChecker()
            >>> success, data = checker.check_nvidia_smi()
            >>> if success:
            ...     print(f"Found {len(data['gpus'])} GPUs")
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit", 
                 "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            
            gpus = []
            total_memory_gb = 0.0
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = [p.strip() for p in line.split(', ')]
                    if len(parts) >= 8:
                        name, total_mem, used_mem, free_mem, gpu_util, temp, power_draw, power_limit = parts
                        
                        # Handle cases where values might be "N/A" or missing
                        try:
                            total_memory_mb = int(total_mem)
                            total_memory_gb += total_memory_mb / 1024
                        except (ValueError, TypeError):
                            total_memory_mb = 0
                        
                        gpus.append({
                            'name': name,
                            'total_memory_mb': str(total_memory_mb),
                            'used_memory_mb': used_mem,
                            'free_memory_mb': free_mem,
                            'gpu_utilization_percent': gpu_util,
                            'temperature_c': temp,
                            'power_draw_w': power_draw,
                            'power_limit_w': power_limit
                        })
            
            return True, {'gpus': gpus, 'total_memory_gb': total_memory_gb}
            
        except subprocess.CalledProcessError as e:
            logger.debug(f"nvidia-smi command failed: {e}")
            return False, None
        except FileNotFoundError:
            logger.debug("nvidia-smi not found in PATH")
            return False, None
    
    def get_cuda_version(self) -> Optional[str]:
        """Get CUDA toolkit version.
        
        Returns:
            CUDA version string (e.g., '12.1') or None if not found.
            
        Example:
            >>> checker = CudaStatusChecker()
            >>> version = checker.get_cuda_version()
            >>> print(f"CUDA Version: {version or 'Not found'}")
        """
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract version from nvcc output
            version_match = re.search(r'V(\d+\.\d+)', result.stdout)
            if version_match:
                return version_match.group(1)
                
        except subprocess.CalledProcessError as e:
            logger.debug(f"nvcc command failed: {e}")
        except FileNotFoundError:
            logger.debug("nvcc not found in PATH")
        
        # Try alternative method via nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Extract CUDA version from driver info
            version_match = re.search(r'CUDA Version: (\d+\.\d+)', result.stdout)
            if version_match:
                return version_match.group(1)
                
        except subprocess.CalledProcessError as e:
            logger.debug(f"nvidia-smi command failed: {e}")
        except FileNotFoundError:
            logger.debug("nvidia-smi not found in PATH")
            
        return None
    
    def get_driver_version(self) -> Optional[str]:
        """Get NVIDIA driver version."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True
            )
            
            version = result.stdout.strip().split('\n')[0]
            return version if version else None
            
        except subprocess.CalledProcessError as e:
            logger.debug(f"nvidia-smi driver query failed: {e}")
            return None
        except FileNotFoundError:
            logger.debug("nvidia-smi not found in PATH")
            return None
    
    def check_pytorch_compatibility(self, cuda_version: str) -> Tuple[bool, Optional[str], List[str]]:
        """Check PyTorch compatibility with CUDA version."""
        notes = []
        
        if not cuda_version:
            return False, None, ["CUDA not available"]
        
        # Find compatible PyTorch versions
        compatible_versions = []
        for cuda_ver, pytorch_versions in self.PYTORCH_CUDA_COMPATIBILITY.items():
            if cuda_version.startswith(cuda_ver):
                compatible_versions.extend(pytorch_versions)
        
        if not compatible_versions:
            # Check if CUDA version is too new
            cuda_major_minor = float(cuda_version)
            max_supported = max(float(v) for v in self.PYTORCH_CUDA_COMPATIBILITY.keys())
            
            if cuda_major_minor > max_supported:
                notes.append(f"CUDA {cuda_version} may be too new for current PyTorch versions")
                notes.append(f"Consider downgrading to CUDA {max_supported} or wait for PyTorch updates")
                return False, None, notes
            else:
                notes.append(f"No explicit compatibility data for CUDA {cuda_version}")
                return False, None, notes
        
        # Recommend latest compatible version
        latest_pytorch = max(compatible_versions, key=lambda x: tuple(map(int, x.split('.'))))
        notes.append(f"Compatible PyTorch versions: {', '.join(sorted(set(compatible_versions)))}")
        
        return True, latest_pytorch, notes
    
    def get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'python_version': platform.python_version()
        }
    
    def get_pytorch_info(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Get PyTorch installation information."""
        try:
            import torch
            pytorch_version = torch.__version__
            cuda_version = torch.version.cuda if torch.cuda.is_available() else None
            return True, pytorch_version, cuda_version
        except ImportError:
            return False, None, None
    
    def get_cuda_runtime_version(self) -> Optional[str]:
        """Get CUDA runtime version if available."""
        try:
            # Try to import torch and get CUDA runtime version
            import torch
            if torch.cuda.is_available():
                return torch.version.cuda
        except ImportError:
            pass
        return None
    
    def evaluate_pytorch_compatibility(self, pytorch_version: Optional[str], pytorch_cuda: Optional[str], 
                                     system_cuda: Optional[str]) -> Tuple[str, Optional[str]]:
        """Evaluate compatibility between installed PyTorch and system CUDA.
        
        Returns:
            Tuple of (status_message, suggested_install_command)
        """
        if not pytorch_version:
            # No PyTorch installed - suggest best version for system CUDA
            if system_cuda:
                suggested_pytorch = self._get_best_pytorch_for_cuda(system_cuda)
                install_cmd = self._get_pytorch_install_command(suggested_pytorch, system_cuda)
                return "No PyTorch installed", install_cmd
            else:
                return "No PyTorch installed", "pip install torch torchvision torchaudio"
        
        if not pytorch_cuda:
            # CPU-only PyTorch but CUDA is available
            if system_cuda:
                suggested_pytorch = self._get_best_pytorch_for_cuda(system_cuda)
                install_cmd = self._get_pytorch_install_command(suggested_pytorch, system_cuda)
                return "CPU-only PyTorch (no CUDA support)", install_cmd
            else:
                return "CPU-only PyTorch (no CUDA support)", None
        
        if not system_cuda:
            return "PyTorch has CUDA support but no CUDA toolkit detected", None
        
        # Check if versions are compatible
        try:
            pytorch_cuda_major_minor = '.'.join(pytorch_cuda.split('.')[:2])
            system_cuda_major_minor = '.'.join(system_cuda.split('.')[:2])
            
            if pytorch_cuda_major_minor == system_cuda_major_minor:
                return "✓ Perfect match", None
            elif pytorch_cuda_major_minor in self.PYTORCH_CUDA_COMPATIBILITY:
                return "✓ Compatible (forward compatibility)", None
            else:
                # Incompatible - suggest better version
                suggested_pytorch = self._get_best_pytorch_for_cuda(system_cuda)
                install_cmd = self._get_pytorch_install_command(suggested_pytorch, system_cuda)
                return "⚠ Version mismatch - may cause issues", install_cmd
        except (ValueError, AttributeError):
            return "? Unable to determine compatibility", None
    
    def _get_best_pytorch_for_cuda(self, cuda_version: str) -> Optional[str]:
        """Get the best PyTorch version for a given CUDA version."""
        cuda_major_minor = '.'.join(cuda_version.split('.')[:2])
        
        # Find compatible versions
        compatible_versions = []
        for cuda_ver, pytorch_versions in self.PYTORCH_CUDA_COMPATIBILITY.items():
            if cuda_major_minor.startswith(cuda_ver):
                compatible_versions.extend(pytorch_versions)
        
        if compatible_versions:
            # Return the latest compatible version
            return max(compatible_versions, key=lambda x: tuple(map(int, x.split('.'))))
        
        # If no exact match, try to find the closest lower CUDA version
        cuda_float = float(cuda_major_minor)
        best_cuda_ver = None
        for cuda_ver in self.PYTORCH_CUDA_COMPATIBILITY.keys():
            cuda_ver_float = float(cuda_ver)
            if cuda_ver_float <= cuda_float:
                if best_cuda_ver is None or cuda_ver_float > float(best_cuda_ver):
                    best_cuda_ver = cuda_ver
        
        if best_cuda_ver:
            pytorch_versions = self.PYTORCH_CUDA_COMPATIBILITY[best_cuda_ver]
            return max(pytorch_versions, key=lambda x: tuple(map(int, x.split('.'))))
        
        return None
    
    def _get_pytorch_install_command(self, pytorch_version: Optional[str], cuda_version: str) -> str:
        """Generate PyTorch installation command."""
        if not pytorch_version:
            return "pip install torch torchvision torchaudio"
        
        cuda_major_minor = '.'.join(cuda_version.split('.')[:2]).replace('.', '')
        
        # Map CUDA versions to PyTorch index URLs
        cuda_url_map = {
            '118': 'cu118',
            '121': 'cu121', 
            '124': 'cu124',
            '126': 'cu126',
            '128': 'cu128'
        }
        
        cuda_suffix = cuda_url_map.get(cuda_major_minor, f"cu{cuda_major_minor}")
        
        return (f"pip install torch=={pytorch_version} torchvision torchaudio "
                f"--index-url https://download.pytorch.org/whl/{cuda_suffix}")

    def get_cuda_status(self, check_pytorch: bool = True) -> CudaInfo:
        """Get comprehensive CUDA status information.
        
        Args:
            check_pytorch: Whether to check PyTorch installation and compatibility
            
        Returns:
            CudaInfo object containing all system and GPU information
            
        Example:
            >>> checker = CudaStatusChecker()
            >>> info = checker.get_cuda_status()
            >>> print(f"CUDA Available: {info.cuda_available}")
            >>> print(f"GPU Count: {len(info.gpus)}")
            >>> 
            >>> # Check without PyTorch
            >>> info = checker.get_cuda_status(check_pytorch=False)
        """
        smi_available, gpu_info = self.check_nvidia_smi()
        cuda_version = self.get_cuda_version()
        driver_version = self.get_driver_version()
        cuda_runtime_version = self.get_cuda_runtime_version()
        system_info = self.get_system_info()
        
        # Get PyTorch information if requested
        pytorch_installed = False
        pytorch_version = None
        pytorch_cuda_version = None
        pytorch_compatibility_status = "Not checked"
        pytorch_install_suggestion = None
        
        if check_pytorch:
            pytorch_installed, pytorch_version, pytorch_cuda_version = self.get_pytorch_info()
            pytorch_compatibility_status, pytorch_install_suggestion = self.evaluate_pytorch_compatibility(
                pytorch_version, pytorch_cuda_version, cuda_version
            )
        
        gpus = gpu_info['gpus'] if gpu_info else []
        total_memory_gb = gpu_info.get('total_memory_gb', 0.0) if gpu_info else 0.0
        
        pytorch_compatible, recommended_pytorch, compatibility_notes = self.check_pytorch_compatibility(cuda_version)
        
        return CudaInfo(
            cuda_available=smi_available and cuda_version is not None,
            cuda_version=cuda_version,
            driver_version=driver_version,
            cuda_runtime_version=cuda_runtime_version,
            gpus=gpus,
            total_memory_gb=total_memory_gb,
            pytorch_compatible=pytorch_compatible,
            recommended_pytorch_version=recommended_pytorch,
            compatibility_notes=compatibility_notes,
            system_info=system_info,
            pytorch_installed=pytorch_installed,
            pytorch_version=pytorch_version,
            pytorch_cuda_version=pytorch_cuda_version,
            pytorch_compatibility_status=pytorch_compatibility_status,
            pytorch_install_suggestion=pytorch_install_suggestion
        )