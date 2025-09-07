"""Tests for CUDA status checker functionality."""

import pytest
import subprocess
from unittest.mock import patch, mock_open
from pathlib import Path

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cuda_info.cuda_status import CudaStatusChecker, CudaInfo


class TestCudaStatusChecker:
    """Test the CudaStatusChecker class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.checker = CudaStatusChecker()
        self.fixtures_path = Path(__file__).parent / "fixtures"

    def load_fixture(self, filename: str) -> str:
        """Load a test fixture file."""
        return (self.fixtures_path / filename).read_text().strip()

    def test_init(self):
        """Test that CudaStatusChecker initializes correctly."""
        checker = CudaStatusChecker()
        assert hasattr(checker, 'PYTORCH_CUDA_COMPATIBILITY')
        assert isinstance(checker.PYTORCH_CUDA_COMPATIBILITY, dict)

    @patch('subprocess.run')
    def test_check_nvidia_smi_single_gpu(self, mock_run):
        """Test nvidia-smi parsing with single GPU."""
        # Mock successful nvidia-smi call
        mock_run.return_value.stdout = self.load_fixture("nvidia_smi_single_gpu.txt")
        mock_run.return_value.returncode = 0

        success, data = self.checker.check_nvidia_smi()

        assert success is True
        assert data is not None
        assert len(data['gpus']) == 1
        
        gpu = data['gpus'][0]
        assert gpu['name'] == 'NVIDIA GeForce RTX 4090'
        assert gpu['total_memory_mb'] == '24564'
        assert gpu['used_memory_mb'] == '1695'
        assert gpu['gpu_utilization_percent'] == '0'
        assert gpu['temperature_c'] == '45'
        assert gpu['power_draw_w'] == '68.2'
        assert gpu['power_limit_w'] == '450.0'
        
        # Check total memory calculation
        assert abs(data['total_memory_gb'] - 24.0) < 0.1

    @patch('subprocess.run')
    def test_check_nvidia_smi_multi_gpu(self, mock_run):
        """Test nvidia-smi parsing with multiple GPUs."""
        mock_run.return_value.stdout = self.load_fixture("nvidia_smi_multi_gpu.txt")
        mock_run.return_value.returncode = 0

        success, data = self.checker.check_nvidia_smi()

        assert success is True
        assert data is not None
        assert len(data['gpus']) == 2
        
        # Test first GPU
        gpu1 = data['gpus'][0]
        assert gpu1['name'] == 'NVIDIA GeForce RTX 4090'
        assert gpu1['gpu_utilization_percent'] == '15'
        
        # Test second GPU
        gpu2 = data['gpus'][1]
        assert gpu2['name'] == 'NVIDIA GeForce RTX 3080'
        assert gpu2['gpu_utilization_percent'] == '85'
        assert gpu2['temperature_c'] == '72'

    @patch('subprocess.run')
    def test_check_nvidia_smi_failure(self, mock_run):
        """Test nvidia-smi when command fails."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'nvidia-smi')

        success, data = self.checker.check_nvidia_smi()

        assert success is False
        assert data is None

    @patch('subprocess.run')
    def test_check_nvidia_smi_not_found(self, mock_run):
        """Test nvidia-smi when command not found."""
        mock_run.side_effect = FileNotFoundError()

        success, data = self.checker.check_nvidia_smi()

        assert success is False
        assert data is None

    @patch('subprocess.run')
    def test_get_cuda_version_nvcc(self, mock_run):
        """Test CUDA version detection via nvcc."""
        mock_run.return_value.stdout = self.load_fixture("nvcc_version.txt")
        mock_run.return_value.returncode = 0

        version = self.checker.get_cuda_version()

        assert version == "12.1"

    @patch('subprocess.run')
    def test_get_cuda_version_nvidia_smi_fallback(self, mock_run):
        """Test CUDA version detection fallback to nvidia-smi."""
        def side_effect(*args, **kwargs):
            if 'nvcc' in args[0]:
                raise FileNotFoundError()
            elif 'nvidia-smi' in args[0]:
                mock_result = type('MockResult', (), {})()
                mock_result.stdout = self.load_fixture("nvidia_smi_version.txt")
                mock_result.returncode = 0
                return mock_result
            
        mock_run.side_effect = side_effect

        version = self.checker.get_cuda_version()

        assert version == "12.2"

    @patch('subprocess.run')
    def test_get_cuda_version_not_found(self, mock_run):
        """Test CUDA version when not found."""
        mock_run.side_effect = FileNotFoundError()

        version = self.checker.get_cuda_version()

        assert version is None

    def test_check_pytorch_compatibility_compatible(self):
        """Test PyTorch compatibility for supported CUDA version."""
        compatible, recommended, notes = self.checker.check_pytorch_compatibility("12.1")

        assert compatible is True
        assert recommended is not None
        assert any("2.5.0" in version for version in self.checker.PYTORCH_CUDA_COMPATIBILITY["12.1"])
        assert len(notes) > 0
        assert "Compatible PyTorch versions" in notes[0]

    def test_check_pytorch_compatibility_too_new(self):
        """Test PyTorch compatibility for CUDA version that's too new."""
        compatible, recommended, notes = self.checker.check_pytorch_compatibility("15.0")

        assert compatible is False
        assert recommended is None
        assert any("too new" in note for note in notes)

    def test_check_pytorch_compatibility_no_cuda(self):
        """Test PyTorch compatibility when CUDA not available."""
        compatible, recommended, notes = self.checker.check_pytorch_compatibility(None)

        assert compatible is False
        assert recommended is None
        assert "CUDA not available" in notes

    @patch('subprocess.run')
    def test_get_driver_version(self, mock_run):
        """Test driver version detection."""
        mock_run.return_value.stdout = "535.104.12"
        mock_run.return_value.returncode = 0

        version = self.checker.get_driver_version()

        assert version == "535.104.12"

    def test_get_system_info(self):
        """Test system information collection."""
        info = self.checker.get_system_info()

        assert isinstance(info, dict)
        assert 'platform' in info
        assert 'python_version' in info
        assert 'architecture' in info
        assert 'platform_version' in info

    @patch('subprocess.run')
    def test_get_pytorch_info_not_installed(self, mock_run):
        """Test PyTorch info when PyTorch not installed."""
        with patch.dict('sys.modules', {'torch': None}):
            with patch('builtins.__import__', side_effect=ImportError()):
                installed, version, cuda_version = self.checker.get_pytorch_info()

        assert installed is False
        assert version is None
        assert cuda_version is None

    def test_evaluate_pytorch_compatibility_no_pytorch(self):
        """Test PyTorch compatibility evaluation when not installed."""
        status, suggestion = self.checker.evaluate_pytorch_compatibility(None, None, "12.1")

        assert status == "No PyTorch installed"
        assert suggestion is not None
        assert "pip install" in suggestion
        assert "torch==" in suggestion

    def test_evaluate_pytorch_compatibility_perfect_match(self):
        """Test PyTorch compatibility evaluation for perfect match."""
        status, suggestion = self.checker.evaluate_pytorch_compatibility("2.1.0", "12.1", "12.1")

        assert "Perfect match" in status
        assert suggestion is None

    def test_evaluate_pytorch_compatibility_mismatch(self):
        """Test PyTorch compatibility evaluation for version mismatch."""
        status, suggestion = self.checker.evaluate_pytorch_compatibility("1.12.0", "11.6", "12.1")

        assert "mismatch" in status
        assert suggestion is not None

    def test_get_best_pytorch_for_cuda(self):
        """Test getting best PyTorch version for CUDA."""
        best_version = self.checker._get_best_pytorch_for_cuda("12.1")

        assert best_version is not None
        assert best_version in self.checker.PYTORCH_CUDA_COMPATIBILITY["12.1"]

    def test_get_pytorch_install_command(self):
        """Test PyTorch installation command generation."""
        cmd = self.checker._get_pytorch_install_command("2.1.0", "12.1")

        assert "pip install" in cmd
        assert "torch==2.1.0" in cmd
        assert "cu121" in cmd
        assert "pytorch.org/whl" in cmd

    @patch('cuda_info.cuda_status.CudaStatusChecker.check_nvidia_smi')
    @patch('cuda_info.cuda_status.CudaStatusChecker.get_cuda_version')
    @patch('cuda_info.cuda_status.CudaStatusChecker.get_pytorch_info')
    def test_get_cuda_status_complete(self, mock_pytorch_info, mock_cuda_version, mock_nvidia_smi):
        """Test complete CUDA status collection."""
        # Mock nvidia-smi
        mock_nvidia_smi.return_value = (True, {
            'gpus': [{
                'name': 'NVIDIA GeForce RTX 4090',
                'total_memory_mb': '24564',
                'used_memory_mb': '1695',
                'free_memory_mb': '22869',
                'gpu_utilization_percent': '0',
                'temperature_c': '45',
                'power_draw_w': '68.2',
                'power_limit_w': '450.0'
            }],
            'total_memory_gb': 24.0
        })
        
        # Mock CUDA version
        mock_cuda_version.return_value = "12.1"
        
        # Mock PyTorch info
        mock_pytorch_info.return_value = (True, "2.1.0+cu121", "12.1")

        info = self.checker.get_cuda_status(check_pytorch=True)

        assert isinstance(info, CudaInfo)
        assert info.cuda_available is True
        assert len(info.gpus) == 1
        assert info.pytorch_installed is True
        assert info.pytorch_version == "2.1.0+cu121"
        assert info.pytorch_cuda_version == "12.1"

    @patch('subprocess.run')
    def test_get_cuda_status_no_pytorch_check(self, mock_run):
        """Test CUDA status collection without PyTorch check."""
        mock_run.return_value.stdout = self.load_fixture("nvidia_smi_single_gpu.txt")
        mock_run.return_value.returncode = 0

        info = self.checker.get_cuda_status(check_pytorch=False)

        assert isinstance(info, CudaInfo)
        assert info.pytorch_installed is False
        assert info.pytorch_compatibility_status == "Not checked"

    def test_cuda_info_to_dict(self):
        """Test CudaInfo to_dict conversion."""
        info = CudaInfo(
            cuda_available=True,
            cuda_version="12.1",
            driver_version="535.104.12",
            cuda_runtime_version="12.1",
            gpus=[],
            total_memory_gb=24.0,
            pytorch_compatible=True,
            recommended_pytorch_version="2.1.0",
            compatibility_notes=[],
            system_info={'platform': 'Linux'},
            pytorch_installed=False,
            pytorch_version=None,
            pytorch_cuda_version=None,
            pytorch_compatibility_status="Not checked",
            pytorch_install_suggestion=None
        )

        result = info.to_dict()

        assert isinstance(result, dict)
        assert result['cuda_available'] is True
        assert result['cuda_version'] == "12.1"
        assert result['total_memory_gb'] == 24.0