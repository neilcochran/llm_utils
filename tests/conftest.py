"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def fixtures_path():
    """Return path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_cuda_info():
    """Return sample CudaInfo data for testing."""
    from cuda_info.cuda_status import CudaInfo
    
    return CudaInfo(
        cuda_available=True,
        cuda_version="12.1",
        driver_version="535.104.12",
        cuda_runtime_version="12.1",
        gpus=[
            {
                'name': 'NVIDIA GeForce RTX 4090',
                'total_memory_mb': '24564',
                'used_memory_mb': '1695',
                'free_memory_mb': '22869',
                'gpu_utilization_percent': '15',
                'temperature_c': '45',
                'power_draw_w': '68.2',
                'power_limit_w': '450.0'
            }
        ],
        total_memory_gb=24.0,
        pytorch_compatible=True,
        recommended_pytorch_version="2.1.0",
        compatibility_notes=["Compatible PyTorch versions available"],
        system_info={
            'platform': 'Linux',
            'python_version': '3.9.0',
            'architecture': 'x86_64',
            'platform_version': 'Linux-5.15.0'
        },
        pytorch_installed=True,
        pytorch_version="2.1.0+cu121",
        pytorch_cuda_version="12.1",
        pytorch_compatibility_status="Perfect match",
        pytorch_install_suggestion=None
    )