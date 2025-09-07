#!/usr/bin/env python3
"""CLI script for checking CUDA status and PyTorch compatibility."""

import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

# Add src to Python path based on script location
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from cuda_info.cuda_status import CudaStatusChecker, CudaInfo


class CudaDisplayFormatter:
    """Handles all display formatting and color logic."""
    
    # Color codes for terminal output
    COLORS = {
        'GREEN': '\033[92m',
        'RED': '\033[91m',
        'YELLOW': '\033[93m',
        'RESET': '\033[0m'
    }
    
    # Thresholds for different metrics
    THRESHOLDS = {
        'memory': {'critical': 90, 'warning': 70},
        'utilization': {'critical': 80, 'warning': 50},
        'temperature': {'critical': 75, 'warning': 60},
        'power': {'critical': 80, 'warning': 40}
    }
    
    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors
    
    def _colorize(self, text: str, color_type: str) -> str:
        """Add color to text based on status type."""
        if not self.use_colors:
            return text
        try:
            if not os.isatty(1):  # Not a terminal
                return text
                
            if color_type == 'good':
                return f"{self.COLORS['GREEN']}{text}{self.COLORS['RESET']}"
            elif color_type == 'bad':
                return f"{self.COLORS['RED']}{text}{self.COLORS['RESET']}"
            elif color_type == 'warn':
                return f"{self.COLORS['YELLOW']}{text}{self.COLORS['RESET']}"
        except:
            pass
        return text
    
    def _create_progress_bar(self, percent: float, width: int = 20) -> str:
        """Create a simple progress bar without percentage."""
        filled = int(width * percent / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"
    
    def _format_percentage_value(self, value: float, metric_type: str, suffix: str = "%", brackets: bool = False) -> str:
        """Generic percentage formatter with color coding.
        
        Args:
            value: The percentage value to format
            metric_type: Type of metric ('memory', 'utilization', 'temperature', 'power')
            suffix: Suffix to add (default '%')
            brackets: Whether to wrap in parentheses
        """
        thresholds = self.THRESHOLDS.get(metric_type, {'critical': 90, 'warning': 70})
        
        formatted_value = f"{value:.0f}{suffix}"
        if brackets:
            formatted_value = f"({formatted_value})"
        
        if value > thresholds['critical']:
            return self._colorize(formatted_value, 'bad')
        elif value > thresholds['warning']:
            return self._colorize(formatted_value, 'warn')
        else:
            return self._colorize(formatted_value, 'good')
    
    def _format_gpu_memory_percentage(self, usage_percent: float) -> str:
        """Format and colorize memory usage percentage."""
        return self._format_percentage_value(usage_percent, 'memory')
    
    def _format_gpu_utilization(self, util_val: int) -> str:
        """Format and colorize GPU utilization."""
        return self._format_percentage_value(util_val, 'utilization')
    
    def _format_temperature(self, temp_val: int) -> str:
        """Format and colorize temperature."""
        return self._format_percentage_value(temp_val, 'temperature', '°C')
    
    def _format_power_percentage(self, power_percent: float) -> str:
        """Format and colorize power usage percentage."""
        return self._format_percentage_value(power_percent, 'power', '%', brackets=True)
    
    def _format_pytorch_compatibility(self, status: str) -> str:
        """Format and colorize PyTorch compatibility status."""
        if "✓" in status:
            return self._colorize(status, 'good')
        elif "⚠" in status:
            return self._colorize(status, 'warn')
        elif "?" in status:
            return self._colorize(status, 'warn')
        else:
            return status
    
    def format_json(self, info: CudaInfo) -> None:
        """Print JSON formatted output."""
        print(json.dumps(info.to_dict(), indent=2))
    
    def format_pretty(self, info: CudaInfo, check_pytorch: bool) -> None:
        """Print pretty formatted output.
        
        Args:
            info: CUDA information to display
            check_pytorch: Whether PyTorch checking was enabled
        """
        width = 60
        
        # System Information Section (moved to top)
        sys_info = info.system_info
        print("=" * width)
        print("SYSTEM INFORMATION".center(width))
        print("=" * width)
        print(f"Platform: {sys_info['platform']} {sys_info['architecture']}")
        print(f"Python: {sys_info['python_version']}")
        
        # CUDA Status Section
        print(f"\n{'=' * width}")
        print("CUDA STATUS".center(width))
        print("=" * width)
        
        cuda_status = self._colorize("✓ Available", 'good') if info.cuda_available else self._colorize("✗ Not Available", 'bad')
        print(f"Status: {cuda_status}")
        print(f"CUDA Toolkit: {info.cuda_version or 'Not found'}")
        print(f"Driver Version: {info.driver_version or 'Not found'}")
        if info.cuda_runtime_version:
            print(f"Runtime Version: {info.cuda_runtime_version}")
        
        # GPU Information Section  
        print(f"\n{'=' * width}")
        print("GPU INFORMATION".center(width))
        print("=" * width)
        
        if info.gpus:
            for i, gpu in enumerate(info.gpus):
                print(f"\nGPU {i}: {gpu['name']}")
                
                # Memory info with clean two-line format
                try:
                    total_mb = int(gpu['total_memory_mb'])
                    used_mb = int(gpu['used_memory_mb'])
                    usage_percent = (used_mb / total_mb) * 100 if total_mb > 0 else 0
                    
                    percent_colored = self._format_gpu_memory_percentage(usage_percent)
                    memory_bar = self._create_progress_bar(usage_percent, width=20)
                    
                    print(f"  Memory: {used_mb:,}MB / {total_mb:,}MB")
                    print(f"  {memory_bar} {percent_colored}")
                except (ValueError, TypeError):
                    print(f"  Memory: {gpu['used_memory_mb']}MB / {gpu['total_memory_mb']}MB")
                
                # GPU Utilization with color coding
                try:
                    util_val = int(gpu['gpu_utilization_percent'])
                    util_display = self._format_gpu_utilization(util_val)
                    print(f"  Utilization: {util_display}")
                except (ValueError, TypeError):
                    print(f"  Utilization: {gpu['gpu_utilization_percent']}%")
                
                # Temperature with color coding
                if gpu.get('temperature_c', 'N/A') != 'N/A':
                    try:
                        temp_val = int(gpu['temperature_c'])
                        temp_display = self._format_temperature(temp_val)
                        print(f"  Temperature: {temp_display}")
                    except (ValueError, TypeError):
                        print(f"  Temperature: {gpu['temperature_c']}°C")
                
                # Power usage with color coding on percentage only
                if gpu.get('power_draw_w', 'N/A') != 'N/A':
                    try:
                        power_draw = float(gpu['power_draw_w'])
                        power_limit = float(gpu['power_limit_w'])
                        power_percent = (power_draw / power_limit) * 100 if power_limit > 0 else 0
                        
                        percent_display = self._format_power_percentage(power_percent)
                        print(f"  Power: {power_draw:.1f}W / {power_limit:.1f}W {percent_display}")
                    except (ValueError, TypeError):
                        print(f"  Power: {gpu['power_draw_w']}W / {gpu['power_limit_w']}W")
        else:
            print(f"\n{self._colorize('No GPUs detected', 'bad')}")
        
        if info.total_memory_gb > 0:
            print(f"\nTotal GPU Memory: {info.total_memory_gb:.1f} GB")
        
        # PyTorch Installation Status (only if checking PyTorch)
        if check_pytorch:
            print(f"\n{'=' * width}")
            print("PYTORCH STATUS".center(width))
            print("=" * width)
            
            if info.pytorch_installed:
                print(f"Installed Version: {info.pytorch_version}")
                print(f"CUDA Support: {info.pytorch_cuda_version or 'CPU-only'}")
                
                compat_display = self._format_pytorch_compatibility(info.pytorch_compatibility_status)
                print(f"System Compatibility: {compat_display}")
            else:
                print(self._colorize("PyTorch not installed", 'warn'))
        
        # Recommendations Section
        print(f"\n{'=' * width}")
        print("RECOMMENDATIONS".center(width)) 
        print("=" * width)
        
        # Show PyTorch installation suggestions
        if check_pytorch and info.pytorch_install_suggestion:
            if not info.pytorch_installed:
                print("Install PyTorch with CUDA support:")
            elif "⚠" in info.pytorch_compatibility_status or "CPU-only" in info.pytorch_compatibility_status:
                print("Install compatible PyTorch version:")
            
            print(f"  {info.pytorch_install_suggestion}")
        elif info.pytorch_compatible and info.recommended_pytorch_version:
            print(f"Recommended PyTorch: {info.recommended_pytorch_version}")
        
        if info.compatibility_notes:
            if check_pytorch and info.pytorch_install_suggestion:
                print("\nAdditional Notes:")
            else:
                print("Notes:")
            for note in info.compatibility_notes:
                print(f"  • {note}")
        
        # Footer
        print(f"\n{'-' * width}")
        print("-" * width)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Check CUDA status and PyTorch compatibility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Full check with colors
  %(prog)s --no-pytorch       # CUDA only, skip PyTorch checks
  %(prog)s --json             # JSON output for scripts
  %(prog)s --no-color         # Plain text output
  %(prog)s -o report.txt      # Save output to file
        """)
    
    parser.add_argument('--json', action='store_true', 
                       help='Output in JSON format')
    parser.add_argument('--no-color', action='store_true', 
                       help='Disable colored output')
    parser.add_argument('--no-pytorch', action='store_true', 
                       help='Skip PyTorch installation checks (CUDA only)')
    parser.add_argument('-o', '--output', type=Path,
                       help='Save output to file instead of stdout')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')
    
    args = parser.parse_args()
    
    # Initialize checker and formatter
    checker = CudaStatusChecker()
    formatter = CudaDisplayFormatter(use_colors=not args.no_color and not args.output)
    
    # Get data
    check_pytorch = not args.no_pytorch
    info = checker.get_cuda_status(check_pytorch=check_pytorch)
    
    # Setup output destination
    output_file = None
    original_stdout = None
    
    try:
        if args.output:
            output_file = open(args.output, 'w', encoding='utf-8')
            original_stdout = sys.stdout
            sys.stdout = output_file
        
        # Display results
        if args.json:
            formatter.format_json(info)
        else:
            formatter.format_pretty(info, check_pytorch)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Restore stdout and close file
        if output_file:
            sys.stdout = original_stdout
            output_file.close()
            if not args.json:
                print(f"Output saved to: {args.output}")
        if original_stdout:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()