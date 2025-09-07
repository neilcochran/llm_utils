# LLM Utils

A collection of utilities for working with Large Language Models, focusing on system compatibility, model management, and performance analysis.

## Features

### CUDA Status Checker
- **System Compatibility**: Check CUDA toolkit, driver versions, and GPU information
- **PyTorch Integration**: Verify PyTorch installation and CUDA compatibility
- **Smart Recommendations**: Get specific installation commands for compatible PyTorch versions
- **Multiple Output Formats**: Pretty-printed terminal output or JSON for scripting

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd llm_utils
```

2. Activate the virtual environment:
```bash
# Linux/Mac
source venv/bin/activate

# Windows
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### CUDA Check
Check your CUDA installation and PyTorch compatibility:

```bash
# Basic check with full information
python scripts/cuda_check.py

# CUDA only (skip PyTorch checks)
python scripts/cuda_check.py --no-pytorch

# JSON output for scripting
python scripts/cuda_check.py --json

# Save output to file
python scripts/cuda_check.py -o cuda_report.txt

# Plain text without colors
python scripts/cuda_check.py --no-color
```

#### Example Output
```
============================================================
                    SYSTEM INFORMATION                     
============================================================
Platform: Linux x86_64
Python: 3.11.5

============================================================
                      CUDA STATUS                          
============================================================
Status: ✓ Available
CUDA Toolkit: 12.1
Driver Version: 535.104.12

============================================================
                    GPU INFORMATION                        
============================================================

GPU 0: NVIDIA GeForce RTX 4090
  Memory: 1,695MB / 24,564MB
  [██░░░░░░░░░░░░░░░░░░] 7%
  Utilization: 0%
  Temperature: 45°C
  Power: 68.2W / 450.0W (15%)

Total GPU Memory: 24.0 GB

============================================================
                    PYTORCH STATUS                         
============================================================
Installed Version: 2.1.0+cu121
CUDA Support: 12.1
System Compatibility: ✓ Perfect match

============================================================
                    RECOMMENDATIONS                        
============================================================
Notes:
  • Compatible PyTorch versions: 2.1.0, 2.2.0, 2.3.0, 2.5.0
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--json` | Output in JSON format for scripting |
| `--no-color` | Disable colored output |
| `--no-pytorch` | Skip PyTorch installation checks |
| `-o, --output FILE` | Save output to file |
| `--version` | Show version information |
| `--help` | Show help message |

## Features Detail

### CUDA Detection
- Automatically detects CUDA toolkit version via `nvcc` or `nvidia-smi`
- Reports driver version and compatibility
- Identifies all available GPUs with detailed specifications

### GPU Monitoring
- **Memory Usage**: Shows used/total memory with visual progress bars
- **Utilization**: Current GPU compute usage percentage  
- **Temperature**: Real-time temperature monitoring with color-coded warnings
- **Power Consumption**: Power draw vs. power limit with efficiency indicators

### PyTorch Integration
- Detects installed PyTorch version and CUDA support
- Compares system CUDA with PyTorch CUDA version
- Provides specific installation commands for compatible versions
- Handles edge cases like CPU-only PyTorch on CUDA systems

### Smart Recommendations
- Suggests optimal PyTorch versions for your CUDA installation
- Warns about CUDA versions that are too new for current PyTorch releases
- Provides copy-paste installation commands with correct index URLs

## Color Coding

The output uses color coding to quickly identify status:
- 🟢 **Green**: Good/optimal values
- 🟡 **Yellow**: Warning/medium usage
- 🔴 **Red**: Critical/high usage values

### Thresholds
- **Memory**: >70% yellow, >90% red
- **GPU Utilization**: >50% yellow, >80% red  
- **Temperature**: >60°C yellow, >75°C red
- **Power**: >40% yellow, >80% red

## Architecture

The project follows clean separation of concerns:

- **`/src/`**: Pure data collection and processing logic
- **`/scripts/`**: CLI interfaces and display formatting
- **Portable**: All scripts can be run from any directory

## System Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (for GPU features)
- `nvidia-smi` available in PATH
- Optional: PyTorch for runtime compatibility checking

## JSON Output

For programmatic use, the `--json` flag outputs structured data:

```json
{
  "cuda_available": true,
  "cuda_version": "12.1",
  "driver_version": "535.104.12",
  "gpus": [
    {
      "name": "NVIDIA GeForce RTX 4090",
      "total_memory_mb": "24564",
      "used_memory_mb": "1695",
      "gpu_utilization_percent": "0",
      "temperature_c": "45",
      "power_draw_w": "68.2",
      "power_limit_w": "450.0"
    }
  ],
  "pytorch_installed": true,
  "pytorch_version": "2.1.0+cu121",
  "pytorch_cuda_version": "12.1",
  "pytorch_compatibility_status": "✓ Perfect match"
}
```

## License

[License information to be added]