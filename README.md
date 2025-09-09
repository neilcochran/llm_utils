# LLM Utils

A collection of utilities for working with Large Language Models, focusing on system compatibility, model management, and performance analysis.

## Features

### Model Performance Evaluation
- **Hardware Monitoring**: Real-time CPU, RAM, and GPU usage tracking during inference
- **Performance Metrics**: Tokens/second, time to first token, total inference time
- **Resource Analytics**: Peak and average resource consumption analysis
- **Flexible Testing**: Custom queries, batch testing, and configurable parameters
- **Multiple Output Formats**: Interactive display, JSON, and CSV export

### CUDA Status Checker
- **System Compatibility**: Check CUDA toolkit, driver versions, and GPU information
- **PyTorch Integration**: Verify PyTorch installation and CUDA compatibility
- **Smart Recommendations**: Get specific installation commands for compatible PyTorch versions
- **Multiple Output Formats**: Pretty-printed terminal output or JSON for scripting


### Ollama Model Management
- **Export to .gguf**: Extract Ollama models as portable .gguf files with Modelfiles
- **Import from .gguf**: Import .gguf files back into Ollama with automatic Modelfile handling
- **Batch Operations**: Export/import single models, multiple models, or all models at once
- **Progress Tracking**: Real-time progress bars and status updates
- **Cross-platform**: Works on Windows, Linux, and macOS



## Architecture

The project follows clean separation of concerns:

- **`/src/`**: Pure data collection and processing logic
- **`/scripts/`**: CLI interfaces and display formatting
- **Portable**: All scripts can be run from any directory

## System Requirements

- Python 3.8+
- **For CUDA features**: NVIDIA GPU with CUDA support, `nvidia-smi` in PATH
- **For Ollama features**: Ollama installed and accessible via PATH
- **For model evaluation**: Ollama models available locally, psutil for system monitoring
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

## Usage:

### Model Performance Evaluation

---

Evaluate LLM performance with comprehensive hardware monitoring:

```bash
# Basic model evaluation with default queries
python scripts/model_evaluation.py -m llama2:7b

# Custom single query evaluation
python scripts/model_evaluation.py -m qwen2.5:14b -q "Explain quantum computing in simple terms"

# Multiple custom queries
python scripts/model_evaluation.py -m codellama:13b --queries "Write a Python function" "Debug this code" "Optimize performance"

# Batch evaluation from file
python scripts/model_evaluation.py -m starcoder2:15b --queries-file evaluation_queries.txt

# Detailed evaluation with verbose output
python scripts/model_evaluation.py -m llama2:7b -v --temperature 0.8 --max-tokens 500

# JSON output for analysis
python scripts/model_evaluation.py -m qwen2.5:14b --json -o results.json

# Skip resource monitoring (faster)
python scripts/model_evaluation.py -m llama2:7b --no-resources
```

#### Model Evaluation Options
| Option | Description |
|--------|-------------|
| `-m, --model MODEL` | Model name to evaluate (required) |
| `--backend BACKEND` | Model backend to use (default: ollama) |
| `-q, --query QUERY` | Single query to evaluate |
| `--queries QUERIES [...]` | Multiple queries (space-separated) |
| `--queries-file FILE` | File containing queries (one per line) |
| `--temperature TEMP` | Generation temperature (default: 0.7) |
| `--max-tokens N` | Maximum tokens to generate |
| `--timeout N` | Timeout per query in seconds (default: 300) |
| `--no-resources` | Skip resource monitoring |
| `-v, --verbose` | Show detailed output and live metrics |
| `--no-color` | Disable colored output |
| `--json` | Output results in JSON format |
| `-o, --output FILE` | Save results to file |

#### Example Output
```
============================================================
 Model Evaluation: qwen3:0.6b
============================================================

Model: qwen3:0.6b
Backend: ollama
Status: Available
Quantization: quantization        Q4_K_M
Model initialization: 2.4s

Starting evaluation of 5 queries...
[5/5] Evaluation complete....

============================================================
 EVALUATION SUMMARY
============================================================

Total Duration: 36.9s
Model Initialization: 2.4s
Queries Evaluated: 5
Successful Queries: 5

Performance Averages (excluding initialization):
  Time to first token: 480.0ms
  Total inference time: 7.1s
  Tokens per second: 78.6
  Total tokens generated: 2867

Resource Usage:
  CPU: 21.0% (max: 43.7%)
  Memory: 15.6GB / 31.7GB (49.3%) (max: 49.3%)
  GPU: 56.9% (max: 90.0%)
  GPU Memory: 3.1GB / 8.0GB (39.1%) (max: 39.1%)
  GPU Power: 109.1W (max: 145.9W)
```

### CUDA Check

---

Check your CUDA installation and PyTorch compatibility:

```bash
# Basic check with full information
python scripts/cuda_check.py

# CUDA only (skip PyTorch checks)
python scripts/cuda_check.py --no-pytorch

# JSON output for scripting
python scripts/cuda_check.py --json
```

#### CUDA Check Options
| Option | Description |
|--------|-------------|
| `--json` | Output in JSON format |
| `--no-color` | Disable colored output |
| `--no-pytorch` | Skip PyTorch checks |
| `-o, --output FILE` | Save output to file |

### JSON Output

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

#### Example Output
```
============================================================
                    SYSTEM INFORMATION                     
============================================================
Platform: Windows AMD64
Python: 3.10.4

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
  Memory: 1,695MB / 24,564MB (7%)
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
```

<br>

### Ollama Model Management

#### Export Models

Export Ollama models to portable .gguf format:

```bash
# List available models
python scripts/ollama_export.py --list --ollama-path "D:\ollama-models"

# Export single model
python scripts/ollama_export.py -m llama2:7b -o ./exports --ollama-path "D:\ollama-models"

# Export specific models (multiple)
python scripts/ollama_export.py --models llama2:7b codellama:13b qwen2.5:14b -o ./exports --ollama-path "D:\ollama-models"

# Export all models with progress
python scripts/ollama_export.py --all -o ./backups --ollama-path "/home/user/.ollama/models" -v

# Export with verbose output
python scripts/ollama_export.py -m starcoder2:15b -o ./exports --ollama-path "D:\ollama-models" -v
```

#### Ollama Export Options
| Option | Description |
|--------|-------------|
| `--ollama-path PATH` | Path to Ollama models directory (required) |
| `-m, --model NAME` | Export specific model |
| `--models NAME [NAME ...]` | Export multiple specific models (space-separated) |
| `--all` | Export all available models |
| `--list` | List available models |
| `-o, --output DIR` | Output directory (default: ./ollama_exports) |
| `-v, --verbose` | Show detailed progress |

#### Example Output
```
============================================================
 Exporting Model: qwen2.5:14b
============================================================

⏳ Retrieving model information from Ollama
✓ Model information retrieved successfully
⏳ Creating export directory
✓ Export directory created: ./exports/qwen2-5-14b
⏳ Writing Modelfile
✓ Modelfile written successfully
⏳ Locating .gguf file
✓ Found .gguf file (8324.5 MB), copying...
Copying |████████████████████████████████████████| 100.0% 8324.5MB / 8324.5MB
✓ .gguf file copied successfully

Model: qwen2.5:14b
Status: SUCCESS
Export Directory: ./exports/qwen2-5-14b
Modelfile: ✓ ./exports/qwen2-5-14b/Modelfile
GGUF File: ✓ ./exports/qwen2-5-14b/qwen2-5-14b.gguf
Parameters: 3 settings
Template: {{- if .Messages }}...
------------------------------------------------------------
```

#### Import Models

Import .gguf files back into Ollama:

```bash
# Import single .gguf file with custom name
python scripts/ollama_import.py -f model.gguf -n my-model

# Import multiple specific .gguf files
python scripts/ollama_import.py --files model1.gguf model2.gguf model3.gguf

# Import all .gguf files from a directory
python scripts/ollama_import.py -d ./exported_models

# Import with verbose output
python scripts/ollama_import.py -f model.gguf -n my-model -v
```

#### Ollama Import Options
| Option | Description |
|--------|-------------|
| `-f, --file PATH` | Import single .gguf file |
| `--files PATH [PATH ...]` | Import multiple .gguf files (space-separated) |
| `-d, --directory PATH` | Import all .gguf files from directory |
| `-n, --name NAME` | Custom name for imported model (only with --file) |
| `-v, --verbose` | Show detailed progress |
| `--no-color` | Disable colored output |
| `--json` | Output results in JSON format |

#### Import Example Output
```
============================================================
 Importing Model: my-model
============================================================

⏳ Validating .gguf file
✓ .gguf file validated
⏳ Looking for Modelfile
✓ Modelfile found and read
⏳ Creating Ollama model 'my-model'
✓ Model 'my-model' created successfully

Model: my-model
Status: SUCCESS
GGUF Source: ./model.gguf
Modelfile: ✓ ./Modelfile
Ollama Import: ✓ Created successfully
------------------------------------------------------------
```

## License

This project is licensed under the MIT License - see the <a href="/LICENSE.md">LICENSE.md</a> file for details