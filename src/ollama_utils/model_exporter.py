"""Ollama model export functionality - data collection only."""

import logging
import subprocess
import re
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelExportInfo:
    """Information about an exported Ollama model."""
    model_name: str
    sanitized_name: str
    template: Optional[str]
    parameters: List[str]
    system_message: Optional[str]
    modelfile_content: str
    gguf_source_path: Optional[str]
    export_directory: Optional[str]
    modelfile_path: Optional[str]
    gguf_destination_path: Optional[str]
    success: bool
    error_message: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class OllamaModelExporter:
    """Export Ollama models to .gguf files and Modelfiles - data only."""
    
    def __init__(self, ollama_models_path: str):
        """Initialize the model exporter.
        
        Args:
            ollama_models_path: Path to Ollama models directory (required).
        """
        if not ollama_models_path:
            raise ValueError("ollama_models_path is required")
        if not os.path.exists(ollama_models_path):
            raise ValueError(f"Ollama models directory does not exist: {ollama_models_path}")
        
        self.ollama_models_path = ollama_models_path
    
    def sanitize_model_name(self, name: str) -> str:
        """Sanitize model name for use as filename.
        
        Args:
            name: Original model name
            
        Returns:
            Sanitized filename-safe string
        """
        name = name.replace(":latest", "")
        return re.sub(r'[<>:"/\\|?*.]', '-', name)
    
    def run_ollama_command(self, command: str) -> Tuple[bool, Optional[str]]:
        """Execute an ollama command and return the output.
        
        Args:
            command: The ollama command to run
            
        Returns:
            Tuple of (success, output) where output is None on failure
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8',
                errors='replace'
            )
            return True, result.stdout.strip()
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr if e.stderr else "No error message"
            logger.error(f"Command failed: {command}, Error: {stderr_msg}")
            return False, None
        except FileNotFoundError:
            logger.error(f"Command not found: {command}")
            return False, None
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error for command: {command}, Error: {e}")
            return False, None
        except Exception as e:
            logger.error(f"Unexpected error running command: {command}, Error: {e}")
            return False, None
    
    def get_model_info(self, model_name: str) -> Tuple[bool, Dict[str, Any]]:
        """Get comprehensive information about an Ollama model.
        
        Args:
            model_name: Name of the Ollama model
            
        Returns:
            Tuple of (success, model_info_dict)
        """
        model_info = {
            'template': None,
            'parameters': [],
            'system_message': None,
            'modelfile': None
        }
        
        # Get template
        success, template = self.run_ollama_command(f'ollama show --template {model_name}')
        if not success or not template:
            return False, {'error': f'Model "{model_name}" not found or template is empty'}
        model_info['template'] = template
        
        # Get parameters
        success, parameters = self.run_ollama_command(f'ollama show --parameters {model_name}')
        if success and parameters:
            model_info['parameters'] = [line.strip() for line in parameters.splitlines() if line.strip()]
        
        # Get system message
        success, system_message = self.run_ollama_command(f'ollama show --system {model_name}')
        if success and system_message:
            model_info['system_message'] = system_message
        
        # Get modelfile
        success, modelfile = self.run_ollama_command(f'ollama show --modelfile {model_name}')
        if success and modelfile:
            model_info['modelfile'] = modelfile
        
        return True, model_info
    
    def extract_gguf_path(self, modelfile_content: str) -> Optional[str]:
        """Extract the .gguf file path from modelfile content.
        
        Args:
            modelfile_content: Content of the modelfile
            
        Returns:
            Path to the .gguf file or None if not found
        """
        if not modelfile_content:
            return None
        
        # ollama_models_path is required - no fallbacks
        assert self.ollama_models_path, "Ollama models path must be configured"
        assert os.path.exists(self.ollama_models_path), f"Ollama models path must exist: {self.ollama_models_path}"
        
        # Normalize path separators for cross-platform compatibility
        normalized_models_path = os.path.normpath(self.ollama_models_path)
        
        # Try with both forward and backward slashes since Windows paths can vary
        path_variants = [
            normalized_models_path,
            normalized_models_path.replace('\\', '/'),
            normalized_models_path.replace('/', '\\')
        ]
        
        for path_variant in path_variants:
            model_file_location_match = re.search(
                fr'FROM\s+({re.escape(path_variant)}[^\s]*)', 
                modelfile_content, 
                re.MULTILINE
            )
            if model_file_location_match:
                extracted_path = model_file_location_match.group(1)
                return extracted_path
        
        # Debug output to help diagnose the issue
        logger.debug(f"Failed to find path in modelfile content:")
        logger.debug(f"Looking for path variants: {path_variants}")
        logger.debug(f"Modelfile content: {modelfile_content[:200]}...")
        
        return None
    
    def create_modelfile_content(self, model_name: str, model_info: Dict[str, Any]) -> str:
        """Create the Modelfile content for export.
        
        Args:
            model_name: Sanitized model name
            model_info: Dictionary containing model information
            
        Returns:
            Generated Modelfile content
        """
        content_lines = [f"FROM {model_name}.gguf"]
        
        # Add template
        if model_info.get('template'):
            template = model_info['template']
            content_lines.append(f'TEMPLATE """{template}"""')
        
        # Add parameters
        for param in model_info.get('parameters', []):
            content_lines.append(f'PARAMETER {param}')
        
        # Add system message
        if model_info.get('system_message'):
            system_msg = model_info['system_message']
            content_lines.append(f'SYSTEM "{system_msg}"')
        
        return '\n'.join(content_lines) + '\n'
    
    def copy_with_progress(self, src: str, dst: str, 
                          progress_callback: Optional[Callable[[int, int, str], None]] = None) -> bool:
        """Copy a file with progress reporting.
        
        Args:
            src: Source file path
            dst: Destination file path  
            progress_callback: Optional callback function(bytes_copied, total_bytes, status)
            
        Returns:
            True if copy succeeded, False otherwise
        """
        try:
            src_path = Path(src)
            dst_path = Path(dst)
            
            if not src_path.exists():
                return False
            
            file_size = src_path.stat().st_size
            bytes_copied = 0
            
            if progress_callback:
                progress_callback(0, file_size, "Starting copy...")
            
            # Copy in chunks for progress reporting
            with open(src_path, 'rb') as fsrc:
                with open(dst_path, 'wb') as fdst:
                    chunk_size = 64 * 1024  # 64KB chunks
                    while True:
                        chunk = fsrc.read(chunk_size)
                        if not chunk:
                            break
                        
                        fdst.write(chunk)
                        bytes_copied += len(chunk)
                        
                        if progress_callback:
                            progress_callback(bytes_copied, file_size, "Copying...")
            
            if progress_callback:
                progress_callback(file_size, file_size, "Copy complete")
            
            return True
            
        except Exception as e:
            logger.error(f"Error copying file from {src} to {dst}: {e}")
            if progress_callback:
                progress_callback(0, 0, f"Copy failed: {e}")
            return False
    
    def export_model(self, model_name: str, output_directory: str,
                    step_callback: Optional[Callable[[str, str], None]] = None,
                    progress_callback: Optional[Callable[[int, int, str], None]] = None) -> ModelExportInfo:
        """Export an Ollama model to .gguf and Modelfile.
        
        Args:
            model_name: Name of the Ollama model to export
            output_directory: Directory where to save the exported files
            step_callback: Optional callback for step updates(step_description, status)
            progress_callback: Optional callback for file copy progress(bytes_copied, total_bytes, message)
            
        Returns:
            ModelExportInfo with export results
        """
        sanitized_name = self.sanitize_model_name(model_name)
        
        # Initialize result object
        export_info = ModelExportInfo(
            model_name=model_name,
            sanitized_name=sanitized_name,
            template=None,
            parameters=[],
            system_message=None,
            modelfile_content="",
            gguf_source_path=None,
            export_directory=None,
            modelfile_path=None,
            gguf_destination_path=None,
            success=False,
            error_message=None
        )
        
        try:
            # Get model information
            if step_callback:
                step_callback("Retrieving model information from Ollama", "working")
            
            success, model_info = self.get_model_info(model_name)
            if not success:
                if step_callback:
                    step_callback("Failed to retrieve model information", "error")
                export_info.error_message = model_info.get('error', 'Failed to get model information')
                return export_info
            
            if step_callback:
                step_callback("Model information retrieved successfully", "done")
            
            # Update export info with model data
            export_info.template = model_info['template']
            export_info.parameters = model_info['parameters']
            export_info.system_message = model_info['system_message']
            
            # Create output directory
            if step_callback:
                step_callback("Creating export directory", "working")
            
            model_dir = Path(output_directory) / sanitized_name
            model_dir.mkdir(parents=True, exist_ok=True)
            export_info.export_directory = str(model_dir)
            
            if step_callback:
                step_callback(f"Export directory created: {model_dir}", "done")
            
            # Create Modelfile content
            modelfile_content = self.create_modelfile_content(sanitized_name, model_info)
            export_info.modelfile_content = modelfile_content
            
            # Write Modelfile
            if step_callback:
                step_callback("Writing Modelfile", "working")
            
            modelfile_path = model_dir / "Modelfile"
            with open(modelfile_path, 'w', encoding='utf-8') as f:
                f.write(modelfile_content)
            export_info.modelfile_path = str(modelfile_path)
            
            if step_callback:
                step_callback("Modelfile written successfully", "done")
            
            # Extract and copy .gguf file
            if step_callback:
                step_callback("Locating .gguf file", "working")
            
            gguf_source = self.extract_gguf_path(model_info.get('modelfile', ''))
            if gguf_source and Path(gguf_source).exists():
                export_info.gguf_source_path = gguf_source
                
                if step_callback:
                    file_size = Path(gguf_source).stat().st_size
                    size_mb = file_size / (1024 * 1024)
                    step_callback(f"Found .gguf file ({size_mb:.1f} MB), copying...", "working")
                
                gguf_dest = model_dir / f"{sanitized_name}.gguf"
                
                # Use progress-aware copy
                copy_success = self.copy_with_progress(gguf_source, str(gguf_dest), progress_callback)
                
                if copy_success:
                    export_info.gguf_destination_path = str(gguf_dest)
                    if step_callback:
                        step_callback(".gguf file copied successfully", "done")
                else:
                    if step_callback:
                        step_callback("Failed to copy .gguf file", "error")
            else:
                logger.warning(f"Could not find .gguf file for model {model_name}")
                if step_callback:
                    step_callback(".gguf file not found", "skip")
            
            export_info.success = True
            
        except Exception as e:
            logger.error(f"Error exporting model {model_name}: {e}")
            export_info.error_message = str(e)
        
        return export_info
    
    def list_available_models(self) -> Tuple[bool, List[str]]:
        """List all available Ollama models.
        
        Returns:
            Tuple of (success, list_of_model_names)
        """
        success, output = self.run_ollama_command('ollama list')
        if not success or not output:
            return False, []
        
        models = []
        lines = output.strip().split('\n')
        
        # Skip header line
        for line in lines[1:]:
            if line.strip():
                # Model name is typically the first column
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    models.append(model_name)
        
        return True, models
    
    def export_all_models(self, output_directory: str) -> List[ModelExportInfo]:
        """Export all available Ollama models.
        
        Args:
            output_directory: Directory where to save all exported models
            
        Returns:
            List of ModelExportInfo for each model export attempt
        """
        results = []
        
        # Get list of models
        success, models = self.list_available_models()
        if not success:
            # Return single failed result
            failed_info = ModelExportInfo(
                model_name="",
                sanitized_name="",
                template=None,
                parameters=[],
                system_message=None,
                modelfile_content="",
                gguf_source_path=None,
                export_directory=None,
                modelfile_path=None,
                gguf_destination_path=None,
                success=False,
                error_message="Failed to list available models"
            )
            return [failed_info]
        
        # Export each model
        for model_name in models:
            result = self.export_model(model_name, output_directory)
            results.append(result)
        
        return results