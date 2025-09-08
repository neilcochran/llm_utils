"""Ollama model import functionality - data collection only."""

import logging
import subprocess
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelImportInfo:
    """Information about an imported Ollama model."""
    model_name: str
    gguf_source_path: str
    modelfile_path: Optional[str]
    modelfile_content: str
    import_success: bool
    ollama_create_success: bool
    error_message: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class OllamaModelImporter:
    """Import .gguf files and Modelfiles into Ollama - data only."""
    
    def __init__(self):
        """Initialize the model importer."""
        pass
    
    def validate_gguf_file(self, gguf_path: str) -> bool:
        """Validate that a file is a .gguf file.
        
        Args:
            gguf_path: Path to the .gguf file
            
        Returns:
            True if valid .gguf file, False otherwise
        """
        if not gguf_path.endswith('.gguf'):
            logger.error(f"File does not have .gguf extension: {gguf_path}")
            return False
        
        if not os.path.exists(gguf_path):
            logger.error(f"GGUF file does not exist: {gguf_path}")
            return False
        
        if not os.path.isfile(gguf_path):
            logger.error(f"Path is not a file: {gguf_path}")
            return False
        
        # Check file size (GGUF files should be reasonably large)
        try:
            file_size = os.path.getsize(gguf_path)
            if file_size < 1024:  # Less than 1KB is suspicious
                logger.warning(f"GGUF file seems very small: {file_size} bytes")
                return False
        except OSError as e:
            logger.error(f"Cannot check file size: {e}")
            return False
        
        return True
    
    def find_modelfile(self, gguf_path: str) -> Optional[str]:
        """Find the Modelfile associated with a .gguf file.
        
        Args:
            gguf_path: Path to the .gguf file
            
        Returns:
            Path to the Modelfile, or None if not found
        """
        gguf_dir = os.path.dirname(gguf_path)
        
        # Common Modelfile names to check
        modelfile_names = ['Modelfile', 'modelfile', 'Modelfile.txt']
        
        for name in modelfile_names:
            modelfile_path = os.path.join(gguf_dir, name)
            if os.path.exists(modelfile_path) and os.path.isfile(modelfile_path):
                return modelfile_path
        
        return None
    
    def read_modelfile(self, modelfile_path: str) -> Tuple[bool, str]:
        """Read the contents of a Modelfile.
        
        Args:
            modelfile_path: Path to the Modelfile
            
        Returns:
            Tuple of (success, content)
        """
        try:
            with open(modelfile_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return True, content
        except Exception as e:
            logger.error(f"Failed to read Modelfile {modelfile_path}: {e}")
            return False, ""
    
    def create_default_modelfile_content(self, gguf_filename: str) -> str:
        """Create default Modelfile content when no Modelfile is found.
        
        Args:
            gguf_filename: Name of the .gguf file
            
        Returns:
            Default Modelfile content
        """
        return f"FROM {gguf_filename}\n"
    
    def run_ollama_create(self, model_name: str, modelfile_content: str, 
                         gguf_dir: str) -> Tuple[bool, Optional[str]]:
        """Run 'ollama create' to import a model.
        
        Args:
            model_name: Name for the new Ollama model
            modelfile_content: Content of the Modelfile
            gguf_dir: Directory containing the .gguf file
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Write temporary Modelfile
            temp_modelfile = os.path.join(gguf_dir, "temp_modelfile")
            with open(temp_modelfile, 'w', encoding='utf-8') as f:
                f.write(modelfile_content)
            
            # Run ollama create command
            result = subprocess.run(
                ['ollama', 'create', model_name, '-f', temp_modelfile],
                cwd=gguf_dir,  # Run in the directory containing the .gguf file
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Clean up temporary file
            try:
                os.remove(temp_modelfile)
            except OSError:
                pass  # Ignore cleanup errors
            
            if result.returncode == 0:
                logger.debug(f"Successfully created Ollama model: {model_name}")
                return True, None
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                logger.error(f"Ollama create failed for {model_name}: {error_msg}")
                return False, error_msg
                
        except FileNotFoundError:
            error_msg = "Ollama command not found. Is Ollama installed and in PATH?"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected error running ollama create: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def sanitize_model_name(self, gguf_path: str) -> str:
        """Generate a model name from a .gguf file path.
        
        Args:
            gguf_path: Path to the .gguf file
            
        Returns:
            Sanitized model name
        """
        # Get filename without extension
        filename = os.path.basename(gguf_path)
        if filename.endswith('.gguf'):
            filename = filename[:-5]  # Remove .gguf extension
        
        # Replace invalid characters with hyphens
        import re
        model_name = re.sub(r'[^a-zA-Z0-9._-]', '-', filename).lower()
        
        # Remove multiple consecutive hyphens
        model_name = re.sub(r'-+', '-', model_name)
        
        # Remove leading/trailing hyphens
        model_name = model_name.strip('-')
        
        return model_name if model_name else 'imported-model'
    
    def import_model(self, gguf_path: str, model_name: Optional[str] = None,
                    step_callback: Optional[Callable[[str, str], None]] = None) -> ModelImportInfo:
        """Import a .gguf file into Ollama.
        
        Args:
            gguf_path: Path to the .gguf file
            model_name: Optional name for the model (auto-generated if None)
            step_callback: Optional callback for step updates(step_description, status)
            
        Returns:
            ModelImportInfo with import results
        """
        # Generate model name if not provided
        if not model_name:
            model_name = self.sanitize_model_name(gguf_path)
        
        # Initialize result object
        import_info = ModelImportInfo(
            model_name=model_name,
            gguf_source_path=gguf_path,
            modelfile_path=None,
            modelfile_content="",
            import_success=False,
            ollama_create_success=False,
            error_message=None
        )
        
        try:
            # Validate .gguf file
            if step_callback:
                step_callback("Validating .gguf file", "working")
            
            if not self.validate_gguf_file(gguf_path):
                if step_callback:
                    step_callback("Invalid .gguf file", "error")
                import_info.error_message = "Invalid .gguf file"
                return import_info
            
            if step_callback:
                step_callback(".gguf file validated", "done")
            
            # Find and read Modelfile
            if step_callback:
                step_callback("Looking for Modelfile", "working")
            
            modelfile_path = self.find_modelfile(gguf_path)
            if modelfile_path:
                import_info.modelfile_path = modelfile_path
                success, modelfile_content = self.read_modelfile(modelfile_path)
                if success:
                    import_info.modelfile_content = modelfile_content
                    if step_callback:
                        step_callback("Modelfile found and read", "done")
                else:
                    if step_callback:
                        step_callback("Failed to read Modelfile", "error")
                    import_info.error_message = "Failed to read Modelfile"
                    return import_info
            else:
                # Create default Modelfile content
                gguf_filename = os.path.basename(gguf_path)
                import_info.modelfile_content = self.create_default_modelfile_content(gguf_filename)
                if step_callback:
                    step_callback("No Modelfile found, using default", "skip")
            
            # Import into Ollama
            if step_callback:
                step_callback(f"Creating Ollama model '{model_name}'", "working")
            
            gguf_dir = os.path.dirname(os.path.abspath(gguf_path))
            success, error_msg = self.run_ollama_create(model_name, import_info.modelfile_content, gguf_dir)
            
            if success:
                import_info.ollama_create_success = True
                import_info.import_success = True
                if step_callback:
                    step_callback(f"Model '{model_name}' created successfully", "done")
            else:
                import_info.error_message = error_msg
                if step_callback:
                    step_callback(f"Failed to create model '{model_name}'", "error")
        
        except Exception as e:
            logger.error(f"Error importing model from {gguf_path}: {e}")
            import_info.error_message = str(e)
            if step_callback:
                step_callback("Import failed with unexpected error", "error")
        
        return import_info
    
    def import_multiple_models(self, gguf_paths: List[str]) -> List[ModelImportInfo]:
        """Import multiple .gguf files into Ollama.
        
        Args:
            gguf_paths: List of paths to .gguf files
            
        Returns:
            List of ModelImportInfo for each import attempt
        """
        results = []
        
        for gguf_path in gguf_paths:
            result = self.import_model(gguf_path)
            results.append(result)
        
        return results
    
    def discover_gguf_files(self, directory: str) -> List[str]:
        """Discover all .gguf files in a directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of .gguf file paths
        """
        gguf_files = []
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith('.gguf'):
                        gguf_files.append(os.path.join(root, file))
        except Exception as e:
            logger.error(f"Error discovering .gguf files in {directory}: {e}")
        
        return sorted(gguf_files)