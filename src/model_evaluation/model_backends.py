"""Model backend abstractions for different inference engines - data only."""

import subprocess
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Iterator, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Response from model inference."""
    content: str
    success: bool
    error_message: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    model_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "content": self.content,
            "success": self.success,
            "error_message": self.error_message,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model_name": self.model_name
        }


class BaseModelBackend(ABC):
    """Abstract base class for model inference backends."""
    
    def __init__(self, model_name: str):
        """Initialize backend with model name.
        
        Args:
            model_name: Name/identifier of the model to use
        """
        self.model_name = model_name
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend and model are available.
        
        Returns:
            True if backend can be used, False otherwise
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response from model.
        
        Args:
            prompt: Input prompt for the model
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse with generated content
        """
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, 
                       token_callback: Optional[Callable[[str], None]] = None,
                       **kwargs) -> ModelResponse:
        """Generate response with streaming tokens.
        
        Args:
            prompt: Input prompt for the model
            token_callback: Optional callback for each token
            **kwargs: Additional generation parameters
            
        Returns:
            ModelResponse with complete generated content
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Dictionary with model metadata
        """
        pass
    
    def get_initialization_time_ms(self) -> Optional[float]:
        """Get model initialization time in milliseconds.
        
        Returns:
            Initialization time in milliseconds, or None if not tracked
        """
        return None


class OllamaBackend(BaseModelBackend):
    """Ollama model backend using subprocess calls."""
    
    def __init__(self, model_name: str):
        """Initialize Ollama backend.
        
        Args:
            model_name: Ollama model name (e.g., "llama2:7b")
        """
        super().__init__(model_name)
        self._model_info_cache: Optional[Dict[str, Any]] = None
        self._is_initialized = False
        self._initialization_time_ms: Optional[float] = None
    
    def is_available(self) -> bool:
        """Check if Ollama is available and model exists."""
        try:
            # Check if ollama command is available
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )
            
            if result.returncode != 0:
                return False
            
            # Check if specific model is available
            return self.model_name in result.stdout
            
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama availability: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        """Generate response using Ollama generate command."""
        try:
            # Build ollama command
            cmd = ['ollama', 'generate', self.model_name, prompt]
            
            # Add optional parameters
            if 'temperature' in kwargs:
                cmd.extend(['--temperature', str(kwargs['temperature'])])
            if 'max_tokens' in kwargs:
                cmd.extend(['--num-predict', str(kwargs['max_tokens'])])
            
            # Run generation
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=kwargs.get('timeout', 300)  # 5 minute default timeout
            )
            
            if result.returncode == 0:
                return ModelResponse(
                    content=result.stdout.strip(),
                    success=True,
                    model_name=self.model_name
                )
            else:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                return ModelResponse(
                    content="",
                    success=False,
                    error_message=error_msg,
                    model_name=self.model_name
                )
                
        except subprocess.TimeoutExpired:
            return ModelResponse(
                content="",
                success=False,
                error_message="Generation timed out",
                model_name=self.model_name
            )
        except Exception as e:
            return ModelResponse(
                content="",
                success=False,
                error_message=str(e),
                model_name=self.model_name
            )
    
    def _ensure_model_loaded(self) -> float:
        """Ensure model is loaded and return initialization time in ms."""
        if self._is_initialized:
            return 0.0
        
        start_time = time.perf_counter()
        
        # Make a small request to warm up the model
        try:
            warmup_request = {
                "model": self.model_name,
                "prompt": "test",
                "stream": False
            }
            
            cmd = [
                'curl', '-s',
                'http://localhost:11434/api/generate',
                '-H', 'Content-Type: application/json',
                '-d', json.dumps(warmup_request)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=60  # Give model time to load
            )
            
            if result.returncode == 0:
                self._is_initialized = True
                init_time = (time.perf_counter() - start_time) * 1000
                self._initialization_time_ms = init_time
                return init_time
            else:
                # Fallback - still mark as initialized to avoid repeated attempts
                self._is_initialized = True
                init_time = (time.perf_counter() - start_time) * 1000
                self._initialization_time_ms = init_time
                return init_time
                
        except Exception as e:
            logger.debug(f"Model warmup failed: {e}")
            self._is_initialized = True  # Mark as initialized to avoid retry loops
            init_time = (time.perf_counter() - start_time) * 1000
            self._initialization_time_ms = init_time
            return init_time
    
    def get_initialization_time_ms(self) -> Optional[float]:
        """Get model initialization time in milliseconds."""
        return self._initialization_time_ms
    
    def generate_stream(self, prompt: str,
                       token_callback: Optional[Callable[[str], None]] = None,
                       **kwargs) -> ModelResponse:
        """Generate response with streaming using Ollama API."""
        try:
            # Ensure model is loaded first (this tracks initialization time)
            init_time_ms = self._ensure_model_loaded()
            
            # Now proceed with actual inference
            # Prepare request data for Ollama API
            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": True
            }
            
            # Add optional parameters
            if 'temperature' in kwargs:
                request_data["temperature"] = kwargs['temperature']
            if 'max_tokens' in kwargs:
                request_data["num_predict"] = kwargs['max_tokens']
            
            # Use curl to call Ollama API for streaming
            cmd = [
                'curl', '-s', '--no-buffer',
                'http://localhost:11434/api/generate',
                '-H', 'Content-Type: application/json',
                '-d', json.dumps(request_data)
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            content_parts = []
            
            try:
                for line in process.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        if 'response' in data:
                            token = data['response']
                            content_parts.append(token)
                            if token_callback:
                                token_callback(token)
                        
                        # Check if generation is complete
                        if data.get('done', False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
                
                process.wait(timeout=kwargs.get('timeout', 300))
                
                full_content = ''.join(content_parts)
                
                if process.returncode == 0:
                    return ModelResponse(
                        content=full_content,
                        success=True,
                        model_name=self.model_name
                    )
                else:
                    error_output = process.stderr.read() if process.stderr else "Unknown error"
                    return ModelResponse(
                        content=full_content,
                        success=False,
                        error_message=error_output,
                        model_name=self.model_name
                    )
                    
            except subprocess.TimeoutExpired:
                process.kill()
                return ModelResponse(
                    content=''.join(content_parts),
                    success=False,
                    error_message="Generation timed out",
                    model_name=self.model_name
                )
                
        except Exception as e:
            return ModelResponse(
                content="",
                success=False,
                error_message=str(e),
                model_name=self.model_name
            )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Ollama model information."""
        if self._model_info_cache is not None:
            # Add initialization time to cached info
            info = self._model_info_cache.copy()
            info["initialization_time_ms"] = self._initialization_time_ms
            return info
        
        # Trigger model initialization to get accurate timing
        if not self._is_initialized:
            self._ensure_model_loaded()
        
        try:
            # Get model info using ollama show
            result = subprocess.run(
                ['ollama', 'show', self.model_name],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse the output for model information
                info = {
                    "model_name": self.model_name,
                    "backend": "ollama",
                    "available": True,
                    "raw_info": result.stdout
                }
                
                # Try to extract specific info from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'parameter size' in line.lower():
                        info["parameter_size"] = line.split(':')[-1].strip()
                    elif 'quantization' in line.lower():
                        info["quantization"] = line.split(':')[-1].strip()
                
                info["initialization_time_ms"] = self._initialization_time_ms
                self._model_info_cache = info
                return info
            else:
                return {
                    "model_name": self.model_name,
                    "backend": "ollama", 
                    "available": False,
                    "error": result.stderr.strip() if result.stderr else "Unknown error",
                    "initialization_time_ms": self._initialization_time_ms
                }
                
        except Exception as e:
            return {
                "model_name": self.model_name,
                "backend": "ollama",
                "available": False,
                "error": str(e),
                "initialization_time_ms": self._initialization_time_ms
            }