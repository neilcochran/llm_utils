"""Tests for Ollama model export functionality."""

import pytest
import subprocess
from unittest.mock import patch, mock_open, MagicMock
from pathlib import Path
import tempfile
import shutil

# Add src to path for testing
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ollama_utils.model_exporter import OllamaModelExporter, ModelExportInfo


class TestOllamaModelExporter:
    """Test the OllamaModelExporter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_models_path = "/tmp/test_ollama_models"
        # Create a mock exporter with a valid path for most tests
        with patch('os.path.exists', return_value=True):
            self.exporter = OllamaModelExporter(self.test_models_path)
        self.temp_dir = None

    def teardown_method(self):
        """Clean up after tests."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_init(self):
        """Test that OllamaModelExporter initializes correctly."""
        with patch('os.path.exists', return_value=True):
            exporter = OllamaModelExporter("/valid/path")
            assert exporter.ollama_models_path == "/valid/path"

    def test_init_invalid_path(self):
        """Test initialization with invalid path fails."""
        with patch('os.path.exists', return_value=False):
            with pytest.raises(ValueError, match="Ollama models directory does not exist"):
                OllamaModelExporter("/invalid/path")

    def test_init_empty_path(self):
        """Test initialization with empty path fails."""
        with pytest.raises(ValueError, match="ollama_models_path is required"):
            OllamaModelExporter("")

    def test_sanitize_model_name(self):
        """Test model name sanitization."""
        # Basic sanitization
        assert self.exporter.sanitize_model_name("llama2:latest") == "llama2"
        assert self.exporter.sanitize_model_name("model/name") == "model-name"
        assert self.exporter.sanitize_model_name("model:7b") == "model-7b"

        # Special characters
        assert self.exporter.sanitize_model_name("model<>:\"/\\|?*.txt") == "model----------txt"

        # Empty and edge cases
        assert self.exporter.sanitize_model_name("") == ""
        assert self.exporter.sanitize_model_name("simple") == "simple"

    @patch('subprocess.run')
    def test_run_ollama_command_success(self, mock_run):
        """Test successful ollama command execution."""
        mock_run.return_value.stdout = "command output"
        mock_run.return_value.returncode = 0

        success, output = self.exporter.run_ollama_command("ollama list")

        assert success is True
        assert output == "command output"
        mock_run.assert_called_once_with(
            "ollama list",
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='replace'
        )

    @patch('subprocess.run')
    def test_run_ollama_command_failure(self, mock_run):
        """Test ollama command execution failure."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'ollama', stderr="command failed")

        success, output = self.exporter.run_ollama_command("ollama invalid")

        assert success is False
        assert output is None

    @patch('subprocess.run')
    def test_run_ollama_command_not_found(self, mock_run):
        """Test ollama command not found."""
        mock_run.side_effect = FileNotFoundError()

        success, output = self.exporter.run_ollama_command("ollama list")

        assert success is False
        assert output is None

    @patch.object(OllamaModelExporter, 'run_ollama_command')
    def test_get_model_info_success(self, mock_run_cmd):
        """Test successful model info retrieval."""

        def mock_command_side_effect(command):
            if 'template' in command:
                return True, "{{ .Prompt }}"
            elif 'parameters' in command:
                return True, "temperature 0.8\nmax_tokens 2048"
            elif 'system' in command:
                return True, "You are a helpful assistant"
            elif 'modelfile' in command:
                return True, "FROM /path/to/model.gguf\\nTEMPLATE {{ .Prompt }}"
            return False, None

        mock_run_cmd.side_effect = mock_command_side_effect

        success, info = self.exporter.get_model_info("llama2")

        assert success is True
        assert info['template'] == "{{ .Prompt }}"
        assert len(info['parameters']) == 2
        assert "temperature 0.8" in info['parameters']
        assert "max_tokens 2048" in info['parameters']
        assert info['system_message'] == "You are a helpful assistant"
        assert info['modelfile'] is not None

    @patch.object(OllamaModelExporter, 'run_ollama_command')
    def test_get_model_info_model_not_found(self, mock_run_cmd):
        """Test model info retrieval for non-existent model."""
        mock_run_cmd.return_value = (False, None)

        success, info = self.exporter.get_model_info("nonexistent")

        assert success is False
        assert 'error' in info
        assert 'not found' in info['error']

    @patch('os.path.exists')
    def test_extract_gguf_path_with_path(self, mock_exists):
        """Test extracting .gguf path from modelfile."""
        modelfile_content = '''FROM /home/user/.ollama/models/blobs/sha256-abc123def456
TEMPLATE "{{ .Prompt }}"
PARAMETER temperature 0.8'''

        # Mock that the ollama models path exists
        mock_exists.return_value = True

        # Test with ollama models path set
        self.exporter.ollama_models_path = "/home/user/.ollama/models"
        path = self.exporter.extract_gguf_path(modelfile_content)

        assert path == "/home/user/.ollama/models/blobs/sha256-abc123def456"

    def test_extract_gguf_path_no_path_configured(self):
        """Test extracting .gguf path when no ollama path configured."""
        modelfile_content = '''FROM llama2:latest
TEMPLATE "{{ .Prompt }}"'''

        # Create exporter with None path (should raise assertion)
        with patch('os.path.exists', return_value=True):
            temp_exporter = OllamaModelExporter("/tmp/test")

        # Manually set to None to test assertion
        temp_exporter.ollama_models_path = None

        with pytest.raises(AssertionError, match="Ollama models path must be configured"):
            temp_exporter.extract_gguf_path(modelfile_content)

    @patch('os.path.exists')
    def test_extract_gguf_path_no_match(self, mock_exists):
        """Test extracting .gguf path when no match found."""
        modelfile_content = '''FROM llama2:latest
TEMPLATE "{{ .Prompt }}"'''

        # Mock that the ollama models path exists
        mock_exists.return_value = True

        # Set ollama path but content doesn't contain it
        self.exporter.ollama_models_path = "/home/user/.ollama/models"
        path = self.exporter.extract_gguf_path(modelfile_content)
        assert path is None

    def test_extract_gguf_path_empty_content(self):
        """Test extracting .gguf path from empty content."""
        path = self.exporter.extract_gguf_path("")
        assert path is None

        path = self.exporter.extract_gguf_path(None)
        assert path is None

    def test_extract_gguf_path_invalid_ollama_path(self):
        """Test extracting .gguf path when ollama path doesn't exist."""
        modelfile_content = '''FROM /home/user/.ollama/models/blobs/sha256-abc123def456
TEMPLATE "{{ .Prompt }}"'''

        # Create exporter and then set path to nonexistent to test assertion
        with patch('os.path.exists', return_value=True):
            temp_exporter = OllamaModelExporter("/tmp/test")

        # Manually set to nonexistent path to test assertion
        temp_exporter.ollama_models_path = "/nonexistent/path"

        with pytest.raises(AssertionError, match="Ollama models path must exist"):
            temp_exporter.extract_gguf_path(modelfile_content)

    def test_create_modelfile_content(self):
        """Test Modelfile content creation."""
        model_info = {
            'template': '{{ .Prompt }}',
            'parameters': ['temperature 0.8', 'max_tokens 2048'],
            'system_message': 'You are helpful'
        }

        content = self.exporter.create_modelfile_content("test-model", model_info)

        assert "FROM test-model.gguf" in content
        assert 'TEMPLATE """{{ .Prompt }}"""' in content
        assert "PARAMETER temperature 0.8" in content
        assert "PARAMETER max_tokens 2048" in content
        assert 'SYSTEM "You are helpful"' in content

    def test_create_modelfile_content_minimal(self):
        """Test Modelfile content creation with minimal info."""
        model_info = {'template': '{{ .Prompt }}'}

        content = self.exporter.create_modelfile_content("test", model_info)

        assert "FROM test.gguf" in content
        assert 'TEMPLATE """{{ .Prompt }}"""' in content
        assert "PARAMETER" not in content
        assert "SYSTEM" not in content

    @patch.object(OllamaModelExporter, 'run_ollama_command')
    def test_list_available_models_success(self, mock_run_cmd):
        """Test listing available models successfully."""
        mock_output = """NAME            ID              SIZE    MODIFIED
llama2:latest   abc123def456    3.8 GB  2 days ago
codellama:7b    def456ghi789    4.1 GB  1 week ago"""

        mock_run_cmd.return_value = (True, mock_output)

        success, models = self.exporter.list_available_models()

        assert success is True
        assert len(models) == 2
        assert "llama2:latest" in models
        assert "codellama:7b" in models

    @patch.object(OllamaModelExporter, 'run_ollama_command')
    def test_list_available_models_failure(self, mock_run_cmd):
        """Test listing models when command fails."""
        mock_run_cmd.return_value = (False, None)

        success, models = self.exporter.list_available_models()

        assert success is False
        assert models == []

    @patch.object(OllamaModelExporter, 'run_ollama_command')
    def test_list_available_models_empty(self, mock_run_cmd):
        """Test listing models when no models exist."""
        mock_output = """NAME    ID    SIZE    MODIFIED"""

        mock_run_cmd.return_value = (True, mock_output)

        success, models = self.exporter.list_available_models()

        assert success is True
        assert models == []

    @patch('builtins.open', new_callable=mock_open)
    @patch.object(OllamaModelExporter, 'copy_with_progress')
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.mkdir')
    @patch.object(OllamaModelExporter, 'extract_gguf_path')
    @patch.object(OllamaModelExporter, 'get_model_info')
    def test_export_model_success(self, mock_get_info, mock_extract_path, mock_mkdir, mock_exists, mock_copy,
                                  mock_file):
        """Test successful model export."""
        # Setup mocks
        mock_get_info.return_value = (True, {
            'template': '{{ .Prompt }}',
            'parameters': ['temperature 0.8'],
            'system_message': 'Be helpful',
            'modelfile': 'FROM /path/to/model.gguf\\nTEMPLATE {{ .Prompt }}'
        })
        mock_extract_path.return_value = "/path/to/model.gguf"
        mock_exists.return_value = True
        mock_copy.return_value = True  # Mock successful copy

        # Create temp directory for test
        self.temp_dir = tempfile.mkdtemp()

        result = self.exporter.export_model("llama2", self.temp_dir)

        assert result.success is True
        assert result.model_name == "llama2"
        assert result.sanitized_name == "llama2"
        assert result.template == '{{ .Prompt }}'
        assert len(result.parameters) == 1
        assert result.system_message == 'Be helpful'
        assert result.error_message is None
        assert result.gguf_source_path == "/path/to/model.gguf"

        # Verify file operations were called
        mock_file.assert_called()
        mock_copy.assert_called()

    @patch.object(OllamaModelExporter, 'get_model_info')
    def test_export_model_get_info_failure(self, mock_get_info):
        """Test model export when get_model_info fails."""
        mock_get_info.return_value = (False, {'error': 'Model not found'})

        self.temp_dir = tempfile.mkdtemp()

        result = self.exporter.export_model("nonexistent", self.temp_dir)

        assert result.success is False
        assert result.error_message == "Model not found"

    @patch.object(OllamaModelExporter, 'list_available_models')
    @patch.object(OllamaModelExporter, 'export_model')
    def test_export_all_models_success(self, mock_export, mock_list):
        """Test exporting all models successfully."""
        mock_list.return_value = (True, ["llama2", "codellama"])

        # Create mock export results
        result1 = ModelExportInfo("llama2", "llama2", None, [], None, "", None, None, None, None, True, None)
        result2 = ModelExportInfo("codellama", "codellama", None, [], None, "", None, None, None, None, True, None)
        mock_export.side_effect = [result1, result2]

        results = self.exporter.export_all_models("/tmp")

        assert len(results) == 2
        assert all(r.success for r in results)
        assert mock_export.call_count == 2

    @patch.object(OllamaModelExporter, 'list_available_models')
    def test_export_all_models_list_failure(self, mock_list):
        """Test export all when listing models fails."""
        mock_list.return_value = (False, [])

        results = self.exporter.export_all_models("/tmp")

        assert len(results) == 1
        assert results[0].success is False
        assert "Failed to list" in results[0].error_message

    @patch('builtins.open', side_effect=PermissionError("Access denied"))
    @patch.object(OllamaModelExporter, 'get_model_info')
    def test_export_model_file_write_error(self, mock_get_info, mock_file):
        """Test model export when file writing fails."""
        mock_get_info.return_value = (True, {
            'template': '{{ .Prompt }}',
            'parameters': [],
            'system_message': None,
            'modelfile': None
        })

        self.temp_dir = tempfile.mkdtemp()

        result = self.exporter.export_model("test", self.temp_dir)

        assert result.success is False
        assert "Access denied" in result.error_message


class TestModelExportInfo:
    """Test the ModelExportInfo dataclass."""

    def test_to_dict(self):
        """Test converting ModelExportInfo to dictionary."""
        info = ModelExportInfo(
            model_name="test",
            sanitized_name="test",
            template="{{ .Prompt }}",
            parameters=["temp 0.8"],
            system_message="Be helpful",
            modelfile_content="FROM test.gguf",
            gguf_source_path="/source/path",
            export_directory="/export/dir",
            modelfile_path="/export/dir/Modelfile",
            gguf_destination_path="/export/dir/test.gguf",
            success=True,
            error_message=None
        )

        result = info.to_dict()

        assert isinstance(result, dict)
        assert result['model_name'] == "test"
        assert result['success'] is True
        assert result['template'] == "{{ .Prompt }}"
        assert len(result['parameters']) == 1
        assert result['error_message'] is None
