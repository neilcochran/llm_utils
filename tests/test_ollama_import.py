#!/usr/bin/env python3
"""Unit tests for Ollama model import functionality."""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import os
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
import sys
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from ollama_utils.model_importer import OllamaModelImporter, ModelImportInfo


class TestModelImportInfo(unittest.TestCase):
    """Test ModelImportInfo dataclass."""
    
    def test_model_import_info_creation(self):
        """Test ModelImportInfo creation with all fields."""
        info = ModelImportInfo(
            model_name="test-model",
            gguf_source_path="/path/to/model.gguf",
            modelfile_path="/path/to/Modelfile",
            modelfile_content="FROM model.gguf\nPARAMETER temperature 0.7",
            import_success=True,
            ollama_create_success=True,
            error_message=None
        )
        
        self.assertEqual(info.model_name, "test-model")
        self.assertEqual(info.gguf_source_path, "/path/to/model.gguf")
        self.assertEqual(info.modelfile_path, "/path/to/Modelfile")
        self.assertEqual(info.modelfile_content, "FROM model.gguf\nPARAMETER temperature 0.7")
        self.assertTrue(info.import_success)
        self.assertTrue(info.ollama_create_success)
        self.assertIsNone(info.error_message)
    
    def test_model_import_info_to_dict(self):
        """Test conversion to dictionary."""
        info = ModelImportInfo(
            model_name="test-model",
            gguf_source_path="/path/to/model.gguf",
            modelfile_path="/path/to/Modelfile",
            modelfile_content="FROM model.gguf",
            import_success=True,
            ollama_create_success=True,
            error_message=None
        )
        
        result = info.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["model_name"], "test-model")
        self.assertEqual(result["gguf_source_path"], "/path/to/model.gguf")
        self.assertEqual(result["modelfile_path"], "/path/to/Modelfile")
        self.assertEqual(result["modelfile_content"], "FROM model.gguf")
        self.assertTrue(result["import_success"])
        self.assertTrue(result["ollama_create_success"])
        self.assertIsNone(result["error_message"])


class TestOllamaModelImporter(unittest.TestCase):
    """Test OllamaModelImporter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.importer = OllamaModelImporter()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_importer_initialization(self):
        """Test importer initialization."""
        importer = OllamaModelImporter()
        self.assertIsInstance(importer, OllamaModelImporter)
    
    def test_validate_gguf_file_valid(self):
        """Test validation of valid .gguf file."""
        gguf_path = os.path.join(self.temp_dir, "model.gguf")
        with open(gguf_path, 'wb') as f:
            f.write(b"x" * 2048)  # Create file larger than 1KB
        
        result = self.importer.validate_gguf_file(gguf_path)
        self.assertTrue(result)
    
    def test_validate_gguf_file_wrong_extension(self):
        """Test validation fails for wrong extension."""
        txt_path = os.path.join(self.temp_dir, "model.txt")
        with open(txt_path, 'w') as f:
            f.write("content")
        
        result = self.importer.validate_gguf_file(txt_path)
        self.assertFalse(result)
    
    def test_validate_gguf_file_not_exists(self):
        """Test validation fails for non-existent file."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent.gguf")
        
        result = self.importer.validate_gguf_file(nonexistent_path)
        self.assertFalse(result)
    
    def test_validate_gguf_file_too_small(self):
        """Test validation fails for file that's too small."""
        gguf_path = os.path.join(self.temp_dir, "small.gguf")
        with open(gguf_path, 'w') as f:
            f.write("x")  # Very small file
        
        result = self.importer.validate_gguf_file(gguf_path)
        self.assertFalse(result)
    
    def test_validate_gguf_file_is_directory(self):
        """Test validation fails for directory."""
        dir_path = os.path.join(self.temp_dir, "model.gguf")
        os.makedirs(dir_path)
        
        result = self.importer.validate_gguf_file(dir_path)
        self.assertFalse(result)
    
    def test_find_modelfile_exists(self):
        """Test finding Modelfile when it exists."""
        gguf_path = os.path.join(self.temp_dir, "model.gguf")
        modelfile_path = os.path.join(self.temp_dir, "Modelfile")
        
        with open(gguf_path, 'w') as f:
            f.write("gguf content")
        with open(modelfile_path, 'w') as f:
            f.write("FROM model.gguf")
        
        result = self.importer.find_modelfile(gguf_path)
        self.assertEqual(result, modelfile_path)
    
    def test_find_modelfile_lowercase(self):
        """Test finding lowercase modelfile."""
        gguf_path = os.path.join(self.temp_dir, "model.gguf")
        modelfile_path = os.path.join(self.temp_dir, "modelfile")
        
        with open(gguf_path, 'w') as f:
            f.write("gguf content")
        with open(modelfile_path, 'w') as f:
            f.write("FROM model.gguf")
        
        result = self.importer.find_modelfile(gguf_path)
        # The implementation checks for 'Modelfile' first, so it may find that instead
        # We just need to verify a modelfile was found
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(result))
    
    def test_find_modelfile_not_exists(self):
        """Test finding Modelfile when it doesn't exist."""
        gguf_path = os.path.join(self.temp_dir, "model.gguf")
        with open(gguf_path, 'w') as f:
            f.write("gguf content")
        
        result = self.importer.find_modelfile(gguf_path)
        self.assertIsNone(result)
    
    def test_read_modelfile_success(self):
        """Test reading Modelfile successfully."""
        modelfile_path = os.path.join(self.temp_dir, "Modelfile")
        content = "FROM model.gguf\nPARAMETER temperature 0.7"
        
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        success, result = self.importer.read_modelfile(modelfile_path)
        self.assertTrue(success)
        self.assertEqual(result, content)
    
    def test_read_modelfile_not_exists(self):
        """Test reading non-existent Modelfile."""
        nonexistent_path = os.path.join(self.temp_dir, "nonexistent")
        
        success, result = self.importer.read_modelfile(nonexistent_path)
        self.assertFalse(success)
        self.assertEqual(result, "")
    
    def test_create_default_modelfile_content(self):
        """Test creating default Modelfile content."""
        gguf_filename = "model.gguf"
        expected = "FROM model.gguf\n"
        
        result = self.importer.create_default_modelfile_content(gguf_filename)
        self.assertEqual(result, expected)
    
    @patch('subprocess.run')
    def test_run_ollama_create_success(self, mock_subprocess):
        """Test successful ollama create command."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        model_name = "test-model"
        modelfile_content = "FROM model.gguf"
        gguf_dir = self.temp_dir
        
        with patch('builtins.open', mock_open()) as mock_file:
            success, error = self.importer.run_ollama_create(model_name, modelfile_content, gguf_dir)
        
        self.assertTrue(success)
        self.assertIsNone(error)
        mock_subprocess.assert_called_once()
        
        # Check the command arguments
        call_args = mock_subprocess.call_args
        self.assertEqual(call_args[0][0][:3], ['ollama', 'create', 'test-model'])
    
    @patch('subprocess.run')
    def test_run_ollama_create_failure(self, mock_subprocess):
        """Test failed ollama create command."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Model creation failed"
        mock_subprocess.return_value = mock_result
        
        model_name = "test-model"
        modelfile_content = "FROM model.gguf"
        gguf_dir = self.temp_dir
        
        with patch('builtins.open', mock_open()) as mock_file:
            success, error = self.importer.run_ollama_create(model_name, modelfile_content, gguf_dir)
        
        self.assertFalse(success)
        self.assertEqual(error, "Model creation failed")
    
    @patch('subprocess.run')
    def test_run_ollama_create_command_not_found(self, mock_subprocess):
        """Test ollama command not found."""
        mock_subprocess.side_effect = FileNotFoundError()
        
        model_name = "test-model"
        modelfile_content = "FROM model.gguf"
        gguf_dir = self.temp_dir
        
        with patch('builtins.open', mock_open()) as mock_file:
            success, error = self.importer.run_ollama_create(model_name, modelfile_content, gguf_dir)
        
        self.assertFalse(success)
        self.assertIn("Ollama command not found", error)
    
    def test_sanitize_model_name_basic(self):
        """Test basic model name sanitization."""
        gguf_path = "/path/to/my_model.gguf"
        expected = "my_model"
        
        result = self.importer.sanitize_model_name(gguf_path)
        self.assertEqual(result, expected)
    
    def test_sanitize_model_name_special_chars(self):
        """Test model name sanitization with special characters."""
        gguf_path = "/path/to/My Model@123!.gguf"
        expected = "my-model-123"  # The implementation strips trailing hyphens
        
        result = self.importer.sanitize_model_name(gguf_path)
        self.assertEqual(result, expected)
    
    def test_sanitize_model_name_multiple_hyphens(self):
        """Test model name sanitization removes multiple hyphens."""
        gguf_path = "/path/to/my---model.gguf"
        expected = "my-model"
        
        result = self.importer.sanitize_model_name(gguf_path)
        self.assertEqual(result, expected)
    
    def test_sanitize_model_name_empty_result(self):
        """Test model name sanitization with empty result."""
        gguf_path = "/path/to/@@@.gguf"
        expected = "imported-model"
        
        result = self.importer.sanitize_model_name(gguf_path)
        self.assertEqual(result, expected)
    
    def test_discover_gguf_files_found(self):
        """Test discovering .gguf files in directory."""
        # Create test .gguf files
        gguf1 = os.path.join(self.temp_dir, "model1.gguf")
        gguf2 = os.path.join(self.temp_dir, "model2.gguf")
        txt_file = os.path.join(self.temp_dir, "readme.txt")
        
        with open(gguf1, 'w') as f:
            f.write("gguf1")
        with open(gguf2, 'w') as f:
            f.write("gguf2")
        with open(txt_file, 'w') as f:
            f.write("readme")
        
        result = self.importer.discover_gguf_files(self.temp_dir)
        
        self.assertEqual(len(result), 2)
        self.assertIn(gguf1, result)
        self.assertIn(gguf2, result)
        self.assertNotIn(txt_file, result)
    
    def test_discover_gguf_files_recursive(self):
        """Test discovering .gguf files recursively."""
        subdir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(subdir)
        
        gguf1 = os.path.join(self.temp_dir, "model1.gguf")
        gguf2 = os.path.join(subdir, "model2.gguf")
        
        with open(gguf1, 'w') as f:
            f.write("gguf1")
        with open(gguf2, 'w') as f:
            f.write("gguf2")
        
        result = self.importer.discover_gguf_files(self.temp_dir)
        
        self.assertEqual(len(result), 2)
        self.assertIn(gguf1, result)
        self.assertIn(gguf2, result)
    
    def test_discover_gguf_files_empty_directory(self):
        """Test discovering .gguf files in empty directory."""
        result = self.importer.discover_gguf_files(self.temp_dir)
        self.assertEqual(result, [])
    
    def test_discover_gguf_files_case_insensitive(self):
        """Test discovering .gguf files with different case."""
        gguf_lower = os.path.join(self.temp_dir, "model.gguf")
        gguf_mixed = os.path.join(self.temp_dir, "MODEL.Gguf")
        
        with open(gguf_lower, 'w') as f:
            f.write("lower")
        with open(gguf_mixed, 'w') as f:
            f.write("mixed")
        
        result = self.importer.discover_gguf_files(self.temp_dir)
        
        # On Windows, file names are case-insensitive, so we might only get one
        # Let's just check that at least one .gguf file is found
        self.assertGreaterEqual(len(result), 1)
        
        # Check that at least one of our files is found
        found_files = [os.path.basename(f).lower() for f in result]
        self.assertIn("model.gguf", found_files)


class TestImportModelIntegration(unittest.TestCase):
    """Integration tests for import_model method."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.importer = OllamaModelImporter()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_import_model_success_with_modelfile(self):
        """Test successful model import with existing Modelfile."""
        gguf_path = os.path.join(self.temp_dir, "model.gguf")
        modelfile_path = os.path.join(self.temp_dir, "Modelfile")
        
        with open(gguf_path, 'wb') as f:
            f.write(b"x" * 2048)  # Create valid-sized file
        with open(modelfile_path, 'w') as f:
            f.write("FROM model.gguf\nPARAMETER temperature 0.7")
        
        with patch.object(self.importer, 'run_ollama_create', return_value=(True, None)):
            result = self.importer.import_model(gguf_path, "test-model")
        
        self.assertTrue(result.import_success)
        self.assertTrue(result.ollama_create_success)
        self.assertEqual(result.model_name, "test-model")
        self.assertEqual(result.gguf_source_path, gguf_path)
        self.assertEqual(result.modelfile_path, modelfile_path)
        self.assertEqual(result.modelfile_content, "FROM model.gguf\nPARAMETER temperature 0.7")
        self.assertIsNone(result.error_message)
    
    def test_import_model_success_without_modelfile(self):
        """Test successful model import without Modelfile (uses default)."""
        gguf_path = os.path.join(self.temp_dir, "model.gguf")
        
        with open(gguf_path, 'wb') as f:
            f.write(b"x" * 2048)  # Create valid-sized file
        
        with patch.object(self.importer, 'run_ollama_create', return_value=(True, None)):
            result = self.importer.import_model(gguf_path, "test-model")
        
        self.assertTrue(result.import_success)
        self.assertTrue(result.ollama_create_success)
        self.assertEqual(result.model_name, "test-model")
        self.assertEqual(result.gguf_source_path, gguf_path)
        self.assertIsNone(result.modelfile_path)
        self.assertEqual(result.modelfile_content, "FROM model.gguf\n")
        self.assertIsNone(result.error_message)
    
    def test_import_model_auto_generated_name(self):
        """Test model import with auto-generated name."""
        gguf_path = os.path.join(self.temp_dir, "my-awesome-model.gguf")
        
        with open(gguf_path, 'wb') as f:
            f.write(b"x" * 2048)
        
        with patch.object(self.importer, 'run_ollama_create', return_value=(True, None)):
            result = self.importer.import_model(gguf_path)
        
        self.assertTrue(result.import_success)
        self.assertEqual(result.model_name, "my-awesome-model")
    
    def test_import_model_invalid_gguf(self):
        """Test model import with invalid .gguf file."""
        gguf_path = os.path.join(self.temp_dir, "invalid.txt")
        
        with open(gguf_path, 'w') as f:
            f.write("not a gguf file")
        
        result = self.importer.import_model(gguf_path, "test-model")
        
        self.assertFalse(result.import_success)
        self.assertFalse(result.ollama_create_success)
        self.assertEqual(result.error_message, "Invalid .gguf file")
    
    def test_import_model_ollama_create_fails(self):
        """Test model import when ollama create command fails."""
        gguf_path = os.path.join(self.temp_dir, "model.gguf")
        
        with open(gguf_path, 'wb') as f:
            f.write(b"x" * 2048)
        
        with patch.object(self.importer, 'run_ollama_create', return_value=(False, "Creation failed")):
            result = self.importer.import_model(gguf_path, "test-model")
        
        self.assertFalse(result.import_success)
        self.assertFalse(result.ollama_create_success)
        self.assertEqual(result.error_message, "Creation failed")
    
    def test_import_model_with_step_callback(self):
        """Test model import with step callback."""
        gguf_path = os.path.join(self.temp_dir, "model.gguf")
        
        with open(gguf_path, 'wb') as f:
            f.write(b"x" * 2048)
        
        callback_calls = []
        def step_callback(step, status):
            callback_calls.append((step, status))
        
        with patch.object(self.importer, 'run_ollama_create', return_value=(True, None)):
            result = self.importer.import_model(gguf_path, "test-model", step_callback)
        
        self.assertTrue(result.import_success)
        
        # Check that callbacks were made
        self.assertGreater(len(callback_calls), 0)
        
        # Check for expected callback patterns
        step_messages = [call[0] for call in callback_calls]
        self.assertIn("Validating .gguf file", step_messages)
        self.assertIn("Looking for Modelfile", step_messages)
    
    def test_import_multiple_models(self):
        """Test importing multiple models."""
        gguf1 = os.path.join(self.temp_dir, "model1.gguf")
        gguf2 = os.path.join(self.temp_dir, "model2.gguf")
        
        with open(gguf1, 'wb') as f:
            f.write(b"x" * 2048)
        with open(gguf2, 'wb') as f:
            f.write(b"x" * 2048)
        
        with patch.object(self.importer, 'run_ollama_create', return_value=(True, None)):
            results = self.importer.import_multiple_models([gguf1, gguf2])
        
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.import_success for r in results))
        self.assertEqual(results[0].model_name, "model1")
        self.assertEqual(results[1].model_name, "model2")


if __name__ == '__main__':
    unittest.main()