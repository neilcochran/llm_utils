#!/usr/bin/env python3
"""Unit tests for model evaluation functionality."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time
import threading
import tempfile
import json
from pathlib import Path

# Add src to path for imports
import sys
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from model_evaluation.metrics_collector import MetricsCollector, TokenMetrics, TimingMetrics
from model_evaluation.performance_monitor import PerformanceMonitor, ResourceSnapshot, ResourceMetrics
from model_evaluation.model_backends import BaseModelBackend, OllamaBackend, ModelResponse
from model_evaluation.model_evaluator import ModelEvaluator, EvaluationConfig, EvaluationResult, QueryResult


class TestMetricsCollector(unittest.TestCase):
    """Test MetricsCollector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = MetricsCollector()
    
    def test_collector_initialization(self):
        """Test collector initialization."""
        self.assertIsNone(self.collector.start_time)
        self.assertIsNone(self.collector.first_token_time)
        self.assertIsNone(self.collector.end_time)
        self.assertEqual(self.collector.tokens_received, [])
        self.assertEqual(self.collector.prompt_tokens, 0)
    
    def test_inference_timing(self):
        """Test timing measurement."""
        with patch('time.perf_counter', side_effect=[1.0, 1.1, 1.5]):
            self.collector.start_inference()
            self.collector.record_first_token()
            self.collector.end_inference()
        
        timing = self.collector.calculate_timing_metrics()
        self.assertAlmostEqual(timing.time_to_first_token_ms, 100.0)  # 0.1s * 1000
        self.assertAlmostEqual(timing.time_to_completion_ms, 400.0)   # 0.4s * 1000
        self.assertAlmostEqual(timing.total_inference_time_ms, 500.0) # 0.5s * 1000
    
    def test_token_recording(self):
        """Test token recording and metrics calculation."""
        # Times: start, record_token(Hello), record_first_token(), record_token(" "), record_token("World"), end
        with patch('time.perf_counter', side_effect=[1.0, 1.1, 1.1, 1.2, 1.4, 1.5]):
            self.collector.start_inference()
            self.collector.record_token("Hello")
            self.collector.record_token(" ")
            self.collector.record_token("World")
            self.collector.set_prompt_tokens(10)
            self.collector.end_inference()
        
        token_metrics = self.collector.calculate_token_metrics()
        self.assertEqual(token_metrics.completion_tokens, 3)
        self.assertEqual(token_metrics.total_tokens, 13)  # 10 prompt + 3 completion
        
        # New fields should exist
        self.assertGreaterEqual(token_metrics.peak_tokens_per_second, 0)
        self.assertGreaterEqual(token_metrics.tokens_per_second_variance, 0)
        self.assertEqual(token_metrics.prompt_tokens, 10)
        self.assertEqual(token_metrics.tokens_per_second, 6.0)  # 3 tokens / 0.5s
        self.assertAlmostEqual(token_metrics.average_token_length, 11/3)  # "Hello" + " " + "World" = 11 chars / 3 tokens
    
    def test_no_tokens_received(self):
        """Test metrics with no tokens received."""
        with patch('time.perf_counter', side_effect=[1.0, 1.5]):
            self.collector.start_inference()
            self.collector.set_prompt_tokens(5)
            self.collector.end_inference()
        
        token_metrics = self.collector.calculate_token_metrics()
        self.assertEqual(token_metrics.completion_tokens, 0)
        self.assertEqual(token_metrics.total_tokens, 5)
        self.assertEqual(token_metrics.tokens_per_second, 0.0)
        self.assertEqual(token_metrics.average_token_length, 0.0)
    
    def test_timing_metrics_to_dict(self):
        """Test TimingMetrics to_dict conversion."""
        timing = TimingMetrics(
            time_to_first_token_ms=100.0,
            time_to_completion_ms=400.0,
            total_inference_time_ms=500.0
        )
        result = timing.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result['time_to_first_token_ms'], 100.0)
    
    def test_token_metrics_to_dict(self):
        """Test TokenMetrics to_dict conversion."""
        tokens = TokenMetrics(
            total_tokens=13,
            prompt_tokens=10,
            completion_tokens=3,
            average_token_length=2.5,
            tokens_per_second=6.0,
            peak_tokens_per_second=8.5,
            tokens_per_second_variance=1.2
        )
        result = tokens.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result['total_tokens'], 13)
        self.assertEqual(result['peak_tokens_per_second'], 8.5)
        self.assertEqual(result['tokens_per_second_variance'], 1.2)


class TestPerformanceMonitor(unittest.TestCase):
    """Test PerformanceMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.monitor = PerformanceMonitor(sample_interval_seconds=0.01)
    
    def tearDown(self):
        """Clean up after tests."""
        if self.monitor.monitoring:
            self.monitor.stop_monitoring()
    
    @patch('model_evaluation.performance_monitor.psutil.cpu_percent')
    @patch('model_evaluation.performance_monitor.psutil.virtual_memory')
    def test_sample_resources(self, mock_memory, mock_cpu):
        """Test resource sampling."""
        mock_cpu.return_value = 25.5
        mock_memory.return_value = Mock(
            percent=60.0,
            used=1024*1024*1000,  # 1000MB
            available=1024*1024*500  # 500MB
        )
        
        with patch.object(self.monitor.cuda_checker, 'get_cuda_status') as mock_cuda:
            mock_cuda.return_value = Mock(
                cuda_available=True,
                gpus=[{
                    'gpu_utilization_percent': '75',
                    'used_memory_mb': '2000',
                    'total_memory_mb': '8000',
                    'temperature_c': '65',
                    'power_draw_w': '150'
                }]
            )
            
            snapshot = self.monitor._sample_resources()
        
        self.assertEqual(snapshot.cpu_percent, 25.5)
        self.assertEqual(snapshot.memory_percent, 60.0)
        self.assertEqual(snapshot.memory_used_mb, 1000.0)
        self.assertEqual(snapshot.memory_available_mb, 500.0)
        self.assertEqual(snapshot.gpu_utilization_percent, 75.0)
        self.assertEqual(snapshot.gpu_memory_used_mb, 2000.0)
    
    @patch('model_evaluation.performance_monitor.psutil.cpu_percent')
    @patch('model_evaluation.performance_monitor.psutil.virtual_memory')
    def test_monitoring_lifecycle(self, mock_memory, mock_cpu):
        """Test monitoring start/stop lifecycle."""
        mock_cpu.return_value = 25.0
        mock_memory.return_value = Mock(percent=50.0, used=1024*1024*500, available=1024*1024*500)
        
        with patch.object(self.monitor.cuda_checker, 'get_cuda_status') as mock_cuda:
            mock_cuda.return_value = Mock(cuda_available=False, gpus=[])
            
            self.assertFalse(self.monitor.monitoring)
            
            self.monitor.start_monitoring()
            self.assertTrue(self.monitor.monitoring)
            self.assertIsNotNone(self.monitor.monitor_thread)
            
            # Let it run for a short time
            time.sleep(0.05)
            
            self.monitor.stop_monitoring()
            self.assertFalse(self.monitor.monitoring)
            self.assertGreater(len(self.monitor.snapshots), 0)
    
    def test_calculate_metrics(self):
        """Test metrics calculation from snapshots."""
        # Create mock snapshots
        snapshots = [
            ResourceSnapshot(1.0, 10.0, 40.0, 400.0, 600.0, 50.0, 1000.0, 8000.0, 60.0, 100.0),
            ResourceSnapshot(1.1, 20.0, 50.0, 500.0, 500.0, 75.0, 1500.0, 8000.0, 65.0, 150.0),
            ResourceSnapshot(1.2, 15.0, 45.0, 450.0, 550.0, 60.0, 1200.0, 8000.0, 62.0, 120.0)
        ]
        self.monitor.snapshots = snapshots
        self.monitor.start_time = 1.0
        
        with patch('time.perf_counter', return_value=1.5):
            metrics = self.monitor.calculate_metrics()
        
        self.assertEqual(metrics.duration_seconds, 0.5)
        self.assertEqual(metrics.cpu_percent_avg, 15.0)  # (10+20+15)/3
        self.assertEqual(metrics.cpu_percent_max, 20.0)
        self.assertEqual(metrics.memory_percent_avg, 45.0)  # (40+50+45)/3
        self.assertAlmostEqual(metrics.gpu_utilization_avg, 61.67, places=1)  # (50+75+60)/3 rounded
    
    def test_resource_snapshot_to_dict(self):
        """Test ResourceSnapshot to_dict conversion."""
        snapshot = ResourceSnapshot(
            timestamp=1.0,
            cpu_percent=25.0,
            memory_percent=50.0,
            memory_used_mb=1000.0,
            memory_available_mb=500.0,
            gpu_utilization_percent=75.0
        )
        result = snapshot.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result['cpu_percent'], 25.0)
        self.assertEqual(result['gpu_utilization_percent'], 75.0)


class TestModelBackends(unittest.TestCase):
    """Test model backend classes."""
    
    def test_model_response_to_dict(self):
        """Test ModelResponse to_dict conversion."""
        response = ModelResponse(
            content="Hello World",
            success=True,
            prompt_tokens=5,
            completion_tokens=2,
            model_name="test:model"
        )
        result = response.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result['content'], "Hello World")
        self.assertTrue(result['success'])
        self.assertEqual(result['prompt_tokens'], 5)
    
    @patch('subprocess.run')
    def test_ollama_backend_is_available_success(self, mock_subprocess):
        """Test Ollama backend availability check - success."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "NAME            SIZE    MODIFIED\nllama2:7b       3.8GB   2 weeks ago"
        mock_subprocess.return_value = mock_result
        
        backend = OllamaBackend("llama2:7b")
        self.assertTrue(backend.is_available())
    
    @patch('subprocess.run')
    def test_ollama_backend_is_available_model_not_found(self, mock_subprocess):
        """Test Ollama backend availability check - model not found."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "NAME            SIZE    MODIFIED\ncodellama:13b   7.2GB   1 week ago"
        mock_subprocess.return_value = mock_result
        
        backend = OllamaBackend("llama2:7b")
        self.assertFalse(backend.is_available())
    
    @patch('subprocess.run')
    def test_ollama_backend_is_available_command_failed(self, mock_subprocess):
        """Test Ollama backend availability check - command failed."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_subprocess.return_value = mock_result
        
        backend = OllamaBackend("llama2:7b")
        self.assertFalse(backend.is_available())
    
    @patch('subprocess.run')
    def test_ollama_backend_is_available_command_not_found(self, mock_subprocess):
        """Test Ollama backend availability check - command not found."""
        mock_subprocess.side_effect = FileNotFoundError()
        
        backend = OllamaBackend("llama2:7b")
        self.assertFalse(backend.is_available())
    
    @patch('subprocess.run')
    def test_ollama_generate_success(self, mock_subprocess):
        """Test Ollama generate - success."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Hello, how can I help you today?"
        mock_subprocess.return_value = mock_result
        
        backend = OllamaBackend("llama2:7b")
        response = backend.generate("Hello")
        
        self.assertTrue(response.success)
        self.assertEqual(response.content, "Hello, how can I help you today?")
        self.assertEqual(response.model_name, "llama2:7b")
    
    @patch('subprocess.run')
    def test_ollama_generate_failure(self, mock_subprocess):
        """Test Ollama generate - failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Model not found"
        mock_subprocess.return_value = mock_result
        
        backend = OllamaBackend("llama2:7b")
        response = backend.generate("Hello")
        
        self.assertFalse(response.success)
        self.assertEqual(response.error_message, "Model not found")
        self.assertEqual(response.content, "")
    
    @patch('subprocess.run')
    def test_ollama_generate_timeout(self, mock_subprocess):
        """Test Ollama generate - timeout."""
        import subprocess
        mock_subprocess.side_effect = subprocess.TimeoutExpired(['ollama'], 300)
        
        backend = OllamaBackend("llama2:7b")
        response = backend.generate("Hello")
        
        self.assertFalse(response.success)
        self.assertEqual(response.error_message, "Generation timed out")
    
    @patch('subprocess.run')
    def test_ollama_get_model_info(self, mock_subprocess):
        """Test Ollama get model info."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """Model: llama2:7b
Parameter size: 7B
Quantization: Q4_0
Template: [INST] {{ .Prompt }} [/INST]"""
        mock_subprocess.return_value = mock_result
        
        backend = OllamaBackend("llama2:7b")
        info = backend.get_model_info()
        
        self.assertEqual(info['model_name'], "llama2:7b")
        self.assertEqual(info['backend'], "ollama")
        self.assertTrue(info['available'])
        self.assertIn('parameter size', info['raw_info'].lower())


class TestModelEvaluator(unittest.TestCase):
    """Test ModelEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_backend = Mock(spec=BaseModelBackend)
        self.mock_backend.model_name = "test:model"
        self.evaluator = ModelEvaluator(self.mock_backend)
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        self.assertEqual(self.evaluator.backend, self.mock_backend)
        self.assertIsInstance(self.evaluator.DEFAULT_QUERIES, list)
        self.assertGreater(len(self.evaluator.DEFAULT_QUERIES), 0)
    
    def test_evaluate_model_backend_unavailable(self):
        """Test evaluation when backend is unavailable."""
        self.mock_backend.is_available.return_value = False
        
        config = EvaluationConfig(model_name="test:model")
        result = self.evaluator.evaluate_model(config)
        
        self.assertFalse(result.success)
        self.assertIn("not available", result.error_message)
    
    def test_evaluate_model_success(self):
        """Test successful model evaluation."""
        self.mock_backend.is_available.return_value = True
        self.mock_backend.get_model_info.return_value = {"model_name": "test:model", "available": True}
        self.mock_backend.generate_stream.return_value = ModelResponse(
            content="Test response", success=True, model_name="test:model"
        )
        
        config = EvaluationConfig(
            model_name="test:model",
            queries=["Test query"],
            monitor_resources=False
        )
        
        with patch('time.perf_counter', side_effect=[1.0, 1.1, 1.1, 1.5, 1.6]):
            result = self.evaluator.evaluate_model(config)
        
        self.assertTrue(result.success)
        self.assertEqual(len(result.query_results), 1)
        self.assertTrue(result.query_results[0].response.success)
    
    def test_evaluate_single_query(self):
        """Test single query evaluation."""
        self.mock_backend.generate_stream.return_value = ModelResponse(
            content="Test response", success=True, model_name="test:model"
        )
        
        config = EvaluationConfig(
            model_name="test:model",
            monitor_resources=False
        )
        
        with patch('time.perf_counter', side_effect=[1.0, 1.1, 1.5]):
            result = self.evaluator.evaluate_single_query("Test query", config)
        
        self.assertTrue(result.response.success)
        self.assertEqual(result.query, "Test query")
        self.assertIsInstance(result.timing_metrics, TimingMetrics)
        self.assertIsInstance(result.token_metrics, TokenMetrics)
    
    def test_evaluation_config_to_dict(self):
        """Test EvaluationConfig to_dict conversion."""
        config = EvaluationConfig(
            model_name="test:model",
            queries=["Query 1", "Query 2"],
            temperature=0.8
        )
        result = config.to_dict()
        self.assertIsInstance(result, dict)
        self.assertEqual(result['model_name'], "test:model")
        self.assertEqual(result['temperature'], 0.8)
    
    def test_evaluation_result_to_dict(self):
        """Test EvaluationResult to_dict conversion."""
        config = EvaluationConfig(model_name="test:model")
        result = EvaluationResult(
            config=config,
            model_info={"name": "test"},
            query_results=[],
            success=True,
            total_duration_seconds=1.5
        )
        result_dict = result.to_dict()
        self.assertIsInstance(result_dict, dict)
        self.assertTrue(result_dict['success'])
        self.assertEqual(result_dict['total_duration_seconds'], 1.5)
    
    def test_evaluation_result_summary_stats(self):
        """Test evaluation result summary statistics."""
        # Create mock query results
        timing_metrics = TimingMetrics(100.0, 400.0, 500.0)
        token_metrics = TokenMetrics(15, 10, 5, 3.0, 10.0, 12.0, 2.0)
        response = ModelResponse("Test", True, model_name="test:model")
        
        query_result = QueryResult(
            query="Test",
            response=response,
            timing_metrics=timing_metrics,
            token_metrics=token_metrics
        )
        
        config = EvaluationConfig(model_name="test:model")
        result = EvaluationResult(
            config=config,
            model_info={},
            query_results=[query_result, query_result],  # Two identical results
            success=True
        )
        
        stats = result.get_summary_stats()
        self.assertEqual(stats['query_count'], 2)
        self.assertEqual(stats['successful_queries'], 2)
        self.assertEqual(stats['avg_time_to_first_token_ms'], 100.0)
        self.assertEqual(stats['avg_tokens_per_second'], 10.0)
        self.assertEqual(stats['total_tokens_generated'], 10)  # 5 * 2


if __name__ == '__main__':
    # Import subprocess for tests that need it
    import subprocess
    unittest.main()