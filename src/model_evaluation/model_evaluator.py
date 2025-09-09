"""Main model evaluation orchestration - data only."""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Callable
from .model_backends import BaseModelBackend, ModelResponse
from .performance_monitor import PerformanceMonitor, ResourceMetrics
from .metrics_collector import MetricsCollector, TokenMetrics, TimingMetrics

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    model_name: str
    backend_type: str = "ollama"
    queries: Optional[List[str]] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout_seconds: int = 300
    monitor_resources: bool = True
    sample_interval: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class QueryResult:
    """Result of evaluating a single query."""
    query: str
    response: ModelResponse
    timing_metrics: TimingMetrics
    token_metrics: TokenMetrics
    resource_metrics: Optional[ResourceMetrics] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query": self.query,
            "response": self.response.to_dict(),
            "timing_metrics": self.timing_metrics.to_dict(),
            "token_metrics": self.token_metrics.to_dict(),
            "resource_metrics": self.resource_metrics.to_dict() if self.resource_metrics else None
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for a model."""
    config: EvaluationConfig
    model_info: Dict[str, Any]
    query_results: List[QueryResult]
    success: bool
    error_message: Optional[str] = None
    total_duration_seconds: float = 0.0
    model_initialization_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": self.config.to_dict(),
            "model_info": self.model_info,
            "query_results": [qr.to_dict() for qr in self.query_results],
            "success": self.success,
            "error_message": self.error_message,
            "total_duration_seconds": self.total_duration_seconds,
            "model_initialization_time_ms": self.model_initialization_time_ms
        }
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all queries."""
        if not self.query_results:
            return {}
        
        # Aggregate timing metrics
        all_timings = [qr.timing_metrics for qr in self.query_results]
        avg_time_to_first = sum(t.time_to_first_token_ms for t in all_timings) / len(all_timings)
        avg_total_time = sum(t.total_inference_time_ms for t in all_timings) / len(all_timings)
        
        # Aggregate token metrics
        all_tokens = [qr.token_metrics for qr in self.query_results]
        avg_tokens_per_sec = sum(t.tokens_per_second for t in all_tokens) / len(all_tokens)
        peak_tokens_per_sec = max(t.peak_tokens_per_second for t in all_tokens) if all_tokens else 0.0
        avg_tokens_per_sec_variance = sum(t.tokens_per_second_variance for t in all_tokens) / len(all_tokens) if all_tokens else 0.0
        total_tokens = sum(t.total_tokens for t in all_tokens)
        total_completion_tokens = sum(t.completion_tokens for t in all_tokens)
        
        # Aggregate resource metrics if available
        resource_summaries = {}
        resource_results = [qr.resource_metrics for qr in self.query_results if qr.resource_metrics]
        if resource_results:
            resource_summaries = {
                "avg_cpu_percent": sum(r.cpu_percent_avg for r in resource_results) / len(resource_results),
                "max_cpu_percent": max(r.cpu_percent_max for r in resource_results),
                "avg_memory_mb": sum(r.memory_used_mb_avg for r in resource_results) / len(resource_results),
                "max_memory_mb": max(r.memory_used_mb_max for r in resource_results),
                "total_memory_mb": resource_results[0].memory_total_mb if resource_results and resource_results[0].memory_total_mb else None,
            }
            
            gpu_results = [r for r in resource_results if r.gpu_utilization_avg is not None]
            if gpu_results:
                # GPU power metrics
                gpu_power_results = [r for r in gpu_results if r.gpu_power_avg is not None]
                avg_gpu_power = sum(r.gpu_power_avg for r in gpu_power_results) / len(gpu_power_results) if gpu_power_results else None
                max_gpu_power = max(r.gpu_power_max for r in gpu_power_results) if gpu_power_results else None
                
                resource_summaries.update({
                    "avg_gpu_percent": sum(r.gpu_utilization_avg for r in gpu_results) / len(gpu_results),
                    "max_gpu_percent": max(r.gpu_utilization_max for r in gpu_results),
                    "avg_gpu_memory_mb": sum(r.gpu_memory_used_mb_avg for r in gpu_results) / len(gpu_results) if all(r.gpu_memory_used_mb_avg for r in gpu_results) else None,
                    "max_gpu_memory_mb": max(r.gpu_memory_used_mb_max for r in gpu_results) if all(r.gpu_memory_used_mb_max for r in gpu_results) else None,
                    "total_gpu_memory_mb": gpu_results[0].gpu_memory_total_mb if gpu_results and gpu_results[0].gpu_memory_total_mb else None,
                    "avg_gpu_power_w": avg_gpu_power,
                    "max_gpu_power_w": max_gpu_power,
                })
        
        return {
            "query_count": len(self.query_results),
            "successful_queries": sum(1 for qr in self.query_results if qr.response.success),
            "avg_time_to_first_token_ms": avg_time_to_first,
            "avg_total_inference_time_ms": avg_total_time,
            "avg_tokens_per_second": avg_tokens_per_sec,
            "peak_tokens_per_second": peak_tokens_per_sec,
            "tokens_per_second_variance": avg_tokens_per_sec_variance,
            "total_tokens_generated": total_completion_tokens,
            "total_tokens_processed": total_tokens,
            **resource_summaries
        }


class ModelEvaluator:
    """Orchestrates model evaluation with performance monitoring."""
    
    DEFAULT_QUERIES = [
        "What is the capital of France?",
        "Explain the concept of machine learning in simple terms.",
        "Write a short story about a robot learning to paint.",
        "Describe the process of photosynthesis step by step.",
        "What are the main differences between Python and JavaScript programming languages?"
    ]
    
    def __init__(self, backend: BaseModelBackend):
        """Initialize evaluator with a model backend.
        
        Args:
            backend: Model backend instance to use for inference
        """
        self.backend = backend
    
    def evaluate_model(self, config: EvaluationConfig,
                      progress_callback: Optional[Callable[[str, int, int], None]] = None) -> EvaluationResult:
        """Evaluate model performance with given configuration.
        
        Args:
            config: Evaluation configuration
            progress_callback: Optional callback for progress updates (message, current, total)
            
        Returns:
            EvaluationResult with all metrics and results
        """
        import time
        start_time = time.perf_counter()
        
        # Validate backend availability
        if not self.backend.is_available():
            return EvaluationResult(
                config=config,
                model_info={},
                query_results=[],
                success=False,
                error_message=f"Model backend not available: {self.backend.model_name}"
            )
        
        # Get model information and track initialization time
        model_info = self.backend.get_model_info()
        initialization_time_ms = self.backend.get_initialization_time_ms()
        
        # Use default queries if none provided
        queries = config.queries or self.DEFAULT_QUERIES
        
        query_results = []
        
        for i, query in enumerate(queries):
            if progress_callback:
                progress_callback(f"Evaluating query {i+1}/{len(queries)}", i, len(queries))
            
            try:
                result = self._evaluate_single_query(query, config)
                query_results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate query '{query}': {e}")
                # Create failed result
                failed_result = QueryResult(
                    query=query,
                    response=ModelResponse(content="", success=False, error_message=str(e)),
                    timing_metrics=TimingMetrics(0, 0, 0),
                    token_metrics=TokenMetrics(0, 0, 0, 0.0, 0.0)
                )
                query_results.append(failed_result)
        
        end_time = time.perf_counter()
        total_duration = end_time - start_time
        
        if progress_callback:
            progress_callback("Evaluation complete", len(queries), len(queries))
        
        return EvaluationResult(
            config=config,
            model_info=model_info,
            query_results=query_results,
            success=len([qr for qr in query_results if qr.response.success]) > 0,
            total_duration_seconds=total_duration,
            model_initialization_time_ms=initialization_time_ms
        )
    
    def _evaluate_single_query(self, query: str, config: EvaluationConfig) -> QueryResult:
        """Evaluate a single query with monitoring."""
        # Initialize monitoring components
        performance_monitor = None
        if config.monitor_resources:
            performance_monitor = PerformanceMonitor(config.sample_interval)
        
        metrics_collector = MetricsCollector()
        
        # Define token callback for streaming
        def token_callback(token: str):
            metrics_collector.record_token(token)
        
        try:
            # Start monitoring
            if performance_monitor:
                performance_monitor.start_monitoring()
            
            metrics_collector.start_inference()
            
            # Generate response with streaming to capture token-level metrics
            response = self.backend.generate_stream(
                query,
                token_callback=token_callback,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                timeout=config.timeout_seconds
            )
            
            metrics_collector.end_inference()
            
            # Stop monitoring
            resource_metrics = None
            if performance_monitor:
                performance_monitor.stop_monitoring()
                resource_metrics = performance_monitor.calculate_metrics()
            
            # Calculate metrics
            timing_metrics = metrics_collector.calculate_timing_metrics()
            token_metrics = metrics_collector.calculate_token_metrics()
            
            return QueryResult(
                query=query,
                response=response,
                timing_metrics=timing_metrics,
                token_metrics=token_metrics,
                resource_metrics=resource_metrics
            )
            
        except Exception as e:
            if performance_monitor:
                performance_monitor.stop_monitoring()
            
            logger.error(f"Error evaluating query: {e}")
            raise
    
    def evaluate_single_query(self, query: str, config: EvaluationConfig) -> QueryResult:
        """Evaluate a single query - public interface."""
        return self._evaluate_single_query(query, config)