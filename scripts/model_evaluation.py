#!/usr/bin/env python3
"""Model Evaluation Tool - CLI interface for testing LLM performance."""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from model_evaluation.model_evaluator import ModelEvaluator, EvaluationConfig, EvaluationResult
from model_evaluation.model_backends import OllamaBackend


class ModelEvaluationFormatter:
    """Handles formatting and display for model evaluation results."""
    
    # ANSI color codes
    COLORS = {
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m',
        'reset': '\033[0m'
    }
    
    @staticmethod
    def format_time(ms: float) -> str:
        """Format time in the most logical unit.
        
        Args:
            ms: Time in milliseconds
            
        Returns:
            Formatted time string with appropriate unit
        """
        if ms < 1000:
            return f"{ms:.1f}ms"
        
        seconds = ms / 1000
        if seconds < 60:
            return f"{seconds:.1f}s"
        
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.1f}min"
        
        hours = minutes / 60
        return f"{hours:.1f}h"
    
    @staticmethod
    def _format_memory_size(mb: float) -> str:
        """Format memory size with appropriate unit (MB or GB).
        
        Args:
            mb: Memory size in MB
            
        Returns:
            Formatted memory string: "512MB" or "2.5GB"
        """
        if mb < 1024:
            return f"{mb:.0f}MB"
        else:
            gb = mb / 1024
            return f"{gb:.1f}GB"
    
    @staticmethod
    def format_memory(used_mb: float, total_mb: Optional[float] = None) -> str:
        """Format memory usage with total and percentage.
        
        Args:
            used_mb: Memory used in MB
            total_mb: Total memory in MB (optional)
            
        Returns:
            Formatted memory string: "1.5GB / 4.0GB (37.5%)" or "512MB / 800MB (64.0%)"
        """
        if total_mb is None:
            return ModelEvaluationFormatter._format_memory_size(used_mb)
        
        percent = (used_mb / total_mb) * 100 if total_mb > 0 else 0
        used_str = ModelEvaluationFormatter._format_memory_size(used_mb)
        total_str = ModelEvaluationFormatter._format_memory_size(total_mb)
        return f"{used_str} / {total_str} ({percent:.1f}%)"
    
    @staticmethod
    def format_memory_percent_only(used_mb: float, total_mb: Optional[float] = None) -> str:
        """Format memory showing only percentage.
        
        Args:
            used_mb: Memory used in MB
            total_mb: Total memory in MB (optional)
            
        Returns:
            Formatted memory string: "25%" or "1.5GB" if no total
        """
        if total_mb is None:
            return ModelEvaluationFormatter._format_memory_size(used_mb)
        
        percent = (used_mb / total_mb) * 100 if total_mb > 0 else 0
        return f"{percent:.1f}%"
    
    def __init__(self, use_color: bool = True):
        """Initialize formatter.
        
        Args:
            use_color: Whether to use colored output
        """
        self.use_color = use_color
    
    def colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_color or color not in self.COLORS:
            return text
        return f"{self.COLORS[color]}{text}{self.COLORS['reset']}"
    
    def print_header(self, title: str) -> None:
        """Print a formatted header."""
        print()
        print(self.colorize("=" * 60, 'cyan'))
        print(self.colorize(f" {title}", 'bold'))
        print(self.colorize("=" * 60, 'cyan'))
        print()
    
    def print_model_info(self, model_info: dict) -> None:
        """Print model information."""
        print(f"Model: {self.colorize(model_info.get('model_name', 'Unknown'), 'bold')}")
        print(f"Backend: {model_info.get('backend', 'Unknown')}")
        
        # Show status first - most important info
        available = model_info.get('available', False)
        status_color = 'green' if available else 'red'
        status_text = "Available" if available else "Not Available"
        print(f"Status: {self.colorize(status_text, status_color)}")
        
        if not available and 'error' in model_info:
            print(f"Error: {self.colorize(model_info['error'], 'red')}")
        elif available:
            # Only show additional details if model is available
            if 'parameter_size' in model_info:
                print(f"Parameters: {model_info['parameter_size']}")
            if 'quantization' in model_info:
                print(f"Quantization: {model_info['quantization']}")
            
            # Show initialization time if available
            init_time = model_info.get('initialization_time_ms')
            if init_time is not None:
                print(f"Model initialization: {self.format_time(init_time)}")
        
        print()
    
    def print_live_metrics(self, query_num: int, total_queries: int, query: str,
                          current_tokens: int = 0, tokens_per_sec: float = 0.0,
                          elapsed_seconds: float = 0.0) -> None:
        """Print live metrics during evaluation."""
        # Clear line and print progress
        print(f"\r{self.colorize(f'[{query_num}/{total_queries}]', 'bold')} "
              f"Query: {query[:50]}{'...' if len(query) > 50 else ''}", end="", flush=True)
        
        if current_tokens > 0:
            print(f"\n  Tokens: {current_tokens} | "
                  f"Speed: {tokens_per_sec:.1f} tok/s | "
                  f"Time: {elapsed_seconds:.1f}s", end="", flush=True)
    
    def print_query_result(self, result, query_num: int, total_queries: int, verbose: bool = False) -> None:
        """Print results for a single query."""
        print(f"\n\n{self.colorize(f'Query {query_num}/{total_queries}:', 'bold')} {result.query}")
        
        if result.response.success:
            print(f"\n{self.colorize('Response:', 'green')}")
            # Truncate long responses
            response_text = result.response.content
            if not verbose and len(response_text) > 200:
                response_text = response_text[:200] + "..."
            print(response_text)
            
            # Timing metrics
            timing = result.timing_metrics
            print(f"\n{self.colorize('Performance:', 'blue')}")
            print(f"  Time to first token: {self.format_time(timing.time_to_first_token_ms)}")
            print(f"  Total inference time: {self.format_time(timing.total_inference_time_ms)}")
            
            # Token metrics
            tokens = result.token_metrics
            print(f"  Tokens generated: {tokens.completion_tokens}")
            print(f"  Tokens per second: {tokens.tokens_per_second:.1f}")
            
            # Resource metrics if available
            if result.resource_metrics and verbose:
                resources = result.resource_metrics
                print(f"\n{self.colorize('Resources:', 'yellow')}")
                print(f"  CPU: {resources.cpu_percent_avg:.1f}% (max: {resources.cpu_percent_max:.1f}%)")
                avg_mem_str = self.format_memory(resources.memory_used_mb_avg, resources.memory_total_mb)
                max_mem_str = self.format_memory_percent_only(resources.memory_used_mb_max, resources.memory_total_mb)
                print(f"  Memory: {avg_mem_str} (max: {max_mem_str})")
                
                if resources.gpu_utilization_avg is not None:
                    print(f"  GPU: {resources.gpu_utilization_avg:.1f}% (max: {resources.gpu_utilization_max:.1f}%)")
                    if resources.gpu_memory_used_mb_avg is not None:
                        avg_gpu_mem_str = self.format_memory(resources.gpu_memory_used_mb_avg, resources.gpu_memory_total_mb)
                        max_gpu_mem_str = self.format_memory_percent_only(resources.gpu_memory_used_mb_max, resources.gpu_memory_total_mb)
                        print(f"  GPU Memory: {avg_gpu_mem_str} (max: {max_gpu_mem_str})")
        else:
            print(f"\n{self.colorize('Error:', 'red')} {result.response.error_message}")
        
        print(self.colorize("-" * 60, 'cyan'))
    
    def print_summary(self, result: EvaluationResult) -> None:
        """Print evaluation summary."""
        self.print_header("EVALUATION SUMMARY")
        
        stats = result.get_summary_stats()
        
        total_duration_ms = result.total_duration_seconds * 1000
        print(f"Total Duration: {self.format_time(total_duration_ms)}")
        
        # Show model initialization time separately
        if result.model_initialization_time_ms is not None:
            print(f"Model Initialization: {self.format_time(result.model_initialization_time_ms)}")
        
        print(f"Queries Evaluated: {stats.get('query_count', 0)}")
        print(f"Successful Queries: {stats.get('successful_queries', 0)}")
        
        if stats.get('successful_queries', 0) > 0:
            print(f"\n{self.colorize('Performance Averages (excluding initialization):', 'green')}")
            print(f"  Time to first token: {self.format_time(stats.get('avg_time_to_first_token_ms', 0))}")
            print(f"  Total inference time: {self.format_time(stats.get('avg_total_inference_time_ms', 0))}") 
            print(f"  Tokens per second: {stats.get('avg_tokens_per_second', 0):.1f}")
            print(f"  Total tokens generated: {stats.get('total_tokens_generated', 0)}")
            
            if 'avg_cpu_percent' in stats:
                print(f"\n{self.colorize('Resource Usage:', 'yellow')}")
                print(f"  CPU: {stats['avg_cpu_percent']:.1f}% (max: {stats['max_cpu_percent']:.1f}%)")
                
                total_mem = stats.get('total_memory_mb')
                avg_mem_str = self.format_memory(stats['avg_memory_mb'], total_mem)
                max_mem_str = self.format_memory_percent_only(stats['max_memory_mb'], total_mem)
                print(f"  Memory: {avg_mem_str} (max: {max_mem_str})")
                
                if 'avg_gpu_percent' in stats:
                    print(f"  GPU: {stats['avg_gpu_percent']:.1f}% (max: {stats['max_gpu_percent']:.1f}%)")
                    
                    # GPU memory metrics
                    if 'avg_gpu_memory_mb' in stats and stats['avg_gpu_memory_mb'] is not None:
                        total_gpu_mem = stats.get('total_gpu_memory_mb')
                        avg_gpu_mem_str = self.format_memory(stats['avg_gpu_memory_mb'], total_gpu_mem)
                        max_gpu_mem_str = self.format_memory_percent_only(stats['max_gpu_memory_mb'], total_gpu_mem)
                        print(f"  GPU Memory: {avg_gpu_mem_str} (max: {max_gpu_mem_str})")
                    
                    # GPU power metrics
                    if 'avg_gpu_power_w' in stats and stats['avg_gpu_power_w'] is not None:
                        print(f"  GPU Power: {stats['avg_gpu_power_w']:.1f}W (max: {stats['max_gpu_power_w']:.1f}W)")


def create_backend(backend_type: str, model_name: str):
    """Create model backend instance."""
    if backend_type.lower() == 'ollama':
        return OllamaBackend(model_name)
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")


def load_queries_from_file(file_path: str) -> List[str]:
    """Load queries from a text file."""
    queries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                queries.append(line)
    return queries


def run_evaluation(model_name: str, backend_type: str = "ollama",
                  queries: Optional[List[str]] = None,
                  query_file: Optional[str] = None,
                  temperature: float = 0.7,
                  max_tokens: Optional[int] = None,
                  timeout: int = 300,
                  monitor_resources: bool = True,
                  formatter: Optional[ModelEvaluationFormatter] = None,
                  verbose: bool = False) -> EvaluationResult:
    """Run model evaluation with given parameters."""
    if formatter is None:
        formatter = ModelEvaluationFormatter()
    
    # Load queries if from file
    if query_file:
        queries = load_queries_from_file(query_file)
    
    # Create backend and evaluator
    try:
        backend = create_backend(backend_type, model_name)
        evaluator = ModelEvaluator(backend)
    except Exception as e:
        print(formatter.colorize(f"Failed to initialize backend: {e}", 'red'))
        return EvaluationResult(
            config=EvaluationConfig(model_name=model_name, backend_type=backend_type),
            model_info={},
            query_results=[],
            success=False,
            error_message=str(e)
        )
    
    # Check if model is available
    formatter.print_header(f"Model Evaluation: {model_name}")
    model_info = backend.get_model_info()
    formatter.print_model_info(model_info)
    
    if not model_info.get('available', False):
        return EvaluationResult(
            config=EvaluationConfig(model_name=model_name, backend_type=backend_type),
            model_info=model_info,
            query_results=[],
            success=False,
            error_message="Model not available"
        )
    
    # Create evaluation config
    config = EvaluationConfig(
        model_name=model_name,
        backend_type=backend_type,
        queries=queries,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout,
        monitor_resources=monitor_resources,
        sample_interval=0.1
    )
    
    # Progress callback for live updates
    def progress_callback(message: str, current: int, total: int):
        if verbose:
            formatter.print_live_metrics(current + 1, total, message)
        else:
            # Show progress even in non-verbose mode to indicate activity
            # Don't add 1 if we're already at the total (completion message)
            display_current = current if current >= total else current + 1
            print(f"\r{formatter.colorize(f'[{display_current}/{total}]', 'bold')} {message}...", end="", flush=True)
    
    # Run evaluation
    if queries:
        query_count = len(queries)
    else:
        query_count = len(evaluator.DEFAULT_QUERIES)
    
    print(formatter.colorize(f"Starting evaluation of {query_count} queries...", 'green'))
    result = evaluator.evaluate_model(config, progress_callback)
    
    # Clear progress line in non-verbose mode
    if not verbose:
        print()  # Move to next line after progress updates
    
    return result


def main():
    """Main entry point for the model evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM performance with resource monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m llama2:7b                                    # Basic evaluation
  %(prog)s -m qwen2.5:14b -q "Explain quantum computing"   # Custom query
  %(prog)s -m starcoder2:15b --queries-file queries.txt   # Batch queries from file
  %(prog)s -m llama2:7b -v --json                         # Verbose with JSON output
  %(prog)s -m codellama:13b --no-resources                # Skip resource monitoring
        """
    )
    
    parser.add_argument('-m', '--model', required=True,
                       help='Model name to evaluate')
    parser.add_argument('--backend', default='ollama',
                       help='Model backend to use (default: ollama)')
    parser.add_argument('-q', '--query',
                       help='Single query to evaluate')
    parser.add_argument('--queries', nargs='+',
                       help='Multiple queries to evaluate (space-separated)')
    parser.add_argument('--queries-file',
                       help='File containing queries (one per line)')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Generation temperature (default: 0.7)')
    parser.add_argument('--max-tokens', type=int,
                       help='Maximum tokens to generate')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout per query in seconds (default: 300)')
    parser.add_argument('--no-resources', action='store_true',
                       help='Skip resource monitoring')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed output and live metrics')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')
    parser.add_argument('-o', '--output',
                       help='Save results to file')
    
    args = parser.parse_args()
    
    # Prepare queries
    queries = None
    if args.query:
        queries = [args.query]
    elif args.queries:
        queries = args.queries
    # queries_file will be handled in run_evaluation
    
    # Initialize formatter
    formatter = ModelEvaluationFormatter(use_color=not args.no_color and not args.json)
    
    # Enable logging if verbose
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        result = run_evaluation(
            model_name=args.model,
            backend_type=args.backend,
            queries=queries,
            query_file=args.queries_file,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            monitor_resources=not args.no_resources,
            formatter=formatter,
            verbose=args.verbose
        )
        
        if args.json:
            output = json.dumps(result.to_dict(), indent=2)
            print(output)
        else:
            # Print detailed results
            if args.verbose:
                for i, query_result in enumerate(result.query_results, 1):
                    formatter.print_query_result(query_result, i, len(result.query_results), verbose=True)
            
            # Print summary
            formatter.print_summary(result)
        
        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nResults saved to: {args.output}")
        
        return 0 if result.success else 1
    
    except KeyboardInterrupt:
        print(f"\n{formatter.colorize('Evaluation cancelled by user', 'yellow')}")
        return 1
    except Exception as e:
        print(f"{formatter.colorize(f'Unexpected error: {e}', 'red')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())