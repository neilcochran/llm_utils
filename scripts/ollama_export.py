#!/usr/bin/env python3
"""Ollama Model Export Tool - CLI interface for exporting models to .gguf format."""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List

# Add src to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from ollama_utils.model_exporter import OllamaModelExporter, ModelExportInfo


class OllamaExportFormatter:
    """Handles formatting and display for ollama export results."""
    
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
    
    def __init__(self, use_color: bool = True):
        """Initialize formatter.
        
        Args:
            use_color: Whether to use colored output
        """
        self.use_color = use_color
    
    def colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled.
        
        Args:
            text: Text to colorize
            color: Color name from COLORS dict
            
        Returns:
            Colored text or plain text if colors disabled
        """
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
    
    def print_progress_bar(self, current: int, total: int, prefix: str = '', suffix: str = '', 
                          length: int = 50, fill: str = '█', empty: str = '░') -> None:
        """Print a progress bar.
        
        Args:
            current: Current progress value
            total: Total value for completion
            prefix: Text before progress bar
            suffix: Text after progress bar
            length: Length of progress bar in characters
            fill: Character for filled portion
            empty: Character for empty portion
        """
        if total == 0:
            percent = 100.0
        else:
            percent = min(100.0, (current / total) * 100)
        
        filled_length = int(length * current // total) if total > 0 else length
        bar = fill * filled_length + empty * (length - filled_length)
        
        print(f'\r{prefix} |{self.colorize(bar, "cyan")}| {percent:.1f}% {suffix}', end='', flush=True)
        
        if current >= total:
            print()  # New line when complete

    def print_step(self, step: str, status: str = "working") -> None:
        """Print a step with status indicator.
        
        Args:
            step: Description of the step
            status: Status - 'working', 'done', 'error', 'skip'
        """
        status_indicators = {
            'working': ('⏳', 'yellow'),
            'done': ('✓', 'green'), 
            'error': ('✗', 'red'),
            'skip': ('⚠', 'yellow')
        }
        
        icon, color = status_indicators.get(status, ('?', 'white'))
        print(f"{self.colorize(icon, color)} {step}")

    def print_model_info(self, export_info: ModelExportInfo) -> None:
        """Print formatted information about a model export.
        
        Args:
            export_info: Model export information to display
        """
        status_color = 'green' if export_info.success else 'red'
        status_text = "SUCCESS" if export_info.success else "FAILED"
        
        print(f"\nModel: {self.colorize(export_info.model_name, 'bold')}")
        print(f"Status: {self.colorize(status_text, status_color)}")
        
        if export_info.success:
            print(f"Export Directory: {self.colorize(export_info.export_directory or 'N/A', 'blue')}")
            
            if export_info.modelfile_path:
                print(f"Modelfile: {self.colorize('✓', 'green')} {export_info.modelfile_path}")
            
            if export_info.gguf_destination_path:
                print(f"GGUF File: {self.colorize('✓', 'green')} {export_info.gguf_destination_path}")
            elif export_info.gguf_source_path:
                print(f"GGUF File: {self.colorize('⚠', 'yellow')} Source found but copy failed")
            else:
                print(f"GGUF File: {self.colorize('✗', 'red')} Not found")
            
            # Show model details
            if export_info.parameters:
                print(f"Parameters: {len(export_info.parameters)} settings")
            
            if export_info.template:
                template_preview = export_info.template[:50] + "..." if len(export_info.template) > 50 else export_info.template
                print(f"Template: {template_preview}")
        else:
            print(f"Error: {self.colorize(export_info.error_message or 'Unknown error', 'red')}")
        
        print("-" * 60)
    
    def print_summary(self, results: List[ModelExportInfo]) -> None:
        """Print export summary statistics.
        
        Args:
            results: List of export results
        """
        if not results:
            print(self.colorize("No models processed", 'yellow'))
            return
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        print()
        print(self.colorize("EXPORT SUMMARY", 'bold'))
        print(f"Total Models: {len(results)}")
        print(f"Successful: {self.colorize(str(successful), 'green')}")
        if failed > 0:
            print(f"Failed: {self.colorize(str(failed), 'red')}")
        print()


def list_models(exporter: OllamaModelExporter, formatter: OllamaExportFormatter) -> bool:
    """List available Ollama models.
    
    Args:
        exporter: Model exporter instance
        formatter: Display formatter
        
    Returns:
        True if successful, False otherwise
    """
    formatter.print_header("Available Ollama Models")
    
    success, models = exporter.list_available_models()
    if not success:
        print(formatter.colorize("Failed to list models. Is Ollama running?", 'red'))
        return False
    
    if not models:
        print(formatter.colorize("No models found", 'yellow'))
        return True
    
    for i, model in enumerate(models, 1):
        print(f"{i:2d}. {formatter.colorize(model, 'blue')}")
    
    print(f"\nFound {len(models)} model(s)")
    return True


def export_single_model(model_name: str, output_dir: str, 
                       exporter: OllamaModelExporter, 
                       formatter: OllamaExportFormatter,
                       verbose: bool = False) -> bool:
    """Export a single model.
    
    Args:
        model_name: Name of the model to export
        output_dir: Output directory
        exporter: Model exporter instance
        formatter: Display formatter
        verbose: Whether to show detailed information
        
    Returns:
        True if successful, False otherwise
    """
    formatter.print_header(f"Exporting Model: {model_name}")
    
    # Define progress callbacks
    def step_callback(step: str, status: str):
        formatter.print_step(step, status)
    
    def progress_callback(bytes_copied: int, total_bytes: int, message: str):
        if total_bytes > 0:
            # Convert to MB for display
            mb_copied = bytes_copied / (1024 * 1024)
            mb_total = total_bytes / (1024 * 1024)
            formatter.print_progress_bar(
                bytes_copied, total_bytes, 
                prefix="Copying", 
                suffix=f"{mb_copied:.1f}MB / {mb_total:.1f}MB"
            )
    
    result = exporter.export_model(model_name, output_dir, step_callback, progress_callback)
    
    if verbose or not result.success:
        formatter.print_model_info(result)
    else:
        status_color = 'green' if result.success else 'red'
        status_text = "SUCCESS" if result.success else "FAILED"
        print(f"\n{model_name}: {formatter.colorize(status_text, status_color)}")
    
    return result.success


def export_multiple_models(model_names: list, output_dir: str,
                          exporter: OllamaModelExporter,
                          formatter: OllamaExportFormatter,
                          verbose: bool = False) -> bool:
    """Export multiple specific models.
    
    Args:
        model_names: List of model names to export
        output_dir: Output directory
        exporter: Model exporter instance
        formatter: Display formatter
        verbose: Whether to show detailed information
        
    Returns:
        True if any models were exported successfully, False otherwise
    """
    formatter.print_header(f"Exporting {len(model_names)} Models")
    
    results = []
    
    # Define progress callbacks
    def step_callback(step: str, status: str):
        formatter.print_step(step, status)
    
    def progress_callback(bytes_copied: int, total_bytes: int, message: str):
        if total_bytes > 0:
            # Convert to MB for display
            mb_copied = bytes_copied / (1024 * 1024)
            mb_total = total_bytes / (1024 * 1024)
            formatter.print_progress_bar(
                bytes_copied, total_bytes, 
                prefix="Copying", 
                suffix=f"{mb_copied:.1f}MB / {mb_total:.1f}MB"
            )
    
    # Export each model with progress
    for i, model_name in enumerate(model_names, 1):
        print(f"\n{formatter.colorize(f'[{i}/{len(model_names)}]', 'bold')} Exporting {formatter.colorize(model_name, 'cyan')}")
        
        result = exporter.export_model(model_name, output_dir, step_callback, progress_callback)
        results.append(result)
        
        if not verbose:
            status_color = 'green' if result.success else 'red'
            status_text = "✓" if result.success else "✗"
            print(f"{formatter.colorize(status_text, status_color)} {model_name}")
    
    # Show summary
    formatter.print_summary(results)
    
    return any(r.success for r in results)


def export_all_models(output_dir: str,
                     exporter: OllamaModelExporter,
                     formatter: OllamaExportFormatter,
                     verbose: bool = False) -> bool:
    """Export all available models.
    
    Args:
        output_dir: Output directory
        exporter: Model exporter instance
        formatter: Display formatter
        verbose: Whether to show detailed information
        
    Returns:
        True if any models were exported successfully, False otherwise
    """
    formatter.print_header("Exporting All Models")
    
    # Get list of models first
    success, models = exporter.list_available_models()
    if not success:
        print(formatter.colorize("Failed to list models", 'red'))
        return False
    
    if not models:
        print(formatter.colorize("No models to export", 'yellow'))
        return False
    
    print(f"Found {len(models)} models to export\n")
    
    results = []
    
    # Define progress callbacks
    def step_callback(step: str, status: str):
        formatter.print_step(step, status)
    
    def progress_callback(bytes_copied: int, total_bytes: int, message: str):
        if total_bytes > 0:
            # Convert to MB for display
            mb_copied = bytes_copied / (1024 * 1024)
            mb_total = total_bytes / (1024 * 1024)
            formatter.print_progress_bar(
                bytes_copied, total_bytes, 
                prefix="Copying", 
                suffix=f"{mb_copied:.1f}MB / {mb_total:.1f}MB"
            )
    
    # Export each model with progress
    for i, model_name in enumerate(models, 1):
        print(f"\n{formatter.colorize(f'[{i}/{len(models)}]', 'bold')} Exporting {formatter.colorize(model_name, 'cyan')}")
        
        result = exporter.export_model(model_name, output_dir, step_callback, progress_callback)
        results.append(result)
        
        if not verbose:
            status_color = 'green' if result.success else 'red'
            status_text = "✓" if result.success else "✗"
            print(f"{formatter.colorize(status_text, status_color)} {model_name}")
    
    # Show summary
    formatter.print_summary(results)
    
    return any(r.success for r in results)


def main():
    """Main entry point for the ollama export CLI."""
    parser = argparse.ArgumentParser(
        description="Export Ollama models to .gguf format with Modelfiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list --ollama-path "C:\\Users\\username\\.ollama\\models"
  %(prog)s -m llama2 -o ./exports --ollama-path "C:\\Users\\username\\.ollama\\models"
  %(prog)s --models llama2 codellama qwen2.5:14b -o ./exports --ollama-path "/path/to/models"
  %(prog)s --all -o ./exports --ollama-path "/home/user/.ollama/models"  
  %(prog)s -m llama2 -o ./exports -v --ollama-path "C:\\Users\\username\\.ollama\\models"
        """
    )
    
    parser.add_argument('-m', '--model', 
                       help='Name of the model to export')
    parser.add_argument('--models', nargs='+',
                       help='List of specific models to export (space-separated)')
    parser.add_argument('-o', '--output', 
                       help='Output directory for exported models',
                       default='./ollama_exports')
    parser.add_argument('--all', action='store_true',
                       help='Export all available models')
    parser.add_argument('--list', action='store_true',
                       help='List available models and exit')
    parser.add_argument('--ollama-path', required=True,
                       help='Path to Ollama models directory (e.g., C:\\Users\\username\\.ollama\\models)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.list and not args.model and not args.models and not args.all:
        parser.error("Must specify --list, --model, --models, or --all")
    
    exclusive_args = sum([bool(args.model), bool(args.models), bool(args.all)])
    if exclusive_args > 1:
        parser.error("Cannot specify more than one of --model, --models, or --all")
    
    # Validate Ollama models path
    ollama_path = Path(args.ollama_path)
    if not ollama_path.exists():
        print(f"Error: Ollama models directory does not exist: {args.ollama_path}")
        print("Common locations:")
        print("  Windows: C:\\Users\\<username>\\.ollama\\models")
        print("  Linux:   /home/<username>/.ollama/models") 
        print("  macOS:   /Users/<username>/.ollama/models")
        return 1
    
    if not ollama_path.is_dir():
        print(f"Error: Path is not a directory: {args.ollama_path}")
        return 1
    
    # Initialize components
    exporter = OllamaModelExporter(ollama_models_path=str(ollama_path))
    formatter = OllamaExportFormatter(use_color=not args.no_color and not args.json)
    
    # Enable debug logging if verbose
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        if args.list:
            success = list_models(exporter, formatter)
            return 0 if success else 1
        
        # Ensure output directory exists
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if args.model:
            success = export_single_model(args.model, args.output, exporter, formatter, args.verbose)
            return 0 if success else 1
        
        elif args.models:
            success = export_multiple_models(args.models, args.output, exporter, formatter, args.verbose)
            return 0 if success else 1
        
        elif args.all:
            success = export_all_models(args.output, exporter, formatter, args.verbose)
            return 0 if success else 1
    
    except KeyboardInterrupt:
        print(f"\n{formatter.colorize('Export cancelled by user', 'yellow')}")
        return 1
    except Exception as e:
        print(f"{formatter.colorize(f'Unexpected error: {e}', 'red')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())