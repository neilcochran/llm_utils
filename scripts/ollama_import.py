#!/usr/bin/env python3
"""Ollama Model Import Tool - CLI interface for importing .gguf files into Ollama."""

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

from ollama_utils.model_importer import OllamaModelImporter, ModelImportInfo


class OllamaImportFormatter:
    """Handles formatting and display for ollama import results."""
    
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

    def print_model_info(self, import_info: ModelImportInfo) -> None:
        """Print formatted information about a model import.
        
        Args:
            import_info: Model import information to display
        """
        status_color = 'green' if import_info.import_success else 'red'
        status_text = "SUCCESS" if import_info.import_success else "FAILED"
        
        print(f"\nModel: {self.colorize(import_info.model_name, 'bold')}")
        print(f"Status: {self.colorize(status_text, status_color)}")
        print(f"GGUF Source: {import_info.gguf_source_path}")
        
        if import_info.import_success:
            if import_info.modelfile_path:
                print(f"Modelfile: {self.colorize('✓', 'green')} {import_info.modelfile_path}")
            else:
                print(f"Modelfile: {self.colorize('⚠', 'yellow')} Used default")
            
            if import_info.ollama_create_success:
                print(f"Ollama Import: {self.colorize('✓', 'green')} Created successfully")
            else:
                print(f"Ollama Import: {self.colorize('✗', 'red')} Failed")
        else:
            print(f"Error: {self.colorize(import_info.error_message or 'Unknown error', 'red')}")
        
        print("-" * 60)
    
    def print_summary(self, results: List[ModelImportInfo]) -> None:
        """Print import summary statistics.
        
        Args:
            results: List of import results
        """
        if not results:
            print(self.colorize("No models processed", 'yellow'))
            return
        
        successful = sum(1 for r in results if r.import_success)
        failed = len(results) - successful
        
        print()
        print(self.colorize("IMPORT SUMMARY", 'bold'))
        print(f"Total Models: {len(results)}")
        print(f"Successful: {self.colorize(str(successful), 'green')}")
        if failed > 0:
            print(f"Failed: {self.colorize(str(failed), 'red')}")
        print()


def import_single_model(gguf_path: str, model_name: str,
                       importer: OllamaModelImporter,
                       formatter: OllamaImportFormatter,
                       verbose: bool = False) -> bool:
    """Import a single .gguf file.
    
    Args:
        gguf_path: Path to the .gguf file
        model_name: Name for the imported model
        importer: Model importer instance
        formatter: Display formatter
        verbose: Whether to show detailed information
        
    Returns:
        True if successful, False otherwise
    """
    formatter.print_header(f"Importing Model: {model_name or Path(gguf_path).stem}")
    
    # Define progress callbacks
    def step_callback(step: str, status: str):
        formatter.print_step(step, status)
    
    result = importer.import_model(gguf_path, model_name, step_callback)
    
    if verbose or not result.import_success:
        formatter.print_model_info(result)
    else:
        status_color = 'green' if result.import_success else 'red'
        status_text = "SUCCESS" if result.import_success else "FAILED"
        print(f"\n{result.model_name}: {formatter.colorize(status_text, status_color)}")
    
    return result.import_success


def import_multiple_models(gguf_paths: List[str],
                          importer: OllamaModelImporter,
                          formatter: OllamaImportFormatter,
                          verbose: bool = False) -> bool:
    """Import multiple .gguf files.
    
    Args:
        gguf_paths: List of paths to .gguf files
        importer: Model importer instance
        formatter: Display formatter
        verbose: Whether to show detailed information
        
    Returns:
        True if any models were imported successfully, False otherwise
    """
    formatter.print_header(f"Importing {len(gguf_paths)} Models")
    
    results = []
    
    # Define progress callbacks
    def step_callback(step: str, status: str):
        formatter.print_step(step, status)
    
    # Import each model with progress
    for i, gguf_path in enumerate(gguf_paths, 1):
        model_name = importer.sanitize_model_name(gguf_path)
        print(f"\n{formatter.colorize(f'[{i}/{len(gguf_paths)}]', 'bold')} Importing {formatter.colorize(model_name, 'cyan')}")
        
        result = importer.import_model(gguf_path, None, step_callback)
        results.append(result)
        
        if not verbose:
            status_color = 'green' if result.import_success else 'red'
            status_text = "✓" if result.import_success else "✗"
            print(f"{formatter.colorize(status_text, status_color)} {result.model_name}")
    
    # Show summary
    formatter.print_summary(results)
    
    return any(r.import_success for r in results)


def discover_and_import_all(directory: str,
                           importer: OllamaModelImporter,
                           formatter: OllamaImportFormatter,
                           verbose: bool = False) -> bool:
    """Discover and import all .gguf files in a directory.
    
    Args:
        directory: Directory to search for .gguf files
        importer: Model importer instance
        formatter: Display formatter
        verbose: Whether to show detailed information
        
    Returns:
        True if any models were imported successfully, False otherwise
    """
    formatter.print_header(f"Discovering Models in: {directory}")
    
    # Discover .gguf files
    gguf_files = importer.discover_gguf_files(directory)
    
    if not gguf_files:
        print(formatter.colorize("No .gguf files found in directory", 'yellow'))
        return False
    
    print(f"Found {len(gguf_files)} .gguf file(s):")
    for i, gguf_file in enumerate(gguf_files, 1):
        print(f"  {i}. {gguf_file}")
    
    # Import all discovered files
    return import_multiple_models(gguf_files, importer, formatter, verbose)


def main():
    """Main entry point for the ollama import CLI."""
    parser = argparse.ArgumentParser(
        description="Import .gguf files into Ollama with Modelfiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f model.gguf -n my-model                    # Import single file
  %(prog)s --files model1.gguf model2.gguf model3.gguf # Import multiple files
  %(prog)s -d ./exported_models                         # Import all from directory
  %(prog)s -f model.gguf -n my-model -v                # Import with verbose output
        """
    )
    
    parser.add_argument('-f', '--file',
                       help='Path to .gguf file to import')
    parser.add_argument('--files', nargs='+',
                       help='List of .gguf files to import (space-separated)')
    parser.add_argument('-d', '--directory',
                       help='Directory to search for .gguf files')
    parser.add_argument('-n', '--name',
                       help='Name for the imported model (only with --file)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--no-color', action='store_true',
                       help='Disable colored output')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.file and not args.files and not args.directory:
        parser.error("Must specify --file, --files, or --directory")
    
    exclusive_args = sum([bool(args.file), bool(args.files), bool(args.directory)])
    if exclusive_args > 1:
        parser.error("Cannot specify more than one of --file, --files, or --directory")
    
    if args.name and not args.file:
        parser.error("--name can only be used with --file")
    
    # Initialize components
    importer = OllamaModelImporter()
    formatter = OllamaImportFormatter(use_color=not args.no_color and not args.json)
    
    # Enable debug logging if verbose
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    try:
        if args.file:
            # Validate file exists
            if not Path(args.file).exists():
                print(f"Error: File does not exist: {args.file}")
                return 1
            
            success = import_single_model(args.file, args.name, importer, formatter, args.verbose)
            return 0 if success else 1
        
        elif args.files:
            # Validate all files exist
            for file_path in args.files:
                if not Path(file_path).exists():
                    print(f"Error: File does not exist: {file_path}")
                    return 1
            
            success = import_multiple_models(args.files, importer, formatter, args.verbose)
            return 0 if success else 1
        
        elif args.directory:
            # Validate directory exists
            if not Path(args.directory).exists():
                print(f"Error: Directory does not exist: {args.directory}")
                return 1
            
            if not Path(args.directory).is_dir():
                print(f"Error: Path is not a directory: {args.directory}")
                return 1
            
            success = discover_and_import_all(args.directory, importer, formatter, args.verbose)
            return 0 if success else 1
    
    except KeyboardInterrupt:
        print(f"\n{formatter.colorize('Import cancelled by user', 'yellow')}")
        return 1
    except Exception as e:
        print(f"{formatter.colorize(f'Unexpected error: {e}', 'red')}")
        return 1


if __name__ == "__main__":
    sys.exit(main())