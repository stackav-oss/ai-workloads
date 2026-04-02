#!/usr/bin/env python3
"""Generate a summary table of throughput results from JSON files."""

import json
import sys
from pathlib import Path
from typing import Dict, List


def collect_results(output_dir: str) -> Dict[str, List[Dict]]:
    """Collect all throughput JSON results from the output directory.
    
    Args:
        output_dir: Directory containing throughput_*.json files
        
    Returns:
        Dictionary mapping control types to list of results
    """
    results: Dict[str, List[Dict]] = {}
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Warning: Output directory '{output_dir}' does not exist.", file=sys.stderr)
        return results
    
    # Find all throughput JSON files
    json_files = sorted(output_path.glob("throughput_*.json"))
    
    if not json_files:
        print(f"Warning: No throughput JSON files found in '{output_dir}'.", file=sys.stderr)
        return results
    
    # Load and organize results
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                control_type = data.get("control_type")
                if control_type not in results:
                    results[control_type] = []
                results[control_type].append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to read {json_file}: {e}", file=sys.stderr)
    
    return results


def create_summary_table(results: Dict[str, List[Dict]]) -> str:
    """Create a formatted ASCII table from collected results.
    
    Args:
        results: Dictionary mapping control types to list of results
        
    Returns:
        Formatted table as a string
    """
    if not results:
        return "No results found."
    
    # Sort by control type and GPU count for consistent table layout
    sorted_items = []
    for control_type in sorted(results.keys()):
        for result in sorted(results[control_type], key=lambda x: x["nproc_per_node"]):
            sorted_items.append((control_type, result))
    
    # Table header
    lines = [
        "=" * 100,
        "COSMOS TRANSFER THROUGHPUT BENCHMARK SUMMARY",
        "=" * 100,
        f"{'Control Type':<15} {'GPUs':<6} {'Avg Time (s)':<15} {'Std Dev (s)':<15} {'Throughput':<20} {'Total Runs':<12}",
        "-" * 100,
    ]
    
    # Table rows
    for control_type, result in sorted_items:
        avg_time = result.get("avg_time", 0)
        std_time = result.get("std_time", 0)
        throughput = result.get("throughput", 0)
        total_runs = result.get("total_runs", 0)
        nproc = result.get("nproc_per_node", 0)
        
        lines.append(
            f"{control_type:<15} {nproc:<6} {avg_time:<15.4f} {std_time:<15.4f} "
            f"{throughput:<20.4f} {total_runs:<12}"
        )
    
    lines.extend([
        "=" * 100,
        f"Evaluation Time: {sorted_items[0][1].get('evaluation_time', 'N/A') if sorted_items else 'N/A'}",
        "=" * 100,
    ])
    
    return "\n".join(lines)


def save_summary(output_dir: str, summary_table: str) -> Path:
    """Save the summary table to a file.
    
    Args:
        output_dir: Directory where the summary file will be saved
        summary_table: The formatted summary table
        
    Returns:
        Path to the saved summary file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_file = output_path / "throughput_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary_table + "\n")
    
    return summary_file


def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Usage: python generate_throughput_summary.py <output_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    # Collect results from JSON files
    results = collect_results(output_dir)
    
    if not results:
        print("No results to summarize.", file=sys.stderr)
        sys.exit(1)
    
    # Create summary table
    summary_table = create_summary_table(results)
    
    # Print to stdout
    print(summary_table)
    
    # Save to file
    summary_file = save_summary(output_dir, summary_table)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
