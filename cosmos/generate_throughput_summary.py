"""Generate throughput summary results from individual test runs."""

import argparse
import json
import logging
import statistics
import time
from pathlib import Path
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate throughput summary results')
    parser.add_argument('--inference_type', type=str, required=True,
                       help='Inference type (text2world or image2world)')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Results directory containing test data')
    return parser.parse_args()


def load_individual_results(results_dir: str) -> List[Dict[str, Any]]:
    """Load individual throughput results from separate files."""
    results_path = Path(results_dir)
    results_data = []
    
    if not results_path.exists():
        logger.warning("Results directory does not exist: %s", results_dir)
        return []
    
    # Look for individual GPU configuration result files
    gpu_configs = [1, 2, 4, 8]  # Standard configurations
    
    for nproc in gpu_configs:
        result_file = results_path / f"{nproc}gpu" / f"throughput_results_{nproc}gpu.txt"
        
        if result_file.exists():
            try:
                with open(result_file, 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                # Parse the content to extract metrics
                avg_time = None
                std_time = None
                throughput = None
                
                for line in content.split('\n'):
                    if line.startswith('Average Time:'):
                        avg_time = float(line.split(':')[1].strip().split()[0])
                    elif line.startswith('Std Deviation:'):
                        std_time = float(line.split(':')[1].strip().split()[0])
                    elif line.startswith('Throughput:'):
                        throughput = float(line.split(':')[1].strip().split()[0])
                
                if all(v is not None for v in [avg_time, std_time, throughput]):
                    results_data.append({
                        'nproc_per_node': nproc,
                        'status': 'COMPLETED',
                        'avg_time': avg_time,
                        'std_time': std_time,
                        'throughput': throughput
                    })
                    logger.info("Loaded results for %d GPU configuration", nproc)
                else:
                    logger.warning("Incomplete data in %s", result_file)
                    
            except (IOError, ValueError) as e:
                logger.error("Error loading results from %s: %s", result_file, e)
        else:
            # Mark as failed/missing
            results_data.append({
                'nproc_per_node': nproc,
                'status': 'FAILED',
                'avg_time': 0,
                'std_time': 0,
                'throughput': 0
            })
    
    return results_data


def generate_throughput_summary(results_data: List[Dict[str, Any]], inference_type: str, output_dir: str) -> str:
    """Generate formatted throughput summary."""
    output_dir = Path(output_dir)
    results_file = output_dir / "throughput_results.txt"
    
    # Create formatted results content
    content_lines = [
        "=" * 60,
        f"COSMOS {inference_type.upper()} THROUGHPUT BENCHMARK RESULTS",
        "=" * 60,
        f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Inference Type: {inference_type}",
        f"Dataset: Physical AI Benchmark",
        "",
        "GPU Configuration and Performance Results:",
        "-" * 40,
    ]
    
    successful_configs = []
    failed_configs = []
    
    for result in results_data:
        nproc = result['nproc_per_node']
        status = result['status']
        
        if status == "COMPLETED":
            avg_time = result['avg_time']
            std_time = result['std_time']
            throughput = result['throughput']
            successful_configs.append(result)
            
            content_lines.extend([
                "",
                f"🟢 {nproc} GPU{'s' if nproc > 1 else ''}:",
                f"   Average Time: {avg_time:.4f} seconds",
                f"   Std Deviation: {std_time:.4f} seconds",
                f"   Throughput: {throughput:.4f} samples/sec/GPU",
                f"   Status: ✅ COMPLETED",
            ])
        else:
            failed_configs.append(result)
            content_lines.extend([
                "",
                f"🔴 {nproc} GPU{'s' if nproc > 1 else ''}:",
                f"   Status: ❌ {status}",
            ])
    
    # Add summary
    #content_lines.extend([
    #    "",
    #    "-" * 40,
    #    "Summary:",
    #    f"✅ Successful Configurations: {len(successful_configs)}",
    #    f"❌ Failed Configurations: {len(failed_configs)}",
    #    f"📊 Total Tests Run: {len(results_data)}",
    #])
    
    # Add best performance if any successful tests
    #if successful_configs:
    #    best_config = max(successful_configs, key=lambda x: x['throughput'])
    #    content_lines.extend([
    #        "",
    #        "🏆 Best Performance:",
    #        f"   Configuration: {best_config['nproc_per_node']} GPU{'s' if best_config['nproc_per_node'] > 1 else ''}",
    #        f"   Throughput: {best_config['throughput']:.4f} samples/sec/GPU",
    #        f"   Average Time: {best_config['avg_time']:.4f} seconds",
    #    ])
    
    #content_lines.extend([
    #    "",
    #    "=" * 60,
    #])
    
    content_lines.append("")
    content = "\n".join(content_lines)
    
    try:
        with open(results_file, 'w', encoding='utf-8') as file:
            file.write(content)
        logger.info("Throughput results saved to %s", results_file)
        return str(results_file)
    except IOError as e:
        logger.error("Failed to save throughput results: %s", e)
        raise


def main():
    """Main function."""
    args = parse_arguments()
    
    logger.info("Generating throughput summary for %s", args.inference_type)
    
    # Load individual test results
    results_data = load_individual_results(args.results_dir)
    
    if not results_data:
        logger.warning("No test data found, creating empty results file")
        results_data = []
    
    # Generate summary
    results_file = generate_throughput_summary(results_data, args.inference_type, args.results_dir)
    
    logger.info("Throughput summary generated successfully")


if __name__ == '__main__':
    main()