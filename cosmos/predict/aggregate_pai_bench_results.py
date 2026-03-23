"""Aggregate results script for Physical AI Benchmark evaluation.

This script aggregates quality and domain scores to compute the overall PAI-Bench score.
The Quality Score is derived from text-to-video and image-to-video metrics adapted from VBench.
The Domain Score is obtained through VQA-based evaluation across seven domains.
"""

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
QUALITY_RESULTS_PATTERN = "*_eval_results.json"
DOMAIN_RESULTS_FILE = "vqa_summary.json"
FINAL_RESULTS_FILE = "final_results.txt"

# VBench metric prefixes to ignore for text2world evaluation
IMAGE_TO_VIDEO_PREFIX = "i2v_"

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description='Aggregate results for PAI-Bench evaluation')
    parser.add_argument('--input_dir', type=str, required=True, 
                       help='Input directory containing evaluation results')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Output directory for aggregated results')
    parser.add_argument('--inference_type', type=str, required=True, 
                       choices=['text2world', 'image2world'],
                       help='Inference type (text2world or image2world)')
    return parser.parse_args()


def find_quality_results_file(input_dir: str) -> Path:
    """Find the quality results file in the input directory.
    
    Args:
        input_dir: Directory to search for quality results
        
    Returns:
        Path to the quality results file
        
    Raises:
        FileNotFoundError: If no quality results file is found
        ValueError: If multiple quality results files are found
    """
    pattern = str(Path(input_dir) / QUALITY_RESULTS_PATTERN)
    quality_files = glob.glob(pattern)
    
    if not quality_files:
        raise FileNotFoundError(f"No quality results files found matching pattern: {pattern}")
    if len(quality_files) > 1:
        raise ValueError(f"Multiple quality results files found: {quality_files}")
    
    return Path(quality_files[0])


def load_quality_results(quality_file: Path, inference_type: str) -> Dict[str, float]:
    """Load and filter quality results based on inference type.
    
    Args:
        quality_file: Path to the quality results JSON file
        inference_type: Type of inference to filter for
        
    Returns:
        Filtered quality results dictionary
        
    Raises:
        json.JSONDecodeError: If the quality file is not valid JSON
        ValueError: If inference_type is not supported
    """
    try:
        with open(quality_file, 'r', encoding='utf-8') as file:
            quality_data = json.load(file)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in quality results file: %s", e)
        raise
    
    # Filter results based on inference type
    if inference_type == 'text2world':
        # Ignore image-to-video metrics for text2world evaluation
        filtered_results = {
            metric: scores[0] for metric, scores in quality_data.items() 
            if not metric.startswith(IMAGE_TO_VIDEO_PREFIX)
        }
    elif inference_type == 'image2world':
        # Include all metrics for image2world evaluation
        filtered_results = {metric: scores[0] for metric, scores in quality_data.items()}
    else:
        raise ValueError(f'Unknown inference type: {inference_type}')
    
    logger.info("Loaded %d quality metrics for %s", len(filtered_results), inference_type)
    return filtered_results


def load_domain_results(input_dir: str) -> Dict[str, Any]:
    """Load domain evaluation results.
    
    Args:
        input_dir: Directory containing domain results
        
    Returns:
        Domain results dictionary
        
    Raises:
        FileNotFoundError: If domain results file doesn't exist
        json.JSONDecodeError: If the domain file is not valid JSON
    """
    domain_file = Path(input_dir) / DOMAIN_RESULTS_FILE
    
    if not domain_file.exists():
        raise FileNotFoundError(f"Domain results file not found: {domain_file}")
    
    try:
        with open(domain_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in domain results file: %s", e)
        raise


def calculate_scores(quality_results: Dict[str, float], domain_results: Dict[str, Any]) -> tuple[float, float, float]:
    """Calculate quality, domain, and overall scores.
    
    Args:
        quality_results: Quality evaluation results
        domain_results: Domain evaluation results
        
    Returns:
        Tuple of (quality_score, domain_score, overall_score)
        
    Raises:
        KeyError: If required keys are missing from domain_results
        ZeroDivisionError: If no quality metrics are available
    """
    if not quality_results:
        raise ZeroDivisionError("No quality metrics available for score calculation")
    
    quality_score = sum(quality_results.values()) / len(quality_results)
    
    try:
        domain_score = domain_results['overall_accuracy']
    except KeyError as e:
        logger.error("Missing 'overall_accuracy' key in domain results")
        raise
    
    overall_score = (quality_score + domain_score) / 2
    
    return quality_score, domain_score, overall_score


def save_results(output_dir: str, quality_score: float, domain_score: float, overall_score: float) -> None:
    """Save final results to file.
    
    Args:
        output_dir: Directory to save results
        quality_score: Calculated quality score
        domain_score: Domain evaluation score
        overall_score: Final overall score
    """
    results_content = f"""
{"="*13} Final Results {"="*13} 
Domain  score {domain_score:.3f}
Quality score {quality_score:.3f}
Overall score {overall_score:.3f}
"""
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save results to file
    results_file = output_path / FINAL_RESULTS_FILE
    try:
        with open(results_file, 'w', encoding='utf-8') as file:
            file.write(results_content)
        logger.info("Results saved to %s", results_file)
    except IOError as e:
        logger.error("Failed to save results to %s: %s", results_file, e)
        raise
    
    return results_content


def main() -> None:
    """Main function for aggregating PAI-Bench results."""
    try:
        args = parse_arguments()
        
        logger.info("="*10 + " Aggregating Results " + "="*10)
        logger.info("Input directory: %s", args.input_dir)
        logger.info("Output directory: %s", args.output_dir)
        logger.info("Inference type: %s", args.inference_type)
        
        # Load quality results
        quality_file = find_quality_results_file(args.input_dir)
        quality_results = load_quality_results(quality_file, args.inference_type)
        logger.info("Quality results loaded from %s", quality_file)
        
        # Load domain results
        domain_results = load_domain_results(args.input_dir)
        logger.info("Domain results loaded successfully")
        
        # Calculate scores
        quality_score, domain_score, overall_score = calculate_scores(quality_results, domain_results)
        
        # Save and display results
        results_content = save_results(args.output_dir, quality_score, domain_score, overall_score)
        print(results_content)
        
        logger.info("Results aggregation completed successfully")
        
    except (FileNotFoundError, ValueError, json.JSONDecodeError, KeyError, ZeroDivisionError) as e:
        logger.error("Error during results aggregation: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise


if __name__ == '__main__':
    main()




