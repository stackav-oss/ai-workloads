import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate and aggregate evaluation results from JSON metrics.")
    parser.add_argument(
        "--metrics_dir", 
        type=str, 
        required=True,
        help="Path to the directory containing JSON metric files."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)

    if not metrics_dir.is_dir():
        logging.error(f"Directory not found: {metrics_dir}")
        return

    # Collect all global data from JSON files
    all_global_data = []
    json_files = list(metrics_dir.glob("*/*.json"))
    
    # Filter out results from previous runs if they exist
    json_files = [f for f in json_files if f.name != "all.json"]

    if not json_files:
        logging.warning(f"No JSON files found in {metrics_dir}")
        return

    total_video_items = 0
    for filepath in json_files:
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if "global" in data:
                    all_global_data.append(data["global"])
                    total_video_items += len(data["per_video"])
                else:
                    logging.debug(f"Missing 'global' field in {filepath.name}")
        except (json.JSONDecodeError, IOError) as e:
            logging.error(f"Error reading {filepath.name}: {e}")

    if not all_global_data:
        logging.warning("No data found for the 'global' field in the provided files.")
        return

    # Calculate averages for each metric
    totals = defaultdict(float)
    counts = defaultdict(int)

    for entry in all_global_data:
        for key, value in entry.items():
            if isinstance(value, (int, float)):
                totals[key] += value
                counts[key] += 1

    # Compute final averages
    averages = {key: totals[key] / counts[key] for key in totals}

    # Output results
    output_path = metrics_dir / "all.json"
    try:
        with open(output_path, "w") as f:
            json.dump(averages, f, indent=4)
        logging.info(f"Aggregated results saved to: {output_path}")
    except IOError as e:
        logging.error(f"Failed to save results: {e}")

    # Print to console
    print(f"\n{'='*40}")
    print(f"Evaluation Averages ({len(all_global_data)} files, {total_video_items} videos)")
    print(metrics_dir)
    print(f"{'='*40}")
    for key, value in sorted(averages.items()):
        print(f"{key:<20}: {value:.4f}")
    print(f"{'='*40}\n")

if __name__ == "__main__":
    main()
