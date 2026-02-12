import json
import glob
import argparse

def main():
    parser = argparse.ArgumentParser(description='Aggregate results')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--inference_type', type=str, required=True, help='Inference type')
    args = parser.parse_args()

    print(f"\n\n{'='*10} Aggregating Results {'='*10}")
    print(f'Input directory: {args.input_dir}')
    print(f'Output directory: {args.output_dir}')
    print(f'Inference type: {args.inference_type}')


    # The Quality Score is derived from eight text-to-video and image-to-video metrics adapted from VBench.
    # Find the quality results file
    quality_input_file = glob.glob(f'{args.input_dir}/*_eval_results.json')[0]
    quality_results =json.load(open(quality_input_file))
    #ignore image to video scores for text2world
    if args.inference_type == 'text2world':
        quality_results = {k:v[0] for k,v in quality_results.items() if not k.startswith('i2v_')}
    elif args.inference_type == 'image2world':
        quality_results = {k:v[0] for k,v in quality_results.items()} 
    else:
        raise ValueError(f'Unknown inference type: {args.inference_type}')

    quality_scores = quality_results.values()
    quality_average_score = sum(quality_scores) / len(quality_scores)

    # The Domain Score is obtained through VQA-based evaluation across seven domains: av, common,
    # human, industry, misc, physics, and robotics.
    # Find the domain results file
    domain_input_file = f'{args.input_dir}/vqa_summary.json'
    domain_results = json.load(open(domain_input_file))
    domain_score = domain_results['overall_accuracy']

    # The final PAI-Bench Overall Score is computed as the average of the Quality and Domain scores.
    overall_score = (quality_average_score + domain_score) / 2

    results = f"""
{"="*13} Final Results {"="*13} 
Domain  score {domain_score:.3f}
Quality score {quality_average_score:.3f}
Overall score {overall_score:.3f}
    """
    print(results)

    with open(f'{args.output_dir}/final_results.txt', 'w') as f:
        f.write(results)

if __name__ == '__main__':
    main()




