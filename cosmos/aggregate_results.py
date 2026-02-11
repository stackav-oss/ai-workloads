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


    # find the domain results file
    domain_input_file = glob.glob(f'{args.input_dir}/*_eval_results.json')[0]
    domain_results =json.load(open(domain_input_file))
    #ignore image to video scores for text2world
    if args.inference_type == 'text2world':
        domain_results = {k:v[0] for k,v in domain_results.items() if not k.startswith('i2v_')}
    elif args.inference_type == 'image2world':
        domain_results = {k:v[0] for k,v in domain_results.items()} 
    else:
        raise ValueError(f'Unknown inference type: {args.inference_type}')

    domain_scores = domain_results.values()
    domain_average_score = sum(domain_scores) / len(domain_scores)

    quality_input_file = f'{args.input_dir}/vqa_summary.json'
    quality_results = json.load(open(quality_input_file))
    quality_score = quality_results['overall_accuracy']

    results = f"""
{"="*13} Final Results {"="*13} 
Domain  score {domain_average_score:.4f}
Quality score {quality_score:.4f}
    """
    print(results)

    with open(f'{args.output_dir}/final_results.txt', 'w') as f:
        f.write(results)

if __name__ == '__main__':
    main()




