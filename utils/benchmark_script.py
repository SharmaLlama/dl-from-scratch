from utils.benchmark import MultiGPUTransformerBenchmarker, BenchmarkConfig
import torch
import argparse
from papers.CommonTransformerComponents.train_sp import build_model
import yaml
import sentencepiece as spm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Evaluation Script")
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--llm_model_file", type=str, required=False, default="")
    parser.add_argument("--attention_type", type=str, required=False, default="vanilla")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.llm_model_file != "":
        checkpoint = torch.load(args.llm_model_file, map_location=torch.device(device))
    else:
        checkpoint = {'model_state_dict' : None, 'optimiser_state_dict' : None}

    if args.attention_type == "sparse":
        YAML_PATH = f"dl-from-scratch/papers/big_bird_attention/config.yaml"
    elif args.attention_type == "vanilla":
        YAML_PATH = f"dl-from-scratch/papers/attention_is_all_you_need/config.yaml"
    elif args.attention_type == "rope":
        YAML_PATH = f"dl-from-scratch/papers/RoPE/config.yaml"


    with open(YAML_PATH, "r") as file:
        config = yaml.safe_load(file)

    sp = spm.SentencePieceProcessor(model_file=args.model_file)


    splat = args.llm_model_file.split("/")[6].split("_")
    config['N_HEADS'] = int(splat[2])
    config['D_MODEL'] = int(splat[3])
    config['FF_HIDDEN'] = int(splat[4])
    config['N_ENCODERS'] = int(splat[5])
    config['N_DECODERS'] = int(splat[6])
    model = build_model(sp, device, config, args.attention_type, checkpoint['model_state_dict'])


    config = BenchmarkConfig(
        batch_sizes=[64, 128, 256, 512, 1024],
        sequence_lengths=[140, 200, 500, 1000, 2000],
        num_warmup_runs=10,
        num_benchmark_runs=100,
        vocab_size=sp.vocab_size(),
        device="cuda"  
    )

    benchmarker = MultiGPUTransformerBenchmarker(config)
    print(f"Available devices: {benchmarker.devices}")
    print(f"Primary device: {benchmarker.primary_device}")


    print("Starting comprehensive benchmarks...")
    results = benchmarker.benchmark_model(model, model_name="_".join(splat))
    benchmarker.export_results_to_csv({f"{'_'.join(splat)}" :results})