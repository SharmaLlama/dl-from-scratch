from utils.benchmark import MultiGPUTransformerBenchmarker, BenchmarkConfig
import torch
import argparse
from papers.CommonTransformerComponents.train_sp import build_model
import yaml
import sentencepiece as spm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Evaluation Script")
    parser.add_argument("--model_file", type=str, required=True, 
                       help="Path to SentencePiece model file")
    parser.add_argument("--model_base_path", type=str, required=True,
                       help="Base path to model directory (e.g., '../Model')")
    parser.add_argument("--output_csv", type=str, required=True,
                       help="Path to output CSV file")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    sp = spm.SentencePieceProcessor(model_file=args.model_file)
    
    # Configure benchmarker
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
    print("Starting comprehensive benchmarks for all discovered models...")
    benchmarker.benchmark_all_models(
            base_model_path=args.model_base_path,
            csv_output_path=args.output_csv,
            sp=sp,
            device=device
        )