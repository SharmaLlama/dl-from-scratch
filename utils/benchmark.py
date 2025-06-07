import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import gc
from dataclasses import dataclass
from contextlib import contextmanager
import os
import pandas as pd
from datetime import datetime
import argparse
from pathlib import Path
from papers.CommonTransformerComponents.train_sp import build_model
import yaml
import sentencepiece as spm

@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking - auto-generates __init__ and other methods"""
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None
    max_new_tokens: int = 50
    vocab_size: int = 32000
    num_warmup_runs: int = 3
    num_benchmark_runs: int = 10
    device: Union[str, List[str]] = "cuda"  # Can be single device or list of devices
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 4, 8, 16]
        if self.sequence_lengths is None:
            self.sequence_lengths = [64, 128, 256, 512]
        
        # Handle multi-GPU device specification
        if isinstance(self.device, str):
            if self.device == "cuda" and torch.cuda.device_count() > 1:
                self.device = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            else:
                self.device = [self.device]
        
        # Ensure we have valid devices
        self.available_devices = []
        for dev in self.device:
            if torch.cuda.is_available() and dev.startswith("cuda"):
                self.available_devices.append(dev)
            elif dev == "cpu":
                self.available_devices.append(dev)
        
        if not self.available_devices:
            self.available_devices = ["cpu"]

@dataclass
class BenchmarkResults:
    """Results container - auto-generates __init__ and other methods"""
    model_name: str
    batch_size: int
    seq_length: int
    # Memory metrics (per-GPU)
    total_memory_mb: float
    memory_per_gpu_mb: Dict[str, float]
    memory_per_token_mb: float
    # Latency metrics
    avg_latency_ms: float
    latency_per_token_ms: float
    # Throughput metrics
    tokens_per_second: float
    sentences_per_second: float
    # GPU utilisation (per GPU)
    gpu_utilisation_percent: Dict[str, float]
    # Generation metrics (if applicable)
    generation_latency_ms: Optional[float] = None
    generation_tokens_per_second: Optional[float] = None
    # Multi-GPU specific
    num_gpus_used: int = 1
    model_parallel_type: str = "single"  # "single", "data_parallel", "distributed"
    # Add timestamp and model info
    model_path: Optional[str] = None
    model_timestamp: Optional[str] = None
    benchmark_timestamp: Optional[str] = None

@dataclass
class ModelInfo:
    """Information about discovered models"""
    model_type: str  # sparse, vanilla, rope
    model_config: str  # configuration name
    model_path: str  # full path to model file
    model_name: str  # derived model name
    file_timestamp: float  # file modification time

class MultiGPUMemoryTracker:
    def __init__(self, devices: List[str]):
        self.devices = [d for d in devices if d.startswith("cuda")]
        self.peak_memory = {}
        
    def reset(self):
        """Reset memory tracking across all GPUs"""
        for device in self.devices:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
        self.peak_memory = {}
        
    def get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage across all GPUs in MB"""
        memory = {}
        for device in self.devices:
            if torch.cuda.is_available():
                memory[device] = torch.cuda.memory_allocated(device) / 1024**2
            else:
                memory[device] = 0
        return memory
    
    def get_peak_memory(self) -> Dict[str, float]:
        """Get peak memory usage across all GPUs in MB"""
        memory = {}
        for device in self.devices:
            if torch.cuda.is_available():
                memory[device] = torch.cuda.max_memory_allocated(device) / 1024**2
            else:
                memory[device] = 0
        return memory
    
    def get_total_memory(self) -> float:
        """Get total memory usage across all GPUs"""
        return sum(self.get_peak_memory().values())

@contextmanager
def multi_gpu_memory_context(devices: List[str]):
    """Context manager ensures proper memory tracking setup and cleanup"""
    tracker = MultiGPUMemoryTracker(devices)
    tracker.reset()
    try:
        yield tracker
    finally:
        # Synchronise all CUDA devices
        for device in devices:
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(device)

class MultiGPUTransformerBenchmarker:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.devices = config.available_devices
        self.primary_device = self.devices[0]
        
        print(f"Initialising benchmarker with devices: {self.devices}")
        
    def discover_models(self, base_path: str) -> List[ModelInfo]:
        """
        Discover models in the folder structure: ../Model/model_type/model_config/Model_{x}
        
        Args:
            base_path: Path to the Model directory (e.g., "../Model")
        
        Returns:
            List of ModelInfo objects
        """
        models = []
        model_types = ["sparse", "vanilla", "rope"]
        model_number = {"sparse" : 100, "vanilla" : 100, "rope" : 100}
        base_path = Path(base_path)
        
        for model_type in model_types:
            model_type_path = base_path / model_type
            if not model_type_path.exists():
                print(f"Warning: {model_type_path} does not exist")
                continue
                
            # Find all config directories
            for config_dir in model_type_path.iterdir():
                if config_dir.is_dir():
                    model_file = config_dir / f"Model_{model_number[model_type]}"
                    
                    if model_file.exists():
                        # Extract model configuration from filename/path
                        model_name = f"{model_type}_{config_dir.name}_Model_{model_number[model_type]}"
                        
                        model_info = ModelInfo(
                            model_type=model_type,
                            model_config=config_dir.name,
                            model_path=str(model_file),
                            model_name=model_name,
                            file_timestamp=model_file.stat().st_mtime
                        )
                        models.append(model_info)
                        print(f"Found model: {model_name} at {model_file}")
                    else:
                        print(f"Model file not found: {model_file}")
        
        return models
    
    def load_existing_results(self, csv_path: str) -> pd.DataFrame:
        """Load existing benchmark results from CSV if it exists"""
        if os.path.exists(csv_path):
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                print(f"Warning: Could not load existing results from {csv_path}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    def needs_benchmarking(self, model_info: ModelInfo, existing_results: pd.DataFrame) -> bool:
        """
        Check if a model needs benchmarking based on timestamps
        
        Args:
            model_info: Information about the model
            existing_results: DataFrame with existing benchmark results
        
        Returns:
            True if benchmarking is needed, False otherwise
        """
        if existing_results.empty:
            return True
        
        # Check if this model exists in results
        model_results = existing_results[existing_results['model_name'] == model_info.model_name]
        
        if model_results.empty:
            return True
        
        # Check if model file is newer than last benchmark
        if 'model_timestamp' in model_results.columns:
            last_model_timestamp = model_results['model_timestamp'].iloc[0]
            try:
                last_timestamp = float(last_model_timestamp)
                if model_info.file_timestamp > last_timestamp:
                    print(f"Model {model_info.model_name} has been updated since last benchmark")
                    return True
                else:
                    print(f"Model {model_info.model_name} is up to date, skipping benchmark")
                    return False
            except (ValueError, TypeError):
                # If timestamp parsing fails, re-benchmark
                return True
        
        return True
    
    def load_model_from_path(self, model_info: ModelInfo, sp, device) -> nn.Module:
        """
        Load a model from the given path and configuration
        
        Args:
            model_info: Information about the model to load
            sp: SentencePiece processor
            device: Target device
        
        Returns:
            Loaded model
        """
        # Determine config path based on model type
        config_paths = {
            "sparse": "dl-from-scratch/papers/big_bird_attention/config.yaml",
            "vanilla": "dl-from-scratch/papers/attention_is_all_you_need/config.yaml",
            "rope": "dl-from-scratch/papers/RoPE/config.yaml"
        }
        
        yaml_path = config_paths.get(model_info.model_type)
        if not yaml_path:
            raise ValueError(f"Unknown model type: {model_info.model_type}")
        
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        
        # Load checkpoint
        checkpoint = torch.load(model_info.model_path, map_location=device)
        
        # Parse model configuration from path/filename
        config_parts = model_info.model_config.split("_")
        if len(config_parts) >= 5:
            config['N_HEADS'] = int(config_parts[2])
            config['D_MODEL'] = int(config_parts[3])
            config['FF_HIDDEN'] = int(config_parts[4])
            config['N_ENCODERS'] = int(config_parts[5])
            config['N_DECODERS'] = int(config_parts[6])
        
        model = build_model(sp, device, config, model_info.model_type, 
                          checkpoint.get('model_state_dict'))
        
        return model
    
    def detect_model_parallelism(self, model: nn.Module) -> Tuple[str, List[str]]:
        """Detect what type of parallelism the model uses"""
        model_devices = []
        
        # Check if model is wrapped in DataParallel or DistributedDataParallel
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            parallel_type = "data_parallel" if isinstance(model, nn.DataParallel) else "distributed"
            # Get devices from the wrapped model
            for param in model.parameters():
                if param.device.type == 'cuda':
                    device_str = f"cuda:{param.device.index}"
                    if device_str not in model_devices:
                        model_devices.append(device_str)
                break  # We just need to check one parameter for DataParallel
        else:
            # Check individual parameter locations for model parallelism
            device_counts = {}
            for param in model.parameters():
                device_str = str(param.device)
                device_counts[device_str] = device_counts.get(device_str, 0) + 1
            
            model_devices = list(device_counts.keys())
            parallel_type = "model_parallel" if len(model_devices) > 1 else "single"
        
        return parallel_type, model_devices
    
    def create_random_inputs(self, batch_size: int, seq_length: int, 
                           target_device: str = None) -> Dict[str, torch.Tensor]:
        """Create random input tensors for testing"""
        device = target_device or self.primary_device
        return {
            'src': torch.randint(0, self.config.vocab_size, 
                                     (batch_size, seq_length), 
                                     device=device),
            'tgt': torch.randint(0, self.config.vocab_size, 
                                     (batch_size, seq_length), 
                                     device=device),
            'encoder_mask': torch.ones((batch_size, 1, 1, seq_length), 
                                       device=device),
            'decoder_mask': torch.ones((batch_size, 1, seq_length, seq_length), 
                                       device=device),
        }
        
    def measure_multi_gpu_utilisation(self) -> Dict[str, float]:
        """Measure GPU utilisation across all available GPUs"""
        utilisation = {}
        try:
            import pynvml
            pynvml.nvmlInit()
            
            for device in self.devices:
                if device.startswith("cuda"):
                    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        utilisation[device] = util.gpu
                    except:
                        utilisation[device] = 0.0
                else:
                    utilisation[device] = 0.0
        except ImportError:
            print("pynvml not available, GPU utilisation will be 0")
            for device in self.devices:
                utilisation[device] = 0.0
        
        return utilisation
    
    def move_model_to_devices(self, model: nn.Module) -> nn.Module:
        """Move model to appropriate device(s) if not already configured"""
        parallel_type, model_devices = self.detect_model_parallelism(model)
        
        if parallel_type == "single" and len(self.devices) > 1:
            # If we have multiple GPUs but model is on single device, wrap in DataParallel
            print(f"Wrapping model in DataParallel across devices: {self.devices}")
            cuda_devices = [d for d in self.devices if d.startswith("cuda")]
            if cuda_devices:
                device_ids = [int(d.split(":")[-1]) if ":" in d else 0 for d in cuda_devices]
                model = model.to(cuda_devices[0])
                model = nn.DataParallel(model, device_ids=device_ids)
        
        return model
    
    def benchmark_forward_pass(self, model: nn.Module, batch_size: int, 
                             seq_length: int, model_info: ModelInfo = None) -> BenchmarkResults:
        """Benchmark a single forward pass with multi-GPU support"""
        model.eval()
        
        # Detect model configuration
        parallel_type, model_devices = self.detect_model_parallelism(model)
        print(f"Model parallel type: {parallel_type}, devices: {model_devices}")
        
        input_device = model_devices[0] if model_devices else self.primary_device
        
        # Warmup runs
        for _ in range(self.config.num_warmup_runs):
            with torch.inference_mode():
                inputs = self.create_random_inputs(batch_size, seq_length, input_device)
                _ = model(**inputs)
        
        # Synchronise all devices
        for device in self.devices:
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize(device)
        
        # Benchmark runs
        latencies = []
        gpu_utilisations = []
        
        with multi_gpu_memory_context(self.devices) as mem_tracker:
            for _ in range(self.config.num_benchmark_runs):
                inputs = self.create_random_inputs(batch_size, seq_length, input_device)
                encoder_input = inputs['src']
                tgt_input = inputs['tgt'] 
                encoder_mask = inputs['encoder_mask']
                decoder_mask = inputs['decoder_mask']
                
                start_time = time.perf_counter()
                with torch.inference_mode():
                    outputs = model(encoder_input, tgt_input, encoder_mask=encoder_mask, decoder_mask=decoder_mask)
                
                # Synchronise all devices
                for device in self.devices:
                    if device.startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.synchronize(device)
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)
                gpu_utilisations.append(self.measure_multi_gpu_utilisation())
            
            peak_memory_per_gpu = mem_tracker.get_peak_memory()
            total_memory = mem_tracker.get_total_memory()
        
        # Calculate metrics
        avg_latency = np.mean(latencies)
        total_tokens = batch_size * seq_length
        
        # Average GPU utilisation across runs
        avg_gpu_util = {}
        for device in self.devices:
            if device.startswith("cuda"):
                device_utils = [util.get(device, 0) for util in gpu_utilisations]
                avg_gpu_util[device] = np.mean(device_utils)
        
        return BenchmarkResults(
            model_name=model_info.model_name if model_info else model.__class__.__name__,
            batch_size=batch_size,
            seq_length=seq_length,
            total_memory_mb=total_memory,
            memory_per_gpu_mb=peak_memory_per_gpu,
            memory_per_token_mb=total_memory / total_tokens,
            avg_latency_ms=avg_latency,
            latency_per_token_ms=avg_latency / total_tokens,
            tokens_per_second=total_tokens * 1000 / avg_latency,
            sentences_per_second=batch_size * 1000 / avg_latency,
            gpu_utilisation_percent=avg_gpu_util,
            num_gpus_used=len([d for d in model_devices if d.startswith("cuda")]),
            model_parallel_type=parallel_type,
            model_path=model_info.model_path if model_info else None,
            model_timestamp=str(model_info.file_timestamp) if model_info else None,
            benchmark_timestamp=datetime.now().isoformat()
        )
    
    def benchmark_model(self, model: nn.Module, model_name: str = None, 
                       auto_parallelise: bool = True, model_info: ModelInfo = None) -> List[BenchmarkResults]:
        """Run comprehensive benchmark on a model with multi-GPU support"""
        if model_name is None:
            model_name = model.__class__.__name__
        
        if auto_parallelise:
            model = self.move_model_to_devices(model)
        
        results = []
        
        print(f"Benchmarking {model_name}...")
        parallel_type, model_devices = self.detect_model_parallelism(model)
        
        for batch_size in self.config.batch_sizes:
            for seq_length in self.config.sequence_lengths:                
                try:
                    result = self.benchmark_forward_pass(model, batch_size, seq_length, model_info)
                    result.model_name = model_name
                    results.append(result)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"OOM at batch_size={batch_size}, seq_length={seq_length}")
                        # Clear memory on all GPUs
                        for device in self.devices:
                            if device.startswith("cuda") and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                
                # Clean up between runs
                for device in self.devices:
                    if device.startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                gc.collect()
        
        return results
    
    def export_results_to_csv(self, results: Dict[str, List[BenchmarkResults]], filename: str, append: bool = True):
        """
        Export results to CSV with multi-GPU information
        
        Args:
            results: Dictionary of model results
            filename: Output CSV filename
            append: If True, append to existing file; if False, overwrite
        """
        rows = []
        for model_name, model_results in results.items():
            for result in model_results:
                row = {
                    'model_name': result.model_name,
                    'batch_size': result.batch_size,
                    'seq_length': result.seq_length,
                    'total_memory_mb': result.total_memory_mb,
                    'memory_per_token_mb': result.memory_per_token_mb,
                    'avg_latency_ms': result.avg_latency_ms,
                    'latency_per_token_ms': result.latency_per_token_ms,
                    'tokens_per_second': result.tokens_per_second,
                    'sentences_per_second': result.sentences_per_second,
                    'generation_latency_ms': result.generation_latency_ms,
                    'generation_tokens_per_second': result.generation_tokens_per_second,
                    'num_gpus_used': result.num_gpus_used,
                    'model_parallel_type': result.model_parallel_type,
                    'model_path': result.model_path,
                    'model_timestamp': result.model_timestamp,
                    'benchmark_timestamp': result.benchmark_timestamp
                }
                
                # Add per-GPU memory usage
                for device, memory in result.memory_per_gpu_mb.items():
                    row[f'memory_{device}_mb'] = memory
                
                # Add per-GPU utilisation
                for device, util in result.gpu_utilisation_percent.items():
                    row[f'utilisation_{device}_percent'] = util
                
                rows.append(row)
        
        new_df = pd.DataFrame(rows)
        
        if append and os.path.exists(filename):
            try:
                # Load existing data
                existing_df = pd.read_csv(filename)
                
                # Remove old results for models that were re-benchmarked
                models_updated = set(new_df['model_name'].unique())
                existing_df = existing_df[~existing_df['model_name'].isin(models_updated)]
                
                # Combine old and new results
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df.to_csv(filename, index=False)
                print(f"Results appended to {filename}")
            except Exception as e:
                print(f"Error appending to existing file: {e}")
                new_df.to_csv(filename, index=False)
                print(f"Results saved to new file {filename}")
        else:
            new_df.to_csv(filename, index=False)
            print(f"Results exported to {filename}")
    
    def benchmark_all_models(self, base_model_path: str, csv_output_path: str, sp, device):
        """
        Discover and benchmark all models in the directory structure
        
        Args:
            base_model_path: Path to the Model directory
            csv_output_path: Path to output CSV file
            sp: SentencePiece processor
            device: Device to use for computation
        """
        models = self.discover_models(base_model_path)
        print(f"Discovered {len(models)} models")
        existing_results = self.load_existing_results(csv_output_path)
        all_results = {}
        
        for model_info in models:
            try:
                # Check if benchmarking is needed
                if not self.needs_benchmarking(model_info, existing_results):
                    continue
                
                print(f"\nBenchmarking {model_info.model_name}...")
                model = self.load_model_from_path(model_info, sp, device)
                results = self.benchmark_model(model, model_info.model_name, 
                                             auto_parallelise=True, model_info=model_info)
                
                all_results[model_info.model_name] = results
                
                del model
                for device_name in self.devices:
                    if device_name.startswith("cuda") and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                gc.collect()
                
            except Exception as e:
                print(f"Error benchmarking {model_info.model_name}: {e}")
                continue
        
        if all_results:
            self.export_results_to_csv(all_results, csv_output_path, append=True)
        else:
            print("No models needed benchmarking")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Inferencing Script")
    parser.add_argument("--model_file", type=str, required=True, 
                       help="Path to SentencePiece model file")
    parser.add_argument("--model_base_path", type=str, required=True,
                       help="Base path to model directory (e.g., '../Model')")
    parser.add_argument("--output_csv", type=str, required=True,
                       help="Path to output CSV file")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Load SentencePiece processor
    sp = spm.SentencePieceProcessor(model_file=args.model_file)
    
    # Configure benchmarker
    config = BenchmarkConfig(
        batch_sizes=[64, 128, 256, 512, 1024],
        sequence_lengths=[140, 200, 500, 1000, 2000],
        num_warmup_runs=10,
        num_benchmark_runs=100,
        vocab_size=sp.vocab_size(),
        device=device
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