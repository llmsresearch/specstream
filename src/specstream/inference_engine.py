"""
Inference Engine for Speculative Streaming

High-level interface for running Speculative Streaming inference with
optimizations for production deployment and real-time applications.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any, Callable
import time
import logging
from dataclasses import dataclass, asdict
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Optional imports
try:
    from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Mock classes for testing
    class PreTrainedModel:
        pass
    class PreTrainedTokenizer:
        pass

from .speculative_streaming import SpeculativeStreaming, SpeculativeStreamingConfig
from .lora_integration import LoRASpeculativeAdapter, SpeculativeLoRAConfig


@dataclass 
class InferenceConfig:
    """Configuration for inference engine."""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = True
    batch_size: int = 1
    use_cache: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    return_full_text: bool = False
    stream_output: bool = False


@dataclass
class InferenceMetrics:
    """Metrics collected during inference."""
    total_tokens: int = 0
    generation_time: float = 0.0
    tokens_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    speculation_accuracy: float = 0.0
    speedup_factor: float = 1.0
    energy_efficiency: float = 0.0


class SpeculativeInferenceEngine:
    """
    Production-ready inference engine for Speculative Streaming.
    
    Provides high-level interface for text generation with built-in optimizations,
    batching support, and comprehensive monitoring capabilities.
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        speculative_config: Optional[SpeculativeStreamingConfig] = None,
        lora_config: Optional[SpeculativeLoRAConfig] = None,
        device: Optional[torch.device] = None,
        enable_logging: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize Speculative Streaming
        self.speculative_config = speculative_config or SpeculativeStreamingConfig()
        self.speculative_engine = SpeculativeStreaming(
            base_model=model,
            config=self.speculative_config,
            device=self.device
        )
        
        # Initialize LoRA if specified
        if lora_config is not None:
            self.lora_adapter = LoRASpeculativeAdapter(
                base_model=self.speculative_engine,
                config=lora_config,
                device=self.device
            )
            self.model = self.lora_adapter
        else:
            self.lora_adapter = None
            self.model = self.speculative_engine
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup logging
        if enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
        
        # Performance monitoring
        self.metrics_history = []
        self.cache = {}
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def generate(
        self,
        prompt: Union[str, List[str]],
        config: Optional[InferenceConfig] = None,
        return_metrics: bool = False,
        **kwargs
    ) -> Union[List[str], Dict[str, Any]]:
        """
        Generate text using Speculative Streaming.
        
        Args:
            prompt: Input text prompt(s)
            config: Inference configuration
            return_metrics: Whether to return performance metrics
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text(s) or dict with text and metrics
        """
        start_time = time.time()
        
        # Setup configuration
        config = config or InferenceConfig()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Handle single prompt vs batch
        if isinstance(prompt, str):
            prompts = [prompt]
            single_prompt = True
        else:
            prompts = prompt
            single_prompt = False
        
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        # Generate with speculative streaming
        with torch.no_grad():
            outputs = self.speculative_engine.generate_speculative(
                input_ids=inputs.input_ids,
                tokenizer=self.tokenizer,
                max_new_tokens=config.max_new_tokens,
                do_sample=config.do_sample,
                return_dict_in_generate=True,
            )
        
        # Decode outputs
        generated_sequences = outputs['sequences']
        if not config.return_full_text:
            # Remove input tokens from output
            input_length = inputs.input_ids.shape[1]
            generated_sequences = generated_sequences[:, input_length:]
        
        generated_texts = self.tokenizer.batch_decode(
            generated_sequences,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Calculate metrics
        generation_time = time.time() - start_time
        total_tokens = generated_sequences.numel()
        
        metrics = InferenceMetrics(
            total_tokens=total_tokens,
            generation_time=generation_time,
            tokens_per_second=total_tokens / generation_time if generation_time > 0 else 0,
            memory_usage_mb=torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            speculation_accuracy=outputs['statistics'].get('average_acceptance_rate', 0),
            speedup_factor=outputs['statistics'].get('speedup_ratio', 1.0),
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        
        if self.logger:
            self.logger.info(f"Generated {total_tokens} tokens in {generation_time:.2f}s "
                           f"({metrics.tokens_per_second:.1f} tok/s, "
                           f"speedup: {metrics.speedup_factor:.2f}x)")
        
        # Return results
        result_texts = generated_texts[0] if single_prompt else generated_texts
        
        if return_metrics:
            return {
                'text': result_texts,
                'metrics': asdict(metrics),
                'statistics': outputs['statistics'],
            }
        else:
            return result_texts
    
    async def generate_async(
        self,
        prompt: Union[str, List[str]],
        config: Optional[InferenceConfig] = None,
        **kwargs
    ) -> Union[List[str], Dict[str, Any]]:
        """Asynchronous text generation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.generate,
            prompt,
            config,
            **kwargs
        )
    
    def generate_stream(
        self,
        prompt: str,
        config: Optional[InferenceConfig] = None,
        **kwargs
    ):
        """
        Stream text generation token by token.
        
        Yields:
            Dict containing partial text and metadata for each token
        """
        config = config or InferenceConfig()
        config.stream_output = True
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(self.device)
        
        current_input_ids = inputs.input_ids
        generated_text = ""
        
        for step in range(config.max_new_tokens):
            # Single token generation
            with torch.no_grad():
                outputs = self.speculative_engine.generate_speculative(
                    input_ids=current_input_ids,
                    tokenizer=self.tokenizer,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                )
            
            # Get new token
            new_token_id = outputs['sequences'][0, -1]
            new_token_text = self.tokenizer.decode([new_token_id], skip_special_tokens=True)
            generated_text += new_token_text
            
            # Yield streaming result
            yield {
                'token': new_token_text,
                'text': generated_text,
                'token_id': new_token_id.item(),
                'step': step,
                'finished': new_token_id == self.tokenizer.eos_token_id,
                'speculation_stats': outputs['statistics'],
            }
            
            # Update input for next iteration
            current_input_ids = outputs['sequences']
            
            # Check for early stopping
            if new_token_id == self.tokenizer.eos_token_id:
                break
    
    def benchmark(
        self,
        test_prompts: List[str],
        config: Optional[InferenceConfig] = None,
        num_runs: int = 5,
    ) -> Dict[str, Any]:
        """
        Comprehensive benchmark of the inference engine.
        
        Args:
            test_prompts: List of prompts for benchmarking
            config: Inference configuration
            num_runs: Number of runs for averaging
            
        Returns:
            Detailed benchmark results
        """
        config = config or InferenceConfig()
        
        self.logger.info(f"Starting benchmark with {len(test_prompts)} prompts, {num_runs} runs each")
        
        all_metrics = []
        
        for run in range(num_runs):
            run_metrics = []
            
            for prompt in test_prompts:
                result = self.generate(prompt, config, return_metrics=True)
                run_metrics.append(result['metrics'])
            
            all_metrics.extend(run_metrics)
        
        # Aggregate results
        total_tokens = sum(m['total_tokens'] for m in all_metrics)
        total_time = sum(m['generation_time'] for m in all_metrics)
        avg_tokens_per_second = sum(m['tokens_per_second'] for m in all_metrics) / len(all_metrics)
        avg_speedup = sum(m['speedup_factor'] for m in all_metrics) / len(all_metrics)
        avg_accuracy = sum(m['speculation_accuracy'] for m in all_metrics) / len(all_metrics)
        
        benchmark_results = {
            'total_prompts': len(test_prompts) * num_runs,
            'total_tokens': total_tokens,
            'total_time': total_time,
            'average_tokens_per_second': avg_tokens_per_second,
            'average_speedup_factor': avg_speedup,
            'average_speculation_accuracy': avg_accuracy,
            'throughput_tokens_per_second': total_tokens / total_time,
            'individual_metrics': all_metrics,
        }
        
        if self.logger:
            self.logger.info(f"Benchmark completed: {avg_tokens_per_second:.1f} tok/s average, "
                           f"{avg_speedup:.2f}x speedup, {avg_accuracy:.1%} speculation accuracy")
        
        return benchmark_results
    
    def optimize_for_deployment(self):
        """Apply optimizations for production deployment."""
        if self.logger:
            self.logger.info("Applying deployment optimizations...")
        
        # Enable torch optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Compile model if available (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
                if self.logger:
                    self.logger.info("Model compiled with torch.compile")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Could not compile model: {e}")
        
        # Set to eval mode
        self.model.eval()
        
        # Merge LoRA weights if possible for deployment
        if self.lora_adapter is not None:
            try:
                merged_model = self.lora_adapter.merge_and_unload()
                self.model = merged_model
                if self.logger:
                    self.logger.info("LoRA weights merged for deployment")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Could not merge LoRA weights: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        info = {
            'model_type': type(self.model).__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'speculative_config': asdict(self.speculative_config),
            'memory_usage_mb': torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
        }
        
        if self.lora_adapter is not None:
            info['lora_stats'] = self.lora_adapter.get_parameter_stats()
            info['adaptation_efficiency'] = self.lora_adapter.get_adaptation_efficiency()
        
        return info
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'speculative_config': asdict(self.speculative_config),
        }
        
        if self.lora_adapter is not None:
            checkpoint['lora_config'] = self.lora_adapter.config.__dict__
        
        torch.save(checkpoint, path)
        
        if self.logger:
            self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.logger:
            self.logger.info(f"Checkpoint loaded from {path}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.metrics_history:
            return {'message': 'No metrics available'}
        
        metrics = self.metrics_history
        
        return {
            'total_generations': len(metrics),
            'average_tokens_per_second': sum(m.tokens_per_second for m in metrics) / len(metrics),
            'average_speedup_factor': sum(m.speedup_factor for m in metrics) / len(metrics),
            'average_speculation_accuracy': sum(m.speculation_accuracy for m in metrics) / len(metrics),
            'total_tokens_generated': sum(m.total_tokens for m in metrics),
            'total_generation_time': sum(m.generation_time for m in metrics),
            'peak_memory_usage_mb': max(m.memory_usage_mb for m in metrics),
        }
    
    def clear_cache(self):
        """Clear internal caches."""
        self.cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
