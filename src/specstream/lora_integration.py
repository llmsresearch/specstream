"""
LoRA Integration for Speculative Streaming

Implements Parameter-Efficient Fine-tuning (PEFT) using LoRA adapters
specifically optimized for the Speculative Streaming architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import math

# Optional imports - will work without them for core functionality
try:
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for parameter-efficient fine-tuning.
    
    Implements the LoRA decomposition: W' = W + BA where B and A are low-rank matrices.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Original frozen layer (will be replaced by reference)
        self.base_layer = nn.Linear(in_features, out_features, bias=bias)
        
        # LoRA decomposition matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA weights
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA parameters following the paper."""
        # Initialize A with random normal, B with zeros
        nn.init.normal_(self.lora_A.weight, std=1/self.rank)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Base computation (frozen)
        base_output = self.base_layer(x)
        
        # LoRA adaptation
        lora_output = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        
        return base_output + lora_output


class SpeculativeLoRAConfig:
    """Configuration for LoRA adaptation in Speculative Streaming."""
    
    def __init__(
        self,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.1,
        target_modules: List[str] = None,
        stream_specific_adaptation: bool = True,
        share_stream_adapters: bool = False,
        adaptation_strategy: str = "full",  # "full", "stream_only", "attention_only"
        task_type: str = "CAUSAL_LM",
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        self.stream_specific_adaptation = stream_specific_adaptation
        self.share_stream_adapters = share_stream_adapters
        self.adaptation_strategy = adaptation_strategy
        self.task_type = task_type


class LoRASpeculativeAdapter(nn.Module):
    """
    LoRA adapter specifically designed for Speculative Streaming architecture.
    
    Provides parameter-efficient fine-tuning while maintaining the multi-stream
    attention capabilities and speculative generation performance.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: SpeculativeLoRAConfig,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Track original layers for potential restoration
        self.original_layers = {}
        
        # Apply LoRA adaptations
        self._apply_lora_adaptations()
        
        # Initialize stream-specific adaptations if enabled
        if config.stream_specific_adaptation:
            self._init_stream_adaptations()
    
    def _apply_lora_adaptations(self):
        """Apply LoRA adaptations to target modules."""
        for name, module in self.base_model.named_modules():
            if self._should_adapt_module(name, module):
                self._replace_with_lora(name, module)
    
    def _should_adapt_module(self, name: str, module: nn.Module) -> bool:
        """Determine if a module should be adapted with LoRA."""
        if not isinstance(module, nn.Linear):
            return False
        
        # Check if module name matches target modules
        for target in self.config.target_modules:
            if target in name:
                return True
        
        return False
    
    def _replace_with_lora(self, name: str, module: nn.Linear):
        """Replace a linear layer with LoRA adaptation."""
        # Store original layer
        self.original_layers[name] = module
        
        # Create LoRA layer
        lora_layer = LoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            rank=self.config.rank,
            alpha=self.config.alpha,
            dropout=self.config.dropout,
            bias=module.bias is not None,
        )
        
        # Copy original weights to base layer
        lora_layer.base_layer.weight.data = module.weight.data.clone()
        if module.bias is not None:
            lora_layer.base_layer.bias.data = module.bias.data.clone()
        
        # Freeze base layer
        lora_layer.base_layer.weight.requires_grad = False
        if lora_layer.base_layer.bias is not None:
            lora_layer.base_layer.bias.requires_grad = False
        
        # Replace in model
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        
        parent_module = self.base_model
        for part in parent_name.split('.'):
            if part:
                parent_module = getattr(parent_module, part)
        
        setattr(parent_module, child_name, lora_layer)
    
    def _init_stream_adaptations(self):
        """Initialize stream-specific LoRA adaptations for MSA."""
        for name, module in self.base_model.named_modules():
            if hasattr(module, 'stream_q_adapters') or hasattr(module, 'stream_v_adapters'):
                self._adapt_stream_modules(module)
    
    def _adapt_stream_modules(self, msa_module):
        """Apply LoRA to stream-specific modules in Multi-Stream Attention."""
        if hasattr(msa_module, 'stream_q_adapters'):
            for i, adapter in enumerate(msa_module.stream_q_adapters):
                lora_adapter = LoRALinear(
                    in_features=adapter.in_features,
                    out_features=adapter.out_features,
                    rank=min(self.config.rank // 2, adapter.in_features // 4),  # Smaller rank for adapters
                    alpha=self.config.alpha / 2,
                    dropout=self.config.dropout,
                )
                
                # Copy weights and replace
                lora_adapter.base_layer.weight.data = adapter.weight.data.clone()
                lora_adapter.base_layer.weight.requires_grad = False
                
                msa_module.stream_q_adapters[i] = lora_adapter
        
        if hasattr(msa_module, 'stream_v_adapters'):
            for i, adapter in enumerate(msa_module.stream_v_adapters):
                lora_adapter = LoRALinear(
                    in_features=adapter.in_features,
                    out_features=adapter.out_features,
                    rank=min(self.config.rank // 2, adapter.in_features // 4),
                    alpha=self.config.alpha / 2,
                    dropout=self.config.dropout,
                )
                
                # Copy weights and replace
                lora_adapter.base_layer.weight.data = adapter.weight.data.clone()
                lora_adapter.base_layer.weight.requires_grad = False
                
                msa_module.stream_v_adapters[i] = lora_adapter
    
    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all trainable LoRA parameters."""
        trainable_params = {}
        
        for name, param in self.named_parameters():
            if param.requires_grad and ('lora_A' in name or 'lora_B' in name):
                trainable_params[name] = param
        
        return trainable_params
    
    def get_parameter_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics about parameter usage."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        # Count LoRA-specific parameters
        lora_params = sum(
            p.numel() for name, p in self.named_parameters() 
            if 'lora_' in name and p.requires_grad
        )
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': frozen_params,
            'lora_parameters': lora_params,
            'trainable_ratio': trainable_params / total_params,
            'lora_ratio': lora_params / total_params,
            'parameter_efficiency': lora_params / trainable_params if trainable_params > 0 else 0,
        }
    
    def merge_and_unload(self) -> nn.Module:
        """
        Merge LoRA weights into base model and return clean model.
        Useful for deployment where LoRA overhead is not needed.
        """
        # Create a copy of the base model
        merged_model = type(self.base_model)(self.base_model.config)
        merged_model.load_state_dict(self.base_model.state_dict())
        
        # Merge LoRA weights
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                # Compute merged weight
                lora_weight = (module.lora_B.weight @ module.lora_A.weight) * module.scaling
                merged_weight = module.base_layer.weight.data + lora_weight
                
                # Find corresponding module in merged model
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                parent_module = merged_model
                for part in parent_name.split('.'):
                    if part:
                        parent_module = getattr(parent_module, part)
                
                target_module = getattr(parent_module, child_name)
                target_module.weight.data = merged_weight
        
        return merged_model
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights to disk."""
        lora_state_dict = {}
        
        for name, param in self.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_state_dict[name] = param.data
        
        torch.save({
            'lora_weights': lora_state_dict,
            'config': self.config.__dict__,
        }, path)
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        lora_weights = checkpoint['lora_weights']
        
        # Load LoRA weights
        for name, param in self.named_parameters():
            if name in lora_weights:
                param.data = lora_weights[name].to(self.device)
    
    def enable_lora_training(self):
        """Enable LoRA parameters for training."""
        for name, param in self.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def disable_lora_training(self):
        """Disable LoRA parameters (freeze all)."""
        for param in self.parameters():
            param.requires_grad = False
    
    def get_lora_scaling_factors(self) -> Dict[str, float]:
        """Get LoRA scaling factors for each adapted module."""
        scaling_factors = {}
        
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                scaling_factors[name] = module.scaling
        
        return scaling_factors
    
    def adjust_lora_scaling(self, scaling_factor: float):
        """Adjust LoRA scaling factors globally."""
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                module.scaling *= scaling_factor
    
    def get_adaptation_efficiency(self) -> Dict[str, float]:
        """Calculate adaptation efficiency metrics."""
        stats = self.get_parameter_stats()
        
        # Theoretical speedup from parameter reduction
        param_reduction = 1 - stats['trainable_ratio']
        memory_efficiency = param_reduction
        
        # Compute theoretical FLOPS reduction (simplified)
        # LoRA adds rank * (input + output) operations vs input * output
        total_flops_saved = 0
        total_original_flops = 0
        
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                original_flops = module.in_features * module.out_features
                lora_flops = module.rank * (module.in_features + module.out_features)
                
                total_original_flops += original_flops
                total_flops_saved += original_flops - lora_flops
        
        computational_efficiency = total_flops_saved / max(total_original_flops, 1)
        
        return {
            'parameter_efficiency': param_reduction,
            'memory_efficiency': memory_efficiency,
            'computational_efficiency': computational_efficiency,
            'adaptation_overhead': stats['lora_ratio'],
        }
