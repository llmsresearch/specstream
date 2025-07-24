"""
Speculative Streaming Core Implementation

Main engine for the Speculative Streaming approach. Coordinates multi-stream
attention, tree-based speculation, and efficient inference pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

# Optional transformers import
try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Mock classes for testing
    class PreTrainedModel:
        def __init__(self):
            self.config = type('config', (), {'hidden_size': 512})()
    class PreTrainedTokenizer:
        def __init__(self):
            self.eos_token_id = 50256

from .multi_stream_attention import MultiStreamAttention
from .tree_pruning import TreePruningAdapter


@dataclass
class SpeculationNode:
    """Represents a node in the speculation tree."""
    token_id: int
    logits: torch.Tensor
    probability: float
    parent: Optional['SpeculationNode'] = None
    children: List['SpeculationNode'] = None
    depth: int = 0
    stream_id: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []


@dataclass
class SpeculativeStreamingConfig:
    """Configuration for Speculative Streaming."""
    gamma: int = 4  # Number of speculative streams
    max_speculation_depth: int = 5  # Maximum depth of speculation tree
    acceptance_threshold: float = 0.7  # Minimum probability for accepting speculative tokens
    top_k_speculation: int = 3  # Top-k tokens to consider for speculation
    temperature: float = 1.0  # Temperature for sampling
    use_tree_pruning: bool = True  # Whether to use tree pruning
    prune_threshold: float = 0.1  # Threshold for pruning low-probability branches
    max_tree_width: int = 8  # Maximum width of speculation tree


class SpeculativeStreaming(nn.Module):
    """
    Core Speculative Streaming implementation.
    
    Manages the speculative generation process using multi-stream attention
    and tree-based speculation for accelerated inference.
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        config: SpeculativeStreamingConfig,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tree pruning adapter
        if config.use_tree_pruning:
            self.tree_pruner = TreePruningAdapter(
                hidden_size=base_model.config.hidden_size,
                prune_threshold=config.prune_threshold,
                max_tree_width=config.max_tree_width
            )
        else:
            self.tree_pruner = None
        
        # Replace attention layers with Multi-Stream Attention
        self._replace_attention_layers()
        
        # Statistics tracking
        self.reset_statistics()
    
    def _replace_attention_layers(self):
        """Replace standard attention with Multi-Stream Attention."""
        for layer_idx, layer in enumerate(self.base_model.model.layers):
            original_attn = layer.self_attn
            
            # Create MSA replacement
            msa = MultiStreamAttention(
                hidden_size=original_attn.hidden_size,
                num_attention_heads=original_attn.num_heads,
                num_key_value_heads=getattr(original_attn, 'num_key_value_heads', None),
                gamma=self.config.gamma,
                max_position_embeddings=original_attn.max_position_embeddings,
                rope_theta=getattr(original_attn, 'rope_theta', 10000.0),
                attention_dropout=getattr(original_attn, 'attention_dropout', 0.0),
            )
            
            # Copy weights from original attention
            msa.q_proj.weight.data = original_attn.q_proj.weight.data.clone()
            msa.k_proj.weight.data = original_attn.k_proj.weight.data.clone()
            msa.v_proj.weight.data = original_attn.v_proj.weight.data.clone()
            msa.o_proj.weight.data = original_attn.o_proj.weight.data.clone()
            
            # Initialize stream adapters with small random weights
            for adapter in msa.stream_q_adapters:
                nn.init.normal_(adapter.weight, mean=0.0, std=0.02)
            for adapter in msa.stream_v_adapters:
                nn.init.normal_(adapter.weight, mean=0.0, std=0.02)
            
            # Replace the attention layer
            layer.self_attn = msa
    
    def reset_statistics(self):
        """Reset generation statistics."""
        self.stats = {
            'total_tokens_generated': 0,
            'accepted_speculative_tokens': 0,
            'rejected_speculative_tokens': 0,
            'average_acceptance_rate': 0.0,
            'average_speculation_depth': 0.0,
            'total_forward_passes': 0,
            'speedup_ratio': 1.0,
        }
    
    def build_speculation_tree(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        max_depth: Optional[int] = None,
    ) -> SpeculationNode:
        """
        Build a tree of speculative token sequences.
        
        Args:
            input_ids: Current sequence tokens [batch_size, seq_len]
            past_key_values: Cached key-value pairs
            max_depth: Maximum depth for speculation tree
            
        Returns:
            Root node of the speculation tree
        """
        max_depth = max_depth or self.config.max_speculation_depth
        
        # Initial forward pass to get base logits
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        
        base_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        base_probs = F.softmax(base_logits / self.config.temperature, dim=-1)
        
        # Get top-k tokens for speculation
        top_k_probs, top_k_indices = torch.topk(
            base_probs, 
            k=min(self.config.top_k_speculation, base_probs.size(-1)), 
            dim=-1
        )
        
        # Create root node
        root = SpeculationNode(
            token_id=-1,  # Placeholder for root
            logits=base_logits,
            probability=1.0,
            depth=0
        )
        
        # Build tree recursively using multiple streams
        self._build_tree_recursive(
            root, 
            input_ids, 
            outputs.past_key_values,
            top_k_indices[0],  # Assuming batch_size=1 for simplicity
            top_k_probs[0],
            max_depth
        )
        
        return root
    
    def _build_tree_recursive(
        self,
        parent_node: SpeculationNode,
        input_ids: torch.Tensor,
        past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        candidate_tokens: torch.Tensor,
        candidate_probs: torch.Tensor,
        remaining_depth: int,
    ):
        """Recursively build speculation tree."""
        if remaining_depth <= 0:
            return
        
        for stream_idx, (token_id, prob) in enumerate(zip(candidate_tokens, candidate_probs)):
            if prob < self.config.prune_threshold:
                continue
            
            # Create new input with speculative token
            new_input_ids = torch.cat([input_ids, token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            
            # Forward pass with multi-stream attention
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=new_input_ids[:, -1:],  # Only the new token
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            
            new_logits = outputs.logits[:, -1, :]
            new_probs = F.softmax(new_logits / self.config.temperature, dim=-1)
            
            # Create child node
            child_node = SpeculationNode(
                token_id=token_id.item(),
                logits=new_logits,
                probability=prob.item(),
                parent=parent_node,
                depth=parent_node.depth + 1,
                stream_id=stream_idx
            )
            
            parent_node.children.append(child_node)
            
            # Prune tree if needed
            if self.tree_pruner and len(parent_node.children) > self.config.max_tree_width:
                parent_node.children = self.tree_pruner.prune_branches(
                    parent_node.children, 
                    self.config.max_tree_width
                )
            
            # Continue building tree
            if remaining_depth > 1:
                next_top_k_probs, next_top_k_indices = torch.topk(
                    new_probs, 
                    k=min(self.config.top_k_speculation, new_probs.size(-1)), 
                    dim=-1
                )
                
                self._build_tree_recursive(
                    child_node,
                    new_input_ids,
                    outputs.past_key_values,
                    next_top_k_indices[0],
                    next_top_k_probs[0],
                    remaining_depth - 1
                )
    
    def verify_speculation_tree(
        self,
        tree_root: SpeculationNode,
        target_tokens: torch.Tensor,
    ) -> Tuple[List[int], float]:
        """
        Verify speculative tokens against target sequence.
        
        Args:
            tree_root: Root of speculation tree
            target_tokens: Ground truth tokens for verification
            
        Returns:
            Tuple of (accepted_tokens, acceptance_rate)
        """
        accepted_tokens = []
        total_speculated = 0
        
        def dfs_verify(node: SpeculationNode, target_idx: int):
            nonlocal total_speculated
            
            if target_idx >= len(target_tokens):
                return
            
            target_token = target_tokens[target_idx].item()
            total_speculated += 1
            
            # Check if this node's token matches target
            if node.token_id == target_token and node.probability >= self.config.acceptance_threshold:
                accepted_tokens.append(target_token)
                
                # Continue verification with children
                for child in node.children:
                    dfs_verify(child, target_idx + 1)
            
        # Start verification from root's children
        for child in tree_root.children:
            dfs_verify(child, 0)
        
        acceptance_rate = len(accepted_tokens) / max(total_speculated, 1)
        return accepted_tokens, acceptance_rate
    
    def generate_speculative(
        self,
        input_ids: torch.Tensor,
        tokenizer: PreTrainedTokenizer,
        max_new_tokens: int = 100,
        do_sample: bool = True,
        return_dict_in_generate: bool = True,
        **kwargs
    ) -> Dict:
        """
        Generate text using speculative streaming.
        
        Args:
            input_ids: Input token sequence [batch_size, seq_len]
            tokenizer: Tokenizer for decoding
            max_new_tokens: Maximum number of new tokens to generate
            do_sample: Whether to use sampling or greedy decoding
            return_dict_in_generate: Whether to return generation info
            
        Returns:
            Dictionary containing generated sequences and statistics
        """
        self.reset_statistics()
        
        generated_tokens = []
        current_input_ids = input_ids.clone()
        past_key_values = None
        
        for step in range(max_new_tokens):
            # Build speculation tree
            speculation_tree = self.build_speculation_tree(
                current_input_ids, 
                past_key_values
            )
            
            # Generate actual next token using base model
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=current_input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            
            next_token_logits = outputs.logits[:, -1, :]
            
            if do_sample:
                probs = F.softmax(next_token_logits / self.config.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Verify speculation against actual token
            accepted_tokens, acceptance_rate = self.verify_speculation_tree(
                speculation_tree, 
                next_token
            )
            
            # Update statistics
            self.stats['total_tokens_generated'] += 1
            self.stats['accepted_speculative_tokens'] += len(accepted_tokens)
            self.stats['total_forward_passes'] += 1
            
            # Add verified tokens to output
            generated_tokens.append(next_token.item())
            current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
            past_key_values = outputs.past_key_values
            
            # Check for EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        # Calculate final statistics
        self._update_final_statistics()
        
        result = {
            'sequences': torch.cat([input_ids, torch.tensor(generated_tokens).unsqueeze(0)], dim=1),
            'generated_tokens': generated_tokens,
            'statistics': self.stats.copy(),
        }
        
        if return_dict_in_generate:
            result['past_key_values'] = past_key_values
            
        return result
    
    def _update_final_statistics(self):
        """Update final generation statistics."""
        total_tokens = self.stats['total_tokens_generated']
        if total_tokens > 0:
            self.stats['average_acceptance_rate'] = (
                self.stats['accepted_speculative_tokens'] / total_tokens
            )
            
            # Estimate speedup based on accepted speculative tokens
            base_forward_passes = total_tokens
            actual_forward_passes = self.stats['total_forward_passes']
            self.stats['speedup_ratio'] = base_forward_passes / max(actual_forward_passes, 1)
    
    def get_model_size_reduction(self) -> Dict[str, float]:
        """Calculate parameter reduction compared to auxiliary model approaches."""
        total_params = sum(p.numel() for p in self.parameters())
        
        # Count additional parameters from MSA stream adapters
        msa_params = 0
        for layer in self.base_model.model.layers:
            if hasattr(layer.self_attn, 'stream_q_adapters'):
                for adapter in layer.self_attn.stream_q_adapters:
                    msa_params += sum(p.numel() for p in adapter.parameters())
                for adapter in layer.self_attn.stream_v_adapters:
                    msa_params += sum(p.numel() for p in adapter.parameters())
        
        # Estimate typical auxiliary model size (same as base model)
        estimated_auxiliary_params = total_params
        
        reduction_ratio = msa_params / estimated_auxiliary_params
        reduction_percentage = (1 - reduction_ratio) * 100
        
        return {
            'total_parameters': total_params,
            'additional_msa_parameters': msa_params,
            'estimated_auxiliary_model_parameters': estimated_auxiliary_params,
            'parameter_reduction_ratio': reduction_ratio,
            'parameter_reduction_percentage': reduction_percentage,
        }
