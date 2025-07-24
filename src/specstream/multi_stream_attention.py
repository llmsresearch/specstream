"""
Multi-Stream Attention (MSA) Implementation

Core innovation of Speculative Streaming: enables a single model to maintain
multiple parallel attention streams (γ=4) for speculative token generation.
Each stream explores different potential continuation paths simultaneously.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math

# Optional transformers import
try:
    from transformers.models.llama.modeling_llama import LlamaAttention
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    LlamaAttention = nn.Module  # Fallback


class MultiStreamAttention(nn.Module):
    """
    Multi-Stream Attention mechanism that extends standard attention to support
    γ parallel streams for speculative token generation.
    
    Key features:
    - Maintains γ=4 parallel attention streams
    - Efficient memory usage through shared base computations
    - Compatible with existing transformer architectures
    - Supports both training and inference modes
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
        gamma: int = 4,  # Number of speculative streams
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.gamma = gamma
        self.head_dim = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.attention_dropout = attention_dropout
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size={hidden_size} and num_heads={num_attention_heads})"
            )
        
        # Base attention projections (shared across streams)
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden_size, bias=False)
        
        # Stream-specific projection layers for speculation
        self.stream_q_adapters = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim, bias=False) 
            for _ in range(gamma)
        ])
        self.stream_v_adapters = nn.ModuleList([
            nn.Linear(self.head_dim, self.head_dim, bias=False) 
            for _ in range(gamma)
        ])
        
        # Rotary Position Embedding
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta
        )
        
        # Stream selection and fusion mechanisms
        self.stream_gate = nn.Linear(hidden_size, gamma)
        self.stream_fusion = nn.Linear(gamma * hidden_size, hidden_size)
        
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """Reshape tensor for multi-head attention."""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        """Apply rotary position embedding to query and key tensors."""
        # Implementation of RoPE - simplified for brevity
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        speculative_mode: bool = False,
        stream_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with multi-stream attention support.
        
        Args:
            hidden_states: Input embeddings [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, 1, seq_len, seq_len]
            position_ids: Position indices [batch_size, seq_len]
            past_key_value: Cached key-value pairs for efficiency
            output_attentions: Whether to return attention weights
            use_cache: Whether to cache key-value pairs
            speculative_mode: Whether to use multi-stream speculation
            stream_weights: Optional weights for stream fusion [batch_size, gamma]
            
        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Base projections (shared across all streams)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        
        # Apply rotary position embedding
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = self._apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        
        if past_key_value is not None:
            # Reuse k, v for efficiency
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        if speculative_mode and self.training:
            # Multi-stream attention for speculative generation
            stream_outputs = []
            stream_attentions = []
            
            for stream_idx in range(self.gamma):
                # Apply stream-specific adaptations
                adapted_q = self.stream_q_adapters[stream_idx](
                    query_states.view(-1, self.head_dim)
                ).view_as(query_states)
                adapted_v = self.stream_v_adapters[stream_idx](
                    value_states.view(-1, self.head_dim)
                ).view_as(value_states)
                
                # Compute attention for this stream
                attn_weights = torch.matmul(adapted_q, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
                
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask
                
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(adapted_q.dtype)
                attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
                
                stream_output = torch.matmul(attn_weights, adapted_v)
                stream_output = stream_output.transpose(1, 2).contiguous()
                stream_output = stream_output.reshape(bsz, q_len, self.hidden_size)
                
                stream_outputs.append(stream_output)
                if output_attentions:
                    stream_attentions.append(attn_weights)
            
            # Fusion of multiple streams
            if stream_weights is None:
                # Compute adaptive stream weights
                stream_weights = F.softmax(self.stream_gate(hidden_states), dim=-1)
            
            # Weighted combination of stream outputs
            combined_streams = torch.stack(stream_outputs, dim=-1)  # [bsz, q_len, hidden_size, gamma]
            stream_weights = stream_weights.unsqueeze(2)  # [bsz, q_len, 1, gamma]
            attn_output = torch.sum(combined_streams * stream_weights, dim=-1)
            
            # Final output projection
            attn_output = self.o_proj(attn_output)
            
            attention_weights = torch.stack(stream_attentions, dim=0) if output_attentions else None
            
        else:
            # Standard single-stream attention
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)
            
            attention_weights = attn_weights if output_attentions else None
        
        return attn_output, attention_weights, past_key_value


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding implementation."""
    
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
    
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
