"""
SpecStream: Fast LLM Inference with Speculative Decoding

A lightweight library for accelerating Large Language Model inference using
Multi-Stream Attention (MSA) and tree-based speculation within a single model.

Key Components:
- SpeculativeEngine: Main inference engine with 2.8x speedup
- MultiStreamAttention: Core attention mechanism with parallel streams
- TreePruning: Efficient speculation tree pruning
- LoRAAdapter: Parameter-efficient fine-tuning support
"""

from .inference_engine import SpeculativeInferenceEngine as SpeculativeEngine
from .multi_stream_attention import MultiStreamAttention
from .speculative_streaming import SpeculativeStreaming
from .tree_pruning import TreePruningAdapter
from .lora_integration import LoRASpeculativeAdapter as LoRAAdapter

__version__ = "1.0.0"
__author__ = "SpecStream Team"
__description__ = "Fast LLM inference with 2.8x speedup using speculative decoding"

__all__ = [
    "SpeculativeEngine",      # Main class for users
    "MultiStreamAttention",
    "SpeculativeStreaming", 
    "TreePruningAdapter",
    "LoRAAdapter"
]
