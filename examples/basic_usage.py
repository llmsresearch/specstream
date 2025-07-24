"""
SpecStream - Basic Usage Example

This example demonstrates the basic functionality of SpecStream library
without requiring external model downloads.
"""

import sys
import os
import torch

# Add src to path for development
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """Basic usage demonstration."""
    print("SpecStream - Basic Usage")
    print("=" * 25)
    
    try:
        from specstream import (
            SpeculativeEngine, 
            MultiStreamAttention, 
            TreePruningAdapter
        )
        
        print("✅ All SpecStream components imported successfully")
        
        # Test TreePruningAdapter
        print("\n🌳 Testing TreePruningAdapter...")
        tree_adapter = TreePruningAdapter(
            hidden_size=768,
            prune_threshold=0.1,
            max_tree_width=8,
            adaptive_pruning=True,
            pruning_strategy="entropy_based"
        )
        print(f"✅ TreePruningAdapter created with hidden_size=768")
        
        # Test MultiStreamAttention
        print("\n🔀 Testing MultiStreamAttention...")
        attention = MultiStreamAttention(
            hidden_size=768,
            num_attention_heads=12,
            gamma=4
        )
        print(f"MultiStreamAttention created with 4 streams")
        
        # Test SpeculativeStreaming (create a mock config)
        print("\nTesting SpeculativeStreaming components...")
        print("SpeculativeStreaming requires a base model and config")
        print("Skipping full initialization (needs pretrained model)")
        
        # Test basic tensor operations
        print("\nTesting basic tensor operations...")
        batch_size, seq_len, hidden_size = 2, 10, 768
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, seq_len, hidden_size)
        print(f"Created dummy input tensor: {dummy_input.shape}")
        
        # Test attention forward pass
        with torch.no_grad():
            attention_output = attention(dummy_input)
            if isinstance(attention_output, tuple):
                print(f"MultiStreamAttention forward pass: tuple with {len(attention_output)} elements")
                print(f"   First element shape: {attention_output[0].shape}")
            else:
                print(f"MultiStreamAttention forward pass: {attention_output.shape}")
        
        print("\nComponent Statistics:")
        print(f"• TreePruningAdapter parameters: {sum(p.numel() for p in tree_adapter.parameters()):,}")
        print(f"• MultiStreamAttention parameters: {sum(p.numel() for p in attention.parameters()):,}")
        
        print("\nBasic usage example completed successfully!")
        print("\nNext steps:")
        print("• Try the LoRA fine-tuning example")
        print("• Check out the quickstart guide")
        print("• Read the documentation for advanced usage")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure the package is installed: pip install -e .")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
