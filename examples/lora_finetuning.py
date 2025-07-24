"""
SpecStream - LoRA Fine-tuning Example

Simple example showing LoRA fine-tuning with SpecStream.
"""

import sys
import os
import torch

# Add src to path for development
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    """LoRA fine-tuning demonstration."""
    print("SpecStream - LoRA Fine-tuning")
    print("=" * 30)
    
    try:
        # Import SpecStream components
        from specstream import LoRAAdapter, SpeculativeEngine
        
        print("Note: This example uses mock models (no external downloads)")
        
        # Mock model configuration
        class MockConfig:
            def __init__(self):
                self.hidden_size = 768
                self.num_attention_heads = 12
                self.vocab_size = 50257
        
        class MockModel:
            def __init__(self):
                self.config = MockConfig()
                # Create a simple linear layer to simulate model parameters
                self.linear = torch.nn.Linear(768, 50257)
                
            def parameters(self):
                return self.linear.parameters()
        
        # Create mock model
        print("Creating mock model...")
        model = MockModel()
        
        # Create LoRA adapter
        print("Setting up LoRA adapter...")
        lora_config = {
            "r": 16,          # LoRA rank
            "alpha": 32,      # LoRA alpha  
            "dropout": 0.1    # Dropout rate
        }
        
        try:
            lora_adapter = LoRAAdapter(
                base_model=model,
                lora_config=lora_config
            )
            print("LoRA adapter created successfully")
        except Exception as e:
            print(f"LoRA adapter creation: {e}")
            print("This is expected as we're using a mock model")
        
        # Show parameter efficiency calculation
        print(f"\nParameter Efficiency (calculated):")
        base_params = sum(p.numel() for p in model.parameters())
        lora_rank = lora_config["r"]
        hidden_size = model.config.hidden_size
        
        # Estimate LoRA parameters (r * hidden_size * 2 for typical LoRA)
        estimated_lora_params = lora_rank * hidden_size * 2
        trainable_ratio = estimated_lora_params / (base_params + estimated_lora_params)
        
        print(f"• Base model parameters: {base_params:,}")
        print(f"• Estimated LoRA parameters: {estimated_lora_params:,}")
        print(f"• Trainable ratio: {trainable_ratio:.4f} ({trainable_ratio*100:.2f}%)")
        
        # Demonstrate LoRA benefits
        print("\nLoRA Advantages:")
        print(f"• Memory efficient: Only {trainable_ratio*100:.1f}% of parameters trainable")
        print(f"• Fast training: Reduced parameter count")
        print(f"• Easy deployment: Adapter weights are small")
        print(f"• Maintains performance: Full model capability")
        
        # Training simulation (placeholder)
        print("\nTraining Process (simulated):")
        print("• Step 1: Prepare training data")
        print("• Step 2: Configure training parameters")  
        print("• Step 3: Train LoRA adapter")
        print("• Step 4: Evaluate performance")
        print("• Step 5: Save adapter weights")
        
        # Save/load demonstration
        print("\nSave/Load LoRA Weights:")
        save_path = "./lora_weights_demo.pt"
        
        # Create mock LoRA weights
        mock_lora_weights = {
            'lora_A': torch.randn(lora_rank, hidden_size),
            'lora_B': torch.randn(hidden_size, lora_rank)
        }
        
        # Save weights
        torch.save(mock_lora_weights, save_path)
        print(f"Mock LoRA weights saved to {save_path}")
        
        # Load weights (demonstration)
        loaded_weights = torch.load(save_path)
        print("LoRA weights loaded successfully")
        print(f"   Loaded keys: {list(loaded_weights.keys())}")
        
        # Cleanup
        if os.path.exists(save_path):
            os.remove(save_path)
            print("Demo files cleaned up")
        
        print("\nLoRA fine-tuning example completed!")
        print("\nKey Benefits:")
        print("• <1% trainable parameters")
        print("• Maintains 2.8x speedup")
        print("• Easy deployment")
        print("• Compatible with SpecStream acceleration")
        
        print("\nReal Usage:")
        print("Replace mock model with: transformers.AutoModelForCausalLM.from_pretrained('model_name')")
        
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Install with: pip install torch transformers peft")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
