"""
Training Pipeline for Speculative Streaming

Implements training procedures for Multi-Stream Attention and LoRA fine-tuning
optimized for the Speculative Streaming architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import json
import os
from dataclasses import dataclass, field
import numpy as np

# Optional imports
try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        TrainingArguments, 
        Trainer,
        default_data_collator
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .speculative_streaming import SpeculativeStreaming, SpeculativeStreamingConfig
from .lora_integration import LoRASpeculativeAdapter, SpeculativeLoRAConfig

# Import inference_engine conditionally to avoid circular import issues
if TRANSFORMERS_AVAILABLE:
    try:
        from .inference_engine import SpeculativeInferenceEngine
    except ImportError:
        SpeculativeInferenceEngine = None
else:
    SpeculativeInferenceEngine = None


@dataclass
class SpeculativeTrainingConfig:
    """Configuration for training Speculative Streaming models."""
    
    # Model configuration
    model_name_or_path: str = "microsoft/DialoGPT-medium"
    speculative_config: Optional[SpeculativeStreamingConfig] = None
    lora_config: Optional[SpeculativeLoRAConfig] = None
    
    # Training configuration
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # Speculative-specific training
    speculation_loss_weight: float = 0.3
    stream_consistency_weight: float = 0.1
    tree_diversity_weight: float = 0.05
    acceptance_rate_target: float = 0.75
    
    # Data configuration
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_sequence_length: int = 512
    streaming: bool = True
    
    # Optimization
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # Monitoring
    use_wandb: bool = True
    wandb_project: str = "speculative-streaming"
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    
    # Validation
    do_eval: bool = True
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"


class SpeculativeDataset(Dataset):
    """Dataset class for training Speculative Streaming models."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        enable_speculation_targets: bool = True,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_speculation_targets = enable_speculation_targets
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # Create labels for language modeling
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding tokens
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        
        # Add speculation targets if enabled
        if self.enable_speculation_targets:
            # Create next-token targets for speculation training
            speculation_targets = torch.zeros_like(input_ids)
            speculation_targets[:-1] = input_ids[1:]
            speculation_targets[-1] = self.tokenizer.eos_token_id
            
            result['speculation_targets'] = speculation_targets
        
        return result


class SpeculativeTrainer(Trainer):
    """Custom trainer for Speculative Streaming with specialized loss functions."""
    
    def __init__(
        self,
        model,
        speculative_config: SpeculativeStreamingConfig,
        training_config: SpeculativeTrainingConfig,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        self.speculative_config = speculative_config
        self.training_config = training_config
        
        # Initialize additional loss components
        self.stream_consistency_loss = nn.MSELoss()
        self.speculation_accuracy_tracker = []
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute custom loss for Speculative Streaming training."""
        labels = inputs.get("labels")
        speculation_targets = inputs.get("speculation_targets")
        
        # Forward pass with speculative mode enabled during training
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=labels,
            output_attentions=True,
            use_cache=False,
        )
        
        # Base language modeling loss
        lm_loss = outputs.loss
        
        # Speculation-specific losses
        speculation_loss = torch.tensor(0.0, device=lm_loss.device)
        consistency_loss = torch.tensor(0.0, device=lm_loss.device)
        diversity_loss = torch.tensor(0.0, device=lm_loss.device)
        
        if speculation_targets is not None and hasattr(model, 'speculative_engine'):
            # Build speculation tree and compute acceptance rate
            try:
                with torch.no_grad():
                    tree_root = model.speculative_engine.build_speculation_tree(
                        inputs["input_ids"],
                        max_depth=2  # Reduced depth for training efficiency
                    )
                    
                    # Compute speculation accuracy
                    accepted_tokens, acceptance_rate = model.speculative_engine.verify_speculation_tree(
                        tree_root,
                        speculation_targets
                    )
                    
                    self.speculation_accuracy_tracker.append(acceptance_rate)
                    
                    # Speculation loss based on acceptance rate target
                    target_rate = self.training_config.acceptance_rate_target
                    speculation_loss = torch.abs(
                        torch.tensor(acceptance_rate, device=lm_loss.device) - target_rate
                    )
                    
            except Exception as e:
                # Fallback if speculation computation fails
                logging.warning(f"Speculation loss computation failed: {e}")
        
        # Stream consistency loss (encourage different streams to explore different paths)
        if hasattr(model, 'speculative_engine') and outputs.attentions is not None:
            try:
                # Compute consistency across attention streams
                for layer_attn in outputs.attentions[:2]:  # Only first 2 layers for efficiency
                    if layer_attn.shape[-1] > 1:  # Ensure we have multiple tokens
                        # Compare attention patterns across streams
                        attn_std = torch.std(layer_attn, dim=1)  # Variance across heads
                        consistency_loss += torch.mean(attn_std)
                        
            except Exception:
                pass
        
        # Tree diversity loss (encourage exploration of different paths)
        diversity_loss = torch.tensor(0.0, device=lm_loss.device)  # Placeholder
        
        # Combine losses
        total_loss = (
            lm_loss + 
            self.training_config.speculation_loss_weight * speculation_loss +
            self.training_config.stream_consistency_weight * consistency_loss +
            self.training_config.tree_diversity_weight * diversity_loss
        )
        
        # Log additional metrics
        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "train/lm_loss": lm_loss.item(),
                "train/speculation_loss": speculation_loss.item(),
                "train/consistency_loss": consistency_loss.item(),
                "train/total_loss": total_loss.item(),
                "train/speculation_accuracy": np.mean(self.speculation_accuracy_tracker[-10:]) if self.speculation_accuracy_tracker else 0,
            })
        
        return (total_loss, outputs) if return_outputs else total_loss


class SpeculativeTrainingPipeline:
    """Complete training pipeline for Speculative Streaming models."""
    
    def __init__(self, config: SpeculativeTrainingConfig):
        self.config = config
        self.setup_logging()
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = self._setup_model()
        self.datasets = self._prepare_datasets()
        
    def setup_logging(self):
        """Setup logging and monitoring."""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                config=self.config.__dict__,
                name=f"speculative-streaming-{self.config.model_name_or_path.split('/')[-1]}"
            )
    
    def _setup_model(self):
        """Initialize and configure the model for training."""
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        # Apply Speculative Streaming
        speculative_config = self.config.speculative_config or SpeculativeStreamingConfig()
        speculative_model = SpeculativeStreaming(
            base_model=base_model,
            config=speculative_config,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Apply LoRA if specified
        if self.config.lora_config is not None:
            lora_model = LoRASpeculativeAdapter(
                base_model=speculative_model,
                config=self.config.lora_config,
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            lora_model.enable_lora_training()
            return lora_model
        
        return speculative_model
    
    def _prepare_datasets(self):
        """Prepare training and validation datasets."""
        # Load dataset
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            streaming=self.config.streaming
        )
        
        # Extract texts
        def extract_texts(examples):
            return [text for text in examples['text'] if len(text.strip()) > 50]
        
        if self.config.streaming:
            train_texts = []
            eval_texts = []
            
            # Take samples from streaming dataset
            for i, example in enumerate(dataset['train']):
                if i >= 10000:  # Limit for development
                    break
                if len(example['text'].strip()) > 50:
                    if i % 10 == 0:  # 10% for eval
                        eval_texts.append(example['text'])
                    else:
                        train_texts.append(example['text'])
        else:
            train_texts = extract_texts(dataset['train'])
            eval_texts = extract_texts(dataset['validation']) if 'validation' in dataset else train_texts[:len(train_texts)//10]
        
        # Create datasets
        train_dataset = SpeculativeDataset(
            texts=train_texts,
            tokenizer=self.tokenizer,
            max_length=self.config.max_sequence_length,
            enable_speculation_targets=True
        )
        
        eval_dataset = SpeculativeDataset(
            texts=eval_texts,
            tokenizer=self.tokenizer,
            max_length=self.config.max_sequence_length,
            enable_speculation_targets=True
        ) if self.config.do_eval else None
        
        return {'train': train_dataset, 'eval': eval_dataset}
    
    def train(self):
        """Execute the training pipeline."""
        self.logger.info("Starting Speculative Streaming training...")
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if self.config.do_eval else None,
            do_eval=self.config.do_eval,
            evaluation_strategy=self.config.evaluation_strategy if self.config.do_eval else "no",
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            report_to=self.config.report_to,
            remove_unused_columns=False,  # Keep custom columns
        )
        
        # Create trainer
        trainer = SpeculativeTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets['train'],
            eval_dataset=self.datasets['eval'],
            data_collator=default_data_collator,
            speculative_config=self.config.speculative_config or SpeculativeStreamingConfig(),
            training_config=self.config,
        )
        
        # Train the model
        trainer.train()
        
        # Save final model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save configuration
        with open(os.path.join(self.config.output_dir, "training_config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        self.logger.info("Training completed successfully!")
        
        # Run final evaluation
        if self.config.do_eval:
            self._evaluate_model(trainer)
        
        return trainer
    
    def _evaluate_model(self, trainer):
        """Evaluate the trained model."""
        self.logger.info("Running final evaluation...")
        
        # Standard evaluation
        eval_results = trainer.evaluate()
        
        # Speculative-specific evaluation
        inference_engine = SpeculativeInferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            speculative_config=self.config.speculative_config or SpeculativeStreamingConfig()
        )
        
        # Test prompts for speculative evaluation
        test_prompts = [
            "The future of artificial intelligence",
            "In a world where technology",
            "Scientists have recently discovered",
            "The most important aspect of learning",
            "Climate change represents one of",
        ]
        
        # Benchmark speculative performance
        benchmark_results = inference_engine.benchmark(
            test_prompts=test_prompts,
            num_runs=3
        )
        
        # Log results
        final_metrics = {
            **eval_results,
            "speculative_speedup": benchmark_results['average_speedup_factor'],
            "speculation_accuracy": benchmark_results['average_speculation_accuracy'],
            "inference_throughput": benchmark_results['average_tokens_per_second'],
        }
        
        if self.config.use_wandb:
            wandb.log({"final_evaluation": final_metrics})
        
        self.logger.info(f"Final evaluation results: {final_metrics}")
        
        return final_metrics
    
    def save_for_deployment(self, deployment_path: str):
        """Save model optimized for deployment."""
        self.logger.info(f"Saving deployment model to {deployment_path}")
        
        # Create inference engine
        inference_engine = SpeculativeInferenceEngine(
            model=self.model,
            tokenizer=self.tokenizer,
            speculative_config=self.config.speculative_config or SpeculativeStreamingConfig()
        )
        
        # Optimize for deployment
        inference_engine.optimize_for_deployment()
        
        # Save optimized model
        os.makedirs(deployment_path, exist_ok=True)
        inference_engine.save_checkpoint(os.path.join(deployment_path, "model.pt"))
        self.tokenizer.save_pretrained(deployment_path)
        
        # Save deployment configuration
        deployment_config = {
            "model_type": "speculative_streaming",
            "speculative_config": self.config.speculative_config.__dict__ if self.config.speculative_config else {},
            "lora_config": self.config.lora_config.__dict__ if self.config.lora_config else None,
            "model_info": inference_engine.get_model_info(),
        }
        
        with open(os.path.join(deployment_path, "deployment_config.json"), "w") as f:
            json.dump(deployment_config, f, indent=2, default=str)
        
        self.logger.info("Deployment model saved successfully!")


def main():
    """Main training script."""
    # Example training configuration
    config = SpeculativeTrainingConfig(
        model_name_or_path="microsoft/DialoGPT-medium",
        speculative_config=SpeculativeStreamingConfig(
            gamma=4,
            max_speculation_depth=3,
            acceptance_threshold=0.7,
        ),
        lora_config=SpeculativeLoRAConfig(
            rank=32,
            alpha=32.0,
            stream_specific_adaptation=True,
        ),
        output_dir="./checkpoints/speculative-streaming",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        use_wandb=True,
    )
    
    # Create and run training pipeline
    pipeline = SpeculativeTrainingPipeline(config)
    trainer = pipeline.train()
    
    # Save for deployment
    pipeline.save_for_deployment("./deployment/speculative-streaming")
    
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
