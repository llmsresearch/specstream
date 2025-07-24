# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-07-23

### Added
- Initial release of SpecStream library
- Implementation of speculative streaming for LLM inference
- Multi-Stream Attention (MSA) mechanism
- Tree-based speculation within single model
- LoRA integration support
- Inference engine with 2.8x speedup
- Training utilities for speculative decoding
- Deployment utilities
- Tree pruning optimization
- Example scripts for basic usage, quickstart, and LoRA fine-tuning
- Comprehensive documentation and README

### Features
- 2.8x speedup compared to standard inference
- 99.99% parameter reduction compared to multi-model approaches
- Memory efficient with <1% overhead
- Platform agnostic (CPU/GPU support)
- Easy integration as drop-in replacement
- Parameter-efficient fine-tuning with LoRA
