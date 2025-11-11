# Experiment 1 - 11/11/2025

## What we're doing
Training a language model from scratch on WikiText-2 to test Chinchilla scaling laws.

## Model Configuration
```python
model = LanguageModel(
    vocab_size=50257,
    max_seq_len=512,
    d_model=16,
    num_heads=2,
    d_ff=64,
    num_layers=2,
    dropout=0.3
)
```
**Parameters:** 819,153

## Dataset
- WikiText-2 (raw-v1)
- Train tokens: 2,347,038
- Val tokens: 242,643

## Training Setup
- Loss: Cross Entropy
- Optimizer: AdamW (lr=3e-4, weight_decay=0.1)
- Gradient Clipping: max_norm=1.0
- Batch size: 8
- Epochs: 300

## Why This Configuration?
Previous model (3.4M params) was 29x larger than Chinchilla optimal → severe overfitting.
This model (819K params) is 7x over optimal → should overfit less.

**Chinchilla optimal for our dataset:** 117K params (tokens/20)

## Expected Outcome
Better text generation and less overfitting than previous experiment.

## Results
_(Fill after training)_