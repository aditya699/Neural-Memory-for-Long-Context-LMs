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
Params :869K
Tokens:242 K 
Preplexity of the final saved model:
Epoch 209:
  Train Loss: 5.7344
  Val Loss: 6.1413
  Val Perplexity: 464.64
  Best model so far! Saved to 'mha_model_best.pt


D_optimal = 20 × N

Where:
- D = Number of training tokens
- N = Number of model parameters
- 20 = The magic ratio (approximately)

Current Ratio = D / N
              = 242,643 / 869,153
              = 0.279

Your ratio: 0.279:1
Chinchilla optimal: 20:1

# Experiment 2 - 11/12/2025

## What we're doing
Testing Chinchilla's hypothesis that Experiment 1's poor generation was caused by data starvation, not model size. We're scaling up the dataset by 424x while keeping the model architecture identical.

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
**Parameters:** 869,153 (unchanged from Experiment 1)

## Dataset
- WikiText-103 (raw-v1)
- Train tokens: 103,000,000
- Val tokens: 217,646

## Training Setup
- Loss: Cross Entropy
- Optimizer: AdamW (lr=3e-4, weight_decay=0.1)
- Gradient Clipping: max_norm=1.0
- Batch size: 32 (increased from 8)
- Epochs: 1 (decreased from 209)
- Num workers: 4 (increased from 0)

## Why This Configuration?
Experiment 1 suffered from severe data starvation:
- Chinchilla ratio was 0.279:1 (72x undertrained)
- Limited vocabulary diversity (242K tokens)
- Model had capacity but insufficient training examples

**This experiment:** Fix the data bottleneck while keeping model constant to isolate the variable.

**Chinchilla analysis for Experiment 2:**
- Optimal tokens needed: 869,153 × 20 = 17,383,060
- Tokens per epoch: 103,000,000
- With 1 epoch: 103M / 17.4M = 5.9x OVERTRAINED ✅

Current Ratio = D / N (for 1 epoch)
              = 103,000,000 / 869,153
              = 118.5

Your ratio: 118.5:1
Chinchilla optimal: 20:1
Status: 6x overtrained (good for generation quality!)

