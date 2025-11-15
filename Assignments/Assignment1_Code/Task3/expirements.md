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

## Results
Device: cuda
GPU Memory: 0.03 GB

Loading dataset...

Creating datasets...
Tokenizing dataset...
Token indices sequence length is longer than the specified maximum sequence length for this model (1063 > 1024). Running this sequence through the model will result in indexing errors
Total tokens: 115,716,078
Tokenizing dataset...
Total tokens: 242,643
Train batches: 7063
Val batches: 15

Creating model...
Parameters: 869,153
GPU Memory: 0.03 GB

======================================================================
STARTING TRAINING
======================================================================
Epoch 1/10: 100%|██████████| 7063/7063 [15:18<00:00,  7.69it/s, loss=8.2508] 

Epoch 1:
  Train Loss: 8.2508
  Val Loss: 7.2842
  Val Perplexity: 1457.12
  Best model so far! Saved to 'mha_model_best.pt'

Epoch 2/10: 100%|██████████| 7063/7063 [15:18<00:00,  7.69it/s, loss=7.1734]

Epoch 2:
  Train Loss: 7.1734
  Val Loss: 6.9867
  Val Perplexity: 1082.15
  Best model so far! Saved to 'mha_model_best.pt'

Epoch 3/10: 100%|██████████| 7063/7063 [15:18<00:00,  7.69it/s, loss=6.9394]

Epoch 3:
  Train Loss: 6.9394
  Val Loss: 6.7338
  Val Perplexity: 840.37
  Best model so far! Saved to 'mha_model_best.pt'

Epoch 4/10: 100%|██████████| 7063/7063 [15:18<00:00,  7.69it/s, loss=6.6895]

Epoch 4:
  Train Loss: 6.6895
  Val Loss: 6.4804
  Val Perplexity: 652.23
  Best model so far! Saved to 'mha_model_best.pt'

Epoch 5/10: 100%|██████████| 7063/7063 [15:19<00:00,  7.68it/s, loss=6.4793]

Epoch 5:
  Train Loss: 6.4793
  Val Loss: 6.3027
  Val Perplexity: 546.04
  Best model so far! Saved to 'mha_model_best.pt'

Epoch 6/10: 100%|██████████| 7063/7063 [15:19<00:00,  7.68it/s, loss=6.3284]

Epoch 6:
  Train Loss: 6.3284
  Val Loss: 6.1547
  Val Perplexity: 470.93
  Best model so far! Saved to 'mha_model_best.pt'

Epoch 7/10: 100%|██████████| 7063/7063 [15:19<00:00,  7.68it/s, loss=6.2119]

Epoch 7:
  Train Loss: 6.2119
  Val Loss: 6.0556
  Val Perplexity: 426.51
  Best model so far! Saved to 'mha_model_best.pt'

Epoch 8/10: 100%|██████████| 7063/7063 [15:19<00:00,  7.68it/s, loss=6.1308]

Epoch 8:
  Train Loss: 6.1308
  Val Loss: 5.9870
  Val Perplexity: 398.24
  Best model so far! Saved to 'mha_model_best.pt'

Epoch 9/10: 100%|██████████| 7063/7063 [15:19<00:00,  7.68it/s, loss=6.0689]

Epoch 9:
  Train Loss: 6.0689
  Val Loss: 5.9348
  Val Perplexity: 377.98
  Best model so far! Saved to 'mha_model_best.pt'

Epoch 10/10: 100%|██████████| 7063/7063 [15:19<00:00,  7.68it/s, loss=6.0220]

Epoch 10:
  Train Loss: 6.0220
  Val Loss: 5.8991
  Val Perplexity: 364.71
  Best model so far! Saved to 'mha_model_best.pt'

======================================================================
Training Complete!
Best Epoch: 10
Best Val Loss: 5.8991
Best Val Perplexity: 364.71
======================================================================


# Expirement-3 12/11/2025 

## What we're doing
Testing capacity hypothesis: Experiment 2 showed data is sufficient (116M tokens) but model under-parameterized. Scaling model to minimum viable size for coherent language generation while keeping dataset constant.

## Model Configuration
```python
model = LanguageModel(
    vocab_size=50257,
    max_seq_len=512,
    d_model=128,
    num_heads=8,
    d_ff=512,
    num_layers=4,
    dropout=0.1
)
```
**Parameters:** ~Parameters: 7,342,033


## Dataset
- WikiText-103 (raw-v1)
- Train tokens: 116,000,000
- Val tokens: 242,643

## Training Setup
- Loss: Cross Entropy
- Optimizer: AdamW (lr=6e-4, weight_decay=0.1)
- Gradient Clipping: max_norm=1.0
- Batch size: 32
- Epochs: 20

## Why This Configuration?
Experiment 2 had sufficient data (118.5:1 ratio vs 20:1 optimal) but failed to generate coherent text. Root cause: severe capacity bottleneck.
- d_model=16 → tokens compressed into insufficient dimensions
- head_dim=8 → attention too simplistic
- 2 layers → can't learn hierarchical structure

This model (10M params) crosses minimum thresholds for basic language understanding:
- d_model=128: minimum for semantic distinctions
- head_dim=16: basic syntactic patterns
- 4 layers: token→phrase→clause→sentence hierarchy

**Chinchilla optimal for our dataset:** 5.8M params (116M tokens / 20)
**This model:** 10M params (1.7x over optimal, acceptable for generation quality)

Current Ratio = D / N (for 20 epochs)
              = 2,320,000,000 / 10,000,000
              = 232

Your ratio: 232:1
Chinchilla optimal: 20:1
Status: 11.6x overtrained (compensates for minimum model size)

## Expected Outcome
Coherent sentence generation with basic grammatical structure. Perplexity target: 80-150 (vs 365 in Experiment 2).

## Results
# Experiment 3 - 11/12/2025

## What we're doing
Testing capacity hypothesis: Experiment 2 showed data is sufficient (116M tokens) but model under-parameterized. Scaling model to minimum viable size for coherent language generation while keeping dataset constant.

## Model Configuration
```python
model = LanguageModel(
    vocab_size=50257,
    max_seq_len=512,
    d_model=128,
    num_heads=8,
    d_ff=512,
    num_layers=4,
    dropout=0.1
)
```
**Parameters:** 7,342,033

## Dataset
- WikiText-103 (raw-v1)
- Train tokens: 116,000,000
- Val tokens: 242,643

## Training Setup
- Loss: Cross Entropy
- Optimizer: AdamW (lr=6e-4, weight_decay=0.1)
- Gradient Clipping: max_norm=1.0
- Batch size: 32
- Epochs: 20
- Training time per epoch: ~34 minutes

## Why This Configuration?
Experiment 2 had sufficient data (118.5:1 ratio vs 20:1 optimal) but failed to generate coherent text. Root cause: severe capacity bottleneck.
- d_model=16 → tokens compressed into insufficient dimensions
- head_dim=8 → attention too simplistic
- 2 layers → can't learn hierarchical structure

This model (7.3M params) crosses minimum thresholds for basic language understanding:
- d_model=128: minimum for semantic distinctions
- head_dim=16: basic syntactic patterns
- 4 layers: token→phrase→clause→sentence hierarchy

**Chinchilla optimal for our dataset:** 5.8M params (116M tokens / 20)
**This model:** 7.3M params (1.26x over optimal)

Current Ratio = D / N (single epoch)
              = 116,000,000 / 7,342,033
              = 15.8

Your ratio: 15.8:1
Chinchilla optimal: 20:1
Status: Slightly undertrained by Chinchilla standards

## Expected Outcome
Coherent sentence generation with basic grammatical structure. Perplexity target: 80-150 (vs 365 in Experiment 2).

## Results

Training stopped at Epoch 11 due to perplexity plateau.

**Final Metrics:**
- Best Epoch: 10
- Train Loss: 4.3677
- Val Loss: 4.1853
- Val Perplexity: 65.71

**Perplexity progression:**
| Epoch | Val Perplexity | Improvement |
|-------|----------------|-------------|
| 1 | 376.01 | - |
| 2 | 187.36 | -188.65 |
| 4 | 91.76 | -95.60 |
| 5 | 77.75 | -13.01 |
| 6 | 71.38 | -6.37 |
| 7 | 68.84 | -2.54 |
| 8 | 67.12 | -1.72 |
| 9 | 66.40 | -0.72 |
| 10 | 65.71 | -0.69 |

Perplexity improvements diminished significantly after Epoch 8 (1.72 → 0.72 → 0.69), indicating model reached capacity ceiling.

**Generation Examples:**

Prompt: "The history of India is"
```
The history of India is シュブル, in example, for a manga anthology of modern 
history. It is most prevalent in Sanskrit literature. As the English publication 
of Kerala and Kerala and is an "etymology of Hindu scholars, though some of the 
earliest sources may have been made by scholars as far as to be the most popular 
poets of the twentieth century.
```

Prompt: "In mathematics,"
```
In mathematics,@ 300 was the three @-@ quarter game with that of the Games. 
The game's game was released on September 28, 16, 1798 and on November 24. 
In 1702, the game was featured in the game's first half, the second half to 
the game's release, one of the most memorable games in Game 16.
```

Prompt: "The cat sat on the"
```
The cat sat on the back of three and the final six @-@ quarters of the line 
were made up at the start of the 18th century, but was too expensive. The 
line was also removed from a small road road, and there were also four of 
the cars for this junction.
```

**Analysis:**

✅ **Major improvements over Experiment 2:**
- Perplexity: 365 → 66 (5.5x improvement)
- Grammatically correct sentences
- Proper subject-verb agreement
- Rich, contextual vocabulary
- Local semantic coherence (5-10 words)

✅ **Achieved targets:**
- Grammatical structure: YES
- Readable sentences: YES
- Perplexity 80-150: EXCEEDED (reached 66)

❌ **Limitations observed:**
- Topic drift after 10-15 words
- Semantic confusion (mixing unrelated contexts)
- No long-range coherence
- Factual inaccuracies

**Conclusion:**
Model capacity hypothesis **validated**. Scaling from 869K to 7.3M parameters enabled transition from token-level statistics to actual language understanding. However, model hit capacity ceiling around perplexity 65-70. 

# Experiment 4 - 11/13/2025

## What we're doing
Experiment 3 plateaued at perplexity 66 due to model capacity ceiling. Doubling model size to test if larger architecture enables better long-range coherence and reduces topic drift.

## Model Configuration
```python
model = LanguageModel(
    vocab_size=50257,
    max_seq_len=512,
    d_model=256,
    num_heads=8,
    d_ff=1024,
    num_layers=4,
    dropout=0.1
)
```
**Parameters:** ~18,000,000

## Dataset
- WikiText-103 (raw-v1)
- Train tokens: 116,000,000
- Val tokens: 242,643

## Training Setup
- Loss: Cross Entropy
- Optimizer: AdamW (lr=4e-4, weight_decay=0.1)
- Gradient Clipping: max_norm=1.0
- Batch size: 32
- Epochs: 15
- Multi-GPU: 2x RTX 4000 Ada (DataParallel)

## Why This Configuration?
Experiment 3 showed 7.3M params can learn grammar + local semantics but plateaued at perplexity 66. Doubling model capacity:
- d_model=128→256: richer token representations
- d_ff=512→1024: more feedforward capacity for pattern storage
- Same 4 layers: proven architecture depth

**Chinchilla optimal for our dataset:** 5.8M params (116M tokens / 20)
**This model:** 18M params (3.1x over optimal)

Current Ratio = D / N (single epoch)
              = 116,000,000 / 18,000,000
              = 6.4

Your ratio: 6.4:1
Chinchilla optimal: 20:1
Status: Undertrained, will compensate with multiple epochs

## Expected Outcome
Perplexity target: 45-55 (vs 66 in Experiment 3)
Multi-sentence coherence with reduced topic drift

## Result

Training stopped at Epoch 9 (404 minutes elapsed)

**Metrics at stopping point:**
- Train Loss: 4.0237
- Val Loss: 3.8358
- Val Perplexity: 46.33

**Conclusion:**
16M param model shows significant improvement over 7.3M (perplexity 66→46).
However, training time increased to 42 min/epoch making full experimentation impractical.
Model was still improving at stop point - early stopping not triggered.

# Experiment 5 - 11/14/2025

## What we're doing
Experiment 4 achieved perplexity 46 by scaling width (d_model, d_ff) but topic drift persisted. Testing depth hypothesis: doubling num_layers while keeping width constant to improve long-range coherence.

## Model Configuration
```python
model = LanguageModel(
    vocab_size=50257,
    max_seq_len=512,
    d_model=256,
    num_heads=8,
    d_ff=1024,
    num_layers=8,
    dropout=0.1
)
```
**Parameters:** ~28,000,000

## Dataset
- WikiText-103 (raw-v1)
- Train tokens: 116,000,000
- Val tokens: 242,643

## Training Setup
- Loss: Cross Entropy
- Optimizer: AdamW (lr=4e-4, weight_decay=0.1)
- Gradient Clipping: max_norm=1.0
- Batch size: 32
- Epochs: 15
- Early stopping patience: 5

## Why This Configuration?
Experiment 4 doubled width (d_model, d_ff) but maintained only 4 layers. While perplexity improved to 46, generation still showed topic drift after 15-20 words. Root cause: insufficient hierarchical depth.

This experiment tests depth scaling:
- d_model=256: proven sufficient for token representation (kept from Exp 4)
- d_ff=1024: proven sufficient for pattern storage (kept from Exp 4)
- num_layers=4→8: double depth for hierarchical reasoning
  - Layers 1-2: token and bigram patterns
  - Layers 3-4: phrase and clause structure
  - Layers 5-6: sentence-level coherence
  - Layers 7-8: multi-sentence topic maintenance

**Chinchilla optimal for our dataset:** 5.8M params (116M tokens / 20)
**This model:** 28M params (4.8x over optimal)

Current Ratio = D / N (single epoch)
              = 116,000,000 / 28,000,000
              = 4.1

Your ratio: 4.1:1
Chinchilla optimal: 20:1
Status: Significantly undertrained, will compensate with 15 epochs

## Expected Outcome
Perplexity target: 35-42 (vs 46 in Experiment 4)
Improved multi-sentence coherence with reduced topic drift

## NOTE :This time will let in learn for longer time (12hrs kinda run)
## Results
