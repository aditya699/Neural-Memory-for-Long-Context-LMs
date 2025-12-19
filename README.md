# Neural Memory for Long-Context LMs

Exploring scaling laws, architectural improvements, and position encoding techniques for language modeling.

## Experiments Overview

| Exp | Date | What We're Testing | Dataset | Params | d_model | Layers | Architecture | Epochs | Train Time | Final Perplexity | Key Finding |
|-----|------|-------------------|---------|--------|---------|--------|-------------|--------|------------|------------------|-------------|
| 1 | 11/11 | Chinchilla scaling baseline | WikiText-2 (2.3M tokens) | 869K | 16 | 2 | Vanilla Transformer | 209 | - | 464.64 | Data starvation (0.28:1 ratio) |
| 2 | 11/12 | Scaling dataset 50x | WikiText-103 (116M tokens) | 869K | 16 | 2 | Vanilla Transformer | 10 | 2.5h | 364.71 | Data helps but capacity bottleneck |
| 3 | 11/12 | Scaling model capacity 8x | WikiText-103 | 7.3M | 128 | 4 | Vanilla Transformer | 11 | 6h | 65.71 | Crossed minimum viable threshold |
| 4 | 11/13 | Scaling width (d_model, d_ff) | WikiText-103 | 16M | 256 | 4 | Vanilla Transformer | 9 | 7h | 46.33 | Width improves vocab understanding |
| 5 | 11/14 | Scaling depth (num_layers) | WikiText-103 | 19M | 256 | 8 | Vanilla Transformer | 20 | 17h | 38.21 | Depth > width for coherence |
| 6 | 11/19 | Flash Attention (3.3x scale) | WikiText-103 | 63.8M | 512 | 12 | Flash Attention | 16 | 20h | 28.05 | Memory efficiency enables large models |
| 7 | 11/21 | RoPE position encoding | WikiText-103 | 63.6M | 512 | 12 | Flash Attention + RoPE | 8 | - | 28.06 | Better coherence, 2x faster convergence |
| 8* | 12/08 | Continual pre-training (CPT) | Clinical notes (30K, 13M tokens) | 494M | - | - | Qwen2.5-0.5B (pre-trained) | 1 | 29min | 2.44 (loss) | Medical domain adaptation: MCQ→clinical docs |
| 9 | 12/17 | Knowledge distillation (KL only) | WikiText-2 (5K examples filtered) | 44.6M (student) | 256 | 6 | Qwen-based | 10 | 10min | 347.06 (KL loss) | 11x compression (494M→44M), loss decreased 33% but output garbage (,,,,) |
| 10 | 12/19 | Knowledge distillation (KL + CE) | WikiText-2 (10K examples filtered) | 125M (student) | 512 | 12 | Qwen-based | 20 | 55min | 4.14 (hard loss) | 4x compression, grammatical output but wrong facts |

*Experiment 8 uses continual pre-training on Qwen2.5-0.5B (not trained from scratch)

## Key Insights

### Scaling Laws Validated
- **Data matters**: Exp 1→2 showed 50x data with same tiny model = 21% perplexity improvement
- **Capacity is critical**: Exp 2→3 showed 8x parameters = 82% improvement, gibberish to readable text
- **Width vs Depth**: Exp 3→4 (width) = 30% gain, Exp 4→5 (depth) = 17% gain + better coherence
- **Architectural efficiency**: Exp 5→6 (Flash Attention) = 27% gain + 3x larger model on same hardware

### Chinchilla Optimal Ratio Tracking
For WikiText-103 (116M tokens), Chinchilla optimal = 5.8M parameters (20:1 ratio)

| Experiment | Parameters | D/N Ratio | Status | Outcome |
|------------|------------|-----------|--------|---------|
| Exp 1 | 869K | 0.28:1 | 72x undertrained | Failed |
| Exp 2 | 869K | 118:1 | 6x overtrained | Still failed (capacity) |
| Exp 3 | 7.3M | 15.8:1 | Near optimal | Success |
| Exp 4 | 16M | 6.4:1 | Undertrained | Improved |
| Exp 5 | 19M | 4.1:1 | Undertrained | Further improved |
| Exp 6 | 63.8M | 1.8:1 | Undertrained | Major breakthrough |
| Exp 7 | 63.6M | 1.82:1 | Undertrained | Same quality, better efficiency |

### Architectural Breakthroughs

**Flash Attention (Exp 6)**
- Enabled 3.3x larger model (19M → 64M) on same hardware
- Memory: O(n²) → O(n) complexity
- VRAM: 12GB peak (60% of RTX 4000 Ada capacity)
- Without Flash: Would need 30-40GB (datacenter GPU)
- Quality: 27% perplexity improvement
- **Key insight**: Algorithmic efficiency substitutes for hardware scaling

**RoPE - Rotary Position Embeddings (Exp 7)**
- Removed 262K learned position parameters
- Replaced with mathematical formula (relative positions)
- Same perplexity (28.06 vs 28.05) but better coherence
- 2x faster convergence (8 vs 16 epochs)
- No max sequence length constraint
- **Key insight**: Relative positions generalize better than absolute

### Knowledge Distillation Experiments (Exp 9 & 10)

**Exp 9: KL Divergence Only (Failed)**
- Teacher: Qwen2.5-0.5B (494M params)
- Student: 44.6M params (11x compression, random init)
- Loss: KL divergence only (soft targets)
- Result: Loss decreased 33% but output was garbage (`,,,,,,,,`)
- **Why it failed**: Without cross-entropy (hard labels), student took shortcut — predicted "safe" punctuation tokens that minimized KL loss without learning actual language

**Exp 10: KL + Cross-Entropy (Improved)**
- Teacher: Qwen2.5-0.5B (494M params)
- Student: 125M params (4x compression, random init)
- Loss: Combined (α=0.5 × KL + 0.5 × Cross-Entropy)
- Training: 10K examples, 20 epochs, 55 minutes
- Loss trajectory:
  - Soft (KL): 3.56 → 2.96
  - Hard (CE): 6.68 → 4.14
- Result: Grammatical English but wrong facts

| Prompt | Teacher Output | Student Output |
|--------|----------------|----------------|
| "The capital of France is" | Paris (31.57%) | the (21.39%) |
| "The cat sat on the" | mat (84.33%) | city (0.58%) |

**Why Exp 10 improved over Exp 9:**
- Cross-entropy loss gave strong signal: "predict actual correct token, not just match distribution shape"
- Without CE, student could minimize KL by predicting punctuation everywhere
- With CE, student forced to learn real language patterns

**Why Exp 10 still didn't match teacher:**
- Random initialization: student had to learn language from scratch
- Industry uses pretrained students that already know language
- 10K examples and 55 min training insufficient for 125M param model learning from zero

### Production Comparison
```
GPT-2 Small (117M params):    ~29 perplexity on WikiText-103
Exp 6/7 (64M params):         28.05 perplexity on WikiText-103
GPT-2 Medium (345M params):   ~23 perplexity on WikiText-103
```
**Achieved GPT-2 Small quality with 45% fewer parameters.**

## Lessons Learned

1. **Data + Capacity both necessary** - Neither alone is sufficient
2. **Minimum viable model size** - Below 7M params, can't learn language structure
3. **Depth > Width for coherence** - Layers enable hierarchical reasoning
4. **Flash Attention is essential** - Makes large models feasible on consumer GPUs
5. **RoPE superior to absolute positions** - Better generalization + faster training
6. **Diminishing returns** - Each doubling costs exponentially more
7. **WikiText-103 ceiling** - Dataset supports ~30M-64M params effectively
8. **Distillation requires combined loss** - KL only → mode collapse; KL + CE → actual learning
9. **Compression ratio matters** - 11x too aggressive, 4x works better
10. **Pretrained students are critical** - Random init requires 10-100x more training than pretrained init