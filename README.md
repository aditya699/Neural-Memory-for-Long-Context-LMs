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

## Best Practices Established

**For future experiments:**
- Use Flash Attention + RoPE as baseline (Exp 7 architecture)
- Target Chinchilla ratio 5-20:1 for compute efficiency
- Depth matters more than width once d_model ≥ 256
- Early stopping patience = 5 epochs (convergence indicator)
- Batch size 16-32 optimal for RTX 4000 Ada
- Monitor coherence span, not just perplexity

## Next Steps

To reach perplexity ~20-25 (GPT-2 Medium quality):
- **Model size**: 100-150M parameters
- **Architecture**: 16-20 layers, d_model=768
- **Dataset**: Larger corpus (500M-1B tokens)
- **Training**: 30-50 hours on L40/A100
- **Cost**: $50-100 GPU rental

**Advanced techniques to explore:**
- Grouped Query Attention (GQA)
- Sliding Window Attention
- Mixture of Experts (MoE)
- Better training data quality
