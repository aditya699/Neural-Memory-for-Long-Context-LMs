# Comprehensive Attention Mechanisms Comparison

A complete reference guide comparing all modern attention variants.

---

## Quick Reference Table

| Mechanism | Complexity | KV Cache | Quality | Speed | Best For |
|:----------|:-----------|:---------|:--------|:------|:---------|
| **Full MHA** | O(n²d) | O(n·h·d) | ⭐⭐⭐⭐⭐ | ⭐ | Short sequences, highest quality |
| **GQA** | O(n²d) | O(n·g·d) | ⭐⭐⭐⭐ | ⭐⭐⭐ | Production LLMs, balanced |
| **MQA** | O(n²d) | O(n·d) | ⭐⭐⭐ | ⭐⭐⭐⭐ | Fast inference, quality acceptable |
| **MLA** | O(n²d) | O(n·d_latent) | ⭐⭐⭐⭐ | ⭐⭐⭐ | Extreme long context |
| **Sliding Window** | O(n·w·d) | O(n·d) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Long sequences, local patterns |
| **Strided** | O(n·s·d) | O(n·d) | ⭐⭐⭐ | ⭐⭐⭐⭐ | Long-range but sparse dependencies |
| **Block-Sparse** | O(n·b·d) | O(n·d) | ⭐⭐⭐⭐ | ⭐⭐⭐ | Structured data, documents |

*Where: n=seq_len, h=num_heads, g=num_groups, d=hidden_dim, w=window, s=stride, b=blocks*

---

## Detailed Comparison

### 1. Multi-Head Attention (MHA) - Standard/Full Attention

**What it is:**
- Original attention from "Attention is All You Need"
- Every head has its own Q, K, V projections
- All heads attend to all previous tokens

**Architecture:**
```
num_heads = 32
Each head: Q_h, K_h, V_h (all independent)
KV cache per token: 32 × d_head × 2 (K and V)
```

**Complexity:**
- Computation: O(n² × d)
- Memory (KV cache): O(n × h × d_head) = O(n × d_model)
- Attention matrix: O(n²) per head

**Models Using It:**
- GPT-2
- BERT
- Early GPT-3 variants
- Smaller models (<1B params)

**Pros:**
- ✅ Highest quality - no information loss
- ✅ Each head can specialize independently
- ✅ Best for understanding and reasoning

**Cons:**
- ❌ Largest KV cache (2 × n × d_model bytes)
- ❌ Slowest inference at long contexts
- ❌ Doesn't scale beyond ~8k tokens
- ❌ O(n²) cost

**When to Use:**
- Short sequences (<2k tokens)
- Quality is paramount
- Research/analysis where speed doesn't matter
- Fine-tuning smaller models

---

### 2. Grouped-Query Attention (GQA)

**What it is:**
- Middle ground between MHA and MQA
- Groups of query heads share K,V heads
- Typically: 32 Q heads → 8 KV heads (4 queries per KV group)

**Architecture:**
```
num_query_heads = 32
num_kv_heads = 8  (or 4)
Ratio: 32/8 = 4 queries share each KV

KV cache per token: 8 × d_head × 2
```

**Complexity:**
- Computation: O(n² × d) (same as MHA)
- Memory (KV cache): O(n × g × d_head) where g = num_kv_heads
- Savings: 4× smaller KV cache vs MHA (if g = h/4)

**Models Using It:**
- **Llama 2 (70B)** - 8 KV heads for 64 query heads
- **Llama 3 / 3.1 / 3.2** - All sizes use GQA
- **Mistral 7B** - 8 KV heads, 32 query heads
- **Mixtral 8x7B** - Same as Mistral
- **Gemma 2** - Uses GQA

**Pros:**
- ✅ 2-8× smaller KV cache than MHA
- ✅ Quality very close to full MHA (minimal degradation)
- ✅ Much faster inference than MHA
- ✅ Scales to 32k-128k tokens
- ✅ Good balance of quality and speed

**Cons:**
- ❌ Still O(n²) attention computation
- ❌ Slightly lower quality than full MHA
- ❌ More complex implementation

**When to Use:**
- **DEFAULT CHOICE for modern LLMs**
- Production deployments
- Need both quality and efficiency
- 8k-128k context windows
- Multi-user serving scenarios

---

### 3. Multi-Query Attention (MQA)

**What it is:**
- Extreme version of GQA
- ALL query heads share a SINGLE K,V head
- Maximum KV cache compression

**Architecture:**
```
num_query_heads = 32
num_kv_heads = 1  (single shared KV)

KV cache per token: 1 × d_head × 2
```

**Complexity:**
- Computation: O(n² × d)
- Memory (KV cache): O(n × d_head) 
- Savings: 32× smaller KV cache vs MHA (if 32 heads)

**Models Using It:**
- **PaLM** - Google's model
- **Falcon** - Some variants
- **StarCoder** - Code generation
- Early experimental models

**Pros:**
- ✅ Minimal KV cache (32× smaller than MHA)
- ✅ Fastest inference speed
- ✅ Best memory efficiency
- ✅ Great for high-throughput serving

**Cons:**
- ❌ Noticeable quality drop vs MHA/GQA
- ❌ Training can be unstable
- ❌ Less expressive (single KV bottleneck)
- ❌ Not used in SOTA models anymore

**When to Use:**
- Inference-only scenarios (no training)
- Extreme memory constraints
- High-throughput batch serving
- Quality is acceptable to trade for speed
- Code completion (low latency critical)

**Not Recommended For:**
- New model training (use GQA instead)
- Tasks requiring nuanced understanding
- Reasoning-heavy applications

---

### 4. Multi-Head Latent Attention (MLA)

**What it is:**
- DeepSeek's innovation
- Compresses K,V into low-dimensional latent space
- Projects back during attention computation

**Architecture:**
```
Original KV: n × h × d_head
Latent KV: n × d_latent (d_latent << h × d_head)

Example:
  h × d_head = 32 × 128 = 4096
  d_latent = 512
  Compression: 8×
```

**Process:**
```
1. Compress: K,V → latent_K, latent_V (low-dim)
2. Store: Only latent representations in cache
3. Inference: Project latent back to full K,V when needed
```

**Complexity:**
- Computation: O(n² × d) + projection overhead
- Memory (KV cache): O(n × d_latent)
- Compression: 4-16× depending on d_latent

**Models Using It:**
- **DeepSeek V2** - 512 latent dim
- **DeepSeek V3** - Further optimized
- Research prototypes

**Pros:**
- ✅ Extreme KV cache reduction (4-16×)
- ✅ Quality close to GQA
- ✅ Enables very long contexts
- ✅ Innovative compression approach

**Cons:**
- ❌ Additional projection computation
- ❌ More complex implementation
- ❌ Training requires careful tuning
- ❌ Not widely adopted yet

**When to Use:**
- Extreme long context (100k-1M tokens)
- Memory is the primary bottleneck
- Can afford projection overhead
- Research/experimentation

---

### 5. Sliding Window Attention

**What it is:**
- Each token attends only to last W tokens
- Window "slides" forward as position increases
- Stacking layers extends effective reach to L×W

**Architecture:**
```
Window size W = 4096
Layers L = 32
Effective reach = 32 × 4096 = 131k tokens

Each token: O(W) attention instead of O(n)
```

**Complexity:**
- Computation: O(n × W × d) per layer → Linear in n!
- Memory (attention): O(n × W) → Linear in n!
- KV cache: O(n × d) (store all, but attend to window)

**Models Using It:**
- **Mistral 7B** - W=4096, 32 layers
- **Mixtral 8x7B** - Same
- **Longformer** - Variable windows
- Some Llama research variants

**Pros:**
- ✅ Linear complexity O(n·W) vs O(n²)
- ✅ Scales to 100k+ tokens
- ✅ Fast inference (2× speedup at 16k)
- ✅ Captures local context perfectly
- ✅ Layer stacking gives global reach

**Cons:**
- ❌ Information dilutes over distance
- ❌ Can't directly access far tokens
- ❌ Needs deep models (32+ layers) for long reach
- ❌ Quality degrades beyond effective reach

**When to Use:**
- Long documents (32k-128k tokens)
- Language modeling (incremental context)
- Most dependencies are local
- Can use deep models (32+ layers)
- Speed matters more than perfect recall

**Not Recommended For:**
- Random access tasks ("what was in paragraph 1?")
- Need perfect recall of early context
- Shallow models (<16 layers)

---

### 6. Strided Attention

**What it is:**
- Attend to every k-th token (stride k)
- Skips intermediate tokens
- Creates fixed sparse pattern

**Architecture:**
```
Stride k = 8
Position 64 attends to: [0, 8, 16, 24, 32, 40, 48, 56, 64]

Attention to: n/k tokens instead of n
```

**Complexity:**
- Computation: O(n × (n/k) × d) = O(n²/k × d)
- Memory: O(n × n/k) = O(n²/k)
- Speedup: k× vs full attention

**Models Using It:**
- **Sparse Transformers** (OpenAI)
- **Reformer** (Google)
- **Longformer** (hybrid with local)
- Research models

**Pros:**
- ✅ Can reach back to position 0 directly
- ✅ k× speedup vs full attention
- ✅ Simple to implement
- ✅ Good for periodic patterns

**Cons:**
- ❌ Loses ALL local context (big gaps)
- ❌ Misses potentially important nearby tokens
- ❌ Still O(n²/k) - not truly linear
- ❌ Fixed pattern may not fit all tasks

**When to Use:**
- Data has periodic structure (music, time series)
- Long-range dependencies more important than local
- Combined with local attention (hybrid approach)
- Research/experimentation

**Not Recommended For:**
- Natural language (needs local context)
- As the only attention mechanism
- Tasks requiring fine-grained understanding

---

### 7. Block-Sparse Attention

**What it is:**
- Divide sequence into blocks
- Attend to entire blocks (all or nothing)
- More flexible than strided

**Architecture:**
```
Block size = 64 tokens
Sequence: 1024 tokens → 16 blocks

Position in Block 15 attends to:
  Block 0  (first block - global context)
  Block 14 (previous block)
  Block 15 (own block - local context)
  Block 7  (random/strategic block)
```

**Complexity:**
- Computation: O(n × b × block_size × d) where b = num_blocks attended
- Memory: O(n × b × block_size)
- Typically: b << n/block_size (sparse)

**Models Using It:**
- **BigBird** (Google) - Random + window + global
- **Longformer** - Local + global blocks
- **Sparse Transformers** - Various patterns
- ETC (Extended Transformer Construction)

**Pros:**
- ✅ Balances local and global attention
- ✅ No gaps within attended blocks
- ✅ Flexible patterns (local + global + random)
- ✅ Better than pure strided for language
- ✅ Hardware-efficient (block operations)

**Cons:**
- ❌ More complex implementation
- ❌ Pattern design requires tuning
- ❌ Not as simple as sliding window
- ❌ Less widely adopted

**When to Use:**
- Long documents with structure (sections, paragraphs)
- Need both local detail and global context
- Document classification/summarization
- Scientific papers, legal documents
- Combined patterns work better than single strategy

**Common Patterns:**
1. **Local + Global**: Own block + first block
2. **Local + Strided**: Own block + every k-th block  
3. **Local + Random**: Own block + random blocks
4. **Windowed**: Own + previous few blocks

---

## Decision Framework: Which Attention to Use?

### By Sequence Length

| Context Length | Recommended | Alternative |
|:---------------|:------------|:------------|
| **<2k tokens** | Full MHA | GQA (if memory matters) |
| **2k-8k** | GQA | Sliding Window |
| **8k-32k** | GQA + Sliding Window | MLA |
| **32k-128k** | Sliding Window + GQA | Block-Sparse |
| **128k-1M** | MLA + Sliding Window | Hybrid patterns |
| **>1M** | Research territory | Llama 4 Scout (iRoPE) |

---

### By Use Case

**1. General-Purpose Language Model (Production)**
```
✅ GQA (Llama 3 style)
   - 32 query heads, 8 KV heads
   - Proven at scale
   - Best balance
```

**2. Long Document Processing**
```
✅ Sliding Window (Mistral style)
   - W = 4096, L = 32 layers
   - Or combine with GQA
   - Linear scaling
```

**3. Code Generation**
```
✅ GQA or MQA
   - Fast inference critical
   - Local context important
   - MQA if latency is key
```

**4. Research / Max Quality**
```
✅ Full MHA
   - Short sequences only
   - No compromises
```

**5. High-Throughput Serving**
```
✅ MQA or GQA
   - Many concurrent requests
   - KV cache memory is bottleneck
   - Small quality drop acceptable
```

**6. Extreme Long Context (100k+)**
```
✅ MLA (DeepSeek V3 style)
   - Or Sliding Window + GQA
   - Or wait for Llama 4 Scout
```

**7. Document Classification**
```
✅ Block-Sparse (BigBird style)
   - Local + first block + random
   - Captures document structure
```

---

## Hybrid Approaches (Best Practices)

Modern models often combine multiple techniques:

### **Mistral 7B Approach:**
```
1. GQA: 32 query heads, 8 KV heads
2. Sliding Window: W=4096
3. FlashAttention optimizations

Result: 131k theoretical reach, 32k practical, very fast
```

### **Llama 3.1 Approach:**
```
1. GQA: 8 KV heads (70B model)
2. RoPE with extended context
3. 128k context window
4. Grouped-query for efficiency

Result: Production-ready long context
```

### **DeepSeek V3 Approach:**
```
1. MLA: 512 latent dimensions
2. MoE architecture
3. Extreme compression

Result: Handles very long contexts efficiently
```

### **BigBird Approach:**
```
1. Block-sparse patterns:
   - Local blocks (sliding window-like)
   - Global blocks (first few blocks)
   - Random blocks (for long-range)

Result: Best for document-level tasks
```

---

## Quality vs Efficiency Trade-offs

### Quality Ranking (Best to Worst):
```
1. Full MHA              ⭐⭐⭐⭐⭐ (100% quality baseline)
2. GQA (g=8)             ⭐⭐⭐⭐☆ (95-98% of MHA)
3. MLA                   ⭐⭐⭐⭐☆ (95-97% of MHA)
4. Sliding Window        ⭐⭐⭐⭐☆ (90-95%, degrades with distance)
5. Block-Sparse          ⭐⭐⭐⭐☆ (90-95%, pattern dependent)
6. GQA (g=4)             ⭐⭐⭐⭐☆ (92-95% of MHA)
7. MQA                   ⭐⭐⭐☆☆ (85-90% of MHA)
8. Strided               ⭐⭐⭐☆☆ (80-90%, task dependent)
```

### Speed Ranking (Fastest to Slowest):
```
1. MQA                   ⚡⚡⚡⚡⚡ (Fastest)
2. Sliding Window        ⚡⚡⚡⚡☆ (Linear complexity)
3. Strided               ⚡⚡⚡⚡☆ (Sparse)
4. GQA (g=4)             ⚡⚡⚡☆☆
5. GQA (g=8)             ⚡⚡⚡☆☆
6. Block-Sparse          ⚡⚡⚡☆☆ (Pattern dependent)
7. MLA                   ⚡⚡⚡☆☆ (Projection overhead)
8. Full MHA              ⚡☆☆☆☆ (Slowest)
```

### Memory Ranking (Smallest to Largest KV Cache):
```
1. MLA                   💾      (512 dim latent)
2. MQA                   💾💾    (1 KV head)
3. GQA (g=4)             💾💾💾  (4 KV heads)
4. GQA (g=8)             💾💾💾💾 (8 KV heads)
5. Full MHA              💾💾💾💾💾 (32 KV heads)

Note: Sliding/Strided/Block-Sparse have same KV cache as MHA,
      but save on attention matrix memory
```

---

## Modern Model Examples

### **GPT-4 (Rumored)**
```
Attention: Likely GQA + optimizations
Context: 128k tokens
Approach: Balanced quality + efficiency
```

### **Claude 3.5 Sonnet**
```
Attention: Unknown (proprietary)
Context: 200k tokens standard, 500k enterprise
Approach: Long context focus
```

### **Llama 3.1 (405B)**
```
Attention: GQA (8 KV heads for 64 query heads)
Context: 128k tokens
Layers: 126
Window: None (uses GQA + RoPE)
```

### **Mistral 7B**
```
Attention: GQA (8 KV heads, 32 query)
          + Sliding Window (W=4096)
Context: 32k practical, 131k theoretical
Layers: 32
Speed: 2× faster at 16k vs vanilla
```

### **Mixtral 8x7B**
```
Attention: Same as Mistral 7B
MoE: 8 experts, 2 active per token
Context: 32k
Efficiency: MoE + GQA + Sliding Window
```

### **DeepSeek V3**
```
Attention: MLA (512 latent dim)
MoE: 256 experts, 8 active
Context: Very long (100k+)
Innovation: Extreme compression
```

### **Gemini 1.5 Pro**
```
Attention: Unknown (proprietary)
Context: 1 million tokens
Approach: Likely hybrid with very aggressive compression
```

### **Llama 4 Scout**
```
Attention: iRoPE (interleaved RoPE)
          + Hybrid global/local attention
Context: 10 million tokens (claimed)
Layers: Unknown
Status: Cutting edge (April 2025)
Practical: ~1.4M on 8×H100
```

---

## Implementation Recommendations

### For Training:
```python
# Start with GQA - best default
num_query_heads = 32
num_kv_heads = 8  # 4:1 ratio
head_dim = 128

# For longer context, add sliding window
window_size = 4096  # if needed

# Avoid MQA for training (unstable)
```

### For Inference Optimization:
```python
# If memory constrained:
use_mqa = True  # or use_gqa with g=4

# If need long context:
use_sliding_window = True
window_size = 4096

# If quality critical:
use_gqa = True
num_kv_heads = 8  # or full MHA
```

### For Research/New Architectures:
```python
# Experiment with:
- MLA for extreme long context
- Block-sparse for structured data
- Hybrid patterns (local + global + random)
- Flash Attention 2/3 for all variants
```

---

## Key Takeaways

1. **GQA is the modern default** - Used by Llama 3, Mistral, most SOTA models
2. **Sliding Window for long context** - Best linear-scaling approach
3. **MQA only for inference** - Too unstable for training
4. **MLA is cutting edge** - Watch DeepSeek for innovations
5. **Full MHA is legacy** - Only for short sequences or research
6. **Sparse patterns are niche** - Use for specific structured data
7. **Hybrid approaches win** - Combine techniques (GQA + Sliding Window)

---

## Quick Decision Tree

```
START: What's your priority?

├─ Quality > All
│  └─ Full MHA (if n < 2k) or GQA (g=8)
│
├─ Long Context (>32k)
│  ├─ With quality: MLA or Sliding Window + GQA
│  └─ Max length: Llama 4 Scout approaches
│
├─ Speed > Quality
│  ├─ Training: GQA (g=4)
│  └─ Inference only: MQA
│
├─ Memory Constrained
│  ├─ Extreme: MLA
│  └─ Moderate: MQA or GQA (g=4)
│
└─ Production Default
   └─ GQA (g=8) + Sliding Window (if n > 8k)
```

---

## Further Reading

**Papers:**
- Full MHA: "Attention is All You Need" (Vaswani et al., 2017)
- GQA: "GQA: Training Generalized Multi-Query Transformer Models" (Ainslie et al., 2023)
- MQA: "Fast Transformer Decoding" (Shazeer, 2019)
- Sliding Window: "Mistral 7B" (Jiang et al., 2023)
- MLA: "DeepSeek-V2" (DeepSeek, 2024)
- Sparse: "Generating Long Sequences with Sparse Transformers" (Child et al., 2019)
- Block-Sparse: "Big Bird" (Zaheer et al., 2020), "Longformer" (Beltagy et al., 2020)

**Implementations:**
- Flash Attention: https://github.com/Dao-AILab/flash-attention
- xFormers: https://github.com/facebookresearch/xformers
- Hugging Face Transformers: All variants supported

---

*This comparison is based on research and production models as of April 2025. The field evolves rapidly—new innovations emerge frequently.*