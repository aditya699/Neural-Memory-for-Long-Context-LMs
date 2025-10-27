# Understanding Grouped-Query Attention (GQA)

A comprehensive guide to why and how modern LLMs like Llama 2/3 optimize inference with GQA.

---

## Step 1 — The Problem: KV Cache Memory Bottleneck

In production LLM serving, Multi-Head Attention (MHA) creates a severe memory bottleneck during inference.

---

### 🧩 The KV Cache Problem

Consider Llama 2 70B with standard Multi-Head Attention:

**Configuration:**
- 64 attention heads
- d_model = 8,192
- 80 layers
- Serving 16 users simultaneously (batch size = 16)
- Context length = 4,096 tokens

**KV Cache Calculation:**

For each token, each head stores:
- K (Key): 128 dimensions
- V (Value): 128 dimensions

**Per layer:**
```
Cache size = 2 (K+V) × batch × seq_len × d_model × 2 bytes (fp16)
           = 2 × 16 × 4,096 × 8,192 × 2
           = 2,147,483,648 bytes
           = 2 GB per layer
```

**Total for 80 layers:**
```
2 GB × 80 layers = 160 GB just for KV cache!
```

---

### 💡 The Core Problem

**Memory bandwidth bottleneck:**
- At each generation step, ALL keys and values must be loaded from memory
- This dominates inference time for large models
- Limits: batch size, context length, deployment feasibility

**The question:** Can we reduce KV cache without losing quality?

---

## Step 2 — The Insight: Do All Heads Need Separate K and V?

Let's revisit what each component does in attention:

---

### 🔍 Role of Q, K, V

**Q (Query):** "What am I looking for?"
- Determines WHAT patterns to attend to
- Different queries → different attention patterns

**K (Key):** "What information is available?"
- Acts as an index to the information

**V (Value):** "The actual content"
- The information retrieved

---

### 💡 Key Observation

In Multi-Head Attention:
```
Head 1: Q₁ (unique) × K₁ (unique) → attends using V₁ (unique)
Head 2: Q₂ (unique) × K₂ (unique) → attends using V₂ (unique)
...
Head 64: Q₆₄ (unique) × K₆₄ (unique) → attends using V₆₄ (unique)
```

**Research finding:** Most of the diversity in attention patterns comes from **different Queries**, not from having separate K and V for each head!

**Analogy:**
- K and V = Library (the available information)
- Q = Your search query

**Insight:** Multiple students (heads) can use the **same library** (shared K, V) but search for **different things** (unique Q).

---

## Step 3 — Multi-Query Attention (MQA): The Extreme Approach

Before GQA, researchers tried Multi-Query Attention (MQA).

---

### 🔀 MQA Architecture

**Idea:** ALL heads share ONE single K and ONE single V.

**Example with 64 heads:**

```
Head 1: Q₁ (unique) × K_shared × V_shared
Head 2: Q₂ (unique) × K_shared × V_shared
Head 3: Q₃ (unique) × K_shared × V_shared
...
Head 64: Q₆₄ (unique) × K_shared × V_shared
```

**KV Cache Reduction:**
- MHA: 64 K + 64 V = 128 KV matrices
- MQA: 1 K + 1 V = 2 KV matrices
- **Reduction: 64x smaller KV cache!** ✅

---

### ⚠️ The Problem with MQA

**Quality degradation:**
- All heads forced to use the SAME information pool
- Loses diversity in what information is available
- Training instability

**Research results (T5-XXL model):**
- MHA (baseline): 47.2 quality score, 1.51 ms per sample
- MQA: 46.6 quality score (-1.3%), 0.24 ms per sample

**Trade-off:** 6x faster but noticeable quality loss.

**Conclusion:** MQA is too aggressive for large, quality-sensitive models.

---

## Step 4 — GQA: The Goldilocks Solution

Grouped-Query Attention is the sweet spot between MHA and MQA.

---

### 🎯 The GQA Approach

**Idea:** Group heads together, share K and V within groups.

**Formula:**
```
h = total number of query heads
g = number of groups
heads per group = h / g
```

---

### 📊 Example: 64 Heads → 8 Groups

**Configuration:**
- 64 query heads
- 8 groups (g = 8)
- 8 heads per group (64 / 8 = 8)

**Architecture:**

**Group 1 (Heads 1-8):**
```
Head 1: Q₁ (unique) × K₁ (shared within group) × V₁ (shared within group)
Head 2: Q₂ (unique) × K₁ (shared) × V₁ (shared)
Head 3: Q₃ (unique) × K₁ (shared) × V₁ (shared)
...
Head 8: Q₈ (unique) × K₁ (shared) × V₁ (shared)
```

**Group 2 (Heads 9-16):**
```
Head 9: Q₉ (unique) × K₂ (shared within group) × V₂ (shared within group)
Head 10: Q₁₀ (unique) × K₂ (shared) × V₂ (shared)
...
Head 16: Q₁₆ (unique) × K₂ (shared) × V₂ (shared)
```

...and so on for all 8 groups.

---

### 🔢 KV Cache Reduction

**MHA (standard):**
- 64 K matrices + 64 V matrices = 128 KV matrices

**GQA (g=8):**
- 8 K matrices + 8 V matrices = 16 KV matrices
- **Reduction: 128 / 16 = 8x smaller!** ✅

**MQA (g=1):**
- 1 K matrix + 1 V matrix = 2 KV matrices
- Reduction: 64x smaller (but quality loss)

---

## Step 5 — Why GQA Works: Different Queries Create Diversity

The key insight is that heads within a group still learn different patterns through their unique queries.

---

### 🧠 Concrete Example

**Sentence:** "The cat sat on the mat because it was soft."

**Shared K and V for Group 1 (Heads 1-8):**

```
K_mat = [0.2, 0.8, 0.1, ...]  (represents "mat" as noun/object)
V_mat = [rich representation of "mat" properties]

K_soft = [0.1, 0.3, 0.9, ...]  (represents "soft" as adjective)
V_soft = [rich representation of softness]
```

**Head 1 in Group 1:**
```
Q₁ from "it" = [0.9, 0.1, 0.2, ...]  (looking for nouns)

Attention scores:
Q₁ · K_mat = 0.9×0.2 + 0.1×0.8 + 0.2×0.1 = 0.28
Q₁ · K_soft = 0.9×0.1 + 0.1×0.3 + 0.2×0.9 = 0.30

Result: Attends more to "mat" (noun referent)
```

**Head 2 in Group 1:**
```
Q₂ from "it" = [0.1, 0.2, 0.9, ...]  (looking for adjectives)

Attention scores:
Q₂ · K_mat = 0.1×0.2 + 0.2×0.8 + 0.9×0.1 = 0.27
Q₂ · K_soft = 0.1×0.1 + 0.2×0.3 + 0.9×0.9 = 0.88

Result: Attends more to "soft" (property)
```

**Same K and V, but different Queries → different attention patterns!** ✅

---

### 💡 Diversity Through Queries

Within Group 1, all 8 heads share K₁ and V₁, but:
- Head 1 might focus on pronoun resolution
- Head 2 might focus on semantic properties
- Head 3 might focus on local syntax
- ...each has unique Q to extract different patterns

**The library analogy:**
- 8 students (heads) use the same library (K, V)
- But each searches for different things (unique Q)
- Student 1: "Show me history books"
- Student 2: "Show me science books"
- Same library, different queries → different results

---

## Step 6 — Llama 2/3 Implementation

Meta's Llama models use GQA with specific configurations.

---

### 📊 Llama 2 Configuration

**Llama 2 7B:**
- num_attention_heads = 32
- num_key_value_heads = 32
- **Groups: 32 / 32 = 1 (actually MQA!)**

**Llama 2 13B:**
- num_attention_heads = 40
- num_key_value_heads = 40
- **Groups: 40 / 40 = 1 (MQA!)**

**Llama 2 70B:**
- num_attention_heads = 64
- num_key_value_heads = 8
- **Groups: 64 / 8 = 8 heads per group (true GQA)** ✅

---

### 💡 Why Different Configurations?

**Smaller models (7B, 13B):**
- Use MQA (g=1) for maximum speed
- Can afford some quality trade-off
- Memory is less critical

**Larger models (70B):**
- Use GQA (g=8) for quality preservation
- Memory bottleneck is severe
- Need balance between speed and quality

---

### 📈 Memory Savings for Llama 2 70B

**With standard MHA (hypothetical):**
```
Per layer: 4 × 16 × 4,096 × 8,192 = 2 GB
Total: 2 GB × 80 layers = 160 GB KV cache
```

**With GQA (g=8):**
```
Per layer: 2 GB / 8 = 256 MB
Total: 256 MB × 80 layers = 20 GB KV cache
```

**Savings: 160 GB → 20 GB = 140 GB freed!** 🎉

**This enables:**
- 2x longer context (2K → 4K tokens)
- OR 2-4x larger batch size
- OR combination of both

---

## Step 7 — The Quality-Speed Spectrum

Understanding the trade-offs across the spectrum.

---

### 📊 Comparison Table

| Method | g (groups) | KV matrices | Memory | Speed | Quality |
|:-------|:-----------|:------------|:-------|:------|:--------|
| **MHA** | h (64) | 128 | 100% | 1.0x | 47.2 |
| **GQA-16** | 16 | 32 | 25% | ~3.5x | ~47.0 |
| **GQA-8** | 8 | 16 | 12.5% | ~5.5x | 47.1 |
| **GQA-4** | 4 | 8 | 6.25% | ~7.0x | 46.9 |
| **MQA** | 1 | 2 | 1.56% | ~8.0x | 46.6 |

*(Quality scores from T5-XXL experiments on summarization tasks)*

---

### 🎯 The Sweet Spot

**Research findings:**
- **GQA-8** achieves 98%+ of MHA quality at 5-6x speed improvement
- Diminishing returns beyond g=8 for most models
- Below g=4, quality degradation becomes noticeable

**Common choices:**
- **g=8:** Standard for large models (Llama 2 70B, Mistral 7B)
- **g=1 (MQA):** For smaller models prioritizing speed (Llama 2 7B/13B)

---

## Step 8 — Mathematical Formulation

The complete GQA attention mechanism.

---

### 🔢 Standard Multi-Head Attention

```
For each head i ∈ [1, h]:
    Q_i = X W_q^i    (d_model → d_head)
    K_i = X W_k^i    (d_model → d_head)
    V_i = X W_v^i    (d_model → d_head)
    
    head_i = Attention(Q_i, K_i, V_i)
           = softmax(Q_i K_i^T / √d_head) V_i

MultiHead = Concat(head_1, ..., head_h) W^O
```

**Parameters:**
- h × W_q matrices
- h × W_k matrices  
- h × W_v matrices

---

### 🔢 Grouped-Query Attention

```
Divide h query heads into g groups

For each group j ∈ [1, g]:
    # One shared K and V per group
    K_j = X W_k^j    (d_model → d_head)
    V_j = X W_v^j    (d_model → d_head)
    
    For each head i in group j:
        Q_i = X W_q^i    (d_model → d_head)
        
        head_i = Attention(Q_i, K_j, V_j)
               = softmax(Q_i K_j^T / √d_head) V_j

MultiHead = Concat(head_1, ..., head_h) W^O
```

**Parameters:**
- h × W_q matrices (same as MHA)
- g × W_k matrices (reduced!)
- g × W_v matrices (reduced!)

**Reduction factor: h / g**

---

### 📐 Dimensions

**For Llama 2 70B:**
```
d_model = 8,192
h = 64 query heads
g = 8 groups
d_head = d_model / h = 128

Each Q_i: (seq_len × 128)
Each K_j: (seq_len × 128)  — only 8 of these!
Each V_j: (seq_len × 128)  — only 8 of these!
```

---

## Step 9 — Implementation Details

How GQA is implemented in practice.

---

### 🚀 Key-Value Head Repetition

Within a group, we "repeat" the shared K and V to match the number of query heads:

```python
# Llama 2 70B example
n_heads = 64        # Query heads
n_kv_heads = 8      # KV heads (groups)
n_rep = n_heads // n_kv_heads  # 64 / 8 = 8

# Compute K and V (only 8 projections)
keys = x @ W_k      # (batch, seq_len, 8, head_dim)
values = x @ W_v    # (batch, seq_len, 8, head_dim)

# Repeat to match query heads
def repeat_kv(x, n_rep):
    """
    Repeat KV heads n_rep times to match query heads
    (batch, seq_len, n_kv_heads, head_dim) 
    → (batch, seq_len, n_heads, head_dim)
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    return x[:, :, :, None, :].expand(
        bs, slen, n_kv_heads, n_rep, head_dim
    ).reshape(bs, slen, n_kv_heads * n_rep, head_dim)

keys = repeat_kv(keys, n_rep)     # Now (batch, seq_len, 64, head_dim)
values = repeat_kv(values, n_rep) # Now (batch, seq_len, 64, head_dim)

# Compute queries (64 projections)
queries = x @ W_q   # (batch, seq_len, 64, head_dim)

# Now we can do standard batched attention
scores = queries @ keys.transpose(-2, -1) / sqrt(head_dim)
attn = softmax(scores)
output = attn @ values
```

---

### 💡 Why Repetition?

**Memory efficiency:**
- Store only 8 KV pairs in cache
- Repeat on-the-fly during computation (negligible cost)
- Maintains compatibility with standard attention implementation

**Computation:**
- Repetition is just indexing (free in terms of FLOPs)
- Attention computation is the same as MHA

---

## Step 10 — Training GQA from Scratch vs. Uptraining

Two approaches to obtain a GQA model.

---

### 🏗️ Approach 1: Train from Scratch

**Method:**
- Initialize GQA architecture directly
- Train with g groups from the beginning

**Pros:**
- ✅ Optimal for the GQA architecture
- ✅ No conversion needed

**Cons:**
- ❌ Expensive (full pre-training)
- ❌ Can't leverage existing MHA checkpoints

**Used by:** Llama 2, Llama 3, Mistral

---

### 🔄 Approach 2: Uptrain from MHA Checkpoint

**Method (from GQA paper):**

1. **Start with trained MHA model** (e.g., T5-XXL)

2. **Convert MHA → GQA:**
   - Keep all Q projections (64 heads)
   - Merge K and V by mean-pooling within groups
   
   Example for Group 1:
   ```
   K_group1 = mean(K_head1, K_head2, ..., K_head8)
   V_group1 = mean(V_head1, V_head2, ..., V_head8)
   ```

3. **Uptrain for 5% of original training steps:**
   - Let model adapt to shared KV
   - Usually 5-10% of original training is sufficient

**Research results:**
- After 5% uptraining: GQA-8 reaches 99% of original MHA quality
- Much cheaper than training from scratch

---

## Step 11 — Real-World Impact

How GQA enables production deployment.

---

### 🌍 Production Benefits

**Llama 2 70B with GQA:**

**Before (hypothetical MHA):**
- KV cache: 160 GB
- Can serve: 4 users @ 4K context on A100 (80GB)
- Context limit: 2K tokens per user

**After (with GQA-8):**
- KV cache: 20 GB  
- Can serve: 32 users @ 4K context on A100
- OR: 8 users @ 16K context
- **8x improvement in throughput or 4x longer context!** 🎉

---

### 📊 Inference Benchmarks

**Mistral 7B (GQA-8) vs. Llama 2 7B (MQA):**

On single A10G GPU with increasing workload:

| Metric | Llama 2 7B (MQA) | Mistral 7B (GQA-8) |
|:-------|:-----------------|:-------------------|
| Light load TPOT | 45 ms | 47 ms |
| Medium load TPOT | 120 ms | 95 ms |
| Heavy load TPOT | Failed (OOM) | 180 ms |

**Key finding:** GQA maintains quality advantage at scale while MQA struggles under load.

---

### 💰 Cost Implications

**Serving 1M requests/day:**

**MHA (baseline):**
- Need: 100 A100 GPUs
- Cost: $10M (hardware) + $500K/year (power)

**GQA (8 groups):**
- Need: 15 A100 GPUs
- Cost: $1.5M (hardware) + $75K/year (power)
- **85% cost reduction!**

---

## Step 12 — Choosing the Number of Groups

Guidelines for selecting g.

---

### 🎯 Decision Framework

**For small models (<10B parameters):**
- Consider **MQA (g=1)** for maximum speed
- Quality loss is tolerable
- Example: Llama 2 7B, Falcon 7B

**For medium models (10B-30B):**
- Use **GQA-4 or GQA-8**
- Balance quality and efficiency
- Example: Mistral 7B uses GQA-8

**For large models (>30B parameters):**
- Use **GQA-8**
- Quality is critical
- Memory bottleneck is severe
- Example: Llama 2 70B, Falcon 40B/175B

---

### 🔬 Empirical Rule

**Sweet spot:** `g ≈ h / 8`

Examples:
- h=32 → g=4 (GQA-4)
- h=64 → g=8 (GQA-8)  
- h=96 → g=12 (GQA-12)

**Rationale:**
- Provides 8x memory reduction
- Maintains >98% of MHA quality
- Beyond 8x, diminishing returns

---

## Summary

**Grouped-Query Attention solves the KV cache memory bottleneck:**

1. **Problem:** MHA's KV cache dominates memory in inference (160 GB for Llama 2 70B)
2. **Insight:** Diversity comes from different Queries, not separate K and V
3. **MQA:** All heads share one K, V — 64x reduction but quality loss
4. **GQA:** Groups of heads share K, V — 8x reduction, minimal quality loss
5. **Llama 2 70B:** Uses 8 groups (64 heads → 8 KV pairs)
6. **Memory savings:** 160 GB → 20 GB enables 8x throughput or 4x longer context
7. **Quality:** GQA-8 maintains 98-99% of MHA quality
8. **Production impact:** 85% cost reduction for serving at scale

**Result:** Modern LLMs can serve longer contexts to more users on the same hardware, making deployment economically feasible.

---

## Key Formulas

**KV Cache Size (per layer):**
```
MHA:  4 × batch × seq_len × d_model bytes
GQA:  4 × batch × seq_len × d_model / (h/g) bytes
      = (4 × batch × seq_len × d_model × g) / h bytes

Reduction factor = h / g
```

**For Llama 2 70B (g=8, h=64):**
```
Reduction = 64 / 8 = 8x smaller KV cache
```

**Memory per layer:**
```
MHA:  2 GB per layer
GQA-8: 256 MB per layer
Total savings: 1.75 GB per layer × 80 layers = 140 GB saved
```

---

**This is why Llama 2, Llama 3, Mistral, and many modern LLMs use GQA — it makes deployment possible.**