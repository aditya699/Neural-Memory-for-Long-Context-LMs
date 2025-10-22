# Understanding Multi-Head Attention

A comprehensive guide to why and how transformers use multiple attention heads.

---

## Step 1 — The Limitation: One Attention Head, One Perspective

In self-attention, we learned that tokens communicate by comparing Queries to Keys:

```
Attention(Q, K, V) = softmax(Q K^T / √dₖ) V
```

This works beautifully — but there's a hidden constraint: **one attention mechanism captures one type of relationship**.

---

### 🧩 The Problem with Single-Head Attention

Consider the sentence:

> "The animal didn't cross the street because it was too tired."

**Question:** What does "it" refer to?

When computing attention for "it", the model needs to:
- Recognize grammatical structure (pronoun resolution)
- Understand semantic meaning (what can be "tired"?)
- Track positional relationships (nearby vs. distant words)

**Single attention head:** Must blend ALL these reasoning patterns into ONE set of attention weights.

**Result:** The model is forced to **average** different types of relationships, potentially diluting important signals.

---

### 💡 The Core Insight

Language is **multi-dimensional**. A token can relate to another token in many different ways simultaneously:

- **Syntactic** (subject-verb-object)
- **Semantic** (meaning similarity)
- **Positional** (next-word, nearby context)
- **Co-reference** (pronouns → nouns)
- **Causal** (cause-effect relationships)

**If we only have one attention head, we get one blended view of all these relationships.**

What if we could look at the sentence through **multiple lenses at once**?

---

## Step 2 — The Solution: Multiple Parallel Attention Heads

Instead of one attention mechanism, use **h** independent attention heads, each learning different patterns.

---

### 🔍 How It Works

Given input embeddings with dimension `d_model = 512`:

**Single-Head Attention (what we had before):**
```
Q = X W_q  (512 → 512)
K = X W_k  (512 → 512)
V = X W_v  (512 → 512)
```

**Multi-Head Attention (what we do now):**

Split the model dimension across `h = 8` heads:
```
d_k = d_model / h = 512 / 8 = 64 dimensions per head
```

Each head `i` gets its own projection matrices:
```
Q_i = X W_q^i  (512 → 64)
K_i = X W_k^i  (512 → 64)
V_i = X W_v^i  (512 → 64)
```

---

### 🧠 The Key Idea

**Each head sees a different 64-dimensional slice of the 512-dimensional embedding space.**

Think of it like:
- You have a photo (the embedding)
- Head 1 only sees the **red channel**
- Head 2 only sees the **green channel**
- Head 3 only sees the **blue channel**
- ...and so on

Same input, but each head is looking at **different aspects** of it.

---

### Example with Actual Numbers

**Token "cat" has embedding:** `[0.2, 0.5, -0.3, 0.8, ..., 0.1, -0.4]` (512 numbers)

**Head 1** projection sees: dimensions [0-63]
```
[0.2, 0.5, -0.3, 0.8, ...]  (first 64 numbers)
```

**Head 2** projection sees: dimensions [64-127]
```
[..., 0.1, -0.4, ...]  (next 64 numbers)
```

**Head 3** projection sees: dimensions [128-191]
```
[..., ...]  (dims 128-191)
```

Each head **cannot access** the dimensions assigned to other heads. This forces specialization.

---

## Step 3 — Parallel Attention Computation

Now each of the 8 heads independently computes attention:

---

### For Head 1:

```
Q₁ = X W_q^1  (batch × seq_len × 64)
K₁ = X W_k^1  (batch × seq_len × 64)
V₁ = X W_v^1  (batch × seq_len × 64)

Scores₁ = Q₁ K₁^T / √64
Attention₁ = softmax(Scores₁)
Output₁ = Attention₁ V₁  (batch × seq_len × 64)
```

### For Head 2:

```
Q₂ = X W_q^2
K₂ = X W_k^2
V₂ = X W_v^2

Output₂ = softmax(Q₂ K₂^T / √64) V₂  (batch × seq_len × 64)
```

...and so on for all 8 heads.

---

### 🧩 What Makes Each Head Different?

**Two factors force heads to specialize:**

1. **Different weight initialization** — each head starts with random weights
2. **Limited dimensions** — each head only has 64 dims to work with (not 512)

Even if Head 1 wanted to learn the same patterns as Head 2, it **can't** because it's looking at a completely different slice of the embedding space.

---

### 💡 Concrete Example

**Sentence:** "The cat sat on the mat because it was soft."

**Head 1** (dimensions 0-63):
- These dimensions might encode: *grammatical structure*
- Learns strong attention: "it" → "mat" (pronoun resolution)

**Head 2** (dimensions 64-127):
- These dimensions might encode: *semantic properties*
- Learns strong attention: "soft" → "mat" (adjective-noun)

**Head 3** (dimensions 128-191):
- These dimensions might encode: *action relationships*
- Learns strong attention: "sat" → "cat" (verb-subject)

Each head discovers different patterns because they're looking at **different information**.

---

## Step 4 — Head Specialization: What Different Heads Learn

Research on trained transformers (e.g., BERT, GPT-2) reveals that heads spontaneously specialize:

---

### 🔬 Observed Specialization Patterns

**Example from a trained 12-layer, 12-head transformer:**

| Head | Specialization | Example Pattern |
|:-----|:---------------|:----------------|
| Head 3 | Pronoun resolution | "their" → "students" |
| Head 5 | Subject-verb syntax | "teacher" → "gave" |
| Head 7 | Local context | Strong attention to ±1 position |
| Head 9 | Semantic similarity | "car" → "vehicle" |
| Head 11 | Delimiter tokens | Strong attention to punctuation |

---

### 🧠 Why Does This Happen?

**The model is NOT told what each head should learn.** Specialization emerges naturally through:

1. **Gradient descent** — during training, each head adjusts to minimize loss
2. **Distributed optimization** — different heads find different solutions
3. **Limited capacity** — each head (64 dims) can't capture everything, so they divide labor

It's like evolution: given limited resources, organisms specialize into different niches.

---

### Real Example: Pronoun Resolution

**Sentence:** "The teacher gave the students their homework."

**Head 3's attention pattern when processing "their":**

| Source Token | Attention Weight | Interpretation |
|:-------------|:-----------------|:---------------|
| The | 0.05 | Ignore |
| teacher | 0.12 | Possible referent |
| gave | 0.08 | Verb (context) |
| the | 0.03 | Ignore |
| students | **0.65** | **Target!** |
| their | 0.07 | Self-attention |

**Head 3 learned:** pronouns should attend strongly to their referents.

**Meanwhile, Head 7** might completely ignore this pattern and focus on local syntax instead.

---

## Step 5 — Concatenation: Combining Multiple Perspectives

After all 8 heads compute their outputs independently, we have:

```
Output₁ (batch × seq_len × 64)
Output₂ (batch × seq_len × 64)
...
Output₈ (batch × seq_len × 64)
```

Now we need to combine these 8 different perspectives.

---

### 🔗 Simple Concatenation

Just stack them side-by-side:

```
Concat = [Output₁ ; Output₂ ; ... ; Output₈]
```

**Result:** `(batch × seq_len × 512)`

We're back to the full `d_model` dimension.

---

### 🧩 Example with Small Numbers

**Head 1 output for token "sat":** `[0.2, 0.5]` (2 dims for simplicity)  
**Head 2 output for token "sat":** `[0.8, 0.1]`  
**Head 3 output for token "sat":** `[0.3, 0.9]`

**Concatenated output:**
```
[0.2, 0.5, 0.8, 0.1, 0.3, 0.9]  (now 6 dims)
```

In reality: 8 heads × 64 dims = 512 dimensions after concatenation.

---

### ⚠️ Problem: Heads Are Still Independent

Right now, the concatenated vector is just **heads stacked next to each other**:

```
[Head1: 64 dims | Head2: 64 dims | ... | Head8: 64 dims]
```

They haven't learned to **work together** yet. Head 1 doesn't know what Head 5 found.

**Solution:** We need one more transformation.

---

## Step 6 — Output Projection: Learning to Combine Heads

After concatenation, we apply a learned linear transformation:

```
MultiHead(Q, K, V) = Concat(head₁, ..., head_h) W^O
```

where `W^O` is a `(512 × 512)` matrix.

---

### 🧠 What Does W^O Learn?

`W^O` learns: **"How should I weight and combine these 8 different perspectives?"**

---

### 💡 Intuition

Think of 8 experts giving advice:

- Expert 1 (Head 1) says: "Focus on syntax"
- Expert 2 (Head 2) says: "Focus on semantics"
- Expert 3 (Head 3) says: "Focus on pronouns"
- ...

`W^O` learns the optimal weighting: *"For this task, I need 40% Expert 1, 30% Expert 2, 20% Expert 3, 10% others."*

---

### 🔍 Example

**For predicting the next word after "The cat sat on the..."**

Through training, `W^O` might learn:
- **Head 2** (semantics) is crucial here → weight it **heavily**
- **Head 7** (local context) helps a bit → use **some** of it
- **Head 4** found nothing useful → **ignore** it

The final output mixes information from all heads according to these learned weights.

---

### Matrix Math

**Input to W^O:**  
Concatenated heads: `(batch × seq_len × 512)`

**W^O matrix:**  
`(512 × 512)`

**Output:**  
`(batch × seq_len × 512) × (512 × 512) = (batch × seq_len × 512)`

Same dimension in and out — but now information from all heads is **blended**.

---

## Step 7 — Complete Multi-Head Attention Formula

Putting it all together:

```
MultiHead(Q, K, V) = Concat(head₁, ..., head_h) W^O

where head_i = Attention(Q W_q^i, K W_k^i, V W_v^i)
```

**Step-by-step:**

1. **Project** input X into Q, K, V for each head (8 separate projections)
2. **Compute** attention independently for each head
3. **Concatenate** all head outputs side-by-side
4. **Project** through W^O to blend heads together

---

### 🧩 Visual Summary

```
Input X (seq_len × 512)
    ↓
Split into 8 projections
    ↓
┌─────────┬─────────┬─────────┬─────────┐
│ Head 1  │ Head 2  │ Head 3  │  ...    │
│ (64-dim)│ (64-dim)│ (64-dim)│ (64-dim)│
│         │         │         │         │
│ Q₁ K₁ V₁│ Q₂ K₂ V₂│ Q₃ K₃ V₃│  ...    │
│    ↓    │    ↓    │    ↓    │    ↓    │
│ Attn 1  │ Attn 2  │ Attn 3  │  ...    │
│ Out: 64 │ Out: 64 │ Out: 64 │ Out: 64 │
└─────────┴─────────┴─────────┴─────────┘
    ↓
Concatenate [64|64|64|...|64] → 512 dims
    ↓
W^O Projection (512 × 512)
    ↓
Output (seq_len × 512)
```

---

## Step 8 — The Trade-off: Number of Heads vs. Dimensions per Head

Given `d_model = 512` total dimensions, we must choose how many heads to use.

---

### 🔀 The Design Choice

**More heads (16 heads × 32 dims each):**
- ✅ More diverse perspectives (16 different "views")
- ✅ Better parallelization on GPUs
- ❌ Each head is "weaker" (only 32 dims)
- ❌ Each head captures simpler patterns

**Fewer heads (4 heads × 128 dims each):**
- ✅ Each head is "stronger" (128 dims = more expressive)
- ✅ Can capture more complex relationships
- ❌ Less diversity (only 4 different views)
- ❌ Less parallelism

---

### 🔬 Empirical Findings

**Common configurations in practice:**

| Model | d_model | num_heads | d_k (per head) |
|:------|:--------|:----------|:---------------|
| BERT-base | 768 | 12 | 64 |
| GPT-2 | 768 | 12 | 64 |
| GPT-3 | 12,288 | 96 | 128 |
| Llama 2 (7B) | 4,096 | 32 | 128 |
| Llama 2 (70B) | 8,192 | 64 | 128 |

**Pattern:** Most models use **8-16 heads** for smaller models, **32-96 heads** for larger models.

**Sweet spot:** `d_k = 64` or `d_k = 128` per head.

---

### ⚠️ Extreme Cases

**Too many heads (512 heads × 1 dim each):**
- Each head is just **one number**
- Cannot capture any complex relationships
- Dot product Q·K is just multiplying two scalars
- **Useless!**

**Too few heads (1 head × 512 dims):**
- Back to single-head attention
- No diversity in perspectives
- Forces one attention pattern to do everything

**Goldilocks zone:** 8-32 heads for most models.

---

## Step 9 — Implementation: Batched Computation

In practice, we don't compute 8 heads separately. We use clever matrix operations.

---

### 🚀 Efficient Implementation

**Naive approach (slow):**
```python
for i in range(8):
    Q_i = X @ W_q[i]  # 8 separate matrix multiplies
    K_i = X @ W_k[i]
    V_i = X @ W_v[i]
    outputs[i] = attention(Q_i, K_i, V_i)
```

**Smart approach (fast):**
```python
# Single large projection
Q_all = X @ W_q_combined  # (512 × 512) — one operation
K_all = X @ W_k_combined
V_all = X @ W_v_combined

# Reshape to separate heads
Q_all = Q_all.view(batch, seq_len, 8, 64)
K_all = K_all.view(batch, seq_len, 8, 64)
V_all = V_all.view(batch, seq_len, 8, 64)

# Transpose to (batch, 8, seq_len, 64) for parallel computation
Q_all = Q_all.transpose(1, 2)
K_all = K_all.transpose(1, 2)
V_all = V_all.transpose(1, 2)

# Compute all heads in parallel (single batched operation)
outputs = batched_attention(Q_all, K_all, V_all)
```

---

### 💡 Why Is This Faster?

**GPUs are optimized for:**
- Large matrix operations (better utilization)
- Parallel computation (all heads at once)

**One big (512 × 512) matrix multiply is MUCH faster than 8 separate (512 × 64) multiplies** — even though the total FLOPs are the same.

**GPU occupancy:** Small operations leave GPU cores idle. Large operations saturate the hardware.

---

## Step 10 — Multi-Head Attention in Context

Multi-head attention is just one component of a transformer layer. Here's where it fits:

---

### Complete Transformer Block

```
Input: x (embeddings from previous layer)
    ↓
┌──────────────────────────────────────┐
│  1. Multi-Head Attention             │
│     - Split into 8 heads             │
│     - Parallel attention computation │
│     - Concatenate + project          │
│     Output: Z (seq_len × 512)        │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  2. Add & Normalize (Residual)       │
│     x' = LayerNorm(x + Z)            │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  3. Feed-Forward Network             │
│     ffn_out = FFN(x')                │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  4. Add & Normalize (Residual)       │
│     output = LayerNorm(x' + ffn_out) │
└──────────────────────────────────────┘
    ↓
Output: Ready for next layer
```

---

### 🔄 Stacking Layers

Modern LLMs stack 32-96 of these blocks:

**Layer 1-10:** Learn basic patterns
- Head specialization: syntax, local context

**Layer 11-20:** Build higher-level features
- Head specialization: semantics, entities

**Layer 21-32:** Abstract reasoning
- Head specialization: logical relationships, world knowledge

Each layer refines representations through multiple attention perspectives.

---

## 🧠 Why Multi-Head Attention Works

### Cognitive Science Parallel

Human attention is **multi-modal**:
- **Spatial attention** (where to look)
- **Feature-based attention** (what properties matter)
- **Object-based attention** (track specific entities)

Multi-head attention implements this computationally:
- Different heads = different attentional channels
- Each head optimizes for different features
- Combined via learned weighting (W^O)

---

### Information Theory View

**Single head:** Limited channel capacity, must multiplex all signals  
**Multi-head:** Parallel channels, each optimized for different information types

It's like having multiple microphones at different positions vs. one microphone trying to hear everything.

---

## Summary

**Multi-Head Attention solves the single-perspective limitation:**

1. **Split** d_model dimensions across h heads (e.g., 512 → 8 × 64)
2. **Force specialization** by giving each head limited dimensions
3. **Parallel computation** — all heads run independently
4. **Natural emergence** of different attention patterns (syntax, semantics, etc.)
5. **Concatenate** head outputs back to d_model size
6. **Project** through W^O to learn optimal head combination
7. **Trade-off** between number of heads and expressiveness per head
8. **Efficient implementation** via batched matrix operations

**Result:** The model can simultaneously attend to multiple types of relationships, capturing the multi-dimensional nature of language.

---

## Key Formulas

**Complete Multi-Head Attention:**
```
MultiHead(Q, K, V) = Concat(head₁, ..., head_h) W^O

where head_i = Attention(X W_q^i, X W_k^i, X W_v^i)
            = softmax((X W_q^i)(X W_k^i)^T / √d_k) (X W_v^i)
```

**Dimensions:**
- Input: `(seq_len × d_model)`
- Each head projection: `(d_model × d_k)` where `d_k = d_model / h`
- Each head output: `(seq_len × d_k)`
- Concatenated: `(seq_len × d_model)`
- Final projection W^O: `(d_model × d_model)`
- Final output: `(seq_len × d_model)`

---

**This is the architecture that powers modern LLMs like GPT, Llama, and Claude.**
