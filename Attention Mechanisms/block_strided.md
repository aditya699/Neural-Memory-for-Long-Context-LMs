# Understanding Block-Sparse Attention

A guide to how block-sparse attention works in models like BigBird and Longformer.

---

## The Problem: Need Both Local and Global Context

**Full attention:** O(n²) - too expensive

**Sliding window:** Linear O(n·w) but information dilutes over distance

**Strided attention:** Can reach far but loses all local context (big gaps)

**Question:** Can we have both local detail AND long-range access without quadratic cost?

**Answer:** Yes! **Block-Sparse Attention**

---

## The Core Idea: Think in Blocks, Not Tokens

Instead of deciding "attend to token i or skip it", we decide "attend to entire block or skip it".

**Example: 16 tokens, block_size=4**

```
Block 0: [T0,  T1,  T2,  T3]
Block 1: [T4,  T5,  T6,  T7]
Block 2: [T8,  T9,  T10, T11]
Block 3: [T12, T13, T14, T15]
```

**Token T14 (in Block 3) with block-sparse attention might attend to:**
```
✓ All of Block 0  [T0, T1, T2, T3]     ← Global context (document start)
✗ Skip Block 1    [T4, T5, T6, T7]     
✓ All of Block 2  [T8, T9, T10, T11]   ← Previous block (recent context)
✓ All of Block 3  [T12, T13, T14, T15] ← Own block (local context)
```

**Key:** Attend to or skip ENTIRE BLOCKS at once, not individual tokens.

---

## Why Blocks Beat Individual Token Selection?

**Computational efficiency!**

**Individual token sparse attention (strided):**
```
For each token, compute: attend or skip?
Irregular access patterns
Hard to optimize on GPUs
```

**Block-sparse attention:**
```
For each block, decide: attend or skip?
Regular block operations
Excellent hardware utilization
Matrix operations stay efficient
```

**Blocks = better hardware performance while maintaining flexibility**

---

## Common Block-Sparse Patterns

### Pattern 1: Local + Global

**Most common pattern in production models**

```
For any token, attend to:
1. Own block (local context)
2. First block (global - document start, special tokens)
3. Previous block (recent context)

Example (block_size=64, position 200):
  ✓ Block 0   [T0-T63]      ← Global
  ✗ Blocks 1-2 [T64-T191]   ← Skip
  ✓ Block 2   [T128-T191]   ← Previous  
  ✓ Block 3   [T192-T255]   ← Own block (contains T200)
```

**Used in:** Longformer, BigBird (partial)

---

### Pattern 2: Local + Strided Blocks

**Combine local detail with periodic long-range**

```
Attend to:
1. Own block
2. Previous block
3. Every k-th block (strided)

Example (block_size=64, stride=2):
  ✓ Block 0   [T0-T63]      ← Stride
  ✗ Block 1   [T64-T127]    ← Skip
  ✓ Block 2   [T128-T191]   ← Stride (and previous)
  ✓ Block 3   [T192-T255]   ← Own block
```

**Used in:** ETC (Extended Transformer Construction)

---

### Pattern 3: Local + Global + Random

**BigBird's famous approach**

```
Attend to:
1. Own block (local)
2. Previous few blocks (sliding window in blocks)
3. First block (global tokens like [CLS])
4. r random blocks (for long-range)

Example (r=2 random blocks):
  ✓ Block 0   ← Global
  ✓ Block 1   ← Random selection
  ✗ Blocks 2-4 ← Skip
  ✓ Block 5   ← Random selection
  ✓ Block 6   ← Previous
  ✓ Block 7   ← Own block
```

**Used in:** BigBird

**Benefit:** Random blocks provide probabilistic coverage of long sequence

---

## Complexity Analysis

**Full Attention:**
```
Computation: O(n²·d)
Memory: O(n²)
```

**Block-Sparse (attending to b blocks, block_size=B):**
```
Computation: O(n·b·B·d)
Memory: O(n·b·B)

Where typically: b << n/B (sparse!)
```

**Example: n=4096, B=64, attend to 8 blocks**
```
Full: 4096² = 16.7M attention operations
Block-sparse: 4096 × 8 × 64 = 2.1M operations
Speedup: ~8×
```

---

## Comparison: Strided vs Block-Sparse

### Strided Attention (stride=8):
```
Position 64 attends to: [0, 8, 16, 24, 32, 40, 48, 56, 64]

Problems:
✗ Loses ALL context between strides
✗ Gap between T8 and T16 (7 tokens missed!)
✗ If important word at T12, it's invisible
```

### Block-Sparse (block_size=8, attend to blocks 0 and 7):
```
Position 64 attends to: [0,1,2,3,4,5,6,7] + [56,57,58,59,60,61,62,63,64]

Benefits:
✓ FULL context within each attended block
✓ No gaps inside blocks
✓ If block 0 is important, get ALL 8 tokens from it
✓ Local block gives complete recent context
```

**Key difference:** Strided picks individual tokens (gaps), Block-sparse picks neighborhoods (complete)

---

## The Architecture: What Changes?

**Same as standard attention, but with a block-based mask**

### Standard Attention Mask (Causal):
```python
mask = torch.tril(torch.ones(n, n))
# Every position sees all previous
```

### Block-Sparse Mask:
```python
def block_sparse_mask(seq_len, block_size, pattern='local_global'):
    n_blocks = seq_len // block_size
    mask = torch.zeros(seq_len, seq_len)
    
    for block_i in range(n_blocks):
        start_i = block_i * block_size
        end_i = start_i + block_size
        
        # Own block (local)
        mask[start_i:end_i, start_i:end_i] = 1
        
        # First block (global)
        mask[start_i:end_i, 0:block_size] = 1
        
        # Previous block
        if block_i > 0:
            prev_start = (block_i-1) * block_size
            prev_end = prev_start + block_size
            mask[start_i:end_i, prev_start:prev_end] = 1
    
    return mask

# Example with seq_len=12, block_size=3:
# Block 0: [0,1,2], Block 1: [3,4,5], Block 2: [6,7,8], Block 3: [9,10,11]
#
#     0 1 2 3 4 5 6 7 8 9 10 11
# 0  [1 1 1 0 0 0 0 0 0 0 0  0]  ← Block 0 sees itself
# 1  [1 1 1 0 0 0 0 0 0 0 0  0]
# 2  [1 1 1 0 0 0 0 0 0 0 0  0]
# 3  [1 1 1 1 1 1 0 0 0 0 0  0]  ← Block 1 sees [Block 0, itself]
# 4  [1 1 1 1 1 1 0 0 0 0 0  0]
# 5  [1 1 1 1 1 1 0 0 0 0 0  0]
# 6  [1 1 1 1 1 1 1 1 1 0 0  0]  ← Block 2 sees [Block 0, 1, itself]
# 7  [1 1 1 1 1 1 1 1 1 0 0  0]
# 8  [1 1 1 1 1 1 1 1 1 0 0  0]
# 9  [1 1 1 0 0 0 1 1 1 1 1  1]  ← Block 3 sees [Block 0, 2, itself]
# 10 [1 1 1 0 0 0 1 1 1 1 1  1]
# 11 [1 1 1 0 0 0 1 1 1 1 1  1]
```

Everything else (Q, K, V, softmax) is identical to standard attention.

---

## Information Flow with Block-Sparse

**Unlike sliding window, block-sparse can "jump" directly to far blocks!**

**Example: Document with 4 blocks**

```
Block 0: "Customer ID: 12345" (important global info)
Block 1: [middle content]
Block 2: [middle content]  
Block 3: "What is the customer ID?" (query at end)

With Local + Global pattern:
Block 3 attends to [Block 0, Block 2, Block 3]
→ Directly accesses Block 0!
→ No multi-hop dilution
→ Perfect recall of customer ID
```

**Benefit over sliding window:** Direct access, no information dilution through layers!

---

## Block-Sparse + Layer Stacking

**Even better: Combine blocks with multiple layers**

**Layer 1:**
```
Block 3 sees: [Block 0, Block 2, Block 3] directly
```

**Layer 2:**
```
Block 3 sees: [Block 0', Block 2', Block 3']
But Block 2' contains info from [Block 0, Block 1, Block 2] (from Layer 1)
So Block 3 now has info from Block 1 too!
```

**Result:** Even sparser patterns work well when stacked!

---

## Real-World Usage

### BigBird (Google)

**Architecture:**
```
Block size: 64
Pattern: Local + Global + Random
- Local: Own block + previous 3 blocks
- Global: First block (special tokens)
- Random: 3 random blocks per position
```

**Use case:** Long documents (4k-16k tokens)

**Performance:** ~8× faster than full attention with minimal quality loss

---

### Longformer

**Architecture:**
```
Block size: Variable (typically 512)
Pattern: Local sliding + Global tokens
- Local: Sliding window implemented as blocks
- Global: Specific tokens attend to everything (e.g., [CLS])
```

**Use case:** Document classification, QA on long documents

**Max length:** 4096 tokens standard, up to 16k

---

### ETC (Extended Transformer Construction)

**Architecture:**
```
Pattern: Hierarchical blocks
- Local blocks within segments
- Strided blocks across segments
- Special global memory tokens
```

**Use case:** Very long structured documents

---

## When to Use Block-Sparse

**✅ Great for:**
- Long documents with structure (sections, paragraphs)
- Document classification/summarization
- Need both local detail AND global context
- Scientific papers, legal documents, books
- Tasks where certain parts (start, end) are always important

**✅ Better than sliding window when:**
- Need direct access to document start
- Important info is at specific positions (not just recent)
- Can't afford many layers for information relay
- Document has clear structure (sections)

**✅ Better than strided when:**
- Local context is important (most NLP tasks)
- Can't afford gaps in understanding
- Need complete neighborhoods, not individual tokens

**❌ Not ideal for:**
- Very short sequences (<1k tokens) - use full attention
- Streaming/incremental generation - sliding window better
- When all context is equally important - full attention better
- Simple implementations - sliding window is simpler

---

## The Trade-offs

**Full Attention:**
```
✅ Perfect quality
✅ Sees everything
❌ O(n²) cost
❌ Max ~8k tokens
```

**Sliding Window:**
```
✅ O(n·w) - truly linear
✅ Simplest sparse pattern
✅ Great for autoregressive generation
❌ Indirect access (multi-hop)
❌ Information dilutes over distance
```

**Block-Sparse:**
```
✅ Direct long-range access
✅ No information dilution
✅ Full context in attended blocks
✅ Flexible patterns
✅ Good for structured documents
❌ Not truly linear (depends on pattern)
❌ More complex implementation
❌ Pattern design matters
❌ Less widely adopted
```

**Strided:**
```
✅ Simple pattern
✅ Can reach far
❌ Loses all local context
❌ Big gaps between attended tokens
❌ Rarely used alone
```

---

## Hybrid Patterns in Practice

**Most production models combine multiple strategies:**

### BigBird Approach:
```python
attention_pattern = {
    'local': sliding_blocks(window=3),      # 3 blocks = 192 tokens
    'global': first_block(),                # Always attend to start
    'random': random_blocks(num=3)          # 3 random blocks
}
```

### Longformer Approach:
```python
attention_pattern = {
    'regular_tokens': sliding_window(w=512),
    'global_tokens': full_attention()       # [CLS] sees everything
}
```

**Key insight:** Hybrid = best of all worlds!

---

## Code Example

```python
class BlockSparseAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def create_block_sparse_mask(self, seq_len):
        """Create Local + Global block-sparse mask"""
        n_blocks = seq_len // self.block_size
        mask = torch.zeros(seq_len, seq_len, device='cuda')
        
        for block_i in range(n_blocks):
            start_i = block_i * self.block_size
            end_i = start_i + self.block_size
            
            # Attend to own block (local)
            mask[start_i:end_i, start_i:end_i] = 1
            
            # Attend to first block (global)
            mask[start_i:end_i, 0:self.block_size] = 1
            
            # Attend to previous block
            if block_i > 0:
                prev_start = (block_i - 1) * self.block_size
                prev_end = prev_start + self.block_size
                mask[start_i:end_i, prev_start:prev_end] = 1
        
        return mask
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # THE KEY CHANGE: Block-sparse mask
        block_mask = self.create_block_sparse_mask(seq_len)
        mask_to_block = (block_mask == 0).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask_to_block, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output
```

---

## Summary

**Block-sparse attention provides structured sparsity:**

1. Think in blocks (neighborhoods), not individual tokens
2. Attend to complete blocks (no gaps within blocks)
3. Flexible patterns: local + global + random
4. Direct long-range access (no multi-hop dilution)
5. Better hardware utilization than irregular patterns

**Key advantage over sliding window:** Direct access to far blocks without information dilution.

**Key advantage over strided:** Complete context within attended blocks, no gaps.

**Best for:** Long structured documents where you need both local detail and specific global context.

**Implementation:** Same architecture as standard attention, just a different mask pattern.

---

## Quick Decision Guide

**Choose Block-Sparse when:**
- Document has clear structure
- Need direct access to start/end
- Local context important + need long-range
- Document classification/summarization
- Can design good attention pattern

**Choose Sliding Window when:**
- Autoregressive generation
- Simpler implementation preferred
- Most dependencies are local
- Can use deep models (32+ layers)

**Choose Strided when:**
- Periodic patterns in data
- Combined with local attention
- Experimental/research setting

**Choose Full Attention when:**
- Short sequences (<2k)
- Quality is paramount
- Computational cost acceptable