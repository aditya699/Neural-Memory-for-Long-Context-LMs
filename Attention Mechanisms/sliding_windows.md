# Understanding Sliding Window Attention

A guide to how sliding window attention works in models like Mistral and Llama.

---

## The Problem: Quadratic Complexity

In full self-attention, every token attends to every previous token:

```
Position 100 → attends to [1, 2, 3, ..., 99, 100]
Position 100k → attends to [1, 2, 3, ..., 99,999, 100,000]
```

**Complexity:**
```
Computation: O(n²·d)
Memory: O(n²)
```

For 100k tokens, this means **10 billion** attention operations per layer. This doesn't scale.

---

## The Solution: Local Windows

**Sliding window attention:** Each token only attends to the **w most recent tokens**.

```
Window size w = 4096

Position 100 → attends to [1, 2, ..., 99, 100] (full attention, window not exceeded)
Position 5000 → attends to [1001, 1002, ..., 4999, 5000] (last 4096 tokens only)
Position 100k → attends to [96k, 96k+1, ..., 99,999, 100k] (last 4096 tokens only)
```

**New Complexity:**
```
Computation: O(n·w·d) → Linear in sequence length!
Memory: O(n·w)
```

The window "slides" forward, maintaining constant size regardless of position.

---

## What Changes in the Code

The architecture is **identical** to full attention. Only the mask changes.

### Full Attention Mask (Causal)

```python
# Every token sees all previous tokens
causal_mask = torch.tril(torch.ones(seq_len, seq_len))

# [[1, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0],
#  [1, 1, 1, 0, 0],
#  [1, 1, 1, 1, 0],
#  [1, 1, 1, 1, 1]]
```

### Sliding Window Mask

```python
def sliding_window_mask(seq_len, window_size):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    
    # Zero out positions beyond the window
    for i in range(seq_len):
        if i >= window_size:
            mask[i, :i-window_size+1] = 0
    
    return mask

# With w=3:
# [[1, 0, 0, 0, 0],
#  [1, 1, 0, 0, 0],
#  [1, 1, 1, 0, 0],
#  [0, 1, 1, 1, 0],  ← only sees last 3
#  [0, 0, 1, 1, 1]]  ← only sees last 3
```

That's it. Everything else (Q, K, V projections, softmax, value mixing) is unchanged.

---

## The Critical Problem: Information Loss

**The issue:** Token at position 100k can only see positions [96k, 97k, 98k, 99k, 100k].

```
Token 1-10:     "The password is BLUE_SKY"
Token 11-100k:  [lots of content]
Token 100k:     "What was the password?"
```

With w=4k, the model **cannot see the password**. It's outside the window.

---

## The Solution: Stacking Layers

Even though each layer has limited window, **information flows through layers**.

### How It Works

**Layer 1:**
```
T4 attends to [T1, T2, T3, T4]
→ produces h4' containing info from [T1, T2, T3, T4]

T7 attends to [T5, T6, T7]
→ produces h7' containing info from [T5, T6, T7]
```

**Layer 2:**
```
T7 attends to [h5', h6', h7']

But h5' already contains info from [T3, T4, T5] (from Layer 1)!

So T7 indirectly receives info from T3
```

**Layer 3:**
```
T10 attends to [h8'', h9'', h10'']

h8'' contains info that traces back to T2 through the previous layers

So T10 indirectly receives info from T2
```

### The Key Insight

**Information doesn't travel through attention directly—it travels through the updated representations.**

- Layer 1: h4 becomes h4' (contains mixed info from window)
- Layer 2: Uses h4' as input, creates NEW K,V from h4'
- These new K,V carry the information forward
- After L layers, info can travel L×w positions

---

## Effective Receptive Field

```
Effective Reach = L × w

Where:
  L = number of layers
  w = window size
```

**Example: Mistral 7B**
```
Layers: 32
Window: 4,096

Effective reach: 32 × 4,096 = 131,072 tokens
```

Even with a 4k window per layer, 32 layers can reach 131k tokens back!

---

## The Trade-off

**Full Attention:**
```
✅ Every layer sees entire context directly
✅ No information loss
❌ O(n²) compute
❌ O(n²) memory
❌ Doesn't scale beyond ~8k tokens
```

**Sliding Window:**
```
✅ O(n·w) compute (linear!)
✅ Can scale to 100k+ tokens
✅ Much faster inference
❌ Information travels through layers (indirect)
❌ Context beyond L×w is lost
❌ Needs deeper models for same reach
```

---

## Combining Local + Global Through Stacking

This phrase means:

- **Local:** Each layer only does local attention (window w)
- **Global:** Multiple layers stacked create global reach (L×w)
- **Through stacking:** The mechanism is layer composition

**Each layer is local. The stack is global.**

---

## Example: Information Flow with w=3

```
Sequence: [T1, T2, T3, T4, T5, T6, T7, T8, T9]

Layer 1:
  T1 → sees [T1]                → h1'
  T4 → sees [T2, T3, T4]        → h4' (contains info from T2, T3, T4)
  T7 → sees [T5, T6, T7]        → h7' (contains info from T5, T6, T7)

Layer 2 (operates on h1', h2', ..., h9'):
  T7 → sees [h5', h6', h7']
       h5' contains [T3, T4, T5] from Layer 1
       So T7 now indirectly has T3's info!

Layer 3 (operates on h1'', h2'', ..., h9''):
  T9 → sees [h7'', h8'', h9'']
       h7'' contains info tracing back to T3
       So T9 now indirectly has T3's info!
```

Information "hops" forward through representation updates.

---

## Real-World Usage: Mistral 7B

**Architecture:**
```
Model: Mistral 7B
Layers: 32
Window: 4,096
Heads: 32
Dimension: 4,096
```

**Why it works:**
- 4k window captures most local dependencies (syntax, nearby references)
- 32 layers provide 131k theoretical reach
- In practice, handles 32k-100k token contexts effectively
- Combined with Grouped Query Attention for efficiency

---

## When to Use Sliding Window

**Works well for:**
- Language modeling (incremental context)
- Long document processing
- Code generation
- Most dependencies are local

**Challenging for:**
- Random access ("What was in paragraph 1?" at position 100k)
- Extreme long-range dependencies beyond L×w
- Tasks requiring precise recall of early context

---

## Summary

**Sliding window attention trades completeness for efficiency:**

1. Each token attends only to w recent tokens (not all previous)
2. This makes computation O(n·w) instead of O(n²)
3. Information loss is mitigated by stacking layers
4. After L layers, effective reach is L×w tokens
5. Enables 100k+ context with linear scaling

**The key mechanism:** Information flows through updated representations across layers, creating a relay effect that achieves global context from local attention.

**Implementation:** Same architecture as full attention, just a different mask. That's it.

---

## Code Example

```python
class SlidingWindowAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size=4096, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # THE ONLY CHANGE: Sliding window mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        for i in range(seq_len):
            if i >= self.window_size:
                mask[i, :i-self.window_size+1] = 0
        
        mask_to_block = (mask == 0).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask_to_block, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        return output
