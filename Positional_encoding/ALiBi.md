# ALiBi: Attention with Linear Biases

**The simplest position encoding. Train short, deploy long.**

---

## Core Idea

Instead of adding position embeddings to tokens, **add a distance penalty to attention scores**.

```python
attention_score = Q·K^T + bias
where: bias = -m × |i - j|
```

**That's it.**

---

## Why This Works

### The Formula

Token at position `i` attending to position `j`:
```
Distance: |i - j|
Penalty: -m × |i - j|  (m is negative, like -1.0)
```

### Example (m = -1.0)

Token at position 5 attending to others:

```
Position:  1    2    3    4    5    6    7
Distance:  4    3    2    1    0    1    2
Bias:     -4   -3   -2   -1    0   -1   -2
```

**Farther tokens get bigger penalties → model prefers nearby tokens**

### Not a Hard Rule

The penalty is **soft** - content can overcome distance:

```
Token 5 → Token 1:
  Content score (Q·K^T): +10.0
  Distance penalty:       -4.0
  Final score:            +6.0  ← Still high!
```

---

## Why Extrapolation Works

### The Key: Linearity

ALiBi uses a **linear** penalty that extends forever:

```python
bias = -m × distance  # Works for ANY distance
```

**Training:** See distances 0-2048, learn pattern "each step = -m penalty"  
**Inference:** Distance 5000? Apply same pattern: -m × 5000

Compare to learned embeddings:
```python
pos_embed[2048]  # Learned
pos_embed[2049]  # ❌ Never learned, doesn't exist!
```

### Empirical Results

Train on 1024 tokens:

| Test Length | ALiBi | RoPE |
|-------------|-------|------|
| 1024 | 15.2 | 15.0 |
| 2048 | 15.8 ✅ | 16.2 |
| 4096 | 16.5 ✅ | 18.9 |
| 8192 | 17.2 ✅ | 24.3 |
| 16384 | 18.1 ✅ | 41.7 |

**ALiBi maintains quality at extreme lengths!**

---

## Multiple Slopes (Multi-Head)

Different attention heads get **different slopes** via geometric sequence.

### Formula

For `n` heads:
```python
r = 2^(-8/n)

Head 1: m = -(r^1)
Head 2: m = -(r^2)
Head 3: m = -(r^3)
...
Head n: m = -(r^n)
```

### Example: 8 Heads

```python
r = 2^(-8/8) = 0.5

Head 1: -0.5      → Steep (local focus)
Head 2: -0.25
Head 3: -0.125
Head 4: -0.0625
...
Head 8: -0.0039   → Gentle (global focus)
```

### Why?

- **Steep slopes**: Focus on nearby tokens (grammar, phrases)
- **Gentle slopes**: Can attend far away (document structure, long refs)
- **Model learns** which head to use for what

---

## Implementation

### Computing Slopes

```python
def compute_slopes(num_heads):
    ratio = 2 ** (-8 / num_heads)
    return [-(ratio ** i) for i in range(1, num_heads + 1)]

# Example
slopes = compute_slopes(8)
# [-0.5, -0.25, -0.125, -0.0625, -0.03125, -0.015625, -0.0078125, -0.00390625]
```

### Computing Bias Matrix

```python
def get_alibi_bias(seq_len, slopes):
    # Distance matrix
    pos_i = torch.arange(seq_len).unsqueeze(1)  # [seq_len, 1]
    pos_j = torch.arange(seq_len).unsqueeze(0)  # [1, seq_len]
    distances = torch.abs(pos_i - pos_j)        # [seq_len, seq_len]
    
    # Apply slopes (one per head)
    num_heads = len(slopes)
    bias = torch.zeros(1, num_heads, seq_len, seq_len)
    for i, slope in enumerate(slopes):
        bias[0, i] = slope * distances
    
    return bias
```

### Example Bias Matrix (m=-1, seq_len=5)

```
       0    1    2    3    4
    [[ 0,  -1,  -2,  -3,  -4],
     [-1,   0,  -1,  -2,  -3],
     [-2,  -1,   0,  -1,  -2],
     [-3,  -2,  -1,   0,  -1],
     [-4,  -3,  -2,  -1,   0]]
```

### Where to Add Bias

```python
# Standard attention
Q = q_proj(x)
K = k_proj(x)
V = v_proj(x)

# Reshape for multi-head
Q = Q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
K = K.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)
V = V.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)

# Compute scores
scores = Q @ K.transpose(-2, -1)  # [batch, num_heads, seq_len, seq_len]

# ⭐ Add ALiBi bias HERE (before softmax)
alibi_bias = get_alibi_bias(seq_len, slopes)  # [1, num_heads, seq_len, seq_len]
scores = scores + alibi_bias

# Standard attention
attention = softmax(scores / sqrt(head_dim))
output = attention @ V
```

### Efficient: Cache the Bias

```python
class ALiBiAttention(nn.Module):
    def __init__(self, num_heads, max_seq_len=8192):
        super().__init__()
        self.slopes = compute_slopes(num_heads)
        # Pre-compute once, reuse forever
        self.register_buffer('bias_cache', 
                            get_alibi_bias(max_seq_len, self.slopes))
    
    def forward(self, Q, K, V):
        seq_len = Q.size(2)
        scores = Q @ K.transpose(-2, -1)
        
        # Just slice from cache
        scores = scores + self.bias_cache[:, :, :seq_len, :seq_len]
        
        attention = F.softmax(scores / math.sqrt(Q.size(-1)), dim=-1)
        return attention @ V
```

---

## ALiBi vs RoPE

| | ALiBi | RoPE |
|---|-------|------|
| **Method** | Distance penalty on scores | Rotation of Q/K vectors |
| **Math** | `-m × distance` | `e^(i·θ·pos)` complex rotation |
| **Where** | After Q·K^T | Before Q·K^T |
| **Complexity** | Very simple | Moderate |
| **Extrapolation** | Excellent | Good |
| **Quality (in-domain)** | Slightly worse | Slightly better |
| **Used in** | BLOOM, MPT | LLaMA, Mistral, Falcon |

### When to Use ALiBi

✅ **Extreme extrapolation** (train 2K, deploy 100K+)  
✅ **Variable-length** applications  
✅ **Limited training budget**  
✅ **Simplicity** matters  

### When to Use RoPE

✅ **Best quality** at typical lengths  
✅ **Known context** windows  
✅ **Industry standard** (LLaMA-style)  
✅ **Proven in production**  

---

## Common Mistakes

### 1. Wrong Distance
```python
# ❌ WRONG
distances = pos_i - pos_j  # Can be negative!

# ✅ CORRECT
distances = torch.abs(pos_i - pos_j)
```

### 2. Bias After Softmax
```python
# ❌ WRONG
attention = softmax(Q @ K.T)
attention = attention + bias

# ✅ CORRECT
scores = Q @ K.T
scores = scores + bias
attention = softmax(scores)
```

### 3. Positive Slopes
```python
# ❌ WRONG
slopes = [0.5, 0.25, ...]  # Positive encourages distance!

# ✅ CORRECT
slopes = [-0.5, -0.25, ...]  # Negative penalizes distance
```

### 4. Not Caching
```python
# ❌ INEFFICIENT
def forward(x):
    bias = get_alibi_bias(seq_len, slopes)  # Recompute every time!

# ✅ EFFICIENT
def __init__():
    self.bias_cache = get_alibi_bias(max_len, slopes)  # Compute once

def forward(x):
    bias = self.bias_cache[:, :, :seq_len, :seq_len]  # Just slice
```

---

## Quick Reference

### Core Formula
```
attention_score = Q·K^T + bias
bias = -m × |i - j|
```

### Slope Formula
```
n heads: r = 2^(-8/n)
Head i: m_i = -(r^i)
```

### Distance Matrix Pattern
```
Diagonal: 0 (no self-attention penalty)
Off-diagonal: -m × distance
Symmetric: bias[i,j] = bias[j,i]
```

---

## Summary

**ALiBi = Radical Simplicity**

1. Add `-m × distance` to attention scores
2. Different slopes for different heads
3. Linear pattern extrapolates perfectly
4. Zero parameters, trivial to implement
5. Train on 2K → deploy on 100K+

**The simplest solution that actually works.**

---

## Further Reading

**Original Paper:**  
Press et al. (2021): "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"  
https://arxiv.org/abs/2108.12409

**Models Using ALiBi:**
- BLOOM (BigScience)
- MPT (MosaicML)