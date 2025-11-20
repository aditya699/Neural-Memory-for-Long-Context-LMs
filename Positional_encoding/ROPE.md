# Understanding Rotary Position Embeddings (RoPE)

A comprehensive guide to one of the most elegant position encoding methods in modern transformers.

---

## Introduction: The Position Problem in Transformers

Transformers have a fundamental limitation: **they have no inherent notion of token order**.

Without positional information:
- "The cat chased the mouse" 
- "The mouse chased the cat"

...would be **identical** to the model. The self-attention mechanism treats input as an unordered set.

---

### 🎯 The Quest for Better Position Encoding

**Evolution of position encoding methods:**

| Method | Description | Limitation |
|:-------|:------------|:-----------|
| **Sinusoidal** (Vaswani et al., 2017) | Fixed sin/cos functions | No learnable patterns |
| **Learned Absolute** (BERT, GPT-2) | Learned embedding per position | Can't extrapolate beyond training length |
| **Relative Position** (T5) | Bias terms for relative distances | Adds parameters, doesn't scale well |
| **ALiBi** (Press et al., 2021) | Attention bias by distance | Linear extrapolation only |
| **RoPE** (Su et al., 2021) | Rotation-based encoding | ✅ Elegant, zero parameters, extrapolates! |

**RoPE has become the de facto standard** in modern LLMs (LLaMA, PaLM, Falcon, Mistral).

---

## Step 1 — The Core Problem: Absolute vs. Relative Positions

Let's understand why absolute positions are problematic.

---

### 📍 Absolute Position Embeddings (The Old Way)

**Standard approach (GPT-2, BERT):**

```python
class Embeddings(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model):
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
    
    def forward(self, token_ids):
        tokens = self.token_embed(token_ids)
        positions = torch.arange(len(token_ids))
        pos_encodings = self.pos_embed(positions)
        return tokens + pos_encodings
```

**What this means:**
- Position 0 gets vector: `[0.23, 0.45, ..., 0.67]` (learned)
- Position 1 gets vector: `[0.12, 0.89, ..., 0.34]` (learned)
- Position 512 gets vector: `[0.56, 0.23, ..., 0.91]` (learned)

These are **absolute** positions - each position has its own learned embedding.

---

### ❌ The Extrapolation Problem

**Training:**
```
Trained on sequences: max_seq_len = 512
Learned embeddings: pos_embed[0], pos_embed[1], ..., pos_embed[511]
```

**Inference (try to generate 1000 tokens):**
```python
pos_embed[512]  # ❌ ERROR: Index out of bounds!
pos_embed[800]  # ❌ ERROR: Never learned this position!
```

**The model literally cannot handle positions it hasn't seen during training.**

---

### 💡 What We Actually Need

Consider these two scenarios:

**Scenario 1:**
```
"The cat" appears at positions [10, 11]
Relative distance: 1 token apart
```

**Scenario 2:**
```
"The cat" appears at positions [500, 501]
Relative distance: 1 token apart
```

**Key insight:** The relationship between "The" and "cat" should be **the same** in both cases - they're both 1 token apart!

**Absolute positions:** Model sees (10, 11) vs (500, 501) as completely different  
**Relative positions:** Model sees both as "1 token apart" ✅

---

### 🎯 The Ideal Solution

We want a position encoding method that:
1. ✅ Encodes **relative distances** between tokens
2. ✅ Works for any sequence length (extrapolation)
3. ✅ Requires **zero learnable parameters**
4. ✅ Integrates seamlessly with attention

**RoPE achieves all four!**

---

## Step 2 — The Geometric Intuition: Rotation Encodes Position

RoPE's breakthrough insight: **use rotation geometry to encode positions**.

---

### 🔄 Basic 2D Rotation

Imagine you have a point in 2D space: `[x, y] = [3, 4]`

**Rotating by angle θ:**

```
      y
      |
    4 |     • (3, 4)
      |    /
      |   /
      |  / 
      | /θ
  ----+----------- x
      0    3
```

After rotating by angle θ, the point moves to a new location.

**The rotation formula:**
```
[x']   [cos θ  -sin θ]   [x]
[y'] = [sin θ   cos θ] × [y]
```

---

### 💡 The RoPE Idea: Position = Rotation Angle

Instead of **adding** position embeddings, **rotate** vectors based on their position:

- Token at **position 0** → rotate by **0°** (no rotation)
- Token at **position 1** → rotate by **θ** 
- Token at **position 2** → rotate by **2θ**
- Token at **position m** → rotate by **m·θ**

**The further the position, the more rotation!**

---

### 🎯 Example with Actual Positions

Let's say θ = 10° per position:

```
Position 0: rotate by 0 × 10° = 0°    → [3.0, 4.0]
Position 1: rotate by 1 × 10° = 10°   → [2.6, 4.3]
Position 2: rotate by 2 × 10° = 20°   → [2.2, 4.5]
Position 5: rotate by 5 × 10° = 50°   → [1.5, 4.8]
```

**Each position gets a unique rotation angle!**

---

### 🔑 Why This Helps with Relative Positions

Here's the mathematical magic:

**Query at position i:** Rotated by angle `i·θ`  
**Key at position j:** Rotated by angle `j·θ`

**When you compute their dot product** (attention score), the rotation math gives:

```
Rotated_Q(i) · Rotated_K(j) = Original_Q · Original_K · f(θ(j - i))
                                                              ↑
                                                    Relative distance!
```

**The result depends on (j - i) - the relative distance between positions!**

This happens automatically through the geometry of rotation. ✨

---

### 📐 Proof with Simple Example

**Setup:**
- Query at position 3 → rotated by 3θ
- Key at position 7 → rotated by 7θ

**Rotation angles:**
- Q rotated by: 3θ
- K rotated by: 7θ

**After dot product, the math simplifies to:**
```
Result depends on: 7θ - 3θ = 4θ
```

The **4θ** represents "4 positions apart" - the relative distance!

**Not the absolute positions (3 and 7), but their difference (4)!**

---

## Step 3 — The Frequency Problem: One Rotation Speed Isn't Enough

Using a single rotation frequency θ creates problems.

---

### ⚠️ Problem 1: Fast Rotation (Large θ)

**Example: θ = 10° per position**

```
Position 1:   rotate by 10°
Position 2:   rotate by 20°
Position 10:  rotate by 100°
Position 36:  rotate by 360°  ← Full circle!
Position 37:  rotate by 370° = 10° (same as position 1!)
```

**The issue:** After 36 positions, rotations start **repeating**!

- Position 1 looks the same as Position 37
- Position 10 looks the same as Position 46
- The model can't distinguish them! ❌

**Fast rotation is good for nearby tokens but fails for distant ones.**

---

### ⚠️ Problem 2: Slow Rotation (Small θ)

**Example: θ = 0.01° per position**

```
Position 1:  rotate by 0.01°
Position 2:  rotate by 0.02°
Position 10: rotate by 0.1°
```

**The differences are tiny!**

```
Position 1 vs Position 2: only 0.01° difference
```

When computing attention, such small angle differences are hard to distinguish.

**Slow rotation is good for distant tokens but fails for nearby ones.** ❌

---

### 💡 The Solution: Multiple Frequencies

**Use different rotation speeds for different dimensions!**

Think of it like a clock:
- **Minute hand** (fast) → good for telling nearby times apart (1 min vs 2 min)
- **Hour hand** (slow) → good for telling distant times apart (1 hour vs 5 hours)

**RoPE does the same thing:**
- Some dimensions rotate **fast** → capture nearby positions
- Some dimensions rotate **slow** → capture distant positions

---

### 📊 Frequency Spectrum

For a model with `d_model = 256` dimensions:

Split into **128 pairs** (rotation works in 2D):

```
Pair 0:   θ₀ = 1.0      (fastest - for nearby tokens)
Pair 1:   θ₁ = 0.982
Pair 2:   θ₂ = 0.965
...
Pair 63:  θ₆₃ = 0.5     (medium speed)
...
Pair 127: θ₁₂₇ = 0.0001 (slowest - for distant tokens)
```

**Each dimension pair gets a different rotation frequency!**

---

### 🔬 Why This Works

**Comparing Position 1 vs Position 2 (nearby):**

**Fast rotation (Pair 0, θ = 1.0):**
```
Position 1: 1.0°
Position 2: 2.0°
Difference: 1.0° ✅ Easy to distinguish!
```

**Slow rotation (Pair 127, θ = 0.0001):**
```
Position 1: 0.0001°
Position 2: 0.0002°
Difference: 0.0001° ❌ Too tiny!
```

**Comparing Position 1 vs Position 500 (distant):**

**Fast rotation (Pair 0, θ = 1.0):**
```
Position 1:   1.0°
Position 500: 500° = 140° (after wrapping)
❌ Confusing! Did we go 140° or 500°?
```

**Slow rotation (Pair 127, θ = 0.0001):**
```
Position 1:   0.0001°
Position 500: 0.05°
Difference: 0.05° ✅ Clear, no wrapping!
```

**By using both fast and slow frequencies, we capture both short-range and long-range relationships!**

---

## Step 4 — The Frequency Formula

How do we choose the rotation frequency for each dimension pair?

---

### 🔢 The Formula

For dimension pair `i` (where i = 0, 1, 2, ..., d/2 - 1):

```
θᵢ = base^(-2i / d)
```

**Parameters:**
- `base`: Typically 10,000 (used in LLaMA models)
- `i`: Dimension pair index
- `d`: Model dimension (e.g., 256, 512, 4096)

---

### 📊 Example Calculation (d=256, base=10000)

```python
d = 256
base = 10000
num_pairs = d // 2  # 128 pairs

# Pair 0:
θ₀ = 10000^(-2×0/256) = 10000^(0) = 1.0

# Pair 1:
θ₁ = 10000^(-2×1/256) = 10000^(-0.0078) ≈ 0.982

# Pair 64 (middle):
θ₆₄ = 10000^(-2×64/256) = 10000^(-0.5) = 0.01

# Pair 127 (last):
θ₁₂₇ = 10000^(-2×127/256) = 10000^(-0.992) ≈ 0.0001
```

**This creates a logarithmic spacing of frequencies from 1.0 down to 0.0001.**

---

### 💡 Why This Exponential Decay?

**Linear spacing:**
```
θ = [1.0, 0.99, 0.98, 0.97, ..., 0.01]
Most values are clustered near 1.0
```

**Exponential spacing (what RoPE uses):**
```
θ = [1.0, 0.982, 0.965, ..., 0.1, ..., 0.01, ..., 0.0001]
Evenly distributed in log-space
```

**This ensures good coverage of both:**
- High frequencies (for local patterns)
- Low frequencies (for long-range patterns)

---

### 🎯 The Role of Base (10,000)

**What does `base = 10000` control?**

It determines the **range** of frequencies:

**With base = 10000:**
```
Max frequency: θ₀ = 1.0
Min frequency: θ₁₂₇ ≈ 0.0001
Range: 1.0 → 0.0001 (spans 4 orders of magnitude)
```

**With base = 1000:**
```
Max frequency: θ₀ = 1.0
Min frequency: θ₁₂₇ ≈ 0.001
Range: 1.0 → 0.001 (only 3 orders of magnitude)
```

**Larger base → slower minimum frequency → better for longer sequences**

That's why LLaMA uses `base = 10000` - it can handle very long contexts!

---

## Step 5 — Complex Numbers: The Elegant Implementation

Rotation can be implemented with matrices OR complex numbers. Complex numbers are far more elegant.

---

### 🔄 Rotation Matrix Approach (The Hard Way)

To rotate a 2D vector `[x, y]` by angle θ:

```python
new_x = x * cos(θ) - y * sin(θ)
new_y = x * sin(θ) + y * cos(θ)
```

**For every dimension pair, we need:**
- 4 multiplications
- 2 additions
- 2 trigonometric function calls

Tedious and computationally expensive!

---

### ✨ Complex Number Approach (The Elegant Way)

**Euler's formula:**
```
e^(iθ) = cos(θ) + i·sin(θ)
```

This is one of the most beautiful equations in mathematics!

**Key insight:** We can represent a 2D vector as a complex number:

```python
vector = [x, y]  →  z = x + i·y  (complex number)
```

**To rotate:** Just multiply by `e^(iθ)`!

```python
z_rotated = z × e^(iθ)
```

**That's it!** One complex multiplication does the entire rotation.

---

### 📐 Example

**Rotate vector [3, 4] by 30°:**

**Matrix way:**
```python
θ = 30° = 0.524 radians
new_x = 3*cos(0.524) - 4*sin(0.524) = 3*0.866 - 4*0.5 = 0.598
new_y = 3*sin(0.524) + 4*cos(0.524) = 3*0.5 + 4*0.866 = 4.964
Result: [0.598, 4.964]
```

**Complex number way:**
```python
z = 3 + 4i
z_rotated = z × e^(i×0.524)
         = (3 + 4i) × (0.866 + 0.5i)
         = 0.598 + 4.964i
Result: [0.598, 4.964]  (extract real and imaginary parts)
```

**Same result, cleaner code!**

---

### 🎯 Why Complex Numbers Are Better

| Aspect | Rotation Matrix | Complex Numbers |
|:-------|:----------------|:----------------|
| **Conceptual clarity** | 4 separate operations | Single multiplication |
| **Code lines** | ~10 lines | ~3 lines |
| **GPU efficiency** | Separate ops | Batched complex ops |
| **Mathematical elegance** | Low | High ✨ |

Modern deep learning frameworks (PyTorch, JAX) have native complex number support!

---

## Step 6 — The Mathematical Property: Relative Position Emerges

This is the core mathematical insight that makes RoPE work.

---

### 🔬 The Setup

We have:
- **Query** vector at position `i`
- **Key** vector at position `j`

After projecting through W_q and W_k, we treat each as a complex number (for one dimension pair):

```
q = a + bi  (some complex number representing query)
k = c + di  (some complex number representing key)
```

---

### 🔄 Step 1: Apply RoPE (Rotate by Position)

**Query at position i:**
```
q_rotated = q × e^(i·θ·i)
```

**Key at position j:**
```
k_rotated = k × e^(i·θ·j)
```

Where:
- θ is the frequency for this dimension pair
- i and j are the token positions
- Note: `i` inside `e^(i·θ·i)` is the imaginary unit, while the outer `i` is the position!

---

### 🎯 Step 2: Compute Attention Score (Dot Product)

In complex numbers, the dot product involves the **complex conjugate**:

```
score = q_rotated × conj(k_rotated)
```

Where `conj(a + bi) = a - bi` (flip sign of imaginary part)

---

### 🧮 Step 3: Substitute and Simplify

```
score = (q × e^(i·θ·i)) × conj(k × e^(i·θ·j))

      = q × conj(k) × e^(i·θ·i) × conj(e^(i·θ·j))
      
      = q × conj(k) × e^(i·θ·i) × e^(-i·θ·j)
      
      = q × conj(k) × e^(i·θ(i-j))
                          ↑
                    Relative position!
```

---

### 🎉 The Key Result

The attention score contains the term:

```
e^(i·θ(i-j))
```

This depends **only on (i - j)**, the **relative distance** between positions!

**Not on the absolute positions i or j individually.**

---

### 💡 What This Means

**If two tokens are 5 positions apart:**

Doesn't matter if they're at:
- Positions (10, 15) → difference = 5
- Positions (100, 105) → difference = 5  
- Positions (1000, 1005) → difference = 5

**The attention computation sees the same relative position information!**

This is why RoPE naturally encodes relative positions. ✨

---

### 📊 Comparison with Absolute Positions

**Absolute position embeddings:**
```
Attention(pos_i, pos_j) depends on both i AND j separately
Position (10, 15) ≠ Position (100, 105) even though distance is the same
```

**RoPE:**
```
Attention(pos_i, pos_j) depends only on (j - i)
Position (10, 15) ≈ Position (100, 105) because distance is the same ✅
```

---

## Step 7 — Why RoPE Enables Extrapolation

This is one of RoPE's most powerful advantages.

---

### ❌ The Problem with Learned Position Embeddings

**GPT-2 / BERT style:**

```python
# During training: max_seq_len = 512
pos_embed = nn.Embedding(512, d_model)  

# Learned embeddings for positions 0-511
pos_embed.weight[0]    # Position 0: [0.23, 0.45, ...]
pos_embed.weight[1]    # Position 1: [0.12, 0.89, ...]
...
pos_embed.weight[511]  # Position 511: [0.67, 0.34, ...]
```

**At inference (try to generate 1000 tokens):**

```python
pos_embed.weight[512]  # ❌ ERROR! Index 512 doesn't exist!
pos_embed.weight[800]  # ❌ ERROR! Never trained this position!
```

**The model literally cannot process positions beyond 511.**

---

### ✅ How RoPE Solves This

**RoPE doesn't learn positions - it uses a mathematical formula:**

```python
# Position encoding is a function, not a lookup table!
def rope_encoding(position, frequency):
    return e^(i × position × frequency)
```

**During training (max_seq_len = 512):**
```python
position 0:   e^(i × 0 × θ)
position 100: e^(i × 100 × θ)
position 511: e^(i × 511 × θ)
```

**At inference (even for position 1000!):**
```python
position 1000: e^(i × 1000 × θ)  # ✅ Just works! It's just math!
```

**No lookup table, no learned parameters - just apply the formula!**

---

### 🎯 Why This Works: Relative Positions Generalize

**What the model learns during training:**

The model learns: "Tokens that are 10 positions apart have relationship X"

This is encoded in the attention weights through the term `e^(i·θ·10)`.

**At inference, with longer sequences:**

```
Positions (5, 15):     Distance = 10 → e^(i·θ·10)  ✅ Seen during training
Positions (500, 510):  Distance = 10 → e^(i·θ·10)  ✅ Same! Still works!
```

**The model learned about relative distance 10, not absolute positions.**

So it can apply that knowledge to **any** pair of tokens 10 positions apart, even at positions it never saw during training!

---

### 📊 Extrapolation Example

**Training:**
```
Max sequence length: 2048 tokens
Model learns: relationships between positions 0-2047
```

**Inference:**
```
Generate 4096 tokens
Positions 3000-3010: Distance = 10
Model applies same relationship learned for distance = 10
✅ Works! Even though position 3000 was never seen in training
```

---

### ⚠️ Limitations and Solutions

**RoPE can extrapolate, but not infinitely:**

If you train on 2048 tokens and try to generate 100,000 tokens:
- Some slow frequencies might complete multiple 360° rotations
- Model might get confused at extremely long distances

**Solutions:**
1. **RoPE Scaling** - Adjust frequencies at inference time
2. **YaRN** (Yet another RoPE extensioN) - Interpolate frequencies
3. **Position Interpolation** - Scale down position indices

**But compared to absolute position embeddings (which fail immediately at 2049), RoPE is far superior!**

---

## Step 8 — Implementation: The Complete Algorithm

Now let's see how to actually implement RoPE in code.

---

### 📋 The Complete 6-Step Algorithm

**One-time setup (before training):**

```python
# Step 0: Precompute frequencies
d_model = 256
num_pairs = d_model // 2  # 128 pairs
base = 10000

freqs = []
for i in range(num_pairs):
    θ_i = base ** (-2 * i / d_model)
    freqs.append(θ_i)

# freqs = [1.0, 0.982, 0.965, ..., 0.0001]
```

**Every forward pass:**

```python
# Step 1: Take Q vector (after q_proj)
Q = q_proj(x)  # Shape: [batch, seq_len, d_model]

# Step 2: Split into pairs
Q_pairs = Q.reshape(batch, seq_len, num_pairs, 2)
# Shape: [batch, seq_len, 128, 2]

# Step 3: Convert pairs to complex numbers
Q_complex = Q_pairs[..., 0] + 1j * Q_pairs[..., 1]
# Shape: [batch, seq_len, 128]

# Step 4: Compute rotation angles for each position
positions = torch.arange(seq_len)  # [0, 1, 2, ..., seq_len-1]
angles = positions[:, None] * freqs[None, :]
# Shape: [seq_len, 128]

# Step 5: Apply rotation (multiply by e^(i·angle))
rotation_complex = torch.exp(1j * angles)
Q_rotated_complex = Q_complex * rotation_complex
# Shape: [batch, seq_len, 128]

# Step 6: Convert back to real numbers
Q_rotated_real = torch.stack([
    Q_rotated_complex.real,
    Q_rotated_complex.imag
], dim=-1)
# Shape: [batch, seq_len, 128, 2]

# Reshape back to original
Q_rotated = Q_rotated_real.reshape(batch, seq_len, d_model)
# Shape: [batch, seq_len, 256]

# Now use Q_rotated in attention!
```

---

### 🔧 Dry Run with Tiny Example

Let's trace through with actual numbers.

**Setup:**
```python
d_model = 4  # Just 4 dimensions for simplicity
num_pairs = 2  # 4 // 2 = 2 pairs
seq_len = 2  # Two tokens
base = 10000
```

**Step 0: Precompute frequencies**
```python
Pair 0: θ₀ = 10000^(-2×0/4) = 1.0
Pair 1: θ₁ = 10000^(-2×1/4) = 0.01
freqs = [1.0, 0.01]
```

**Step 1: Take Q (after projection)**
```python
Q = [[2.0, 1.0, 3.0, 1.5],   # Token at position 0
     [1.0, 2.0, 2.0, 1.0]]   # Token at position 1
# Shape: [1, 2, 4]
```

**Step 2: Split into pairs**
```python
Q_pairs = [
    # Position 0:
    [[2.0, 1.0],   # Pair 0
     [3.0, 1.5]],  # Pair 1
    
    # Position 1:
    [[1.0, 2.0],   # Pair 0
     [2.0, 1.0]]   # Pair 1
]
# Shape: [1, 2, 2, 2]
```

**Step 3: Convert to complex**
```python
Q_complex = [
    [2.0+1.0j, 3.0+1.5j],  # Position 0, both pairs
    [1.0+2.0j, 2.0+1.0j]   # Position 1, both pairs
]
# Shape: [1, 2, 2]
```

**Step 4: Compute rotation angles**
```python
positions = [0, 1]
freqs = [1.0, 0.01]

angles = [
    [0×1.0, 0×0.01],     # Position 0
    [1×1.0, 1×0.01]      # Position 1
] = [
    [0, 0],
    [1.0, 0.01]
]
```

**Step 5: Apply rotation**
```python
rotation_complex = exp(1j × angles) = [
    [e^(i×0), e^(i×0)],           # Position 0
    [e^(i×1.0), e^(i×0.01)]       # Position 1
] = [
    [1.0+0j, 1.0+0j],                        # No rotation at position 0
    [cos(1.0)+i·sin(1.0), cos(0.01)+i·sin(0.01)]  # Position 1
] ≈ [
    [1.0+0j, 1.0+0j],
    [0.540+0.841j, 0.999+0.010j]
]

Q_rotated_complex = Q_complex × rotation_complex
Position 0, Pair 0: (2.0+1.0j) × (1.0+0j) = 2.0+1.0j
Position 0, Pair 1: (3.0+1.5j) × (1.0+0j) = 3.0+1.5j

Position 1, Pair 0: (1.0+2.0j) × (0.540+0.841j) 
                  = (1.0×0.540 - 2.0×0.841) + i(1.0×0.841 + 2.0×0.540)
                  = -1.142 + 1.921j

Position 1, Pair 1: (2.0+1.0j) × (0.999+0.010j)
                  = 1.988 + 1.019j
```

**Step 6: Convert back to real**
```python
Q_rotated = [
    [2.0, 1.0, 3.0, 1.5],           # Position 0 (barely rotated)
    [-1.142, 1.921, 1.988, 1.019]   # Position 1 (rotated!)
]
# Shape: [1, 2, 4]
```

**Done! Q_rotated now has position information encoded through rotation!**

---

### 🎯 Key Observations

1. Position 0 barely rotated (angles were 0)
2. Position 1 rotated by different amounts:
   - Pair 0: Large rotation (θ=1.0)
   - Pair 1: Tiny rotation (θ=0.01)
3. The rotated vectors are **different** from the original
4. Position information is now **baked into** the vector values

---

## Step 9 — Integration with Multi-Head Attention

Where exactly does RoPE fit into your transformer?

---

### 🔧 Before RoPE (Standard Attention)

```python
class MultiHeadAttention(nn.Module):
    def forward(self, x):
        # x has position info added externally
        
        # Project
        Q = self.q_proj(x)  # [batch, seq_len, d_model]
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape for multi-head
        Q = Q.view(batch, seq_len, num_heads, head_dim)
        K = K.view(batch, seq_len, num_heads, head_dim)
        V = V.view(batch, seq_len, num_heads, head_dim)
        
        # Compute attention
        scores = Q @ K.transpose(-2, -1)
        attn = softmax(scores / sqrt(head_dim))
        output = attn @ V
        
        return output
```

---

### ✅ After RoPE

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # ⭐ Add RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim)
    
    def forward(self, x):
        # x does NOT have position info yet!
        
        # Project
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Reshape
        Q = Q.view(batch, seq_len, num_heads, head_dim)
        K = K.view(batch, seq_len, num_heads, head_dim)
        V = V.view(batch, seq_len, num_heads, head_dim)
        
        # ⭐⭐⭐ APPLY ROPE HERE ⭐⭐⭐
        positions = torch.arange(seq_len, device=x.device)
        Q = self.rope(Q, positions)  # Rotate Q
        K = self.rope(K, positions)  # Rotate K
        # V is NOT rotated!
        
        # Compute attention (rest is same)
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        scores = Q @ K.transpose(-2, -1)
        attn = softmax(scores / sqrt(head_dim))
        output = attn @ V
        
        return output
```

---

### 🎯 Key Points

1. **RoPE is applied AFTER projection but BEFORE attention**
2. **Only Q and K are rotated** (not V!)
3. **Applied separately to each attention head**
4. **No changes needed to the rest of attention mechanism**

---

### 📊 What Gets Removed?

```python
# REMOVE THIS:
class Embeddings(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model):
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)  # ❌ DELETE
    
    def forward(self, token_ids):
        tokens = self.token_embed(token_ids)
        pos = self.pos_embed(positions)  # ❌ DELETE
        return tokens + pos  # ❌ DELETE

# REPLACE WITH:
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # No position embeddings needed!
    
    def forward(self, token_ids):
        return self.token_embed(token_ids)  # Just token embeddings
```

Position information is now handled **inside** the attention mechanism via RoPE!

---

## Step 10 — LLaMA Configuration Deep Dive

Let's understand the specific choices made in production models.

---

### 🦙 LLaMA-2 RoPE Configuration

```python
# LLaMA-2 7B model
d_model = 4096
num_heads = 32
head_dim = d_model // num_heads = 128

# RoPE settings
rope_base = 10000
rope_dim = head_dim  # Apply to full head dimension (128)
```

---

### 🔍 Why base = 10000?

**The base controls the frequency range:**

```python
θ₀ = 10000^(0) = 1.0           # Fastest frequency
θ₆₃ = 10000^(-126/128) ≈ 0.0001  # Slowest frequency (for head_dim=128)
```

**This gives a frequency range spanning 4 orders of magnitude!**

**What if we used different bases?**

| Base | Min Frequency | Max Context (approx) | Use Case |
|:-----|:--------------|:---------------------|:---------|
| 1,000 | 0.001 | ~6K tokens | Short contexts |
| 10,000 | 0.0001 | ~32K tokens | **Standard (LLaMA)** |
| 100,000 | 0.00001 | ~100K+ tokens | Extended contexts |

**LLaMA uses 10,000 as a sweet spot:**
- Good for typical contexts (2K-8K tokens)
- Can extrapolate to 32K+ with techniques like YaRN
- Not too slow (avoids numerical precision issues)

---

### 🔍 Why dim = 128 (head_dim)?

LLaMA applies RoPE to the **entire head dimension** (128), not just part of it.

**Some models use partial RoPE:**
```python
# PaLM uses partial RoPE
head_dim = 256
rope_dim = 64  # Only apply RoPE to first 64 dims
# Remaining 192 dims don't get position info
```

**LLaMA uses full RoPE:**
```python
# LLaMA
head_dim = 128
rope_dim = 128  # All dimensions get position info
```

**Trade-off:**

| Approach | Pros | Cons |
|:---------|:-----|:-----|
| **Partial RoPE** | Some dims can learn position-agnostic patterns | Less position info |
| **Full RoPE** | Maximum position information | All dims affected by position |

**LLaMA chose full RoPE** - empirically found to work best for their model size.

---

### 📊 Frequency Distribution for head_dim=128

```python
# With base=10000, head_dim=128, we get 64 frequency pairs:

Pair 0:  θ = 1.0000      # Period ≈ 6 tokens
Pair 16: θ = 0.3162      # Period ≈ 20 tokens  
Pair 32: θ = 0.1000      # Period ≈ 63 tokens
Pair 48: θ = 0.0316      # Period ≈ 199 tokens
Pair 63: θ = 0.0100      # Period ≈ 628 tokens
```

**This spread covers from very local (a few tokens) to quite distant (hundreds of tokens) relationships!**

---

### 🎯 Why These Values Were Chosen

**The LLaMA team tested:**
- Different bases: 1000, 10000, 100000
- Different rope_dim: 32, 64, 128 (full head)
- Different head_dim: 64, 128, 256

**Findings:**
1. base=10000 worked best for their training length (2048 tokens) while enabling extrapolation
2. Full head RoPE (rope_dim = head_dim) gave best performance
3. head_dim=128 balanced expressiveness and efficiency

**These are now standard choices adopted by most modern LLMs!**

---

## Step 11 — RoPE vs. Other Position Encodings

How does RoPE compare to alternatives?

---

### 📊 Comparison Table

| Method | Parameters | Extrapolation | Relative Encoding | Complexity |
|:-------|:-----------|:--------------|:------------------|:-----------|
| **Absolute (Learned)** | O(L × d) | ❌ No | ❌ No | Low |
| **Sinusoidal** | 0 | ⚠️ Limited | ❌ No | Low |
| **Relative Position Bias** | O(L²) | ❌ No | ✅ Yes | Medium |
| **ALiBi** | 0 | ✅ Linear | ✅ Yes | Low |
| **RoPE** | **0** | **✅ Yes** | **✅ Yes** | Medium |

---

### 🔍 Detailed Comparison

**Absolute Position Embeddings (GPT-2, BERT):**
```python
pos_embed = nn.Embedding(max_seq_len, d_model)
```
- ❌ Fails immediately beyond training length
- ❌ Doesn't encode relative positions
- ✅ Simple to implement
- Used in: GPT-2, BERT, older models

---

**Sinusoidal (Original Transformer):**
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
- ⚠️ Can extrapolate but loses quality
- ❌ Doesn't explicitly encode relative positions  
- ✅ Zero parameters
- Used in: Original "Attention is All You Need" paper

---

**Relative Position Bias (T5):**
```python
# Add learned bias based on relative distance
attention_score += learned_bias[i - j]
```
- ✅ Explicitly encodes relative positions
- ❌ Requires O(L²) parameters (bias for each distance)
- ❌ Can't extrapolate beyond training
- Used in: T5

---

**ALiBi (Attention with Linear Biases):**
```python
# Add linear penalty based on distance
attention_score -= m × |i - j|
```
- ✅ Zero parameters
- ✅ Good extrapolation (linear)
- ⚠️ Only linear distance, not rotational geometry
- Used in: BLOOM, MPT

---

**RoPE (LLaMA, modern LLMs):**
```python
# Rotate Q and K by position
Q_rotated = apply_rotation(Q, position, frequencies)
K_rotated = apply_rotation(K, position, frequencies)
```
- ✅ Zero learnable parameters
- ✅ Excellent extrapolation
- ✅ Naturally encodes relative positions
- ✅ Mathematically elegant
- Used in: **LLaMA, PaLM, Falcon, Mistral, Mixtral, most modern LLMs**

---

### 🏆 Why RoPE Won

**RoPE combines the best properties:**
1. Zero parameters (like sinusoidal and ALiBi)
2. Extrapolation capability (better than absolute)
3. Relative position encoding (like T5, but cleaner)
4. Mathematical elegance (rotation geometry)
5. Efficient implementation (complex numbers)

**That's why almost every major LLM since 2021 uses RoPE!**

---

## Step 12 — Advanced Topics and Extensions

RoPE has inspired several extensions and improvements.

---

### 🚀 RoPE Scaling (Extending Context Length)

**Problem:** Even RoPE has limits. If you train on 2K tokens and try to generate 100K, quality degrades.

**Solution: RoPE Scaling**

Adjust frequencies at inference time:

```python
# Original frequencies
θ_original = base^(-2i/d)

# Scaled frequencies (for longer context)
scaling_factor = 2.0  # Double the context length
θ_scaled = θ_original / scaling_factor
```

**Effect:** Rotations happen more slowly → can handle longer contexts.

**Used in:** GPT-NeoX, Falcon

---

### 🎯 YaRN (Yet another RoPE extensioN)

**Even smarter frequency adjustment:**

```python
# Instead of uniform scaling, scale different frequencies differently
# High frequencies (local): Scale less
# Low frequencies (global): Scale more

if i < 32:  # High frequency pairs
    θ_scaled = θ_original / 1.5
else:  # Low frequency pairs  
    θ_scaled = θ_original / 4.0
```

**Result:** Better preservation of local patterns while extending global range.

**Used in:** LLaMA-2 extended contexts, Mistral

---

### 🔄 Position Interpolation

**Alternative approach:**

Instead of scaling frequencies, scale positions:

```python
# Original
angle = position × θ

# Position interpolation
max_train_len = 2048
max_inference_len = 8192
scale = max_train_len / max_inference_len

angle = (position × scale) × θ
```

**Effect:** "Compress" long sequence into shorter position space.

**Used in:** Some LLaMA fine-tunes

---

### 📐 2D RoPE (Vision Transformers)

**Extend RoPE to 2D (images):**

```python
# For an image token at position (x, y)
# Apply RoPE separately to x and y coordinates

# Half dimensions for x-axis rotation
Q_x_rotated = apply_rope(Q[:, :dim//2], position_x, freqs_x)

# Half dimensions for y-axis rotation  
Q_y_rotated = apply_rope(Q[:, dim//2:], position_y, freqs_y)

Q_rotated = concat(Q_x_rotated, Q_y_rotated)
```

**Used in:** Some vision transformers (ViT variants)

---

### 🌐 3D RoPE (Video Transformers)

**Extend to 3D (video: x, y, time):**

```python
# Split dimensions three ways
Q_x_rotated = apply_rope(Q[:, :dim//3], position_x, freqs_x)
Q_y_rotated = apply_rope(Q[:, dim//3:2*dim//3], position_y, freqs_y)
Q_t_rotated = apply_rope(Q[:, 2*dim//3:], position_t, freqs_t)

Q_rotated = concat(Q_x_rotated, Q_y_rotated, Q_t_rotated)
```

**Used in:** Video understanding models

---

## Step 13 — Common Pitfalls and Debugging

Things that can go wrong when implementing RoPE.

---

### ⚠️ Pitfall 1: Forgetting to Apply to K

```python
# WRONG
Q_rotated = apply_rope(Q, positions)
# Forgot K!
scores = Q_rotated @ K.transpose()
```

**Fix:** Apply RoPE to both Q and K!

```python
# CORRECT
Q_rotated = apply_rope(Q, positions)
K_rotated = apply_rope(K, positions)
scores = Q_rotated @ K_rotated.transpose()
```

**Why:** Both need position info for relative position encoding to work.

---

### ⚠️ Pitfall 2: Applying to V

```python
# WRONG
Q_rotated = apply_rope(Q, positions)
K_rotated = apply_rope(K, positions)
V_rotated = apply_rope(V, positions)  # ❌ Don't do this!
```

**Fix:** Don't rotate V!

```python
# CORRECT  
Q_rotated = apply_rope(Q, positions)
K_rotated = apply_rope(K, positions)
# V stays as is!
```

**Why:** Position encoding only needs to affect the attention scores (Q·K), not the values.

---

### ⚠️ Pitfall 3: Wrong Dimension Shape

```python
# WRONG: Applying RoPE to full d_model before splitting heads
Q = self.q_proj(x)  # [batch, seq_len, d_model]
Q_rotated = apply_rope(Q, positions)  # Applied to full 512 dims
Q = Q.view(batch, seq_len, num_heads, head_dim)  # Then split
```

**Fix:** Apply after reshaping to heads!

```python
# CORRECT: Apply RoPE to each head separately
Q = self.q_proj(x)
Q = Q.view(batch, seq_len, num_heads, head_dim)  # Split first
Q_rotated = apply_rope(Q, positions)  # Apply to head_dim
```

**Why:** RoPE works on head_dim (64 or 128), not full d_model (512).

---

### ⚠️ Pitfall 4: Frequency Computation Error

```python
# WRONG
θ_i = base ** (-2 * i / num_heads)  # ❌ Used num_heads instead of d_model
```

**Fix:**

```python
# CORRECT
θ_i = base ** (-2 * i / d_model)  # Or head_dim if applying per-head
```

---

### 🔍 Debugging Checklist

When RoPE isn't working:

1. ✅ Check: Are Q and K both rotated?
2. ✅ Check: Is V **not** rotated?
3. ✅ Check: Are frequencies precomputed correctly?
4. ✅ Check: Is rotation applied at correct dimension (head_dim)?
5. ✅ Check: Are positions passed correctly (0, 1, 2, ...)?
6. ✅ Check: Complex number handling (real/imag extraction)

---

## Step 14 — Theoretical Foundations

For those interested in the deeper mathematics.

---

### 🎓 Rotation as a Group Operation

**Mathematical structure:**

Rotations form a group under composition:
```
R(θ₁) ∘ R(θ₂) = R(θ₁ + θ₂)
```

This is called SO(2) - Special Orthogonal group in 2D.

**Relevance to RoPE:**

When you compute attention between rotated Q and K:
```
R(θᵢ)·q · R(θⱼ)·k

= q · R(-θᵢ) · R(θⱼ) · k

= q · R(θⱼ - θᵢ) · k
```

The **group property** ensures relative positions emerge!

---

### 🎓 Connection to Fourier Analysis

**RoPE frequencies are like Fourier basis functions:**

```
e^(i·θ·m) where θ varies logarithmically
```

This is similar to Fourier transform, which decomposes signals into frequency components.

**Different frequencies capture different scales:**
- High frequency: Local patterns (like high-frequency sound)
- Low frequency: Global patterns (like low-frequency sound)

---

### 🎓 Information Theory Perspective

**Channel capacity view:**

- Single frequency: Limited information about position
- Multiple frequencies: Increases channel capacity exponentially

**With n frequency pairs, we can distinguish 2^n different relative positions!**

That's why using multiple frequencies is so powerful.

---

### 🎓 Geometric Interpretation

**RoPE embeds sequences in a spiral:**

```
Position 0:   Point at angle 0
Position 1:   Point at angle θ
Position 2:   Point at angle 2θ
...
```

In high dimensions (128D), this creates a complex spiral structure where:
- Nearby points are close in space
- Distant points maintain angular relationships
- Relative distances are preserved

---

## Summary: Why RoPE is Brilliant

RoPE represents a fundamental breakthrough in position encoding by recognizing that **rotation geometry naturally encodes relative position information**.

---

### 🎯 Key Insights Recap

1. **Geometric Foundation**
   - Position encoded as rotation angle
   - Relative distance emerges from rotation properties
   
2. **Multiple Frequencies**
   - Fast frequencies: nearby relationships
   - Slow frequencies: long-range relationships
   - Logarithmic spacing covers all scales

3. **Complex Number Elegance**
   - e^(iθ) = cos(θ) + i·sin(θ)
   - Single multiplication instead of matrix ops
   - Cleaner implementation

4. **Mathematical Property**
   - e^(iθᵢ) × conj(e^(iθⱼ)) = e^(i(θᵢ-θⱼ))
   - Relative position (i-j) emerges automatically
   - No explicit computation needed

5. **Zero Parameters**
   - No learned position embeddings
   - Just mathematical formula
   - Enables extrapolation

6. **Production Ready**
   - Used in LLaMA, PaLM, Falcon, Mistral
   - Well-tested and proven
   - Efficient GPU implementation

---

### 📊 The Complete Picture

```
Token Embedding
      ↓
  No position info yet!
      ↓
Multi-Head Attention:
      ↓
  Q, K projection
      ↓
  ⭐ Apply RoPE ⭐
  - Split into pairs
  - Convert to complex
  - Multiply by e^(i·position·θ)
  - Convert back to real
      ↓
  Now Q, K have position info!
      ↓
  Compute attention: Q·K^T
  - Relative positions automatically encoded
      ↓
  Rest of attention (softmax, multiply V)
      ↓
  Output with position awareness!
```

---

### 🔬 Comparison: Before and After RoPE

**Without RoPE (absolute positions):**
- ❌ Learned embeddings: max_seq_len × d_model parameters
- ❌ Can't extrapolate beyond training length
- ❌ Absolute positions, not relative
- ❌ Must retrain for longer contexts

**With RoPE:**
- ✅ Zero additional parameters
- ✅ Extrapolates to longer sequences
- ✅ Naturally encodes relative positions
- ✅ Just apply the formula at any length!

---

### 🏆 Why RoPE Became the Standard

1. **Simplicity**: Just rotation, a concept from basic geometry
2. **Elegance**: Complex numbers make implementation clean
3. **Effectiveness**: Empirically works better than alternatives
4. **Efficiency**: No extra parameters to store or learn
5. **Flexibility**: Easy to extend (scaling, YaRN, etc.)

**Result:** Nearly every major LLM since 2021 uses RoPE.

---

## Appendix: Key Formulas Reference

### RoPE Frequency Formula
```
θᵢ = base^(-2i / d)

where:
- i ∈ [0, 1, 2, ..., d/2 - 1]  (dimension pair index)
- base = 10000 (typical choice)
- d = model dimension or head_dim
```

### Rotation via Complex Numbers
```
z = x + iy  (complex representation of 2D vector)

z_rotated = z × e^(i·θ·m)

where:
- m = position index
- θ = frequency for this dimension pair
- i = imaginary unit (√-1)
```

### Euler's Formula
```
e^(iθ) = cos(θ) + i·sin(θ)
```

### Relative Position Emergence
```
score = q_rotated × conj(k_rotated)
      = q × conj(k) × e^(i·θ·(pos_q - pos_k))
                               ↑
                        Relative distance!
```

### Multi-Head Integration
```
Q_rotated = apply_rope(Q, positions)  # Apply to queries
K_rotated = apply_rope(K, positions)  # Apply to keys
V stays unchanged                      # Don't apply to values

attention = softmax(Q_rotated · K_rotated^T / √d_k) · V
```

---

## Further Reading

**Original Paper:**
- Su et al. (2021): "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- https://arxiv.org/abs/2104.09864

**Applications:**
- Touvron et al. (2023): "LLaMA: Open and Efficient Foundation Language Models"
- Chowdhery et al. (2022): "PaLM: Scaling Language Modeling with Pathways"

**Extensions:**
- poe et al. (2023): "YaRN: Efficient Context Window Extension of Large Language Models"
- Chen et al. (2023): "Extending Context Window via Position Interpolation"

---

**This position encoding method - born from geometric intuition and complex analysis - has become foundational to modern AI. Understanding RoPE means understanding how state-of-the-art LLMs handle the structure of language.**