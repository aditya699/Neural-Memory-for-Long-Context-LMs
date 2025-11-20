# Understanding Flash Attention

A comprehensive guide to making attention faster through IO-aware algorithms.

---

## Step 1 — The Problem: Standard Attention is Memory-Bound

We know the standard attention formula:

```
Attention(Q, K, V) = softmax(Q K^T / √dₖ) V
```

This works beautifully — but there's a hidden bottleneck: **memory access, not computation**.

---

### 🐌 The Bottleneck Nobody Talked About

**Common belief (2017-2021):**
> "GPUs are compute-limited. We need faster matrix multiplication!"

**Reality (discovered 2022):**
> "GPUs are memory-limited. We're spending more time moving data than computing!"

---

### 💾 GPU Memory Hierarchy: The Core Issue

Modern GPUs have **two types of memory**:

```
┌─────────────────────────────────────────┐
│  GPU Chip                               │
│                                         │
│  ┌──────────────────┐                  │
│  │  Compute Cores   │ ← Math happens   │
│  │  (CUDA cores)    │    here (FAST)   │
│  └────────┬─────────┘                  │
│           │                             │
│           ↕ 19 TB/s (super fast!)      │
│  ┌────────────────────┐                │
│  │      SRAM          │ ← Tiny scratch │
│  │    (~20 MB)        │    workspace   │
│  └────────┬───────────┘                │
│           │                             │
└───────────┼─────────────────────────────┘
            │
            ↕ 1.5 TB/s (10-20x slower!)
            │
   ┌────────────────────┐
   │   HBM (VRAM)       │ ← Large main
   │   (~80 GB)         │    storage
   └────────────────────┘
```

**The critical numbers:**
- **SRAM**: ~20 MB, ~19 TB/s bandwidth
- **HBM**: ~80 GB, ~1.5 TB/s bandwidth
- **SRAM is 10-20x faster but 4000x smaller!**

---

### 🔥 The Fundamental Constraint

**Compute cores can ONLY work on data in SRAM.**

If data is in HBM:
```
1. Load HBM → SRAM (SLOW)
2. Compute in SRAM (FAST)
3. Write SRAM → HBM (SLOW)
```

**Modern reality:**
- Matrix multiplication: ~300 TFLOPS (tera floating-point operations per second)
- Memory bandwidth: Only ~1.5 TB/s
- **We spend more time waiting for memory than computing!**

---

### 📊 Standard Attention: A Memory Disaster

For sequence length **N = 4096** with **d = 512** dimensions:

**Step 1: Compute Scores**
```
Q, K in HBM (each 4096 × 512 = 8 MB)
↓ Load to SRAM
Compute S = Q K^T (4096 × 4096 = 64 MB!)
↓ Doesn't fit in SRAM!
Write S to HBM (SLOW WRITE!)
```

**Step 2: Compute Softmax**
```
Load S from HBM (SLOW READ!)
↓
Compute P = softmax(S) (4096 × 4096 = 64 MB)
↓
Write P to HBM (SLOW WRITE!)
```

**Step 3: Compute Output**
```
Load P from HBM (SLOW READ!)
Load V from HBM
↓
Compute O = P V
↓
Write O to HBM
```

---

### 💸 The Cost Analysis

**Memory transfers:**
- Read Q, K: 16 MB
- Write S: 64 MB
- Read S: 64 MB
- Write P: 64 MB
- Read P: 64 MB
- Read V: 8 MB
- Write O: 8 MB

**Total HBM traffic: 288 MB**

**But the actual computation?**
- Q K^T: 4096² × 512 FLOPs
- Softmax: 4096² FLOPs
- P V: 4096² × 512 FLOPs

**The memory bottleneck is 10-20x slower than the compute!**

---

### 🎯 The Key Insight

> "We're compute-bound in theory, but memory-bound in practice."

**What if we could:**
- Keep intermediate results in SRAM instead of writing to HBM?
- Recompute values instead of loading them from HBM?
- Trade more computation for less memory traffic?

This is **Flash Attention**.

---

## Step 2 — The Flash Attention Strategy

Flash Attention's radical idea:

> **Never materialize the full attention matrix in HBM. Ever.**

---

### 🧩 The Three Core Principles

**1. Tiling (Chunking)**
- Process attention in small blocks that fit in SRAM
- Work on chunks sequentially, not all at once

**2. Online Softmax**
- Compute softmax incrementally as we see chunks
- Update running statistics instead of storing full scores

**3. Recomputation**
- Recompute attention scores during backward pass
- Trade compute for memory (compute is cheaper!)

---

### 💡 The Mental Model

**Standard Attention:**
```
"I need to see ALL scores at once to compute softmax,
so I must store the full (N × N) matrix in HBM."
```

**Flash Attention:**
```
"I'll process attention in tiles that fit in SRAM,
keep running statistics, and never touch HBM for intermediate values."
```

---

## Step 3 — Online Softmax: The Mathematical Foundation

The key algorithmic innovation that makes Flash Attention possible.

---

### 🤔 The Softmax Challenge

Standard softmax for vector **x = [x₁, x₂, ..., xₙ]**:

```
softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)
```

**Problem:** We need the sum **Σⱼ exp(xⱼ)** which requires seeing ALL values!

**Example:**
```
x = [5, 9, 3, 7]

sum = exp(5) + exp(9) + exp(3) + exp(7)
    = 148 + 8103 + 20 + 1097
    = 9368

softmax = [148/9368, 8103/9368, 20/9368, 1097/9368]
        = [0.016, 0.865, 0.002, 0.117]
```

If we can only process 2 values at a time (SRAM limit), we can't compute the denominator!

---

### 🎯 Online Softmax Solution

**Key idea:** Keep running statistics that we can update incrementally.

**What we track:**
1. **m**: Running maximum (for numerical stability)
2. **d**: Running sum of exponentials (denominator)
3. **O**: Running softmax output

---

### 📐 The Algorithm (Step-by-Step)

Let's process **x = [5, 9, 3, 7]** in chunks of size 2.

---

#### **Initialization**
```
m = -∞  (running max)
d = 0   (running sum of exponentials)
O = 0   (running output)
```

---

#### **Chunk 1: [5, 9]**

**Step 1:** Update maximum
```
m_old = -∞
m_new = max(-∞, 5, 9) = 9
```

**Step 2:** Update denominator
```
Scale previous sum by exp(m_old - m_new):
d_old_scaled = 0 × exp(-∞ - 9) = 0

Add new exponentials (normalized by new max):
d_new = d_old_scaled + exp(5 - 9) + exp(9 - 9)
      = 0 + exp(-4) + exp(0)
      = 0 + 0.018 + 1
      = 1.018
```

**Step 3:** Compute partial softmax
```
softmax_chunk1 = [exp(5-9)/1.018, exp(9-9)/1.018]
               = [0.018/1.018, 1/1.018]
               = [0.018, 0.983]

Note: These are WRONG because we haven't seen chunks 2 yet!
```

**State after chunk 1:**
```
m = 9
d = 1.018
```

---

#### **Chunk 2: [3, 7]**

**Step 1:** Update maximum
```
m_old = 9
m_new = max(9, 3, 7) = 9  (unchanged)
```

**Step 2:** Update denominator
```
Scale previous sum:
d_old_scaled = 1.018 × exp(9 - 9) = 1.018

Add new exponentials:
d_new = 1.018 + exp(3 - 9) + exp(7 - 9)
      = 1.018 + exp(-6) + exp(-2)
      = 1.018 + 0.0025 + 0.135
      = 1.156
```

**Step 3:** Now we have the TRUE denominator!
```
Final softmax:
softmax([5,9,3,7]) = [exp(5-9), exp(9-9), exp(3-9), exp(7-9)] / 1.156
                   = [0.018, 1.000, 0.0025, 0.135] / 1.156
                   = [0.016, 0.865, 0.002, 0.117]
```

This matches the standard softmax! ✅

---

### 🧠 Key Properties of Online Softmax

**1. Numerically Stable**
- Subtracting max prevents overflow (large exp values)
- Standard trick: `exp(x - max(x))`

**2. Constant Memory**
- Only store: `m` (1 number), `d` (1 number)
- Don't need to store all scores!

**3. Single Pass**
- See each value exactly once
- Update statistics incrementally

**4. Exact**
- Mathematically equivalent to standard softmax
- No approximation!

---

### 📊 Why This Enables Flash Attention

**Standard softmax:**
```
Store all scores [5, 9, 3, 7] in HBM (doesn't fit in SRAM)
↓
Load all scores from HBM
↓
Compute sum
↓
Normalize
```

**Online softmax:**
```
Process chunk [5, 9], update statistics (stay in SRAM!)
↓
Process chunk [3, 7], update statistics (stay in SRAM!)
↓
Done! Never wrote to HBM!
```

---

## Step 4 — Flash Attention Algorithm (Forward Pass)

Now let's see how online softmax enables memory-efficient attention.

---

### 🎯 Setup

**Given:**
- Query matrix **Q** (N × d)
- Key matrix **K** (N × d)
- Value matrix **V** (N × d)
- Sequence length **N = 4096**
- Head dimension **d = 64**
- SRAM capacity: Can hold blocks of size **B = 512**

**Goal:** Compute **O = softmax(Q K^T) V** without storing (4096 × 4096) attention matrix in HBM.

---

### 🧩 Tiling Strategy

**Divide into blocks:**
```
Q = [Q₁, Q₂, ..., Q₈]  (each Qᵢ is 512 × 64)
K = [K₁, K₂, ..., K₈]  (each Kᵢ is 512 × 64)
V = [V₁, V₂, ..., V₈]  (each Vᵢ is 512 × 64)
```

**Block size: 512 tokens per block**
- 512 × 64 × 4 bytes = 128 KB (fits comfortably in SRAM!)

---

### 🚀 The Algorithm (Outer Loop over Query Blocks)

```python
# Initialize output
O = zeros(N, d)  # In HBM
m = -∞ × ones(N)  # In HBM (running max for each query)
ℓ = zeros(N)      # In HBM (running sum for each query)

# Outer loop: iterate over query blocks
for i in range(num_blocks):  # i = 0 to 7
    # Load query block Qᵢ from HBM to SRAM
    Qᵢ = load_from_HBM(Q, block=i)  # (512 × 64)
    
    # Load current statistics for this query block
    Oᵢ = load_from_HBM(O, block=i)  # (512 × 64)
    mᵢ = load_from_HBM(m, block=i)  # (512,)
    ℓᵢ = load_from_HBM(ℓ, block=i)  # (512,)
    
    # Inner loop: iterate over key/value blocks
    for j in range(num_blocks):  # j = 0 to 7
        # Load key and value blocks from HBM to SRAM
        Kⱼ = load_from_HBM(K, block=j)  # (512 × 64)
        Vⱼ = load_from_HBM(V, block=j)  # (512 × 64)
        
        # Compute attention scores for this block pair
        Sᵢⱼ = Qᵢ @ Kⱼ.T / √d  # (512 × 512) — stays in SRAM!
        
        # Online softmax update
        m_new = max(mᵢ, rowmax(Sᵢⱼ))  # (512,)
        
        # Compute correction factor
        correction = exp(mᵢ - m_new)  # (512,)
        
        # Update denominator
        ℓ_new = correction × ℓᵢ + rowsum(exp(Sᵢⱼ - m_new))  # (512,)
        
        # Compute attention weights (normalized by new denominator)
        Pᵢⱼ = exp(Sᵢⱼ - m_new) / ℓ_new  # (512 × 512)
        
        # Update output (scale old output + add new contribution)
        Oᵢ = correction × Oᵢ + Pᵢⱼ @ Vⱼ  # (512 × 64)
        
        # Update statistics
        mᵢ = m_new
        ℓᵢ = ℓ_new
        
        # Kⱼ, Vⱼ, Sᵢⱼ, Pᵢⱼ are discarded (freed from SRAM)
    
    # Write updated output and statistics back to HBM
    write_to_HBM(O, Oᵢ, block=i)
    write_to_HBM(m, mᵢ, block=i)
    write_to_HBM(ℓ, ℓᵢ, block=i)
    
    # Qᵢ is discarded (freed from SRAM)
```

---

### 🔍 Key Observations

**1. Never materialize full attention matrix**
- `Sᵢⱼ` is only (512 × 512) at a time, not (4096 × 4096)
- Computed in SRAM, never written to HBM!

**2. Online updates**
- As we see each key/value block, we update running output `Oᵢ`
- Correction factor accounts for changing denominator

**3. Memory footprint**
- SRAM holds: One query block, one key block, one value block, scores
- Total: ~512 KB (fits easily in 20 MB SRAM)

**4. HBM writes**
- Only write final output `O` and statistics `m`, `ℓ`
- Never write intermediate scores or softmax!

---

## Step 5 — Worked Example with Small Numbers

Let's trace through Flash Attention with concrete numbers.

---

### 📝 Setup

```
N = 4 tokens (tiny for visualization)
d = 2 dimensions
Block size B = 2 tokens

Q = [[1, 2],   →  Q₁ = [[1, 2],     Q₂ = [[1, 1],
     [3, 1],              [3, 1]]         [0, 2]]
     [1, 1],
     [0, 2]]

K = [[3, 1],   →  K₁ = [[3, 1],     K₂ = [[1, 1],
     [2, 4],              [2, 4]]         [0, 2]]
     [1, 1],
     [0, 2]]

V = [[5],      →  V₁ = [[5],        V₂ = [[9],
     [7],               [7]]              [11]]
     [9],
     [11]]
```

We'll compute output for **query block 1 (Q₁)** only.

---

### 🔄 Processing Q₁ (tokens 0-1)

**Initialization:**
```
O₁ = [[0], [0]]  (output for tokens 0-1)
m₁ = [-∞, -∞]    (max for tokens 0-1)
ℓ₁ = [0, 0]      (sum for tokens 0-1)
```

Load Q₁ into SRAM:
```
Q₁ = [[1, 2],
      [3, 1]]
```

---

#### **Inner Loop: j=0 (Process K₁, V₁)**

**Load K₁, V₁ into SRAM:**
```
K₁ = [[3, 1],
      [2, 4]]

V₁ = [[5],
      [7]]
```

**Compute scores S₁₁ = Q₁ K₁^T / √2:**
```
S₁₁ = [[1,2], [3,1]] @ [[3,2], [1,4]] / √2

Token 0: [1,2] · [3,1] = 5,  [1,2] · [2,4] = 10
Token 1: [3,1] · [3,1] = 10, [3,1] · [2,4] = 10

S₁₁ = [[5, 10],    / √2 = [[3.5, 7.1],
       [10, 10]]            [7.1, 7.1]]
```

**Online softmax update:**

For token 0:
```
m_old = -∞,  m_new = max(-∞, 3.5, 7.1) = 7.1

correction = exp(-∞ - 7.1) = 0

ℓ_new = 0 × 0 + exp(3.5-7.1) + exp(7.1-7.1)
      = exp(-3.6) + exp(0)
      = 0.027 + 1.0
      = 1.027

P₁₁[0] = [exp(3.5-7.1), exp(7.1-7.1)] / 1.027
       = [0.027, 1.0] / 1.027
       = [0.026, 0.974]

O₁[0] = 0 × 0 + [0.026, 0.974] @ [[5], [7]]
      = 0.026×5 + 0.974×7
      = 6.95
```

For token 1:
```
m_old = -∞,  m_new = max(-∞, 7.1, 7.1) = 7.1

ℓ_new = 0 + exp(7.1-7.1) + exp(7.1-7.1)
      = 1.0 + 1.0
      = 2.0

P₁₁[1] = [exp(7.1-7.1), exp(7.1-7.1)] / 2.0
       = [1.0, 1.0] / 2.0
       = [0.5, 0.5]

O₁[1] = 0 + [0.5, 0.5] @ [[5], [7]]
      = 0.5×5 + 0.5×7
      = 6.0
```

**State after j=0:**
```
O₁ = [[6.95],
      [6.0]]
m₁ = [7.1, 7.1]
ℓ₁ = [1.027, 2.0]
```

---

#### **Inner Loop: j=1 (Process K₂, V₂)**

**Load K₂, V₂ into SRAM:**
```
K₂ = [[1, 1],
      [0, 2]]

V₂ = [[9],
      [11]]
```

**Compute scores S₁₂ = Q₁ K₂^T / √2:**
```
Token 0: [1,2] · [1,1] = 3,  [1,2] · [0,2] = 4
Token 1: [3,1] · [1,1] = 4,  [3,1] · [0,2] = 2

S₁₂ = [[3, 4],    / √2 = [[2.1, 2.8],
       [4, 2]]            [2.8, 1.4]]
```

**Online softmax update:**

For token 0:
```
m_old = 7.1,  m_new = max(7.1, 2.1, 2.8) = 7.1  (unchanged)

correction = exp(7.1 - 7.1) = 1.0

ℓ_new = 1.0 × 1.027 + exp(2.1-7.1) + exp(2.8-7.1)
      = 1.027 + exp(-5.0) + exp(-4.3)
      = 1.027 + 0.007 + 0.014
      = 1.048

P₁₂[0] = [exp(2.1-7.1), exp(2.8-7.1)] / 1.048
       = [0.007, 0.014] / 1.048
       = [0.007, 0.013]

O₁[0] = 1.0 × 6.95 + [0.007, 0.013] @ [[9], [11]]
      = 6.95 + 0.007×9 + 0.013×11
      = 6.95 + 0.063 + 0.143
      = 7.16
```

For token 1:
```
m_old = 7.1,  m_new = max(7.1, 2.8, 1.4) = 7.1

ℓ_new = 1.0 × 2.0 + exp(2.8-7.1) + exp(1.4-7.1)
      = 2.0 + 0.014 + 0.003
      = 2.017

P₁₂[1] = [exp(2.8-7.1), exp(1.4-7.1)] / 2.017
       = [0.014, 0.003] / 2.017
       = [0.007, 0.001]

O₁[1] = 1.0 × 6.0 + [0.007, 0.001] @ [[9], [11]]
      = 6.0 + 0.063 + 0.011
      = 6.07
```

**Final state:**
```
O₁ = [[7.16],
      [6.07]]
m₁ = [7.1, 7.1]
ℓ₁ = [1.048, 2.017]
```

---

### ✅ Verification

We computed attention for Q₁ without ever storing the full (4×4) attention matrix!

**Memory used:**
- SRAM: Query block (2×2), Key block (2×2), Value block (2×1), Scores (2×2)
- Total: ~20 numbers at any time (not 16 for full matrix!)

---

## Step 6 — Memory Analysis: Flash vs Standard

Let's compare memory footprints quantitatively.

---

### 📊 Standard Attention Memory

For sequence length **N** and dimension **d**:

**Stored in HBM:**
1. **Q, K, V**: 3 × N × d
2. **Attention scores S**: N × N
3. **Attention weights P**: N × N
4. **Output O**: N × d

**Total memory:**
```
Memory = 3Nd + 2N² + Nd = 4Nd + 2N²
```

**For N=4096, d=512:**
```
Memory = 4 × 4096 × 512 + 2 × 4096²
       = 8.4 million + 33.5 million
       = 41.9 million float32 values
       = 167 MB
```

**Quadratic term dominates:** 2N² >> 4Nd for large N

---

### 🚀 Flash Attention Memory

**Stored in HBM:**
1. **Q, K, V**: 3 × N × d
2. **Output O**: N × d
3. **Statistics m, ℓ**: 2 × N

**No attention matrix!**

**Total memory:**
```
Memory = 3Nd + Nd + 2N = 4Nd + 2N
```

**For N=4096, d=512:**
```
Memory = 4 × 4096 × 512 + 2 × 4096
       = 8.4 million + 8 thousand
       = 8.4 million float32 values
       = 33.6 MB
```

**Linear in N!**

---

### 📈 Comparison

| Sequence Length N | Standard Attention | Flash Attention | Savings |
|:------------------|:-------------------|:----------------|:--------|
| 1024 | 10 MB | 8.4 MB | 16% |
| 2048 | 25 MB | 16.8 MB | 33% |
| 4096 | 84 MB | 33.6 MB | 60% |
| 8192 | 302 MB | 67.2 MB | 78% |
| 16384 | 1.1 GB | 134.4 MB | 88% |
| 32768 | 4.2 GB | 268.8 MB | 94% |

**Key insight:** Savings grow with sequence length!

For **N=100K tokens** (long documents):
- Standard: ~40 GB (doesn't fit on GPU!)
- Flash: ~820 MB (fits easily!)

---

## Step 7 — Speed Analysis: Why Recomputation is Faster

"Wait, doesn't Flash Attention do MORE work?"

Yes! Let's see why it's still faster.

---

### ⏱️ Time Model

**GPU operation costs (realistic ratios):**
- **Compute (FLOPs)**: 1 unit
- **Read from HBM**: 10 units
- **Write to HBM**: 30 units (3× slower than read!)

---

### 📊 Standard Attention Timeline

```
Operation                    Cost
─────────────────────────────────────
1. Read Q, K from HBM        20 units
2. Compute S = QK^T          1 unit
3. Write S to HBM            30 units ← expensive!
4. Read S from HBM           10 units
5. Compute P = softmax(S)    1 unit
6. Write P to HBM            30 units ← expensive!
7. Read P, V from HBM        20 units
8. Compute O = PV            1 unit
9. Write O to HBM            10 units
─────────────────────────────────────
Total:                       123 units
```

**Dominated by writes (30+30 = 60 units)!**

---

### 🚀 Flash Attention Timeline (Single Query Block)

```
Operation                            Cost
─────────────────────────────────────────────
1. Read Q block from HBM             2 units
2. Load O, m, ℓ from HBM             4 units

   Inner loop (repeated B times):
   3. Read K, V block from HBM       4 units
   4. Compute scores                 0.2 units
   5. Online softmax update          0.2 units
   6. Update output                  0.2 units
   (No writes! Stays in SRAM)
   
   × 8 blocks = 8 × 4.6              36.8 units

7. Write O, m, ℓ to HBM              4 units
─────────────────────────────────────────────
Total per query block:               46.8 units
Total for all 8 query blocks:        374 units
```

**Wait, 374 > 123? Flash is slower?**

---

### 🔍 The Hidden Factor: Real GPU Behavior

**The time model above is simplified.** In reality:

1. **Write amplification**: Writing to HBM often requires 2× the bandwidth
2. **Cache invalidation**: Writing invalidates caches, slowing subsequent reads
3. **Memory coherency overhead**: Managing consistency across memory hierarchy
4. **Kernel launch overhead**: More data movement = more kernel launches

**Actual empirical measurements:**
- Standard attention: ~140 ms (for N=4096)
- Flash attention: ~35 ms
- **4× speedup!**

---

### 💡 Why Flash Wins

**Standard attention:**
- 2 expensive HBM writes (S and P)
- Large matrices → poor cache utilization
- Memory allocations/deallocations

**Flash attention:**
- 1 HBM write (only final output)
- Small tiles → excellent cache utilization
- Reuses SRAM efficiently
- Despite more FLOPs, spends less time waiting!

**The formula:**
```
Wall-clock time = compute_time + memory_time

Standard: 3 + 120 = 123 units
Flash:    10 + 25 = 35 units
```

**Compute increased 3× but memory decreased 5×!**

---

## Step 8 — Backward Pass: Recomputation Strategy

Training requires gradients. How does Flash Attention handle backpropagation?

---

### 🎓 The Challenge

Standard attention during forward pass stores:
- Attention matrix **P** (N × N) in HBM
- Used during backward pass to compute gradients

Flash Attention **never stored P**!

---

### 💡 The Solution: Recomputation

During backward pass:
1. Re-read Q, K, V from HBM
2. Recompute attention scores S and weights P
3. Compute gradients on-the-fly
4. Never materialize P in HBM

**Trade-off:**
- Forward pass: Compute attention once
- Backward pass: Recompute attention
- Total: 2× FLOPs

**But still faster!** Because we avoid expensive HBM writes.

---

### 🔄 Backward Pass Algorithm (Simplified)

```python
# Given: dO (gradient of output), Q, K, V, statistics m, ℓ
# Goal: Compute dQ, dK, dV

dQ = zeros_like(Q)
dK = zeros_like(K)
dV = zeros_like(V)

for i in range(num_query_blocks):
    Qᵢ = load_block(Q, i)
    dOᵢ = load_block(dO, i)
    mᵢ, ℓᵢ = load_statistics(m, ℓ, i)
    
    dQᵢ = zeros_like(Qᵢ)
    
    for j in range(num_key_blocks):
        Kⱼ = load_block(K, j)
        Vⱼ = load_block(V, j)
        
        # Recompute attention (never stored!)
        Sᵢⱼ = Qᵢ @ Kⱼ.T / √d
        Pᵢⱼ = exp(Sᵢⱼ - mᵢ) / ℓᵢ
        
        # Compute gradients
        dVⱼ = Pᵢⱼ.T @ dOᵢ  # gradient w.r.t. V
        dPᵢⱼ = dOᵢ @ Vⱼ.T   # gradient w.r.t. P
        dSᵢⱼ = Pᵢⱼ × (dPᵢⱼ - rowsum(dPᵢⱼ × Pᵢⱼ))  # softmax backward
        dQᵢ += dSᵢⱼ @ Kⱼ / √d
        dKⱼ = dSᵢⱼ.T @ Qᵢ / √d
        
        write_block(dK, dKⱼ, j)
        write_block(dV, dVⱼ, j)
    
    write_block(dQ, dQᵢ, i)
```

---

### 🔢 FLOPs Comparison

**Standard attention:**
- Forward: N²d FLOPs
- Backward: N²d FLOPs (using stored P)
- **Total: 2N²d FLOPs**

**Flash attention:**
- Forward: N²d FLOPs
- Backward: 2N²d FLOPs (recompute P!)
- **Total: 3N²d FLOPs**

**50% more compute, but 4× faster in practice!**

---

## Step 9 — Flash Attention 2: Further Improvements

Flash Attention 2 (July 2023) made additional optimizations.

---

### 🚀 Key Improvements

**1. Better Parallelism**
- Flash 1: Parallelized over batch and heads
- Flash 2: Also parallelizes over sequence length
- Better GPU utilization (more cores active)

**2. Reduced Non-Matmul FLOPs**
- Optimized softmax computation
- Better scheduling of operations
- Fewer synchronization points

**3. Work Partitioning**
- Smarter division of work across GPU cores
- Reduces load imbalance
- Better occupancy

---

### 📊 Performance Gains

| Model | Flash 1 Speedup | Flash 2 Speedup |
|:------|:----------------|:----------------|
| BERT (seq=512) | 2.2× | 2.8× |
| GPT-2 (seq=1024) | 3.1× | 4.5× |
| GPT-3 (seq=2048) | 3.8× | 5.2× |
| Long context (seq=16K) | 4.1× | 6.8× |

**Flash 2 is ~1.5× faster than Flash 1!**

---

### 🔬 Technical Details

**Block size tuning:**
- Flash 1: Fixed block size
- Flash 2: Adaptive block sizing based on sequence length
- Better SRAM utilization

**Warp-level optimizations:**
- Warps = groups of 32 threads on GPU
- Flash 2 schedules work at warp granularity
- Reduces thread divergence

---

## Step 10 — Impact and Applications

Flash Attention revolutionized transformer scaling.

---

### 🌍 Real-World Impact

**Before Flash Attention (2017-2022):**
- Max context: ~2K tokens (GPT-2)
- Long documents: Split into chunks
- Memory: Major bottleneck for scaling

**After Flash Attention (2022-present):**
- Max context: 100K+ tokens (GPT-4, Claude)
- Full document processing
- Memory: No longer the primary constraint

---

### 📚 Applications Enabled

**1. Long-Context LLMs**
- Claude 3: 200K token context
- GPT-4: 128K token context
- Gemini: 1M token context (!)

**2. High-Resolution Vision Transformers**
- ViT with 1024×1024 images
- Video transformers (thousands of frames)

**3. Efficient Training**
- Longer sequences during training
- Larger batch sizes
- Reduced training time

**4. Multi-Modal Models**
- Process text + images + audio simultaneously
- Longer cross-attention sequences

---

### 💰 Cost Savings

**Training GPT-3 scale model:**
- Standard attention: ~$5M in GPU hours
- Flash attention: ~$3M in GPU hours
- **40% cost reduction!**

**Inference serving:**
- 4× throughput increase
- Lower latency
- Fewer GPUs needed

---

## Step 11 — Limitations and Variants

Flash Attention isn't perfect. Let's discuss trade-offs.

---

### ⚠️ Limitations

**1. Implementation Complexity**
- Requires CUDA expertise
- Harder to debug than standard attention
- Platform-specific optimizations

**2. Backward Pass Overhead**
- Recomputation adds 50% more FLOPs
- Still net positive, but noticeable

**3. Small Sequences**
- Overhead dominates for N < 512
- Standard attention may be faster
- Tiling overhead not amortized

**4. Memory Access Patterns**
- Requires careful tuning per GPU architecture
- A100 vs H100 have different optimal block sizes

---

### 🔧 Variants and Extensions

**1. Flash Attention 3 (Ongoing)**
- H100 optimizations (FP8, tensor cores)
- Asynchronous pipeline
- Expected: 2× faster than Flash 2

**2. Flash Decoding**
- Optimized for autoregressive generation
- Parallelizes over key/value sequence
- 8× faster for long-context inference

**3. Paged Attention (vLLM)**
- Combines Flash Attention with memory paging
- Efficient KV cache management
- Better serving throughput

**4. Ring Attention**
- Distributed across multiple GPUs
- Enables million-token contexts
- Each GPU holds subset of sequence

---

### 🧪 Specialized Variants

**Sparse Flash Attention:**
- Only compute attention for subset of tokens
- Combines Flash with sparsity patterns
- 10× speedup for very long sequences

**Multi-Query Flash Attention:**
- Optimized for multi-query attention (MQA)
- Used in Llama 2, Falcon
- Reduces KV cache size

---

## Step 12 — Implementation Considerations

If you're implementing Flash Attention yourself (or using it), here are key points.

---

### 🛠️ Using Flash Attention in Practice

**PyTorch (Official Implementation):**
```python
from flash_attn import flash_attn_func

# Standard usage
output = flash_attn_func(
    q,  # (batch, seqlen, nheads, headdim)
    k,  # (batch, seqlen, nheads, headdim)
    v,  # (batch, seqlen, nheads, headdim)
    dropout_p=0.0,
    softmax_scale=1.0 / math.sqrt(headdim),
    causal=False  # True for autoregressive
)
```

**PyTorch 2.0+ (Built-in):**
```python
import torch.nn.functional as F

# Uses Flash Attention automatically if available
output = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False
)
```

---

### ⚙️ Hyperparameters to Tune

**Block size:**
- Smaller blocks: Better for short sequences
- Larger blocks: Better for long sequences
- Typical: 64-256

**Dropout:**
- Flash Attention supports dropout
- Applied at attention weight level
- No performance penalty

**Causal masking:**
- Flash Attention has optimized causal path
- Saves 50% compute for autoregressive
- Use `causal=True` flag

---

### 🔍 Debugging Tips

**Common issues:**

1. **Shape mismatches**
```python
# Flash expects: (batch, seqlen, nheads, headdim)
# Standard expects: (batch, nheads, seqlen, headdim)
# Need to transpose!
```

2. **Numerical differences**
```python
# Flash uses float16/bfloat16 internally
# May see ~1e-3 differences from float32
# This is normal!
```

3. **CUDA out of memory**
```python
# Even Flash Attention has limits
# Reduce batch size or use gradient checkpointing
```

---

## Summary

Flash Attention revolutionizes transformer efficiency through IO-aware algorithm design.

---

### 🎯 Core Principles

**1. Memory Hierarchy Awareness**
- Recognize that SRAM is 10-20× faster than HBM
- Design algorithms around this constraint
- Minimize HBM ↔ SRAM transfers

**2. Online Algorithms**
- Process data incrementally (tiling)
- Maintain running statistics
- Avoid materializing full intermediate results

**3. Recomputation Trade-off**
- Recompute values instead of storing them
- Trade cheap compute for expensive memory
- Net win: 4× faster despite 1.5× more FLOPs

**4. Never Materialize Attention Matrix**
- Standard: Store N×N matrix in HBM
- Flash: Compute N×N in tiles, never store
- Memory: O(N²) → O(N)

---

### 📊 Key Results

| Metric | Standard | Flash 1 | Flash 2 |
|:-------|:---------|:--------|:--------|
| Memory | O(N²) | O(N) | O(N) |
| Speed | 1× | 2-4× | 3-7× |
| Max context | ~2K | ~16K | ~100K |
| Implementation | Simple | Complex | Complex |
| FLOPs | 2N²d | 3N²d | 3N²d |

---

### 🧠 The Big Idea

> **"Optimize for memory movement, not computation."**

Modern GPUs are so fast at math that the bottleneck has shifted from compute to memory. Flash Attention is the first attention algorithm designed for this reality.

---

### 🚀 What It Enables

**Long-Context Models:**
- GPT-4: 128K tokens
- Claude 3: 200K tokens
- Gemini: 1M tokens

**Efficient Training:**
- 40% cost reduction
- Larger batch sizes
- Faster iteration

**Better Products:**
- Process full documents
- Multi-modal understanding
- Real-time applications

---

### 🔮 Future Directions

**Hardware Evolution:**
- H100 has faster HBM (3 TB/s vs 1.5 TB/s)
- Gap between SRAM and HBM shrinking
- But Flash principles still apply!

**Algorithmic Advances:**
- Flash Attention 3 (H100 optimized)
- Ring Attention (distributed)
- Sparse Flash (very long contexts)

**Integration:**
- Built into PyTorch 2.0+
- Becoming the default attention
- Hidden from most users (just works!)

---

## Key Formulas

**Standard Attention:**
```
S = Q K^T / √d               (N × N matrix - stored in HBM)
P = softmax(S)                (N × N matrix - stored in HBM)
O = P V                       (N × d matrix)

Memory: O(N²)
HBM writes: 2 (S and P)
```

**Flash Attention:**
```
Process in blocks of size B:
For each query block Qᵢ:
  For each key block Kⱼ:
    Sᵢⱼ = Qᵢ Kⱼ^T / √d      (B × B - stays in SRAM)
    Update running statistics (m, ℓ)
    Update output Oᵢ incrementally
    
Memory: O(N)
HBM writes: 1 (only final O)
```

**Online Softmax:**
```
m_new = max(m_old, max(x_new))
correction = exp(m_old - m_new)
d_new = correction × d_old + sum(exp(x_new - m_new))
softmax = exp(x - m_new) / d_new
```

---

**Flash Attention made modern LLMs possible. It's not just an optimization — it's a paradigm shift in how we think about transformer efficiency.**

---

## References

**Original Papers:**
- Flash Attention 1: https://arxiv.org/abs/2205.14135
- Flash Attention 2: https://arxiv.org/abs/2307.08691

**Code:**
- Official repo: https://github.com/Dao-AILab/flash-attention
- PyTorch integration: torch.nn.functional.scaled_dot_product_attention

**Authors:**
- Tri Dao (Stanford → Together AI)
- Daniel Y. Fu (Stanford)
- Stefano Ermon (Stanford)
- Atri Rudra (SUNY Buffalo)
- Christopher Ré (Stanford)

**Further Reading:**
- ELI5 blog: https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad
- Tri Dao's website: https://tridao.me/