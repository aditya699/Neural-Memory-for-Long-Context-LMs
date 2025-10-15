# Understanding Transformers: From Embeddings to Attention

A comprehensive guide to how transformer models predict text, from token embeddings to self-attention mechanisms.

---

## Step 1 — Where Everything Ends: Predicting the Next Token

When a transformer like GPT writes text, everything it does ultimately boils down to one thing:

> Given the previous tokens `[x₁, x₂, ..., xₜ]`, it must produce a probability distribution over what comes next:
> 
> ```
> P(x_{t+1} | x₁, x₂, ..., xₜ)
> ```

That's the **single prediction task** repeated thousands of times per second.

---

### 🧩 How does it get there?

The model doesn't know language. It only knows **vectors** — numbers that capture relationships it has learned from data.

To predict `x_{t+1}`:

1. Each previous token ("The", "cat", "sat", …) has an **embedding vector**
2. These vectors interact through **self-attention**, letting each token "see" the context around it
3. The model uses this *context-aware representation* to output logits (scores) for every word in the vocabulary
4. Those logits are passed through a **softmax** to become probabilities

Then the model samples or picks the highest one — that's your next token.

---

### Example in Action

**Prompt:** "The cat sat on the …"

Internally, the model is asking:

> "Given everything I've seen — *the cat sat on the* — which word statistically fits best next?"

The self-attention mechanism is the part that lets "sat" remember *who sat* (cat) and *where sat* (on something), so that when the model predicts the next token, **'mat'** gets the highest probability.

---

## Step 2 — Where It Begins: Embeddings as the Model's Language

Before the model can *think*, it must *represent* words numerically. That's the first transformation: turning discrete tokens into continuous geometry.

---

### 🧩 What is an embedding?

Each token in the vocabulary — say *"the"*, *"cat"*, *"sat"*, *"mat"* — is mapped to a vector of fixed size, say 768 dimensions (for small models, maybe 128).

We call this matrix **E**, the *embedding matrix*, with one row per token. If the vocabulary has 50,000 tokens and each embedding is 768-dimensional, then:

```
E ∈ ℝ^(50000 × 768)
```

When you feed a sentence, for example:

> "The cat sat on the mat"

the model looks up the corresponding rows in **E** and builds a sequence:

```
x₁, x₂, x₃, x₄, x₅, x₆
```

So "The" might become a vector like `[0.12, -0.33, 0.85, ...]` — a *dense fingerprint* of meaning.

---

### 🧠 Why continuous vectors?

Because similarity can now be *measured*.

- "cat" and "dog" will have vectors pointing in roughly similar directions (semantic neighbors)
- "cat" and "banana" will be far apart

This geometry gives the model a space to reason in. Language becomes *numbers that preserve relationships*.

---

### 🧩 Adding position

Here's the key difference from a bag of words: Self-attention itself is **permutation invariant** — it doesn't know order. So we add **positional encodings**, unique vectors that mark each word's location in the sequence.

Thus each input token's final vector is:

```
xᵢ = embedding(tokenᵢ) + positional_encoding(i)
```

That's what enters the self-attention mechanism.

---

## Step 3 — Why We Create Q, K, and V

We now have our sequence of embeddings:

```
x₁, x₂, ..., xₜ
```

each a 768-dimensional vector carrying the word's meaning and its position. But these vectors alone don't tell us *how one word should look at another*. They're just static representations.

The model needs to learn *how to query this memory of tokens*.

---

### 🧠 The Core Idea

Each token plays three roles at once:

1. It is **a thinker** — it wants to know which other tokens are useful  
   → that's the **Query (Q)**

2. It is **an address in memory** — other tokens may want to look at it  
   → that's the **Key (K)**

3. It is **a carrier of information** — something worth being read  
   → that's the **Value (V)**

So we learn three different linear projections:

```
Qᵢ = xᵢ Wq
Kᵢ = xᵢ Wk  
Vᵢ = xᵢ Wv
```

where `Wq`, `Wk`, `Wv` are distinct weight matrices (each `768 × 64`, for example).

---

### 🧩 Why three separate projections?

Because the *way* a token asks questions is not the same as the *way* it stores answers.

**Imagine a classroom:**

- Every student (token) has some knowledge (its **Value**)
- Each also has a particular phrasing of its question (its **Query**)
- And it also defines keywords by which it can be found (its **Key**)

During computation, every token's Query is compared to all other tokens' Keys. That comparison produces a score: *how relevant is that token to me right now?*

---

### Matrix Form

In matrix form:

```
Q = X Wq
K = X Wk
V = X Wv
```

where `X` is your whole sequence `[x₁, x₂, ..., xₜ]`.

These three new sets of vectors are the raw material for computing **attention weights**.

---

## Step 4 — How Tokens "Look" at Each Other (Computing Attention Scores)

Now that we have:

```
Q = X Wq  
K = X Wk  
V = X Wv
```

each token has:

- a **Query vector** → how it looks for information
- a **Key vector** → how it describes itself to others
- a **Value vector** → what content it offers

The first operation of attention is to compare every Query with every Key.

---

### 🔍 The Comparison Mechanism

For every pair of tokens `(i, j)`, we compute:

```
score_ij = Qᵢ · Kⱼ
```

This is a **dot product** — a measure of alignment or similarity in vector space.

If `Qᵢ` and `Kⱼ` point in similar directions, the model believes token *j* is relevant to token *i*.

---

### 💡 Intuitive Picture

Let's reuse the example:

> "The cat sat on the mat."

When computing the representation for "sat":

- Its Query vector `Q_sat` is compared with all Keys
- The dot product with `K_cat` might be large (subject–verb link)
- The dot product with `K_mat` might also be moderately high (verb–object link)
- The dot product with `K_the` is small (article, low semantic value)

So for token "sat," the raw scores might look like:

| Token | Q_sat · Kⱼ | Interpretation            |
|:------|:-----------|:--------------------------|
| the   | 0.3        | weak relation             |
| cat   | 2.1        | strong relation (subject) |
| sat   | 1.2        | medium self-relation      |
| on    | 1.6        | medium (context)          |
| mat   | 1.8        | strong (object)           |

These numbers are **not yet probabilities** — they are unnormalized attention logits.

---

### ⚙️ In Matrix Form

All pairwise comparisons are computed at once:

```
Scores = Q Kᵀ
```

If your sequence has `T` tokens, this gives a `T × T` matrix — each row shows **how one token attends to all others**.

---

## Step 5 — Scaling and Normalizing the Attention Scores

We now have a full matrix of raw pairwise similarities:

```
Scores = Q Kᵀ
```

Each number tells how strongly one token relates to another. But there's a hidden problem: **the values can get too large**.

---

### ⚠️ The Problem: Exploding Magnitudes

Each Query and Key might be 64 or 128 dimensions wide. When we take their dot product, the expected magnitude of the sum increases roughly with the dimension `dₖ`.

So if we don't scale these values down, the range of the scores grows large, and when we later apply the softmax, it becomes **overconfident** — the largest score dominates completely.

This kills gradient flow, because the softmax becomes almost a step function (saturated).

---

### 🧮 The Fix: Scale by √dₖ

To stabilize the distribution, we divide every score by the square root of the key dimension:

```
ScaledScores = (Q Kᵀ) / √dₖ
```

This ensures that the variance of the dot product remains roughly constant, keeping gradients smooth and learning stable.

---

### 📊 Example

Continuing with our earlier "sat" example, suppose `dₖ = 64`. Then:

```
√dₖ = 8
```

Our raw scores (from before):

```
the: 0.3, cat: 2.1, sat: 1.2, on: 1.6, mat: 1.8
```

After scaling:

```
the: 0.0375, cat: 0.2625, sat: 0.15, on: 0.20, mat: 0.225
```

This keeps them in a narrow, healthy range for softmax to work effectively.

---

### 🔁 Step Summary

Scaling by `√dₖ` is not arbitrary — it's a variance normalization step ensuring:

- Stable training dynamics
- Smooth attention distributions
- Better gradient propagation

**Without it**, the softmax would saturate and the model would stop learning meaningful attention patterns.

---

## Step 6 — Turning Scores into Attention Weights (Softmax Stage)

We have our scaled similarity matrix:

```
ScaledScores = (Q Kᵀ) / √dₖ
```

Each row corresponds to one token's view of all others — for instance, how "sat" relates to "the", "cat", "on", "mat", etc. But right now these are just raw numbers — they don't yet tell us **how much attention to allocate**.

We need to convert these arbitrary real values into a normalized distribution of focus. That's where **softmax** comes in.

---

### 🧮 Formula

For each token `i`:

```
aᵢⱼ = exp(ScaledScoresᵢⱼ) / Σₖ exp(ScaledScoresᵢₖ)
```

Now every row `aᵢ = [aᵢ₁, aᵢ₂, ..., aᵢₜ]` sums to 1.0. Each `aᵢⱼ` can be read as:

> "How much token i attends to token j."

---

### 🧠 Example

Take our "sat" row again (after scaling):

```
the: 0.0375
cat: 0.2625
sat: 0.150
on : 0.200
mat: 0.225
```

Apply softmax:

```
aᵢⱼ = softmax(0.0375, 0.2625, 0.150, 0.200, 0.225)
```

Exponential each and normalize:

| Token | exp(value) | normalized |
|:------|:-----------|:-----------|
| the   | 1.038      | 0.157      |
| cat   | 1.300      | 0.197      |
| sat   | 1.162      | 0.176      |
| on    | 1.222      | 0.185      |
| mat   | 1.252      | 0.192      |

All add up to ~1.0.

Now the model has a **probability distribution** — "sat" gives about 19.7% of its focus to "cat", 19.2% to "mat", etc.

**These are the attention weights.**

---

### 🔍 Interpretation

- **Softmax converts similarity into focus**
- The model doesn't choose one token — it **blends many**, but with learned proportions
- Because it's differentiable, gradients can tell the model later: *"Hey, you should've paid more attention to 'cat' here."*

That's how it learns to attend correctly through training.

---

## Step 7 — Weighted Summation: Building Contextual Representations

We've reached the stage where the model actually **mixes information**. The attention weights `A` tell us who looks at whom; the Values `V` tell us what each token offers.

So the final step of the attention operation is beautifully simple:

```
Z = A V
```

---

### 🧩 What this means

For each token `i`:

```
zᵢ = Σⱼ₌₁ᵀ aᵢⱼ Vⱼ
```

Each `aᵢⱼ` is a scalar weight — the degree to which token `i` cares about token `j`. Each `Vⱼ` is a vector (the content carried by token `j`).

The weighted sum gives us a new vector `zᵢ` — a **contextualized embedding**, where "i" has now absorbed relevant information from others.

---

### 🧠 Example: "sat" again

**Attention weights from before:**

| Token | weight | role                |
|:------|:-------|:--------------------|
| the   | 0.157  | small context       |
| cat   | 0.197  | subject (important) |
| sat   | 0.176  | self                |
| on    | 0.185  | linking word        |
| mat   | 0.192  | object (important)  |

**Values (simplified, 4-D embeddings):**

| Token | Vⱼ                   |
|:------|:---------------------|
| the   | [0.1, 0.0, 0.2, 0.0] |
| cat   | [0.6, 0.1, 0.3, 0.5] |
| sat   | [0.2, 0.8, 0.4, 0.1] |
| on    | [0.3, 0.3, 0.2, 0.4] |
| mat   | [0.4, 0.2, 0.1, 0.6] |

Now multiply and sum:

```
z_sat = 0.157·V_the + 0.197·V_cat + 0.176·V_sat + 0.185·V_on + 0.192·V_mat
```

**Result ≈ `[0.32, 0.30, 0.25, 0.36]`**

This is the new embedding for "sat", which now encodes:

- the subject ("cat")
- the object ("mat")
- some local syntax ("on")

**It's no longer an isolated word — it's context-aware meaning.**

---

### 🧩 Matrix View

All tokens do this simultaneously:

```
Z = softmax((Q Kᵀ) / √dₖ) V
```

That's the **famous attention formula** you see in every paper — compact but profound. 

It means: **every token blends all others, weighted by learned relational similarity**.

---

## Step 8 — Beyond Attention: The Complete Transformer Block

Self-attention gives us contextualized representations `Z`, but this is just **one component** of a transformer layer. Here's the complete flow:

---

### The Transformer Block Architecture

```
Input: x (embeddings from previous layer)
    ↓
┌─────────────────────────────────────┐
│  1. Self-Attention                  │
│     Z = softmax(QK^T/√dₖ) V         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  2. Add & Normalize (Residual)      │
│     x' = LayerNorm(x + Z)           │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  3. Feed-Forward Network            │
│     ffn_out = FFN(x')               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  4. Add & Normalize (Residual)      │
│     output = LayerNorm(x' + ffn_out)│
└─────────────────────────────────────┘
    ↓
Output: Ready for next layer
```

---

### Why This Architecture?

**Residual Connections (`x + Z`)**
- Enable gradient flow through deep networks (32+ layers)
- Allow each layer to learn **refinements** rather than complete transformations
- First introduced in ResNets, crucial for training stability
- Without residuals, gradients vanish in deep networks

**Layer Normalization**
- Stabilizes training by normalizing across feature dimensions
- Modern models use **RMSNorm** (simpler, faster than LayerNorm)
- Placed **before** attention (Pre-Norm) in modern architectures like Llama
- Keeps activations in a healthy range for gradient computation

**Feed-Forward Network (FFN)**
- Processes each token **independently** (no cross-token interaction)
- Typically: `FFN(x) = W₂·σ(W₁·x + b₁) + b₂`
- Modern: Uses **SwiGLU** activation instead of ReLU/GELU
- Usually much larger than attention layer (4× the model dimension)
- Where the model stores factual knowledge

---

### Stacking Layers

A complete transformer model stacks **N** of these blocks (e.g., N=32 in Llama 2):

```
Token Embeddings + Positional Encoding (Step 2)
    ↓
Transformer Block 1
    ↓
Transformer Block 2
    ↓
    ...
    ↓
Transformer Block N
    ↓
Final Layer Normalization
    ↓
Linear Projection (to vocabulary size)
    ↓
Softmax → Sample Next Token (Step 1)
```

**This is how we get from embeddings back to predicting the next token!**

Each block refines the representations, building increasingly abstract and context-aware features:
- **Early layers**: Detect basic patterns (syntax, local dependencies)
- **Middle layers**: Build higher-level features (semantic relationships, entities)
- **Late layers**: Form abstract concepts (reasoning, world knowledge)

Until the final layer can predict which token comes next.

---

### 🔄 Completing the Circle

We started at **Step 1** asking: *"How does the model predict the next token?"*

Now we have the full answer:

1. **Embeddings** convert tokens to vectors
2. **Positional encodings** add sequence order
3. **Self-attention** lets tokens communicate (Steps 3-7)
4. **Residual + Norm** stabilizes training
5. **FFN** processes individual tokens
6. **Repeat N times** to build deep representations
7. **Project to vocabulary** and sample the next token

Self-attention is the **communication mechanism**. The complete transformer block is the **processing unit**. And stacking many blocks creates the **intelligence** we see in modern LLMs.

---

## 🧠 Attention & Neuroscience Parallels

### Attention as Biological Computation

Think of Q, K, V as an artificial analog of **selective attention** in the brain. In human cortex, neurons fire most strongly when a stimulus matches both a **sensory pattern** (what they detect) and a **task-relevant signal** (what you're looking for).

The dot-product `Qᵢ · Kⱼ` mimics this gating: *"activate me if what I'm sensing (Key) aligns with what I'm searching for (Query)."*

---

### Key Neural Parallels

**1. Hippocampal Pattern Completion**
- Partial cues (Query) → match stored patterns (Keys) → retrieve content (Values)
- Transformers implement this as differentiable memory addressing

**2. Biased Competition (Desimone & Duncan, 1995)**
- Multiple stimuli compete for neural representation
- Top-down signals bias competition toward relevant items
- Softmax implements this: Query provides bias, creates winner-take-more distribution

**3. Gain Modulation in Visual Cortex**
- Attention scales neural firing rates multiplicatively
- Similar to: `neuron_response = baseline × attention_gain`
- Transformers: `zᵢ = Vⱼ × aᵢⱼ`

**4. The Cocktail Party Effect**
- Focus on one conversation while filtering others
- Self-attention: `softmax(Qᵢ·Kⱼ/√dₖ)` provides routing weights

---

### Why It Matters

**Marr's Levels**: Different implementations (matrix ops vs. neurons), same computational goal — **selectively amplify relevant information**.

---

## Summary

1. **Embeddings** convert discrete tokens into continuous vectors
2. **Positional encodings** add sequence order information
3. **Q, K, V projections** create specialized representations for attention computation
4. **Attention scores** are computed via dot products between Queries and Keys
5. **Self-attention** uses these scores to determine which tokens are relevant to each other
6. **Softmax over logits** produces the final probability distribution for the next token

The transformer architecture chains these operations together, layer after layer, to build increasingly sophisticated representations that capture the patterns of language.