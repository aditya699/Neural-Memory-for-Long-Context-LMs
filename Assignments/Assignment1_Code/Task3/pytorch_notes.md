1.Think of nn.Module as a factory that provides infrastructure (parameter tracking, GPU support, etc.). By calling super().__init__(), you're saying: "Hey parent class, set up all your infrastructure before I add my custom stuff."

2.super().__init__() is a Python concept for calling the parent class's constructor

3.When you call `model(input)`, PyTorch automatically:
1. Calls `forward(input)` 
2. Tracks gradients (for backward pass later)
3. Returns the output

4.## **Forward vs Backward:**

**Forward pass:** Input → Output (what we just talked about)

**Backward pass:** Calculate gradients, flows backward
```
Input → Output
      ← Gradients flow back

5.nn.Parameter = tells PyTorch "this is learnable"
requires_grad=True = track gradients for training
Random initialization with torch.randn()

6.
In practice, we use nn.Linear because:

Neural networks do: y = xW^T + b
nn.Linear does this automatically
Handles weight initialization properly
More efficient than manual x * weight + bias


7.
Why nn.Linear is better:

Proper initialization - Xavier/Kaiming initialization built-in
Matrix multiplication - Uses optimized BLAS operations
Handles dimensions - Works with batches automatically
Less code - Don't manually track W and b

8
So YES, when building transformers:

We use nn.Linear for Q, K, V projections
We use nn.Linear for FFN layers
We use nn.Linear for output projection

9.
How nn.Linear works:
Math: y = xW^T + b
Where:

x: input [batch, input_dim]
W: weight matrix [output_dim, input_dim]
W^T: transpose of W [input_dim, output_dim]
b: bias vector [output_dim]
y: output [batch, output_dim]

The transpose (^T) is important!

10.
self.weight = nn.Parameter(...)
PyTorch's magic __setattr__() method detects:

"Oh, this is an nn.Parameter"
"I need to register this for optimization"
"Add it to my internal list of parameters"

11.
So now we have 3 types of things in nn.Module:

Parameter - learnable, gets gradients, optimizer updates
Buffer - not learnable, no gradients, but still saved in model
Regular attribute - doesn't get saved, not part of model

