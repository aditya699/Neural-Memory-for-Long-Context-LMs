1. CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model that allows you to use NVIDIA GPUs for general-purpose computing, not just graphics.

2. nn.Linear is PyTorch's fully connected (dense) layer that performs a linear transformation on the input. i.e y = xW^T + b

3. When we say "input projection" we mean we are projecting the input matrix into a different matrix space.

4. nn.Module is the base class in PyTorch - when we inherit from it, we get automatic parameter tracking, GPU movement (.to('cuda')), and train/eval modes.

5. super().__init__() calls the parent class constructor - REQUIRED to initialize nn.Module features.

6. self.variable creates an instance variable that persists and can be accessed anywhere in the class.

7. nn.Dropout(p) randomly sets p% of values to zero during training to prevent overfitting.

8. .view() reshapes a tensor without changing the underlying data - just reorganizes how we look at it.

9. .transpose(dim1, dim2) swaps two dimensions of a tensor. We use it to rearrange axes for computation.

10. torch.matmul() performs matrix multiplication. For attention: Q × K^T gives us attention scores.

11. Scaling by √(head_dim) prevents the dot products from getting too large, which would make softmax saturate.

12. torch.softmax(x, dim=-1) converts scores to probabilities along the last dimension - each row sums to 1.0.

13. Attention scores shape: (batch, heads, seq_len, seq_len) - each position attends to all positions.

14. After softmax, we get attention weights - probability distributions showing how much each token focuses on other tokens.