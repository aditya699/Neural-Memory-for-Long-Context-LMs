import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    """
    This is the standard multi-head attention class.
    """
    def __init__(self, d_model, num_heads, dropout=0.1):  # This runs when you create an object of this class
        super().__init__()  # This is used to call nn.module's init method which initializes the methods and attributes of the nn.module class
        assert d_model % num_heads == 0
        
        # We are storing all these so that they can be anywhere in the code
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
       
        # nn.Linear is PyTorch's fully connected (dense) layer that performs a linear transformation on the input.
        # It takes the input and multiplies it by a weight matrix and adds a bias term.
        # So it does a y=xw^T+b
        
        # So we need to create projections for Q, K, V (the parameters are input_dim, output_dim), so self.q_proj will create a weight matrix of size d_model x d_model,the weight initlization follows Xavier/Kaiming Initilication
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Post combination of all heads we need a final projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout helps us to randomly drop out some neurons to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        
    # This is the method which runs when you call the model
    def forward(self, x):
        # This is tuple unpacking
        batch_size, seq_len, _ = x.size()  # Fixed: using _ instead of d_model to avoid shadowing

        # Now we need to project the input matrix into a different matrix
        # So we need to create projections for Q, K, V
        # Q: What am i looking for?
        # K: What do i contain?
        # V: What information do i have?

        Q = self.q_proj(x)  # Query = x@W_q^T + b_q  #This actually calls the forward method
        K = self.k_proj(x)  # Key = x@W_k^T + b_k
        V = self.v_proj(x)  # Value = x@W_v^T + b_v
        
        # Now we wish to split the query, key and value matrices into multiple attention heads so that we can perform parallel computations
        # Now we are reshaping the matrix to (batch_size, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Now we need to transpose the matrix to put heads first
        # We are doing this since we want to compute attention for each head separately
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        # Scaling prevents softmax from saturating
        # scores[i,j]: how much token i should attend to token j high score means more attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Convert to probabilities
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply dropout to the attention weights
        attn_weights = self.dropout(attn_weights)

        # We need to multiply with V
        # (batch_size, num_heads, seq_len, seq_len) * (batch_size, num_heads, seq_len, head_dim)
        # Here we are taking combination of information from all the heads weighted by attention
        output = torch.matmul(attn_weights, V)
        
        # We need to concatenate heads back
        # This is done to transpose the output and make it contiguous in memory (since a simple transpose is not contiguous)
        output = output.transpose(1, 2).contiguous()
        # This is concatenation of heads
        output = output.view(batch_size, seq_len, self.d_model)  # Fixed: batch -> batch_size, d_model -> self.d_model

        # Final Projection
        output = self.out_proj(output)

        return output


# Test the implementation
if __name__ == "__main__":
    # Create model
    model = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)
    
    # Create input
    batch_size = 32
    seq_len = 10
    x = torch.randn(batch_size, seq_len, 512)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape:  {x.shape}")       # [32, 10, 512]
    print(f"Output shape: {output.shape}")   # [32, 10, 512]
    print("Multi-head attention works!")