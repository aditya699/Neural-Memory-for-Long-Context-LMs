import torch
import torch.nn as nn
import math


class MultiHeadLatentAttention(nn.Module): #Inheritence
    def __init__(self,d_model,num_heads,d_latent,dropout=0.1): #d_model is the dimension of the model,d_latent is the compression vector
        """
        Implmenting Multi-Head Latent Attention from scratch
        """
        super().__init__() #call the parent class constructor
        assert d_model % num_heads == 0 #assert is used to check if the condition is true
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_latent = d_latent  #Added  the compression dimension
        self.head_dim = d_model // num_heads

        #Compress Intialize the weight matrix for shared compression
        self.kv_compress=nn.Linear(d_model,d_latent)
        
        #Expand latent to K and V for all heads
        self.k_proj=nn.Linear(d_latent,d_model)
        self.v_proj=nn.Linear(d_latent,d_model)

        #NOTE:Q remains the same , there is no compression in Q
        self.q_proj=nn.Linear(d_model,d_model)
        
        #Output projection(This is after attention we need to concatenate all heads)
        self.out_proj=nn.Linear(d_model,d_model)

        #Create a dropout layer to prevent overfitting
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        #Unpack the input
        batch_size,seq_len,d_model=x.size()

        #Project Q (NOTE:Projection just means x.w+b)
        Q=self.q_proj(x)

        #:Compress input to latent space (THE KEY STEP!)
        # Here you compress the input into a smaller latent vector
        kv_latent=self.kv_compress(x)

        #Now we need to expand latent into K and V
        K = self.k_proj(kv_latent)
        V = self.v_proj(kv_latent)
        
        #Here we reshaping so that each head can compute attention independently and in parallel
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        #Transpose to put heads first 

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        #Compute the attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        #Convert scores into attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        
        #apply droput
        attn_weights=self.dropout(attn_weights)
        
        #Multiply with V
        output = torch.matmul(attn_weights, V)
        #Transpose back
        output = output.transpose(1, 2).contiguous()
       
        #Concat all heads
        output = output.view(batch_size, seq_len, self.d_model)

        #Final Output
         
        output = self.out_proj(output)
         
        return output
        




        

