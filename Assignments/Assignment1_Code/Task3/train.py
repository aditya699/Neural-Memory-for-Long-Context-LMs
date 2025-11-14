import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
import math
import os
import matplotlib.pyplot as plt

# ============================================
# MODEL COMPONENTS
# ============================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
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

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal masking
        batch_size, num_heads, seq_len, _ = scores.shape
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask_to_block = (causal_mask == 0)
        mask_to_block = mask_to_block.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask_to_block, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x


class Embeddings(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model):
        super().__init__()

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)

    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape

        tokens = self.token_embed(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device)
        pos = self.pos_embed(positions)

        return tokens + pos


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()

        self.embeddings = Embeddings(vocab_size, max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.embeddings.token_embed.weight

    def forward(self, token_ids):
        x = self.embeddings(token_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits


# ============================================
# DATASET
# ============================================

class WikiTextDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.max_length = max_length

        print("Tokenizing dataset...")
        all_tokens = []

        for example in data:
            text = example['text'].strip()
            if len(text) > 0:
                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Total tokens: {len(self.tokens):,}")

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start = idx * self.max_length
        end = start + self.max_length

        input_ids = self.tokens[start:end]
        target_ids = self.tokens[start+1:end+1]

        return input_ids, target_ids


# ============================================
# VALIDATION FUNCTION
# ============================================

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, target_ids in val_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * seq_len, vocab_size)
            targets_flat = target_ids.view(batch_size * seq_len)
            
            loss = criterion(logits_flat, targets_flat)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss))
    
    model.train()
    return avg_loss, perplexity.item()


# ============================================
# TRAINING FUNCTION
# ============================================

def train_model_properly(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, patience=20):
    model.to(device)
    best_val_loss = float('inf')
    best_epoch = 0

    # Track metrics history for plotting
    train_losses = []
    val_losses = []
    val_perplexities = []

    # Early stopping tracking
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for input_ids, target_ids in pbar:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Forward
            logits = model(input_ids)
            
            # Loss
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * seq_len, vocab_size)
            targets_flat = target_ids.view(batch_size * seq_len)
            loss = criterion(logits_flat, targets_flat)
            
            # Backward with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
        
        # Validation phase
        val_loss, val_perplexity = validate_model(model, val_loader, criterion, device)
        
        avg_train_loss = total_loss / num_batches

        # Store metrics for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_perplexities.append(val_perplexity)

        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Perplexity: {val_perplexity:.2f}")

        # Save best model and track early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0  # Reset counter

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': val_loss,
                'val_perplexity': val_perplexity,
            }, 'mha_model_best.pt')

            print(f"  Best model so far! Saved to 'mha_model_best.pt'")
        else:
            epochs_without_improvement += 1
            print(f"  Best model still at Epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")
            print(f"  Epochs without improvement: {epochs_without_improvement}/{patience}")

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} consecutive epochs.")
            print(f"   Best model was at Epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")
            break

        print()
    
    print("=" * 70)
    print(f"Training Complete!")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Perplexity: {torch.exp(torch.tensor(best_val_loss)):.2f}")
    print("=" * 70)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_perplexities': val_perplexities,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch
    }


# ============================================
# TEXT GENERATION FUNCTION
# ============================================

def generate_text_proper(model, tokenizer, prompt, max_length=50, top_k=50, device='cuda'):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)
            next_token_logits = logits[0, -1, :]
            
            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[next_token_idx]
            
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# ============================================
# PLOTTING FUNCTION
# ============================================

def plot_metrics(metrics_history, save_path='training_metrics.png'):
    """
    Plot all training metrics including train loss, val loss, and val perplexity.

    Args:
        metrics_history: Dictionary containing training metrics
        save_path: Path to save the plot
    """
    train_losses = metrics_history['train_losses']
    val_losses = metrics_history['val_losses']
    val_perplexities = metrics_history['val_perplexities']
    best_epoch = metrics_history['best_epoch']

    epochs = range(1, len(train_losses) + 1)

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Overview', fontsize=16, fontweight='bold')

    # Plot 1: Training Loss
    axes[0, 0].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    axes[0, 0].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss over Epochs')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Validation Loss
    axes[0, 1].plot(epochs, val_losses, 'g-', linewidth=2, label='Val Loss')
    axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    axes[0, 1].scatter([best_epoch], [metrics_history['best_val_loss']],
                       color='r', s=100, zorder=5, label='Best Val Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Validation Loss over Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Validation Perplexity
    axes[1, 0].plot(epochs, val_perplexities, 'purple', linewidth=2, label='Val Perplexity')
    axes[1, 0].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Perplexity')
    axes[1, 0].set_title('Validation Perplexity over Epochs')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Train vs Validation Loss Comparison
    axes[1, 1].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    axes[1, 1].plot(epochs, val_losses, 'g-', linewidth=2, label='Val Loss')
    axes[1, 1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Train vs Validation Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining metrics plot saved to: {save_path}")

    # Display the plot
    plt.show()


# ============================================
# MAIN TRAINING SCRIPT
# ============================================

if __name__ == "__main__":
    # Setup
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load data
    print("\nLoading dataset...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = WikiTextDataset(dataset['train'], tokenizer, max_length=512)
    val_dataset = WikiTextDataset(dataset['validation'], tokenizer, max_length=512)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = LanguageModel(
        vocab_size=50257,
        max_seq_len=512,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        num_layers=4,
        dropout=0.1
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    metrics_history = train_model_properly(
        model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20, patience=5
    )

    # Plot metrics
    print("\n" + "=" * 70)
    print("PLOTTING TRAINING METRICS")
    print("=" * 70)
    plot_metrics(metrics_history, save_path='training_metrics.png')

    # Test generation
    print("\n" + "=" * 70)
    print("TESTING GENERATION")
    print("=" * 70)
    
    prompts = [
        "The history of India is ",
        "In mathematics,",
        "The cat sat on the",
    ]
    
    for prompt in prompts:
        generated = generate_text_proper(model, tokenizer, prompt, max_length=40, device=device)
        print(f"\nPrompt: {prompt}")
        print(f"Output: {generated}")
        print("-" * 70)