"""
Main Training Script
Run: python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
import os

from src.models import LanguageModel


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
# TRAINING FUNCTION
# ============================================

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for input_ids, target_ids in pbar:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            logits = model(input_ids)
            
            batch_size, seq_len, vocab_size = logits.shape
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
        
        avg_train_loss = total_loss / num_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                logits = model(input_ids)
                batch_size, seq_len, vocab_size = logits.shape
                loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_perplexity = torch.exp(torch.tensor(avg_val_loss))
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Perplexity: {val_perplexity:.2f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  ✅ Best model saved!")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    print("Loading dataset...")
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = WikiTextDataset(dataset['train'], tokenizer, max_length=512)
    val_dataset = WikiTextDataset(dataset['validation'], tokenizer, max_length=512)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    # Create model
    print("Creating model...")
    model = LanguageModel(
        vocab_size=50257,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        num_layers=12,
        dropout=0.1
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}\n")
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Train
    print("="*70)
    print("STARTING TRAINING")
    print("="*70)
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=20)
    
    print("\n✅ Training complete!")