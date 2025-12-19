"""
Improved Knowledge Distillation Experiment
Changes from original:
1. Added cross-entropy loss (combined with KL divergence)
2. 3x compression ratio (~165M student vs 494M teacher)
3. More training examples (10,000+)
4. Longer training (20 epochs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

print("="*60)
print("IMPROVED DISTILLATION EXPERIMENT")
print("="*60)

# Check GPU
print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# STEP 1: Load Teacher Model
# ============================================================
print("\n" + "="*60)
print("STEP 1: Loading Teacher Model (Qwen2.5-0.5B)")
print("="*60)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

teacher_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B",
    torch_dtype=torch.float16,
    device_map="cuda"
)
teacher_model.eval()

teacher_params = sum(p.numel() for p in teacher_model.parameters())
print(f"Teacher parameters: {teacher_params:,}")

# ============================================================
# STEP 2: Create LARGER Student Model (~3x compression)
# ============================================================
print("\n" + "="*60)
print("STEP 2: Creating Student Model (~3x compression)")
print("="*60)

# Teacher config for reference:
# hidden_size=896, num_hidden_layers=24, num_attention_heads=14, intermediate_size=4864

student_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
student_config.hidden_size = 512          # Was 256, teacher has 896
student_config.num_hidden_layers = 12     # Was 6, teacher has 24
student_config.num_attention_heads = 8    # Keep at 8 (must divide hidden_size)
student_config.intermediate_size = 2048   # Was 1024, teacher has 4864
student_config.num_key_value_heads = 4    # For grouped query attention

student_model = AutoModelForCausalLM.from_config(student_config)
student_model = student_model.to("cuda")

student_params = sum(p.numel() for p in student_model.parameters())
compression_ratio = teacher_params / student_params

print(f"Teacher parameters: {teacher_params:,}")
print(f"Student parameters: {student_params:,}")
print(f"Compression ratio: {compression_ratio:.1f}x")

# ============================================================
# STEP 3: Load and Prepare MORE Training Data
# ============================================================
print("\n" + "="*60)
print("STEP 3: Loading Training Data")
print("="*60)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
print(f"Original dataset size: {len(dataset)}")

# Filter out empty/short examples
def is_good_example(example):
    text = example['text'].strip()
    return len(text) > 100 and not text.startswith('=')

dataset = dataset.filter(is_good_example)
print(f"After filtering: {len(dataset)} examples")

# Use more examples this time
num_examples = min(10000, len(dataset))
dataset = dataset.select(range(num_examples))
print(f"Using: {num_examples} examples for training")

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=128,
        padding='max_length',
        return_tensors=None
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

train_dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)
print(f"Number of batches: {len(train_dataloader)}")

# ============================================================
# STEP 4: Define COMBINED Loss Function (KEY FIX!)
# ============================================================
print("\n" + "="*60)
print("STEP 4: Defining Combined Loss Function")
print("="*60)

def combined_distillation_loss(student_logits, teacher_logits, labels, attention_mask, temperature=3.0, alpha=0.5):
    """
    Combined loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    Args:
        student_logits: [batch, seq_len, vocab_size]
        teacher_logits: [batch, seq_len, vocab_size]
        labels: [batch, seq_len] - the actual next tokens (shifted input_ids)
        attention_mask: [batch, seq_len] - which positions to compute loss on
        temperature: softness of distributions
        alpha: balance between soft and hard loss
    """
    
    # === SOFT LOSS (KL Divergence with teacher) ===
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    
    kl_div = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction='none'
    ).sum(dim=-1)  # Sum over vocab dimension
    
    # Mask out padding positions
    kl_div = kl_div * attention_mask
    soft_loss = kl_div.sum() / attention_mask.sum()
    soft_loss = soft_loss * (temperature ** 2)
    
    # === HARD LOSS (Cross-entropy with actual labels) ===
    # Reshape for cross-entropy: [batch * seq_len, vocab_size]
    student_logits_flat = student_logits.view(-1, student_logits.size(-1))
    labels_flat = labels.view(-1)
    
    # Compute cross-entropy (ignoring padding with ignore_index=-100)
    hard_loss = F.cross_entropy(
        student_logits_flat,
        labels_flat,
        ignore_index=-100,
        reduction='mean'
    )
    
    # === COMBINED LOSS ===
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    return total_loss, soft_loss.item(), hard_loss.item()

print("Combined loss function defined!")
print(f"  - Soft loss: KL divergence (match teacher distribution)")
print(f"  - Hard loss: Cross-entropy (predict actual tokens)")
print(f"  - Alpha=0.5: Equal weight to both")

# ============================================================
# STEP 5: Training Loop
# ============================================================
print("\n" + "="*60)
print("STEP 5: Training")
print("="*60)

# Training config
num_epochs = 20
learning_rate = 1e-4
temperature = 3.0
alpha = 0.5

optimizer = AdamW(student_model.parameters(), lr=learning_rate, weight_decay=0.01)

print(f"Configuration:")
print(f"  Epochs: {num_epochs}")
print(f"  Learning rate: {learning_rate}")
print(f"  Temperature: {temperature}")
print(f"  Alpha (soft/hard balance): {alpha}")
print(f"  Total steps: {len(train_dataloader) * num_epochs}")

# Track losses
all_losses = []
soft_losses = []
hard_losses = []

student_model.train()

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print("-" * 40)
    
    epoch_loss = 0
    epoch_soft = 0
    epoch_hard = 0
    
    progress_bar = tqdm(train_dataloader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        
        # Create labels (shifted input_ids for next token prediction)
        # Labels are input_ids shifted left by 1
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Ignore last position
        
        # Mask out padding in labels
        labels[attention_mask == 0] = -100
        
        # Teacher predictions (frozen)
        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            teacher_logits = teacher_outputs.logits
        
        # Student predictions
        student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = student_outputs.logits
        
        # Combined loss
        loss, soft_loss, hard_loss = combined_distillation_loss(
            student_logits, 
            teacher_logits, 
            labels,
            attention_mask,
            temperature=temperature,
            alpha=alpha
        )
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_soft += soft_loss
        epoch_hard += hard_loss
        all_losses.append(loss.item())
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.2f}',
            'soft': f'{soft_loss:.2f}',
            'hard': f'{hard_loss:.2f}'
        })
    
    avg_loss = epoch_loss / len(train_dataloader)
    avg_soft = epoch_soft / len(train_dataloader)
    avg_hard = epoch_hard / len(train_dataloader)
    
    soft_losses.append(avg_soft)
    hard_losses.append(avg_hard)
    
    print(f"Epoch {epoch+1} - Total: {avg_loss:.2f}, Soft: {avg_soft:.2f}, Hard: {avg_hard:.2f}")

print("\n" + "="*60)
print("Training Complete!")
print("="*60)

# ============================================================
# STEP 6: Evaluation
# ============================================================
print("\n" + "="*60)
print("STEP 6: Evaluation")
print("="*60)

student_model.eval()

test_prompts = [
    "The capital of France is",
    "The cat sat on the",
    "Machine learning is a type of"
]

for prompt in test_prompts:
    print(f"\nPrompt: '{prompt}'")
    print("-" * 50)
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        # Teacher
        teacher_outputs = teacher_model(**inputs)
        teacher_logits = teacher_outputs.logits[0, -1, :]
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        # Student
        student_outputs = student_model(**inputs)
        student_logits = student_outputs.logits[0, -1, :]
        student_probs = F.softmax(student_logits, dim=-1)
    
    # Top 5 for both
    teacher_top_probs, teacher_top_indices = torch.topk(teacher_probs, k=5)
    student_top_probs, student_top_indices = torch.topk(student_probs, k=5)
    
    print("TEACHER:")
    for i, (prob, idx) in enumerate(zip(teacher_top_probs, teacher_top_indices)):
        token = tokenizer.decode([idx])
        print(f"  {i+1}. '{token}' → {prob.item()*100:.2f}%")
    
    print("STUDENT:")
    for i, (prob, idx) in enumerate(zip(student_top_probs, student_top_indices)):
        token = tokenizer.decode([idx])
        match = "✓" if idx in teacher_top_indices else ""
        print(f"  {i+1}. '{token}' → {prob.item()*100:.2f}% {match}")
    
    # Generation
    with torch.no_grad():
        teacher_gen = teacher_model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        student_gen = student_model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    print(f"\nTeacher generates: {tokenizer.decode(teacher_gen[0], skip_special_tokens=True)}")
    print(f"Student generates: {tokenizer.decode(student_gen[0], skip_special_tokens=True)}")

print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)