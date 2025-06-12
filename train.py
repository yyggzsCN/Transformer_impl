import torch
import torch.nn as nn
from torch.nn import functional as F
from model import BigramLanguageModel

# Hyperparameters
block_size = 32
batch_size = 16
learning_rate = 1e-3
eval_interval = 100
max_iters = 5000
eval_iters = 200
n_layer = 4
n_embd = 64
dropout = 0.2
n_head = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)

# Load data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}

encoder = lambda x: [stoi[c] for c in x]
decoder = lambda x: ''.join([itos[i] for i in x])

data = torch.tensor(encoder(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Initialize model
model = BigramLanguageModel(vocab_size, n_embd, block_size, n_layer, n_head, dropout)
model = model.to(device)

# Try to load checkpoint if it exists
try:
    checkpoint = torch.load('model_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded checkpoint successfully")
except:
    print("No checkpoint found, starting from scratch")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Load optimizer state if checkpoint exists
try:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Loaded optimizer state successfully")
except:
    print("No optimizer state found, starting from scratch")

# Training loop
for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# Save checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'vocab_size': vocab_size,
    'block_size': block_size,
    'n_layer': n_layer,
    'n_head': n_head,
    'n_embd': n_embd,
    'chars': chars,
    'stoi': stoi,
    'itos': itos
}

torch.save(checkpoint, 'model_checkpoint.pt')
print("Model checkpoint has been saved to model_checkpoint.pt")