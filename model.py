import torch
import torch.nn as nn
from torch.nn import functional as F

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
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



# single head of self-Attension 
class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias = False)
    self.query = nn.Linear(n_embd, head_size, bias = False)
    self.value = nn.Linear(n_embd, head_size, bias = False)
    # storing fixed mask
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x)
    q = self.query(x)
    v = self.value(x)

    wei = (q @ k.transpose(-2, -1)) / C**-0.5

    tril = torch.tril(torch.ones(T, T))
    wei = wei.masked_fill(tril == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)

    out = wei @ v
    
    return out

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
  
  def forward(self, x):
    # concat all the comupted attention into the last dimension of the output
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    # Randomly Drop Some attention weight
    out = self.dropout(self.proj(out))
    return out
  
# Feed Forward Layer
class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    # See the formula
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout),
    )
  
  def forward(self, x):
    return self.net(x)

# Layer Normalization
class LayerNorm1d: # (used to be BatchNorm1d)

  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)

  def __call__(self, x):
    # calculate the forward pass
    xmean = x.mean(1, keepdim=True) # batch mean
    xvar = x.var(1, keepdim=True) # batch variance
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]

# Single Block
class Block(nn.Module):
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(head_size, n_head)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
  
  def forward(self, x):
    x = x + self.sa(self.ln2(x))
    x = x + self.ffwd(self.ln1(x))
    return x


class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()

    self.tocken_embbeding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embbeding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(
        *[Block(n_embd, n_head = n_head) for _ in range(n_layer)]
    )
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    
    B, T = idx.shape

    tok_emb = self.tocken_embbeding_table(idx)
    pos_emb = self.position_embbeding_table(torch.arange(T,device=device))
    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      # crop idx to the last block_size tokens
      idx_cond = idx[:, -block_size:]
      # get the predictions
      logits, loss = self(idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :]
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

model = BigramLanguageModel(vocab_size)
model = model.to(device)

# Try to load checkpoint if it exists
try:
    checkpoint = torch.load('model_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded checkpoint successfully")
except:
    print("No checkpoint found, starting from scratch")

optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

# Load optimizer state if checkpoint exists
try:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Loaded optimizer state successfully")
except:
    print("No optimizer state found, starting from scratch")

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

for iter in range(max_iters):
  
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
   
    
    if iter % eval_interval == 0:
      losses = estimate_loss()
      print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# Generate and save the result
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decoder(model.generate(context, max_new_tokens=10000)[0].tolist())

# Save to result.txt
with open('result.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)

# Save the model and optimizer state
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
print("Generated text has been saved to result.txt")
print("Model checkpoint has been saved to model_checkpoint.pt")