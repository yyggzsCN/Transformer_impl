import torch
import torch.nn as nn
from torch.nn import functional as F
from model import TransformerLanguageModel

# Load the checkpoint
checkpoint = torch.load('model_checkpoint.pt')

# Extract hyperparameters from checkpoint
vocab_size = checkpoint['vocab_size']
block_size = checkpoint['block_size']
n_layer = checkpoint['n_layer']
n_head = checkpoint['n_head']
n_embd = checkpoint['n_embd']
chars = checkpoint['chars']
stoi = checkpoint['stoi']
itos = checkpoint['itos']

# Create decoder function
decoder = lambda x: ''.join([itos[i] for i in x])

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize model
model = TransformerLanguageModel(vocab_size, n_embd, block_size, n_layer, n_head, dropout=0.0)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()  # Set to evaluation mode

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    generated_text = decoder(model.generate(context, max_new_tokens=1000)[0].tolist())

# Save to file
with open('generated_text.txt', 'w', encoding='utf-8') as f:
    f.write(generated_text)

print("Generated text has been saved to generated_text.txt") 