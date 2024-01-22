import torch 
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Sort and get unique characters from the text
chars = sorted(list(set(text)))

# Calculate the size of the vocabulary
vocab_size = len(chars)

# Create a mapping from characters to integers (stoi: string to integer)
stoi = {ch: i for i, ch in enumerate(chars)}

# Create a mapping from integers to characters (itos: integer to string)
itos = {i: ch for i, ch in enumerate(chars)}

# Simple Encoder: Map characters to their integer indices
def encode(s):
    return [stoi[c] for c in s]

# Simple Decoder: Map integer indices to their corresponding characters
def decode(l):
    return ''.join([itos[i] for i in l])


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# Simplified version of the get_batch function
def get_batch(split):
    # Select the appropriate dataset (train or validation)
    data = train_data if split == 'train' else val_data
    # Generate random indices for creating a small batch
    ix = torch.randint(len(data) - block_size, (batch_size,)) 
    # Extract input sequences (x) and target sequences (y) from the dataset
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]) 
    # Move the tensors to the specified device  (CPU,GPU)
    x, y = x.to(device), y.to(device)
    
    return x, y


# Function to estimate loss without computing gradients
@torch.no_grad()
def estimate_loss():
    # Dictionary to store the computed losses
    out = {}
    # Set the model to evaluation mode
    model.eval()
    # Iterate through training and validation splits
    for split in ['train', 'val']:
        # Initialize a tensor to store losses for each iteration
        losses = torch.zeros(eval_iters)
        # Iterate over evaluation iterations
        for k in range(eval_iters):
            # Get a batch of data for the specified split
            X, Y = get_batch(split)
            # Get model predictions and compute the loss
            logits, loss = model(X, Y)
            # Store the loss in the tensor
            losses[k] = loss.item()

        # Calculate the mean loss for the current split
        out[split] = losses.mean()

    # Set the model back to training mode
    model.train()

    # Return the computed losses
    return out

class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()

        # Linear layers for key, query, and value projections
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Buffer to store a lower triangular matrix for masking
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input of size (batch, time-step, channels)
        # Output of size (batch, time-step, head size)
        B, T, C = x.shape

        # Linear projections for key and query
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)

        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        # Mask the upper triangular part of the attention matrix
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)

        # Apply softmax to get attention weights
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # Apply dropout for regularization
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)

        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()

        # Create a list of individual self-attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # Linear projection layer to combine heads
        self.proj = nn.Linear(head_size * num_heads, n_embd)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate outputs from each self-attention head
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # Apply dropout to the concatenated output
        out = self.dropout(self.proj(out))

        return out

class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd):
        super().__init__()

        # Define a sequential neural network module
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Linear layer with an expansion factor
            nn.ReLU(),                      # Rectified Linear Unit (ReLU) activation
            nn.Linear(4 * n_embd, n_embd),  # Linear layer for dimension reduction
            nn.Dropout(dropout),            # Dropout layer for regularization
        )

    def forward(self, x):
        # Pass the input through the defined neural network module
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads
        super().__init__()

        # Calculate the size of each head in the MultiHeadAttention
        head_size = n_embd // n_head

        # MultiHeadAttention layer
        self.sa = MultiHeadAttention(n_head, head_size)

        # FeedForward layer
        self.ffwd = FeedForward(n_embd)

        # Layer normalization for the first and second sub-layers
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Self-attention followed by layer normalization and residual connection
        x = x + self.sa(self.ln1(x))

        # FeedForward layer followed by layer normalization and residual connection
        x = x + self.ffwd(self.ln2(x))

        return x
    

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(m.generate(context, max_new_tokens=1000)[0].tolist())
