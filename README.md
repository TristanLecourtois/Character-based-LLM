# Character-based-LLM
This repository contains a language model centered around characters, trained on a dataset comprising the works of Shakespeare. The architecture of the model is constructed using the Transformer framework and is implemented using PyTorch.

The code is inspired by Andrej Karpathy's work on NanoGPT.

The model was trained on a tiny dataset of Shakespeare's works, which is a subset of Shakespeare's works. The dataset was split into 90% for training and 10% for validation.

The Transformer architecture is composed of a series of blocks that include multi-head self-attention and feedforward layers. The model also utilizes positional embeddings for each token, which allows the model to understand the order of the input characters.

Here are the key hyperparameters used for training the model:

* Batch size: 16
* Block size: 32
* Maximum iterations: 5000
* Learning rate: 1e-3
* Number of layers: 4
* Number of heads: 4
* Embedding dimension: 64
* Dropout rate: 0.0

## Usage 

The resulting tensor (*context*) is an initial context for text generation. In this context, this is used as a starting point or seed for generating text. The generated text is then decoded and printed. The *max_new_tokens* parameter specifies the maximum number of tokens to generate.

```python

context = torch.zeors((1,1),dtype=torch.long,device=device)
print(decode(m.generate(context,max_new_tokens=1000)[0].tolist())) 

```
You can change any parameters : 

```python

# Hyperparameters
batch_size = 16 
block_size = 32 
max_iters = 15000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

```


The generated text is stored in the *output.txt*.


