import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# read in all words
words = open('names.txt', 'r').read().splitlines()

# Build the vocabulary of characters and mapping to/from integers
chs = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chs)}
stoi['.'] = 0
vocab_size = len(stoi)
itos = {i:s for s, i in stoi.items()}

# Build the dataseet: Training, validation, and test
block_size = 3
def build_dataset(words):
    xs = []
    ys = []
    for word in words:
        context = [0] * block_size
        word = word + '.'
        for ch in word:
            xs.append(context)
            idx = stoi[ch]
            ys.append(idx)
            out_str = ''.join(itos[i] for i in context)+"-->"+itos[idx]
            context = context[1:]+[idx]
            # print(out_str)
    X = torch.tensor(xs) # Shape: (num_samples, 3)
    Y = torch.tensor(ys) # Shape: (num_samples, )
    return X, Y

num_samples = len(words)
import random
random.seed(42)
random.shuffle(words)
num1 = int(num_samples*0.8)
num2 = int(num_samples*0.9)
Xtr, Ytr = build_dataset(words[:num1])          # Training: 80%
Xdev, Ydev = build_dataset(words[num1:num2])    # Val: 10%
Xtest, Ytest = build_dataset(words[num2:])      # Test: 10%
print(Xtr.shape, Ytr.shape)
print(Xdev.shape, Ydev.shape)
print(Xtest.shape, Ytest.shape)

g = torch.Generator().manual_seed(2147483647)

class Linear:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn(fan_in, fan_out, generator=g) / fan_in**0.5
        self.bias = torch.zeros(fan_out, generator=g) if bias else None
    
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    
class BatchNorm1D:
    def __init__(self, dim, eps=1e-5, momentum=0.01):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # Parameters(trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers(trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        
    def __call__(self, x): # x shape: (num_samples, dim)
        if self.training:
            xmean = x.mean(dim=0, keepdims=True)
            xvar = x.var(dim=0, keepdims=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        if self.training:
            # put this part in no_grad to not draw computation graph
            with torch.no_grad():
                self.running_mean = (1-self.momentum)*self.running_mean+self.momentum*xmean
                self.running_var = (1-self.momentum)*self.running_var+self.momentum*xvar
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        return torch.tanh(x)
    
    def parameters(self):
        return []
    
# Initialization
g = torch.Generator().manual_seed(2147483647)
n_emb = 10 # Size the representation vector for each character
n_hidden = 100 # Number of neurons in the hidden layer
vocab_size = 27
C = torch.randn(vocab_size, n_emb)

# Define the model
layers = [Linear(block_size*n_emb, n_hidden, False), BatchNorm1D(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
          Linear(n_hidden, vocab_size, bias=False), BatchNorm1D(vocab_size)
         ]

# Init the weights
with torch.no_grad():
    layers[-1].gamma *= 0.1
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5/3 # gain

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters))
for p in parameters:
    p.requires_grad = True
    
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

    # Select Minibatch
    idx = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb = Xtr[idx]
    Yb = Ytr[idx]

    # Forward pass
    # Embedding
    emb = C[Xb] # (num_samples, block_size, emb_size)
    x = emb.view(-1, block_size*n_emb)
    # Model
    for layer in layers:
        # print(type(layer))
        x = layer(x)
    loss = F.cross_entropy(x, Yb)

    # Backward
    for p in parameters:
        p.grad = None
    loss.backward()

    # Update
    lr = 0.1 if i < 150000 else 0.01 # Step learning rate decay
    for p in parameters:
        p.data += -lr*p.grad

    # track stats
    if i % 10000 == 0: # print every once in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())
    
@torch.no_grad()
def split_loss(split):
    X, Y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev), 
        'test': (Xtest, Ytest)
    }[split]
    emb = C[X] # (num_samples, block_size, emb_size)
    x = emb.view(-1, block_size*n_emb)
    # Go through the model
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Y)
    print(f"{split} loss: {loss.item()}")

# put layers into eval mode
for layer in layers:
  layer.training = False

split_loss('train')
split_loss('val')
split_loss('test')

# Sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

words_num = 20
for i in range(words_num):
    context = [0]*3
    out = []
    while True:
        emb = C[torch.tensor(context)] # (3, emb_size)
        x = emb.reshape(1, block_size*n_emb)
        
        for layer in layers:
            x = layer(x)
        prob = F.softmax(x, dim=1)
        idx = torch.multinomial(prob, num_samples = 1).item()
        out.append(itos[idx])
        context = context[1:]+[idx]
        if idx == 0:
            break
    print(''.join(out))