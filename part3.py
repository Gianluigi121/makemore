import torch
import torch.nn.functional as F 
import matplotlib.pyplot as plt 
import random

words = open('names.txt', 'r').read().split()
chs = sorted(list(set(''.join(words))))
# Step 1: Build a reference map
stoi = {s:i+1 for i, s in enumerate(chs)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

# Step 2: Build dataset
def build_dataset(words):
    xs, ys = [], []
    for word in words:
        word = word + '.'
        context = [0]*3
        
        for ch in word:
            xs.append(context)
            idx = stoi[ch]
            ys.append(idx)
            context = context[1:] + [idx]
    X = torch.tensor(xs)
    Y = torch.tensor(ys)
    return X, Y

random.shuffle(words)
num1 = int(len(words)*0.8)
num2 = int(len(words)*0.9)
Xtr, Ytr = build_dataset(words[:num1])
Xdev, Ydev = build_dataset(words[num1:num2])
Xtest, Ytest = build_dataset(words[num2:])

g = torch.Generator().manual_seed(2147483647) # Modify this part later

# Step 3: Define layer model
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn(fan_in, fan_out, generator=g) / fan_in ** 0.5 # Multiply a value to reduce the variance increasement due to the matrix multiplication
        self.bias = torch.zeros(fan_out, generator=g) if bias else None
        
    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + [] if self.bias is None else [self.bias]

class BatchNorm1D:
    def __init__(self, dim, eps=1e-5, momentum=0.01):
        self.eps = eps
        self.momentum = momentum
        self.training = True # Set to the training mode by default
        # Init parameters
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim) 
        # Init running updates
        self.running_mean = torch.zeros(dim) 
        self.running_var = torch.ones(dim)
        
    def __call__(self, x):
        if self.training:
            xmean = x.mean(dim=0, keepdims=True) # (1, hidden_num)
            xvar = x.var(dim=0, keepdims=True)  # (1, hidden_num)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / (xvar + self.eps) ** 0.5
        self.out = self.gamma * xhat + self.beta 
        if self.training:
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        return self.out         
    
    def parameters(self):
        return [self.gamma, self.beta]

class Tanh:
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []
    
# Step 4: Define Custom Model & Init
emb_size = 10
block_size = 3
n_hidden= 100
vocab_size = 27
C = torch.randn(27, emb_size, generator=g)
layers = [Linear(block_size*emb_size, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
          Linear(n_hidden, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
          Linear(n_hidden, vocab_size, bias=False), BatchNorm1D(vocab_size)]

with torch.no_grad():
    layers[-1].gamma *= 0.1 # Init self.gamma to be small values as the final logits = self.gamma * x + self.bias
                            # Previously we have init self.bias = 0. 
    for layer in layers[:-1]:
        if isinstance(layer, Linear):
            layer.weight *= 5 / 3 # Multiply by the gain to fight with the contraction effect of the tanh function
    parameters = [C] + [p for layer in layers for p in layer.parameters()]
    
for p in parameters:
    p.requires_grad = True

# Step 5: Traning loop
max_steps = 200000
batch_size = 32
lossi = []
for i in range(max_steps):
    # Step 1: Select Mini-batch
    idx = torch.randint(0, Xtr.shape[0], (batch_size, ), generator=g)
    Xb, Yb = Xtr[idx], Ytr[idx]

    # Step 2: Forward network
    # Embedding
    emb = C[Xb] # (num_samples, block_size, emb_size)
    x = emb.view(-1, block_size*emb_size)
    # Go through the model
    for layer in layers:
        x = layer(x)
    logits = x
    loss = F.cross_entropy(logits, Yb)
    lossi.append(loss.item())
    
    # Step 3: Backward
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # Step 4: Update
    lr = 0.1 if i < 150000 else 0.01
    for p in parameters:
        p.data += -lr*p.grad
    
    # Step 5: Track Loss
    if i % 10000 == 0:
        print(f"{i}: {loss.item()}")
    
def compute_loss(split):
    X, Y = {
        'train': (Xtr, Ytr), 
        'val': (Xdev, Ydev),
        'test': (Xtest, Ytest)
    }[split]
    
    # Embedding
    emb = C[X]
    x = emb.view(-1, block_size*emb_size)
    for layer in layers:
        x = layer(x)
    logits = x 
    loss = F.cross_entropy(logits, Y)
    print(f"{split} loss: {loss.item()}")

# Set the model to the validation mode
for layer in layers:
    if isinstance(layer, BatchNorm1D):
        layer.training = False
compute_loss('train')
compute_loss('val')
compute_loss('test')

# Sample from the Model
word_num = 20
for i in range(word_num):
    out = []
    context = [0] * block_size
    while True:
        emb = C[torch.tensor(context)] # (block_size, emb_size)
        x = emb.view(block_size*emb_size)
        for layer in layers:
            x = layer(x)
        prob = F.softmax(x, dim=1)
        idx = torch.multinomial(prob, num_samples=1, generator=g).item()
        out.append(itos[idx])
        context = context[1:] + [idx]
        if idx == 0:
            break
    print(''.join(out))      
        
# words_num = 20
# for i in range(words_num):
#     context = [0]*3
#     out = []
#     while True:
#         emb = C[torch.tensor(context)] # (3, emb_size)
#         x = emb.reshape(1, block_size*emb_size)
        
#         for layer in layers:
#             x = layer(x)
#         prob = F.softmax(x, dim=1)
#         idx = torch.multinomial(prob, num_samples = 1).item()
#         out.append(itos[idx])
#         context = context[1:]+[idx]
#         if idx == 0:
#             break
#     print(''.join(out))