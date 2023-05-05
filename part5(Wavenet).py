import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# read in the words 
words = open('names.txt', 'r').read().split()
print("read_file")

# Build a reference map
chs = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chs)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

block_size = 8
# Build the dataset
def build_dataset(words):
    xs = []
    ys = []
    for word in words:
        word += '.'
        context = [0]*block_size
        for ch in word:
            xs.append(context)
            idx = stoi[ch]
            ys.append(idx)
            context = context[1:] + [idx]
    X = torch.tensor(xs)
    Y = torch.tensor(ys)
    return X, Y

# Construct train, val, test dataset
num1 = int(len(words)*0.8)
num2 = int(len(words)*0.9)
Xtr, Ytr = build_dataset(words[:num1])
Xval, Yval = build_dataset(words[num1:num2])
Xtest, Ytest = build_dataset(words[num2:])

# Define Model layers
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn(fan_in, fan_out) / (fan_in ** 0.5)
        self.bias = torch.zeros(fan_out) if bias else None
        
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
        self.training = True
        # Init params
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # Init running updates
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)
        
    def __call__(self, x):
        if x.ndim == 2:
            dim = 0
        elif x.ndim == 3:
            dim = (0, 1)
        if self.training:
            xmean = x.mean(dim)
            xvar = x.var(dim)
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x-xmean) / torch.sqrt(xvar + self.eps)
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
    
class Embedding:
    def __init__(self, vocab_size, emb_size):
        self.weight = torch.randn(vocab_size, emb_size)
        
    def __call__(self, x):
        # X: (batch_size, block_size) self.out: (batch_size, block_size, emb_size)
        self.out = self.weight[x] 
        return self.out
        
    def parameters(self):
        return [self.weight]

class Flatten:
    def __init__(self, group_size):
        self.group_size = group_size
        
    def __call__(self, x):
        batch_size, block_size, emb_size = x.shape
        self.out = x.view(batch_size, block_size // self.group_size, self.group_size * emb_size)
        if self.out.shape[1] == 1:
            self.out = self.out.squeeze(dim=1)
        return self.out
    
    def parameters(self):
        return []
    
class Sequential:
    # Layers: A list of layers
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# Initialization
batch_size = 32
block_size = 8
emb_size = 10
n_hidden = 100
vocab_size = len(itos)
group_size = 2


model = Sequential([
    Embedding(vocab_size, emb_size), # (32, 8, 10)
    Flatten(2),  # (32, 4, 20)
    Linear(group_size * emb_size, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),  # (32, 4, 100)
    Flatten(2),  # (32, 2, 200)
    Linear(group_size * n_hidden, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),  # (32, 2, 100)
    Flatten(2),  # (32, 200)
    Linear(group_size * n_hidden, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),  # (32, 100)
    Linear(n_hidden, vocab_size), BatchNorm1D(vocab_size)   # (32, 27)
])
        
# Init params
model.layers[-1].gamma *= 0.1
for layer in model.layers:
    if isinstance(layer, Linear):
        layer.weight *= 5/3     # Multiply by the gain

parameters = model.parameters()
# Set the gradient for all params to be true
for p in parameters:
    p.requires_grad = True
    

# Training
epoch_num = 200000
lossi = []
for i in range(epoch_num):
    # Select a mini batch
    idx = torch.randint(0, Xtr.shape[0], (batch_size, ))
    Xb, Yb = Xtr[idx], Ytr[idx]
    
    # Forward pass
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)
    lossi.append(loss.log10().item())
    
    # Backward
    for p in parameters:
        p.grad = None
    loss.backward()
    
    # Update
    lr = 0.1 if i < 150000 else 0.01
    for p in parameters:
        p.data += -lr*p.grad

    if i % 10000 == 0:
        print(f"{i}: {loss.item()}")

for layer in model.layers:
    print(f"{layer.__class__.__name__}: {layer.out.shape}")
    
    
# Convert BatchNorm to eval model(training = False)
for layer in model.layers:
    if isinstance(layer, BatchNorm1D):
        layer.training = False

# Compute loss for training, validation, and testing set
@torch.no_grad()
def split_loss(split):
    X, Y = {
        'train': (Xtr, Ytr), 
        'val': (Xval, Yval),
        'test': (Xtest, Ytest)
    }[split]
    
    logits = model(X)
    loss = F.cross_entropy(logits, Y)
    print(f"{split} loss: {loss.item()}")

split_loss('train')
split_loss('val')
split_loss('test')

# Sample from the model
word_num = 20
for i in range(word_num):
    out = []
    context = [0]*block_size
    while True:
        x = torch.tensor(context).view(1, -1)  # (1, block_size)
        logits = model(x) # (1, vocab_size)
        prob = F.softmax(logits, dim=1).view(vocab_size)
        idx = torch.multinomial(prob, num_samples=1).item()
        out.append(itos[idx])
        context = context[1:] + [idx]
        if idx == 0:
            break
    print(''.join(out))
        
        
        