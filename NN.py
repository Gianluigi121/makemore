import torch
import torch.nn.functional as F
words = open('names.txt', 'r').read().split()

# Create the reference dictionary
chs = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chs)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

# Create a dataset
xs = []
ys = []
for word in words:
    chs = '.' + word + '.'
    for ch1, ch2 in zip(chs, chs[1:]):
        idx1 = stoi[ch1]
        idx2 = stoi[ch2]
        xs.append(idx1)
        ys.append(idx2)
x = torch.tensor(xs)
y = torch.tensor(ys)

# Initialize the weights
g = torch.Generator().manual_seed(0)     
w = torch.randn((27, 27), generator=g, requires_grad=True)

# Training loop
epoch = 200
num_samples = len(xs)
step_size = 50
for i in range(epoch):
    # Forward pass
    x_enc = F.one_hot(x, num_classes=27).float()
    logits = x_enc @ w
    counts = torch.exp(logits)
    P = counts / counts.sum(dim=1, keepdims=True)
    loss = -P[torch.arange(num_samples), ys].log().mean()
    
    # Backward pass
    w.grad = None
    loss.backward()
    
    # Update
    w.data += -step_size*w.grad
print(f"final loss: {loss}")
    
# Inference
# Start with . and use the NN to find us the next char and stop when we see .
word_num = 5
for i in range(word_num):
    idx = 0
    out = []
    while True:
        x_enc = F.one_hot(torch.tensor(idx), num_classes = 27).float()
        logits = x_enc @ w
        counts = logits.exp()
        p = counts / counts.sum()
        idx = torch.multinomial(p, num_samples=1, generator=g).item()
        if idx == 0:
            break
        out.append(itos[idx])
    print(''.join(out))
