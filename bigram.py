import torch
import matplotlib.pyplot as plt

# First part: Build a bigram model
words = open('names.txt', 'r').read().split()

# Step 1: Build a bigram dictionary
# key: Bigram tuple, value: Number of appearence that they occur together
b = {}
for word in words:
    word = '.' + word + '.'
    for ch1, ch2 in zip(word, word[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0)+1

# Step 2: Create reference dictionaries
# Create a dictionary map from chars to integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0

# Create a dictionary map from integers to chars
itos = {i:s for s, i in stoi.items()}

# Step 3: Create a torch tensor where each entry represent the number of occurrence
N = torch.zeros((27, 27))
for bigram, count in b.items():
    ch1, ch2 = bigram
    id1 = stoi[ch1]
    id2 = stoi[ch2]
    N[id1][id2] += count

# Step 4: Show the figure of the matrix
# plt.figure(figsize=(16,16))
# plt.imshow(N, cmap='Blues')
# for i in range(27):
#     for j in range(27):
#         chstr = itos[i] + itos[j]
#         plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
#         plt.text(j, i, N[i][j].item(), ha='center', va='top', color='gray')
# plt.axis('off')
# plt.show()

# Step 5: Construct a prob matrix based on the count matrix with normalization
P = N / torch.sum(N, dim=1, keepdims=True)
print(P[0].sum())

# Step 6: Inference: Use the bigram matrix, start with . and end when we see another .
g = torch.Generator().manual_seed(0)
word_num = 5
for i in range(word_num):
    idx = 0
    out = []
    while True:
        p = P[idx] # prob of the next char given the current one
        idx = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if idx == 0:
            break
        out.append(itos[idx])
    print(''.join(out))
