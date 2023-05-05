# Makemore(Part 5): Wavenet

In this section, we mainly want to build a hierarchical structure like the image below. 

![Untitled](Makemore(Part%205)%20Wavenet%206c518d79b5574485bc4f3f95b72b9720/Untitled.png)

More specifically, assume we have block_size = 8. We will build a model with this structure:

Embedding : (32, 8, 10)
Flatten : (32, 4, 20)
Linear : (32, 4, 100) BatchNorm1D : (32, 4, 100) Tanh : (32, 4, 100)
Flatten : (32, 2, 200)
Linear : (32, 2, 100) BatchNorm1D : (32, 2, 100) Tanh : (32, 2, 100)
Flatten : (32, 200)
Linear : (32, 100) BatchNorm1D : (32, 100) Tanh : (32, 100)
Linear : (32, 27) BatchNorm1D : (32, 27)

### Important Note:

1. We will modify the flatten layer: 
    
    ```python
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
    ```
    
2. We will modify the BatchNorm Layer:
    
    ```python
    class BatchNorm1D:
    		...     
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
    		...
    ```
    
3. When we do the evaluation, testing, and inference(sampling), we **must** turn off the training mode in BatchNorm Layer:
    
    ```python
    for layer in model.layers:
        if isinstance(layer, BatchNorm1D):
            layer.training = False
    ```
    
    Otherwise, we will be using the batch mean and variance instead of the running mean and variance. This will cause a large problem. For example, during inference, we want to sample one word at a time from the model.  We will feed the model with only one sample(size: 1*block_size), the batch var will be `nan`. This will give us `nan` in logits and cause further problem
    

My colab notebook link: 

[Google Colaboratory](https://colab.research.google.com/drive/1k_MBYwbYkSgqJZ5DIvf1vwNPsIIzCtma#scrollTo=pCOHO0LO9Lar)