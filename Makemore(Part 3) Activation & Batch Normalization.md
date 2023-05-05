# Makemore(Part 3): Activation & Batch Normalization

## Initialization

### 1. Fixing the saturated tanh

Problem for activation function `tanh`:

$\frac{\partial L}{\partial h} = \frac{\partial L}{\partial t}\frac{\partial t}{\partial h}$

1. t = 1 / -1 (h very large / very small): 
    
    $\frac{\partial t}{\partial h}=0 => \frac{\partial L}{\partial h}=0$
    
    When t = 1, we will have the local derivative $\frac{\partial t}{\partial h}=0$ and total derivative $\frac{\partial L}{\partial h}=0$. 
    Batch_size = 32, hidden_size = 200
    hpreact: (32, 200) w2: (emb_size, 200) b: (200,)
    hpreact = emb_cat*w2 + b
        
    Here, we can see if we consider ith neuron in the hidden layer, the output of the ith layer $h_i = embcat*w_1[i]+b_1$. 
    
    Shape:
    
    - `embcat`: (32, `emb_size`)
    - $h_i$: (32, 1)
    - $w_1[i]$: (`emb_size`, 1)
    
    Therefore, we can see that if for all samples, the ith neuron will output 1/-1. Then we will not be able to learn anything for $w_1[i]$. In this case, we say the neuron is dead(i.e. This neuron will not activate for any functions)
    
    If we plot h(the output from the activation function), and we see a complete white column, this means the that column is a dead neuron given that for every sample, that neuron will generate a output > 0.99. The corresponding weight and bias will be impossible to learn from gradient descent given the gradient for them is almost 0
    
2. t = 0 (h = 0):
    
    $\frac{\partial t}{\partial h}=1 => \frac{\partial L}{\partial h}=\frac{\partial L}{\partial h}$
    
    When h = 0, we will have t = 0 and $\frac{\partial t}{\partial h}=1 => \frac{\partial L}{\partial h}=\frac{\partial L}{\partial h}$
    
    In this case, the local gradient is 1, and will not change anything from the upstream gradient. Therefore, we should also try to avoid that
    

In summary, we don’t want t = 1 and -1. Namely, we don’t want our h to be very large and very small. Given h is computed from $h = embcat*w_1+b_1$, we want to keep $w_2$ and $b_2$ small to ensure $h$ is also small.

**We don’t want the pre-activation states to be way too small(close to 0) because then `tanh` is not doing anything. We also don’t want the pre-activation state to be too large because then `tanh` is saturated. We want the pre-activation state to be roughly Gaussian(zero mean and one standard deviation)** 

We might encounter this problem in  initialization and optimization:

1. Initialization:
    
    As I have explained above, if we initialize $w_1$ and $b_1$ too large, $hpreact$ will be too large and make t = 1 / -1 which will lead to the zero gradient problem.
    
     So during initialization, we should keep $w_1$ and $b_1$ small: 
    
    ```python
    w1 = torch.randn(emb_size*block_size, hidden_size) * 0.1
    b1 = torch.randn(hidden_size) * 0
    ```
    
2. Optimization: 
    
    During optimization, we might sometimes overshoot(learning rate too large). In that case, w1 and b1 after updates could be too small, making pre-activation state to small. Then we will have the zero gradient problem again 
    

### 2. Kaiming Init

The previous way of determining the initialization is too complicated. It require us to find the initialization by checking the distribution of the output of activation functions. It could be very hard to init manually and keep the pre-activation state for every layer as normalized distribution given the NN becomes deeper and deeper

Instead, we have a more protocol way to init weights: Kaiming Init

- Kaiming Init: 
    We want to maintain $std = \frac{gain}{\sqrt{fan\_in}}$

By maintaining the std, we can just multiply the std to a unit distribution

We can divide this initialization into two parts:

1. `weight *= 1/ (fan_in ** 0.5)`
    
    Case: Only have a sandwich of linear layer. 
    
    Problem with linear layers:
    
    Consider x and w are two standard normalized matrix, when they multiply together, the result will have a higher variance. This means our pre-activation distribution will expand and take extreme value. These extreme value as input will lead zero gradient for the activation function.
    
    To reduce the variance, we `*= 1/ (fan_in ** 0.5)`
    
2. `weight *= 1/ (fan_in ** 0.5) * gain`
    
    Case: Aside from linear layers, we also have activation layers in between
    
    ```python
    layers = [Linear(block_size*n_emb, n_hidden, False), Tanh(),
              Linear(n_hidden, n_hidden, bias=False), Tanh(),
              Linear(n_hidden, n_hidden, bias=False), Tanh(),
              Linear(n_hidden, n_hidden, bias=False), Tanh(),
              Linear(n_hidden, n_hidden, bias=False), Tanh(),
              Linear(n_hidden, vocab_size, bias=False)
             ]
    ```
    
    If we have a sandwich of linear layers alone then initializing our weights without the “gain” will conserve the standard deviation of 1. But due to that we have this interspersed `tanh` layers in there and these `tanh` layers are **squashing functions** and so they take your distribution and they slightly squash it. So some “gain” is necessary to keep expanding it to fight the squashing. $\frac{5}{3}$ is a good value for `tanh` function
    

## 3. Batch Normalization:

1. Motivation: We want pre-activation state to be roughly gaussian. Therefore, we can just normalize it to make it exactly gaussian
2. Basic Intro:
    - Initialization
    No matter what the distribution of the `hpreact` is coming in, we will normalize `hpreact` and make it unit gaussian.(`bngain=1`and `bnbias=0`) Then `hpreact` will be unit gaussian for each neuron
    - Optimization
        
        We will use back propagation to train `bngain` and `bnbias` to shift the gaussian distribution by allowing it to change its mean and variance
        
    - Process:
        1. Calculate the mean and std of the input that are feeding into the batch norm layer over that batch.
        2.  Center that batch to be unit gaussian and then offsetting and scaling it by the learned bias and gain
        3. Meanwhile, keep track of the mean and std of the inputs and maintain the running mean and running std. Running mean and running std will be used further in the testing stage. During the test time, we want to forward individual sample to the network and get the result. Given that, we no longer have a batch for the input in the testing stage. We will use the running mean and running std for testing
3. Why Scale and shift: We are not gonna achieve a very good result with standard(zero mean and unit variance) gaussian distribution. **The reason is that we want the pre-activation state to be roughly gaussian but only at initialization but we don’t want them always to be forced gaussian during training.** We would like to allow the neural net to move this around to potentially to make it more diffuse, to make it more sharp, to make same tanh neurons more trigger happy or less trigger happy. Therefore, we need to have scale and shift
    
    
    1. **We only want standard normalization at the initialization**
        
        During initialization, having a standard normalized pre-activation state could give us better gradient after going through the activation function
        
        Reason: We don’t want pre-activation has a range wide(has too large or too small values)
        
    2. **During training, we don’t want the pre-activation state to always stay as standard normalized**
        
        Adjusting the distribution of the activation could better capture the structure of the data. The scale and shift operation in batch normalization provides the flexibility for the model to do this. By allowing the model to adjust gamma and beta, it can optimize the shape and scale of the activation distribution for each layer as needed.
        
        **Examples of how a shifting normal distributed pre-activation could help us during training:**
        
        1. Adjusting Activations for Different Activation Functions:
            - Different activation functions have different active ranges. For example, the sigmoid function maps inputs to the range (0, 1), while the tanh function maps inputs to the range (-1, 1). Depending on the activation function used, the model may benefit from adjusting the scale and shift of the activations to keep them in the active range.
            - For instance, if the tanh activation function is used in the following layer, the model may learn to scale and shift the normalized activations to fall within the range where tanh is most sensitive to changes (around 0). This allows the model to better utilize the non-linear properties of the activation function.
        2. Adapting to Data Distribution:
            - The structure and distribution of the data can vary across different tasks. For example, in image classification, the distribution of pixel values in natural images may differ from that of medical images.
            - By allowing the model to learn the scaling factor (gamma) and shifting factor (beta), **it can adapt the activation distribution to better match the characteristics of the specific data being processed. For instance, if certain features are more important for a particular classification task, the model may learn to scale up those activations, making them more pronounced in the decision-making process.**
        3. Dealing with Class Imbalance:
            - In classification tasks with class imbalance (where some classes have fewer samples than others), the model may need to be more sensitive to the underrepresented class to achieve good performance.
            - The scale and shift operation in batch normalization can help the model adjust the activations so that the features associated with the underrepresented class are more pronounced. This makes the model more "trigger-happy" to detect the underrepresented class, thereby improving its performance on the minority class.
        
        In summary, by allowing the model to learn the scale and shift parameters, batch normalization provides the flexibility to adjust the distribution of activations based on the characteristics of the data and the specific task, leading to better model performance.
        
4. Initialization of BatchNorm parameters
    
    Key Idea: 
    
    1. At initialization, we want a standard normalized pre-activation state
    2. `hpreact = bngain * ((hpreact-bnmeani) / bnstd) + bnbias`  
    
    Parameters: 
    
    1. `bnmean_running = 0` and `bnstd_running = 1`
        
        $\because$ At initialization, we want a standard normalized(zero-mean, unit variance) pre-activation state
        
        $\therefore$ We know that the mean and variance of the pre-activation is 0 and 1
        
    2. `bngain = 1`   and `bnbias = 0`
        
        $\because$ At initialization, we want a standard normalized(zero-mean, unit variance) pre-activation state & `hpreact = bngain * ((hpreact-bnmeani) / bnstd) + bnbias` 
        
        $\therefore$ We will have `bngain=1` and `bnbias=0` to have:
        
        ```python
        hpreact = bngain * ((hpreact-bnmeani) / bnstd) + bnbias 
        				= (hpreact-bnmeani) / bnstd 
        ```
        

1. Where to apply batch normalization
We usually take the result from the linear layer or conv layer(pre-activation state) and append a batch normalization layer to normalized the result. Then we will input the normalized result to the activation function to control the scale of these activations at every point in the NN.
2. Regularization
    
    Batch normalization (BatchNorm) has the effect of regularization, even though its primary purpose is to stabilize and accelerate the training of deep neural networks by reducing internal covariate shift. The regularization effect of BatchNorm is attributed to the noise introduced during the normalization process, which helps prevent overfitting. Here are the key reasons why BatchNorm acts as a regularizer:
    
    1. Mini-Batch Noise: During training, BatchNorm normalizes the pre-activations of each mini-batch independently using the mini-batch mean and variance. Because the mini-batch statistics are noisy approximations of the overall dataset statistics, this introduces a form of noise into the training process. This noise can be viewed as a form of stochastic regularization, similar to dropout, as it encourages the model to be robust to variations in the input data and helps prevent overfitting to the training set.
    2. Dependency on Other Examples: In traditional training, each example is processed independently with respect to the loss and gradient computation. However, with BatchNorm, the normalization of each example depends on the statistics of the entire mini-batch. **This introduces a form of input dependency, where the output for a specific example depends on the other examples in the mini-batch. This dependency acts as a regularizer by preventing the model from relying too heavily on any individual example.**
    3. Smoothing the Optimization Landscape: BatchNorm transforms the optimization landscape by making the loss surface smoother and more well-conditioned. This can lead to more generalizable solutions, as the optimization process is less likely to get stuck in sharp, non-generalizable minima and is more likely to find wider, more stable regions of the loss surface.
    
    It's important to note that while BatchNorm has a regularizing effect, it is not a substitute for other regularization techniques like L1/L2 regularization or dropout. In practice, BatchNorm is often used in combination with these other regularization methods to improve the generalization performance of the model.
    
    Paper related:
    
    1. Original Batch Normalization Paper:
        - Title: "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
        - Authors: Sergey Ioffe and Christian Szegedy
        - Link: **[https://arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167)**
        - Summary: This is the original paper that introduced the concept of batch normalization. It provides a detailed explanation of the motivation behind batch normalization, its implementation, and its impact on the training of deep neural networks.
    2. Analysis of the Regularization Effect of Batch Normalization:
        - Title: "How Does Batch Normalization Help Optimization?"
        - Authors: Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, Aleksander Madry
        - Link: **[https://arxiv.org/abs/1805.11604](https://arxiv.org/abs/1805.11604)**
        - Summary: This paper provides an analysis of how batch normalization helps optimization. It explains that the regularization effect of batch normalization comes from smoothing the optimization landscape, making it easier for gradient-based optimization methods to find good solutions.
    3. Understanding the Regularization Effect of Batch Normalization:
        - Title: "Understanding Batch Normalization"
        - Authors: Seyed-Iman Mirzadeh, Mehrdad Farajtabar, Ang Li, Hassan Ghasemzadeh
        - Link: **[https://arxiv.org/abs/1806.02375](https://arxiv.org/abs/1806.02375)**
        - Summary: This paper provides a deeper understanding of batch normalization and its regularization effect. It discusses the properties of batch normalization that contribute to its success and provides empirical and theoretical analysis.
        
3. Momentum
    
    Update rate for the `bnmean` and `bnstd` in the current epoch
    
    ```python
    bnmean_running = (1 - momentum) * bnmean_running + momentum * bnmeani
    bnstd_running = (1 - momentum) * bnstd_running + momentum * bnstd
    ```
    
    1. Small batch size ⇒ Small momentum
        - When using a small batch size, the mini-batch statistics (mean and variance) are more susceptible to noise and fluctuations due to the limited number of samples in each mini-batch. This can result in less reliable estimates of the overall population statistics.
        - To mitigate this, a smaller momentum value can be used to allow the running statistics to respond more quickly to new mini-batch statistics. This helps smooth out the noise and provides a more accurate representation of the population statistics over time.
    2. Large batch size ⇒ Large momentum
        - When using a large batch size, the mini-batch statistics are more reliable and less noisy due to the larger number of samples in each mini-batch. This provides better estimates of the overall population statistics.
        - In this case, a larger momentum value can be used to provide more stability to the running statistics. Since the mini-batch statistics are already reliable, giving more weight to the previous running statistics helps maintain consistency and reduces abrupt changes.

1. `torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)`
    
    $$
    y = \frac{x-E[x]}{\sqrt{Var[x] + \epsilon}}*\gamma + \beta
    $$
    
    - num_features: The number of neurons in the hidden layer. We need this to define `bngain` , `bnbias`, `bnmean_running` , `bnstd_running` :
        
        ```python
        bnmean_running = torch.zeros(1, num_features)
        bnstd_running = torch.ones(1, num_features)
        bngain = torch.ones(1, num_features)
        bnbias = torch.zeros(1, num_features)
        ```
        
    - eps = 1e-05
        
        We define this eps to avoid the case that denominator is 0
        
    - momentum: See 7
    - track_running_stats: Decide whether we need to use keep track of the `bnmean_running` , `bnstd_running`
        
        Namely, track_running_stats control whether we need to do this part:
        
        ```python
        with torch.no_grad():
        	bnmean_running = (1-momentum)*bnmean_running + momentum * bnmeani
        	bnstd_running = (1-momentum)*bnstd_running + momentun * bnstdi
        ```
        
    

### Initialization Summary

1. Fix the logits of the last layer
    
    Goal:  We want to initialize the weight of the last layer small so that we will not have a very large logits in the first few rounds
    
    Problem with extreme logits: Logits being too small or too large will lead to very high prob and very small prob after normalization. Given that we don’t have any training at initialization, these probs are assigned randomly. So we could have a very large prob on the incorrect labels and a very small prob on the correct labels
    
    Solution: 
    
    1. If the final layer is a Linear layer:
        1. Weight: Init weight to be a very small value 
            
            Note: We don’t want to init weight as 0. Here I think `1/fan_in**0.5` is a small value, so we just use this value. `1/fan_in**0.5` comes from the next part.
            
        2. Bias: Init as 0
        
        ```python
        # Define the model layers, later we could chain them together
        class Linear:
            def __init__(self, fan_in, fan_out, bias=True):
                self.weight = torch.randn(fan_in, fan_out) / fan_in**0.5
                self.bias = torch.zeros(fan_out) if bias else None
        		....
        ```
        
    
    1. If the final layer is BatchNorm
        
        out = gamma * xhat + beta
        
        1. gamma(scale): `gamma *= 0.1`
        2. beta(shift): `beta = torch.zeros(dim, keepdims=True)`
2. Fix the pre-activation state
    
    Goal: We don’t want the pre-activation state have a wide range(high variance) as activation function will have zero gradient for a very large/small pre-activation input
    
    Solution: 
    
    Init the weight in Linear layer: `weight *= 1/ (fan_in ** 0.5) * gain`
    
    1. `weight *= 1/ (fan_in ** 0.5)`
        
        Case: Only have a sandwich of linear layer. 
        
        Problem with linear layers:
        
        Consider x and w are two standard normalized matrix, when they multiply together, the result will have a higher variance. This means our pre-activation distribution will expand and take extreme value. These extreme value as input will lead zero gradient for the activation function.
        
        To reduce the variance, we `*= 1/ (fan_in ** 0.5)`
        
    2. `weight *= 1/ (fan_in ** 0.5) * gain`
        
        Case: Aside from linear layers, we also have activation layers in between
        
        ```python
        layers = [Linear(block_size*n_emb, n_hidden, False), Tanh(),
                  Linear(n_hidden, n_hidden, bias=False), Tanh(),
                  Linear(n_hidden, n_hidden, bias=False), Tanh(),
                  Linear(n_hidden, n_hidden, bias=False), Tanh(),
                  Linear(n_hidden, n_hidden, bias=False), Tanh(),
                  Linear(n_hidden, vocab_size, bias=False)
                 ]
        ```
        
        If we have a sandwich of linear layers alone then initializing our weights without the “gain” will conserve the standard deviation of 1. But due to that we have this interspersed `tanh` layers in there and these `tanh` layers are **squashing functions** and so they take your distribution and they slightly squash it. So some “gain” is necessary to keep expanding it to fight the squashing. $\frac{5}{3}$ is a good value for `tanh` function
        
    
3. Batch Normalization
    
    Idea: As we want normalized pre-activation state, let’s normalize it!
    
    `hpreact = bngain * ((hpreact-bnmeani) / bnstd) + bnbias`  
    
    Parameters: 
    
    1. `bnmean_running = 0` and `bnstd_running = 1`
        
        $\because$ At initialization, we want a standard normalized(zero-mean, unit variance) pre-activation state
        
        $\therefore$ We know that the mean and variance of the pre-activation is 0 and 1
        
    2. `bngain = 1`   and `bnbias = 0`
        
        $\because$ At initialization, we want a standard normalized(zero-mean, unit variance) pre-activation state & `hpreact = bngain * ((hpreact-bnmeani) / bnstd) + bnbias` 
        
        $\therefore$ We will have `bngain=1` and `bnbias=0` to have:
        
        ```python
        hpreact = bngain * ((hpreact-bnmeani) / bnstd) + bnbias 
        				= (hpreact-bnmeani) / bnstd 
        ```
        

Complete code:

```python
# Define Model layers
class Linear:
    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn(fan_in, fan_out) / (fan_in ** 0.5)
        self.bias = torch.zeros(fan_out) if bias else Non
    
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
        
# Init params
model.layers[-1].gamma *= 0.1
for layer in model.layers:
    if isinstance(layer, Linear):
        layer.weight *= 5/3     # Multiply by the gain
```