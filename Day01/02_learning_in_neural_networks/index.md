+++
weight = 2
title = "Create and Optimize Neural Networks"
description = ""

[[author]]
name = "Dr. Atle E. Rimehaug"
role = "Conceptualization, Writing, and Editing"

[[author]]
name = "Dr. Nicholas Del Grosso"
role = "Conceptualization, Writing, and Editing"
+++
## Introduction

In this notebook, we'll build up a practical understanding of how neural networks are constructed and trained in PyTorch, using PyTorch's `optim` module to implement gradient descent and update the model parameters.

```mermaid
flowchart LR
    X[Input] --> M[Model]
    M --> P[Prediction]
    P --> L[Loss]
    Y[Target] --> L
    L --> G[Gradients]
    G --> U[Update]
    U --> M
```


We'll start with simple linear layers, inspect their parameters directly, and observe how predictions change as those parameters change. From there, we'll walk through the full training process—forward pass, loss computation, gradient calculation, and parameter updates.  We'll first use PyTorch's built-in tools, and at the end will re-implement Mean Squared Error and standard gradient descent ourselves. 



The goal is not just to train models that work, but to develop a clear mental model of why they work and what PyTorch is actually doing at each step.

## Setup

### Import Libraries


```python
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchinfo
```

---

## Section 1: How to Build a Neural Network Model

A neural network model consists of layers that transform input data into output predictions. The simplest layer is a linear layer, which multiplies inputs by weights and adds a bias - exactly what you did manually in the previous session. PyTorch's `nn.Linear` creates these layers for you, and `nn.Sequential` lets you stack multiple layers together to build deeper networks.

| Code | Description |
| :-- | :-- |
| `nn.Linear(n_features_in, n_features_out)` | Creates a linear model that takes in `n_features_in` input features (nodes) and<br>outputs `n_features_out` features (nodes).|
| `nn.Sequential(`<br>&nbsp;&nbsp;&nbsp;&nbsp;`nn.Linear(1,10),`<br>&nbsp;&nbsp;&nbsp;&nbsp;`nn.Linear(10,1)`<br>`)` | Creates a container where multiple layers are stacked in a sequential order. This ensures that input is passed through the network in that order during the forward pass. This model has one input feature, one output feature, and one hidden layer with 10 nodes. |
| `model = nn.Sequential(`<br>&nbsp;&nbsp;&nbsp;&nbsp;`nn.Linear(2,20),`<br>&nbsp;&nbsp;&nbsp;&nbsp;`nn.Linear(20,10),`<br>&nbsp;&nbsp;&nbsp;&nbsp;`nn.Linear(10,1)`<br>`)` | Assigns the container to a variable `model`. This model has 2 input nodes, two hidden<br>layers with 20 and 10 nodes, respectively, and one output node. |

**Example**: Create the model illustrated below, which has 3 input features and 2 output features:

```mermaid
flowchart LR
  %% Layers
  subgraph IN["Input Layer"]
    direction TB
    i1(( ))
    i2(( ))
    i3(( ))
  end

  subgraph OUT["Output Layer"]
    direction TB
    o1(( ))
    o2(( ))
  end

  %% Dense connections: Input -> Hidden
  i1 --> o1
  i2 --> o1
  i3 --> o1

  i1 --> o2
  i2 --> o2
  i3 --> o2

  %% Styling
  classDef input fill:#F4A261,stroke:#333,stroke-width:1px;
  classDef hidden fill:#2ECC71,stroke:#333,stroke-width:1px;
  classDef output fill:#4D7CFE,stroke:#333,stroke-width:1px;

  class i1,i2,i3 input;
  class o1,o2 output;

```


```python
model = nn.Linear(3, 2)
model
```




    Linear(in_features=3, out_features=2, bias=True)



Make sure data passes successfully through the model, and the output has the right shape. The last number inside `torch.Size` should correspond to the number of output features in the model.


```python
data = torch.randn(100, 3)
model(data).shape
```




    torch.Size([100, 2])



How many total parameters need to be trained in this model? (`torchinfo.summary(model)`):


```python
torchinfo.summary(model)
```




    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    Linear                                   8
    =================================================================
    Total params: 8
    Trainable params: 8
    Non-trainable params: 0
    =================================================================



**Exercise**: Create the model illustrated below, which has 4 input features and 1 output features:

```mermaid
flowchart LR
  %% Layers
  subgraph IN["Input Layer"]
    direction TB
    i1(( ))
    i2(( ))
    i3(( ))
    i4(( ))
  end

  subgraph OUT["Output Layer"]
    direction TB
    o1(( ))
  end

  %% Dense connections: Input -> Hidden
  i1 --> o1
  i2 --> o1
  i3 --> o1
  i4 --> o1

  %% Styling
  classDef input fill:#F4A261,stroke:#333,stroke-width:1px;
  classDef hidden fill:#2ECC71,stroke:#333,stroke-width:1px;
  classDef output fill:#4D7CFE,stroke:#333,stroke-width:1px;

  class i1,i2,i3,i4 input;
  class o1 output;

```



```python

```

Make sure data passes successfully through the model, and that the output has the right shape. The last number inside `torch.Size` should correspond to the number of output features in the model.


```python

```

How many total parameters need to be trained in this model? (`torchinfo.summary(model)`):


```python

```

 **Exercise**: Create the model illustrated below, which has 3 input features, 2 hidden features and 2 output features (Note: Use `nn.Sequential()`):

```mermaid
flowchart LR
  %% Layers
  subgraph IN["Input Layer"]
    direction TB
    i1(( ))
    i2(( ))
    i3(( ))
  end

  subgraph HID["Hidden Layer"]
    direction TB
    h1(( ))
    h2(( ))
  end

  subgraph OUT["Output Layer"]
    direction TB
    o1(( ))
    o2(( ))
  end

  %% Dense connections: Input -> Hidden
  i1 --> h1
  i2 --> h1
  i3 --> h1
  
  i1 --> h2
  i2 --> h2
  i3 --> h2

  h1 --> o1
  h2 --> o1

  h1 --> o2
  h2 --> o2
  

  %% Styling
  classDef input fill:#F4A261,stroke:#333,stroke-width:1px;
  classDef hidden fill:#2ECC71,stroke:#333,stroke-width:1px;
  classDef output fill:#4D7CFE,stroke:#333,stroke-width:1px;

  class i1,i2,i3 input;
  class h1,h2 hidden;
  class o1,o2 output;

```


```python

```

Make sure data passes successfully through the model, and the output has the right shape. The last number inside `torch.Size` should correspond to the number of output features in the model.


```python
data = torch.randn(100, 3)
model(data).shape
```




    torch.Size([100, 2])



How many total parameters need to be trained in this model? (`torchinfo.summary(model)`):


```python
torchinfo.summary(model)
```




    =================================================================
    Layer (type:depth-idx)                   Param #
    =================================================================
    Sequential                               --
    ├─Linear: 1-1                            8
    ├─Linear: 1-2                            6
    =================================================================
    Total params: 14
    Trainable params: 14
    Non-trainable params: 0
    =================================================================



 **Exercise**: As the number of nodes increases, the style of visualization changes to more of a summary.  Create the model illustrated below, which has 8 input features, 20 hidden features and 1 output feature (Note: Use `nn.Sequential()`):

```mermaid
flowchart LR
  x["Input\n(batch, 8)"]
  l1["Linear\nin=8 → out=20"]
  l2["Linear\nin=20 → out=1"]
  y["Output\n(batch, 1)"]

  x --> l1 --> l2 --> y

```


```python

```

Make sure data passes successfully through the model, and the output has the right shape:


```python

```

How many total parameters need to be trained in this model? (`torchinfo.summary(model)`):


```python

```

 **Exercise**: Create the model illustrated below:

```mermaid
flowchart LR
  x["Input\n(batch, 20)"]
  l1["Linear\nin=20 → out=50"]
  l2["Linear\nin=50 → out=35"]
  l3["Linear\nin=35 → out=10"]
  l4["Linear\nin=10 → out=1"]
  y["Output\n(batch, 1)"]

  x --> l1 --> l2 --> l3 --> l4 --> y

```


```python

```

Make sure data passes successfully through the model, and the output has the right shape:


```python

```

How many total parameters need to be trained in this model? (`torchinfo.summary(model)`):


```python

```

---

## Section 2: Initialize and Extract Model Parameters

Each of these model parameters (i.e. the weights and biases of each layer) are stored in Pytorch as a `tensor()`.  
When creating the model, PyTorch starts out with random values for each parameter, and then these values are updated during learning in order to better-fit the data.

In this section, we'll explore the parameters directly and manually update them ourselves, to see how they influence the model's prediction.

| Code | Description |
| :-- | :-- |
| `list(model.parameters())` | Get a list of the tensors in the model. |
| `list(model.named_parameters())` | Get a dict of the tensors in the model and their names. |
| `model.state_dict()` | Yet another way to see the model parameters. |
| `model[0].weight` | Get the "weight" tensor from the first layer (index 0) |
| `model[0].bias` | Get the "bias" tensor from the first layer (index 0) |
| `model[1].weight` | Get the "weight" tensor from the second layer (index 1)|
| `model[1].bias` | Get the "bias" tensor from the second layer (index 1)|
| `with torch.no_grad():`<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`model[1].weight[:] = torch.tensor([1, 2, 3])` | Override the automatically generated random parameter values in the second layer.  |

### Exercises

**Example**: Create the following model and view the randomly-initialized values of all parameters in the model.


```mermaid
flowchart LR
  x["Input\n(batch, 3)"]
  l1["Linear\nin=3 → out=2"]
  l2["Linear\nin=2 → out=1"]
  y["Output\n(batch, 1)"]

  x --> l1 --> l2 --> y

```


```python
model = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
list(model.named_parameters())
```




    [('0.weight',
      Parameter containing:
      tensor([[-0.5276, -0.4558, -0.0812],
              [-0.4980, -0.2540, -0.0134]], requires_grad=True)),
     ('0.bias',
      Parameter containing:
      tensor([-0.4436,  0.5642], requires_grad=True)),
     ('1.weight',
      Parameter containing:
      tensor([[ 0.0947, -0.6549]], requires_grad=True)),
     ('1.bias',
      Parameter containing:
      tensor([0.5782], requires_grad=True))]



Generate a sample of random data and calculate the model's output with those parameter values:


```python
torch.manual_seed(42)     # Set the randomizer to a specific value
data = torch.randn(1, 3)  # Generate 1 sample of data with 3 features (a 1x3 tensor)
model(data)
```




    tensor([[0.2757]], grad_fn=<AddmmBackward0>)



Set the bias of the output layer (layer "1") to **100**, and run the same data through the model. How does it change the output?


```python
with torch.no_grad(): 
    model[1].bias[:] = torch.tensor([100])

model(data)
```




    tensor([[99.6976]], grad_fn=<AddmmBackward0>)



**Exercise**: Create the following model and view the randomly-initialized values of all parameters in the model.


```mermaid
flowchart LR
  x["Input\n(batch, 1)"]
  l1["Linear\nin=1 → out=1"]
  l2["Linear\nin=1 → out=1"]
  y["Output\n(batch, 1)"]

  x --> l1 --> l2 --> y

```


```python

```

Generate a sample of random data and calculate the model's output with those parameter values:


```python

```

Set the weights on the input layer (layer "0") to **-10**, and run the same data through the model. How does it change the output?


```python

```

**Exercise**: Create the following model and view the randomly-initialized values of all parameters in the model.


```mermaid
flowchart LR
  x["Input\n(batch, 100)"]
  l1["Linear\nin=100 → out=50"]
  l2["Linear\nin=50 → out=1"]
  l3["Linear\nin=1 → out=1"]
  y["Output\n(batch, 1)"]

  x --> l1 --> l2 --> l3 --> y

```


```python

```

Generate a sample of random data and calculate the model's output with those parameter values:


```python

```

Set **both** the bias and the weights of the output layer (layer "2") to **0**, and run the same data through the model. How does it change the output?


```python

```

---

## Section 3: Training Our Model: Iteratively Optimizing Model Parameters to Minimize Loss



Training models requires:
  1. A "forward" pass: The model make a prediction, and the loss is calculated.
  2. A "backward" pass: Loss gradients are calculated for model parameters, and the parameter values are updated.
  3. Iteration: Training is repeated some number of times.
  4. Learning tracking: The training process is recorded for troubleshooting.


Below is the model training function we'll use for doing each training iteration: `update_model()`:


```python
def update_model(model, loss_function, optimizer, x, y_obs) -> float:
    """
    Do one training iteration, and return its loss.
    
    Arguments:
      - model: the PyTorch model we'll be training (e.g. `nn.Sequential(Linear(1, 1)))`)
      - loss_function: the loss function we'll use to estimate model fit, by calculating loss (e.g. `nn.MSELoss()`)
      - optimizer: a PyTorch "Optimizer" object that decides how much to update each parameter. (e.g. `optim.SGD()`)
      - x: The `x` input data used for training
      - y_obs: The expected result data used for training. (used here for "supervised" training.)    
    """

    # Forward Pass
    optimizer.zero_grad()                  # Reset parameter gradients
    y_pred = model(x)                      # Make a Prediction
    loss = loss_function(y_pred, y_obs)    # Calculate the overall prediction error ("Loss")

    # Backward Pass
    loss.backward()   # Calculate the loss gradients and store them in each parameters' ".grad" attribute
    optimizer.step()  # Update the model's parameters, based on their stored gradients

    # record the loss, for plotting or logging
    return loss.item()  
    


```

We'll monitor the quality of training by plotting:
  1. `plot_losses()`: The loss value over time shows how quickly training is improving.
  2. `plot_1d_model_fit()`: Makes a scatter plot with regression line, to show the model's fit.


```python

def plot_losses(losses, ax=None) -> None:
    """Plot the losses over each training iteration."""
    if ax is None:
        ax = plt.gca()
    ax.plot(losses, ':.')
    ax.set(
        title='Loss Over Learning Iterations',
        xlabel='Iteration #',
        ylabel='Loss',
    )
    

def plot_1d_model_fit(model, x, y_obs, ax=None) -> None:
    """Plot a scatterplot and regression line to show model fit"""
    if ax is None:
        ax = plt.gca()

    # Make a scatterplot of the data points (x vs y_obs)
    ax.plot(x, y_obs, '.', markersize=1.5);

    # Plot a regression line
    x_range = torch.linspace(x.min(), x.max(), 100)  # Get a sequence of x values
    y_pred = model(x_range.reshape(-1, 1))
    ax.plot(x_range, y_pred.detach().numpy(), ':', linewidth=2)
    ax.set(title='Model Fit', xlabel='X', ylabel='Y')

```

### Exercises

There are so many things to choose: How does the learning rate, optimizer, model architecture and complexity, and even the number of learning iterations itself affect learning?  Let's get a sense of each of these parameters' behaviors.  

In each of the following exercises, experiment with the requested single parameter of the model and training, in order to improve training performance.  (Hint: There's an `UPDATE ME` comment next to the key line to update.)

**Exercise**: How many training iterations are needed to train this model? Update the `num_iterations` variable just until the model has been fit. When the loss curve starts to flatten, the model is as fit as well as it can be.


```python
# Generate some Data
torch.manual_seed(42)
x = torch.randn(200, 1)
y_obs = (5 * x + 2) + 2 * torch.randn(200, 1)  # True Model: 5x + 2
y_obs[:20]

# Set up the model and training hyperparameters
model = nn.Sequential(nn.Linear(1, 1))
loss_function = nn.MSELoss()
optimizer = optim.SGD(params=model.parameters(), lr=.01)
num_iterations = 30   # UPDATE ME!

# Train the Model
losses = [update_model(model, loss_function, optimizer, x, y_obs) for _ in range(num_iterations)]

# Make a Figure showing the losses and model fit
plt.subplot(1, 2, 1); plot_losses(losses);
plt.subplot(1, 2, 2); plot_1d_model_fit(model, x, y_obs);
plt.tight_layout();
dict(model.state_dict())
```




    {'0.weight': tensor([[2.0565]]), '0.bias': tensor([1.4153])}




    
![png](index_Exercises_files/index_Exercises_66_1.png)
    



```python

```

**Exercise**: Does the number of layers affect training speed?  Add a hidden layer to the `model` below and see if it improves or worsens model training.


```python
# Generate some Data
torch.manual_seed(42)
x = torch.randn(200, 1)
y_obs = (5 * x + 2) + 2 * torch.randn(200, 1)  # True Model: 5x + 2
y_obs[:20]

# Set up the model and training hyperparameters
model = nn.Sequential(nn.Linear(1, 1))   # UPDATE ME!
loss_function = nn.MSELoss()
optimizer = optim.SGD(params=model.parameters(), lr=.01)
num_iterations = 150   # UPDATE ME!

# Train the Model
losses = [update_model(model, loss_function, optimizer, x, y_obs) for _ in range(num_iterations)]

# Make a Figure showing the losses and model fit
plt.subplot(1, 2, 1); plot_losses(losses);
plt.subplot(1, 2, 2); plot_1d_model_fit(model, x, y_obs);
plt.tight_layout();
dict(model.state_dict())
```




    {'0.weight': tensor([[4.8087]]), '0.bias': tensor([2.2150])}




    
![png](index_Exercises_files/index_Exercises_69_1.png)
    



```python

```

**Exercise**: The `optimizer` is responsible for making a decision on how to update the parameters.  It can take into account the parameter values themselves, their gradients, the parameters' training history, and supplied "hyperparameters" like the learning rate.  

In this exercise, we're using the `SGD()` optimizer (i.e. Gradient Descent).  It takes a `lr` (learning rate) parameter to decide how much to update the parameters. 

Experiment: For this dataset and model, how high can we make the learning rate (`lr`), before the model stops learning productively? In other words, from (approximately) which learning rate does the loss increase instead of decrease?


```python
# Generate some Data
torch.manual_seed(42)
x = torch.randn(200, 1)
y_obs = (5 * x + 2) + 2 * torch.randn(200, 1)  # True Model: 5x + 2
y_obs[:20]

# Set up the model and training hyperparameters
model = nn.Sequential(nn.Linear(1, 1))
loss_function = nn.MSELoss()
optimizer = optim.SGD(params=model.parameters(), lr=.01)  # UPDATE ME (just the "lr" value)!
num_iterations = 250   

# Train the Model
losses = [update_model(model, loss_function, optimizer, x, y_obs) for _ in range(num_iterations)]

# Make a Figure showing the losses and model fit
plt.subplot(1, 2, 1); plot_losses(losses);
plt.subplot(1, 2, 2); plot_1d_model_fit(model, x, y_obs);
plt.tight_layout();
dict(model.state_dict())
```




    {'0.weight': tensor([[5.0558]]), '0.bias': tensor([2.2563])}




    
![png](index_Exercises_files/index_Exercises_72_1.png)
    



```python

```

**Exercise**: Why pick a learning rate ourselves at all?  Pytorch supplies a number of "adaptive" optimizers, that tweak the learning rate based on a number of self-measuring factors.  These adaptive optimizers tend to be more robust than the standard gradient descent (SGD) optimizer.

| Complexity | Code | Description |
| :-- | :-- | :-- |
| Lowest | `torch.optim.SGD(model.parameters(), lr=0.01)` | a gradient descent (GD) optimizer with a *fixed* learning rate `lr` of 0.01 |
| Low | `torch.optim.Adagrad(model.parameters(), lr=.01)` | an adaptive gradient optimizer, with an *initialized* learning rate of 0.01.|
| High | `torch.optim.RMSprop(model.parameters(), lr=.01)` | a root mean square (RMS) propagation optimizer, with an *initialized* learning rate of 0.01. It's Adagrad with a moving average. |
| Highest | `torch.optim.Adam(model.parameters(), lr=.01)` | an adaptive moment optimizer, with an *initialized* learning rate of 0.01. It's RMSprop with momentum. |

*Experiment*: The learning rate has been set too high, and the model is not converging properly.  Instead of adjusting the learning rate, exchange the optimizer itself with a more robust one. See if it's able to adjust and recover convergence. 


```python
# Generate some Data
torch.manual_seed(42)
x = torch.randn(200, 1)
y_obs = (5 * x + 2) + 2 * torch.randn(200, 1)  # True Model: 5x + 2
y_obs[:20]

# Set up the model and training hyperparameters
model = nn.Sequential(nn.Linear(1, 1))
loss_function = nn.MSELoss()
optimizer = optim.SGD(params=model.parameters(), lr=1.5)   # UPDATE ME! (replace `SGD` with another from the table above)
num_iterations = 75 

# Train the Model
losses = [update_model(model, loss_function, optimizer, x, y_obs) for _ in range(num_iterations)]

# Make a Figure showing the losses and model fit
plt.subplot(1, 2, 1); plot_losses(losses);
plt.subplot(1, 2, 2); plot_1d_model_fit(model, x, y_obs);
plt.tight_layout();
dict(model.state_dict())
```




    {'0.weight': tensor([[2.8141e+24]]), '0.bias': tensor([4.1882e+24])}




    
![png](index_Exercises_files/index_Exercises_75_1.png)
    



```python

```

---

## Section 4: Verifying PyTorch: Implementing Mean Squared Error and Standard Gradient Descent

Are you wondering whether there is still some hidden magic that PyTorch is doing behind the scenes with `nn.MSELoss()` and `optim.SGD()`?  In this section, we'll re-implement Mean-Square-Error loss calculations and Standard Gradient Descent, in order to verify that PyTorch is indeed just doing these calculations.

The most challenging part of coding this is usually the calculating the derivatives on each parameter, but using PyTorch Tensors makes this easy because they have [AutoGrad](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html); they will calculate the deriviates for us behind the scenes.  

That's what `loss.backward()` does: it calculates the derivatives of the parameters with respect to the loss and stores them on the parameter tensors themselves in their `.grad` attributes.  Then, the optimizer calculation can access those deriviates directly to decide how much to update the parameters.  To reset the gradients for the next pass, `zero_grad()` is called.  

Let's try it out!

### Exercises

Each of the exercises below build on the last one.  The `UPDATE ME!`-commented lines have directions on how to update the code to replace the PyTorch-supplied calculations with our own.

**Exercise**: **Mean-Squared Error Loss**: The new `update_model2()` function doesn't take a `loss_function` any more; it makes its own internally.  Replace the `nn.MSELoss()(y_pred, y_obs)` with the direct calculation of loss below, and confirm that the training goes exactly the same as with the PyTorch-supplied `nn.MSELoss()` function:

 $$  \mathrm{MSE}(\hat{y}, y) = \frac{1}{n} \sum_{i=1}^{n} \left(\hat{y}_i - y_i\right)^2 $$



```python
def update_model2(model, optimizer, x, y_obs) -> float:

    # Forward Pass
    optimizer.zero_grad()                  # Reset parameter gradients
    y_pred = model(x)                      # Make a Prediction
    loss = nn.MSELoss()(y_pred, y_obs)    # UPDATE ME!  Calculate mean squared error here: ((y_pred - y_obs) ** 2).mean()

    # Backward Pass
    loss.backward()   # Calculate the loss gradients and store them in each parameters' ".grad" attribute
    optimizer.step()  # Update the model's parameters, based on their stored gradients

    # record the loss, for plotting or logging
    return loss.item()  
    


# Generate some Data
torch.manual_seed(42)
x = torch.randn(200, 1)
y_obs = (5 * x + 2) + 2 * torch.randn(200, 1)  # True Model: 5x + 2
y_obs[:20]

# Set up the model and training hyperparameters
model = nn.Sequential(nn.Linear(1, 1)) 
optimizer = optim.SGD(params=model.parameters(), lr=.01)
num_iterations = 150  

# Train the Model
losses = [update_model2(model, optimizer, x, y_obs) for _ in range(num_iterations)]

# Make a Figure showing the losses and model fit
plt.subplot(1, 2, 1); plot_losses(losses);
plt.subplot(1, 2, 2); plot_1d_model_fit(model, x, y_obs);
plt.tight_layout();
dict(model.state_dict())
```




    {'0.weight': tensor([[4.8087]]), '0.bias': tensor([2.2150])}




    
![png](index_Exercises_files/index_Exercises_81_1.png)
    



```python

```

**Exercise**: **Standard Gradient Descent**: Let's build on the previous function by replacing the PyTorch-supplied `optim.SGD()` optimizer with our own direct calculation of the standard gradient descent equation:

$$
w_{i, t+1} = w_{i,t} - \eta \times \nabla L(w_{i,t})
$$

where $t$ is the current time step (epoch) in the training, $L$ is loss function to be minimized, and $\eta$ is the learning rate.

**Hint**: Useful code for updating parameters: `p[:] = p - lr * p.grad`



```python
def update_model3(model, x, y_obs) -> float:

    # Forward Pass
    model.zero_grad()     # Reset parameter gradients 
    
    optimizer = optim.SGD(model.parameters(), lr=.01) # UPDATE ME: Delete this line
    
    y_pred = model(x)                      # Make a Prediction
    loss = ((y_obs - y_pred) ** 2).mean()  # Calculate Loss using the Mean-Squared Error method.

    # Backward Pass
    loss.backward()   # Calculate the loss gradients and store them in each parameters' ".grad" attribute
    optimizer.step()  # UPDATE ME: Delete this line

    lr = .01
    with torch.no_grad():
        for p in model.parameters():
            ...    # UPDATE ME: Do standard gradient descent: p[:] = p - lr * p.grad
    
    
    

    # record the loss, for plotting or logging
    return loss.item()  
    


# Generate some Data
torch.manual_seed(42)
x = torch.randn(200, 1)
y_obs = (5 * x + 2) + 2 * torch.randn(200, 1)  # True Model: 5x + 2
y_obs[:20]

# Set up the model and training hyperparameters
model = nn.Sequential(nn.Linear(1, 1)) 
optimizer = optim.SGD(params=model.parameters(), lr=.01)
num_iterations = 150  

# Train the Model
losses = [update_model3(model, x, y_obs) for _ in range(num_iterations)]

# Make a Figure showing the losses and model fit
plt.subplot(1, 2, 1); plot_losses(losses);
plt.subplot(1, 2, 2); plot_1d_model_fit(model, x, y_obs);
plt.tight_layout();
dict(model.state_dict())
```




    {'0.weight': tensor([[4.8087]]), '0.bias': tensor([2.2150])}




    
![png](index_Exercises_files/index_Exercises_84_1.png)
    



```python

```

That's it, we've now replaced PyTorch's objects with our calculation, taking advantage of PyTorch's [AutoGrad](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) feature to make the code for our implementation simple.

Note: We don't advise doing this as a general practice, though. But it's very nice to have tested code from PyTorch supplied to us.


