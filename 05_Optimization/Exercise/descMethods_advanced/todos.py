"""Gradient descent exploration - todos"""

import torch

# TODO 1: linear model
def linear_model(x, theta):
  """Predictions of linear model.
   
  Args:
    x: (n, 1) torch.tensor with inputs
    theta: (2, 1) torch.tensor with parameters

  Output:
    y: (n, 1) torch.tensor with linear predictions
  """
  X1=torch.ones(x.size(0),1)
  X2=x[:,0].view(-1,1)
  
  # prepend x by a zeros to match the dims of theta
  x = torch.hstack((X1,X2))
  y = x @ theta
  return y

def mse(y, y_predict):
  """Mean squared error loss.
   
  Args:
    y, y_predict: (n, 1) torch.tensors with targets and predictions

  Output:
    mse: (scalar)
  """
  # magda version
  mse = ((y_predict-y)**2).mean() 
  return mse


# TODO 2: complete the code below
def lin_grads(x, y, theta):
  """Gradient for parameters of linear model.
  
  Args:
    x: (n, 1) torch.tensor with inputs
    y: (2, 1) torch.tensor with outputs
    theta: (2, 1) torch tensor with initial parameters (intercept, slope)
  """
  # prepend x by a zeros to match the dims of theta
  x = torch.cat([torch.ones_like(x), x], dim=1) 
  grad = 2*(x.T @ (x @ theta - y)) 
  return grad

def grad_descent(x,y, theta, lr, grad_func, model, n_epochs=1):
  # monitor thetas
  thetas = torch.empty(2, n_epochs+1)
  thetas[:,0] = theta[:,0]

  # monitor losses
  losses = torch.empty((n_epochs+1, 1))
  
  # initial prediction and loss
  y_pred = model(x,theta)
  losses[0] = mse(y, y_pred)
  # grad descent
  for epoch in range(n_epochs):
    gradient = grad_func(x,y, theta)
    theta = theta - lr * gradient
    # monitor thetas and losses
    y_preds = model(x, theta)
    thetas[:,epoch+1] = theta.T
    losses[epoch+1] = mse(y, y_preds)

  # Return output
  return thetas, losses

# TODO 3: complete the code below

def stochastic_grad_descent(x,y, theta, lr, bs, grad_func, model, n_epochs=1):  
  thetas = torch.empty(2, n_epochs+1)
  thetas[:,0] = theta[:,0]

  # monitor losses
  n = x.size(0)
  n_batches = n // bs if n % bs == 0 else n // bs + 1
  losses = torch.empty((n_epochs*n_batches+1, 1))

  # initial prediction and loss
  y_pred = model(x,theta)
  losses[0] = mse(y, y_pred)

  # grad descent
  for epoch in range(n_epochs):
    # divide into batches
    perm = torch.randperm(x.shape[0])
    shuffle_x = x[perm]
    shuffle_y = y[perm]
    batches_x = torch.split(shuffle_x, n_batches)
    batches_y = torch.split(shuffle_y, n_batches)
    for (b_x, b_y) in zip(batches_x, batches_y):
        gradient = grad_func(b_x, b_y, theta)
        theta = theta - lr * gradient
        
    # monitor losses
    y_preds = model(x, theta)
    thetas[:,epoch+1] = theta.T
    losses[epoch+1] = mse(y, y_preds)

  # Return output
  return thetas, losses


# TODO 4: complete the code below
def sgd_momentum(x,y, theta, lr, beta, bs, grad_func, model, n_epochs=1):
  """SGD with momentum for theta parameters.
  
  Args:
    x: (n, 1) torch.tensor with inputs
    y: (2, 1) torch.tensor with outputs
    theta: (2, 1) torch tensor with initial parameters (intercept, slope)
    lr: (scalar) learning rate
    beta: momentum smoothing parameter
    bs: batch size
    grad_func: function to calculate gradients
    model: model for predictions
    n_epochs: (scalar) number of epochs to run the GD for
  """
  
  # monitor thetas
  thetas = torch.empty(2, n_epochs+1)
  thetas[:,0] = theta[:,0]

  # monitor losses
  n = x.size(0)
  n_batches = n // bs if n % bs == 0 else n // bs + 1
  losses = torch.empty((n_epochs*n_batches+1, 1))
  
  # initial prediction and loss
  y_pred = model(x,theta)
  losses[0] = mse(y, y_pred)

  # grad descent
  momentum = 0
  for epoch in range(n_epochs):
    # divide into batches
    perm = torch.randperm(x.shape[0])
    shuffle_x = x[perm]
    shuffle_y = y[perm]
    batches_x = torch.split(shuffle_x, n_batches)
    batches_y = torch.split(shuffle_y, n_batches)
    for (b_x, b_y) in zip(batches_x, batches_y):
        
        # here comes the change!
        gradient = grad_func(b_x, b_y, theta)
        momentum=beta*momentum+(1-beta)*gradient
        theta = theta - lr * momentum

        #gradient = grad_func(b_x, b_y, theta)
        #theta = theta - lr * gradient
        
    # monitor losses
    y_preds = model(x, theta)
    thetas[:,epoch+1] = theta.T
    losses[epoch+1] = mse(y, y_preds)

  # Return output
  return thetas, losses


# TODO 5: complete the code below
def sgd_nestorov(x,y, theta, lr, beta, bs, grad_func, model, n_epochs=1):
  thetas = torch.empty(2, n_epochs+1)
  thetas[:,0] = theta[:,0]

  # monitor losses
  n = x.size(0)
  n_batches = n // bs if n % bs == 0 else n // bs + 1
  losses = torch.empty((n_epochs+1, 1))

  # initial prediction and loss
  y_pred = model(x,theta)
  losses[0] = mse(y, y_pred)

  m = torch.zeros(theta.shape)

  # grad descent
  for epoch in range(n_epochs):
    # divide into batches
    perm = torch.randperm(x.shape[0])
    shuffle_x = x[perm]
    shuffle_y = y[perm]
    batches_x = torch.split(shuffle_x, n_batches)
    batches_y = torch.split(shuffle_y, n_batches)
    for (b_x, b_y) in zip(batches_x, batches_y):
        gradient = grad_func(b_x, b_y, theta - lr*beta*m)
        m = beta * m + (1- beta) * gradient
        theta = theta - lr * m
        
    # monitor losses
    y_preds = model(x, theta)
    thetas[:,epoch+1] = theta.T
    losses[epoch+1] = mse(y, y_preds)

  # Return output
  return thetas, losses

def ADAM(x,y, theta, lr, beta, gamma, bs, grad_func, model, n_epochs=1):
  thetas = torch.empty(2, n_epochs+1)
  thetas[:,0] = theta[:,0]

  # monitor losses
  n = x.size(0)
  n_batches = n // bs if n % bs == 0 else n // bs + 1
  losses = torch.empty((n_epochs+1, 1))

  # initial prediction and loss
  y_pred = model(x,theta)
  losses[0] = mse(y, y_pred)

  m = torch.zeros(theta.shape)
  v = torch.ones(theta.shape)

  # grad descent
  for epoch in range(n_epochs):
    # divide into batches
    perm = torch.randperm(x.shape[0])
    shuffle_x = x[perm]
    shuffle_y = y[perm]
    batches_x = torch.split(shuffle_x, n_batches)
    batches_y = torch.split(shuffle_y, n_batches)
    for (t, (b_x, b_y)) in enumerate(zip(batches_x, batches_y)):
        gradient = grad_func(b_x, b_y, theta)
        m = beta * m + (1- beta) * gradient
        v = gamma * v + (1-gamma) * gradient**2
        m_tilde = m / (1 - beta**(t+1))
        v_tilde = v / (1 - gamma**(t+1))
        theta = theta - lr * (m_tilde/torch.sqrt(v_tilde))
        
    # monitor losses        m_tilde = m / (1 - beta**(t+1))

    y_preds = model(x, theta)
    thetas[:,epoch+1] = theta.T
    losses[epoch+1] = mse(y, y_preds)

  # Return output
  return thetas, losses