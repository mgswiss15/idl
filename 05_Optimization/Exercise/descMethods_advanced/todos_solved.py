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
  x = torch.cat((torch.ones_like(x),x),dim=1)
  return x@theta

def mse(y, y_predict):
  """Mean squared error loss.
   
  Args:
    y, y_predict: (n, 1) torch.tensors with targets and predictions

  Output:
    mse: (scalar)
  """
  return torch.mean((y - y_predict)**2)


# TODO 2: complete the code below
def lin_grads(x, y, theta):
  """Gradient for parameters of linear model.
  
  Args:
    x: (n, 1) torch.tensor with inputs
    y: (2, 1) torch.tensor with outputs
    theta: (2, 1) torch tensor with initial parameters (intercept, slope)
  """
  x0 = torch.cat((torch.ones_like(x),x),dim=1)
  grad = 2*x0.T@(x0@theta - y)
  return grad


def grad_descent(x,y, theta, lr, grad_func, model, n_epochs=1):
  """Gradient descent for theta parameters.
  
  Args:
    x: (n, 1) torch.tensor with inputs
    y: (2, 1) torch.tensor with outputs
    theta: (2, 1) torch tensor with initial parameters (intercept, slope)
    lr: (scalar) learning rate
    grad_func: function to calculate gradients
    model: model for predictions
    n_epochs: (scalar) number of epochs to run the GD for
  """
  
  # monitor thetas
  thetas = torch.empty(2, n_epochs+1)
  thetas[:,0] = theta[:,0]

  # monitor losses
  losses = torch.empty((n_epochs+1, 1))
  
  # initial prediction and loss
  y_pred = model(x,theta)
  losses[0] = mse(y, y_pred)
  
  # grad descent
  for epoch in list(range(n_epochs)):

    # update theta
    grad = grad_func(x,y,theta)
    theta = theta - lr*grad

    # monitor thetas
    thetas[:,epoch+1] = theta[:,0]

    # monitor losses
    y_pred = model(x,theta)
    losses[epoch+1] = mse(y, y_pred)
  
  # Return output
  return thetas, losses


# TODO 3: complete the code below
def stochastic_grad_descent(x,y, theta, lr, bs, grad_func, model, n_epochs=1):
  """Stochastic gradient descent for theta parameters.
  
  Args:
    x: (n, 1) torch.tensor with inputs
    y: (2, 1) torch.tensor with outputs
    theta: (2, 1) torch tensor with initial parameters (intercept, slope)
    lr: (scalar) learning rate
    bs: batch size
    grad_func: function to calculate gradients
    model: model for predictions
    n_epochs: (scalar) number of epochs to run the GD for
  """
  
  # monitor thetas
  n = x.size(0)
  n_batches = n // bs if n % bs == 0 else n // bs + 1
  thetas = torch.empty(2, n_epochs+1)
  thetas[:,0] = theta[:,0]

  # monitor losses
  losses = torch.empty((n_epochs*n_batches+1, 1))
  
  # initial prediction and loss
  y_pred = model(x,theta)
  losses[0] = mse(y, y_pred)

  # grad descent
  lidx = 0
  for epoch in list(range(n_epochs)):
    idx = torch.randperm(n)
    for i in range(0, n, bs):
      lidx += 1
      batch_end = i+bs
      batch_idx = idx[i:batch_end]
      bx = x[batch_idx,:]
      by = y[batch_idx,:]

      # update theta
      grad = grad_func(bx,by,theta)
      theta = theta - lr*grad

      # monitor losses
      y_pred = model(x,theta)
      losses[lidx] = mse(y, y_pred)
 
    # monitor thetas
    thetas[:,epoch+1] = theta[:,0]

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
  n = x.size(0)
  n_batches = n // bs if n % bs == 0 else n // bs + 1
  thetas = torch.empty(2, n_epochs+1)
  thetas[:,0] = theta[:,0]

  # monitor losses
  losses = torch.empty((n_epochs*n_batches+1, 1))
  
  # initial prediction and loss
  y_pred = model(x,theta)
  losses[0] = mse(y, y_pred)

  # grad descent
  lidx = 0
  momentum = torch.zeros_like(theta)
  for epoch in list(range(n_epochs)):
    idx = torch.randperm(n)
    for i in range(0, n, bs):
      lidx += 1
      batch_end = i+bs
      batch_idx = idx[i:batch_end]
      bx = x[batch_idx,:]
      by = y[batch_idx,:]

      # update theta
      grad = grad_func(bx,by,theta)
      momentum = beta * momentum + (1-beta)*grad
      theta = theta - lr*momentum

      # monitor losses
      y_pred = model(x,theta)
      losses[lidx] = mse(y, y_pred)

    # monitor thetas
    thetas[:,epoch+1] = theta[:,0]

  # Return output
  return thetas, losses


# TODO 5: complete the code below
def sgd_nestorov(x,y, theta, lr, beta, bs, grad_func, model, n_epochs=1):
  """SGD with accelerated momentum for theta parameters.
  
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
  n = x.size(0)
  n_batches = n // bs if n % bs == 0 else n // bs + 1
  thetas = torch.empty(2, n_epochs+1)
  thetas[:,0] = theta[:,0]

  # monitor losses
  losses = torch.empty((n_epochs*n_batches+1, 1))
  
  # initial prediction and loss
  y_pred = model(x,theta)
  losses[0] = mse(y, y_pred)

  # grad descent
  lidx = 0
  momentum = torch.zeros_like(theta)
  for epoch in list(range(n_epochs)):
    idx = torch.randperm(n)
    for i in range(0, n, bs):
      lidx += 1
      batch_end = i+bs
      batch_idx = idx[i:batch_end]
      bx = x[batch_idx,:]
      by = y[batch_idx,:]

      # update theta
      grad = grad_func(bx,by,theta - lr*beta*momentum)
      momentum = beta * momentum + (1-beta)*grad
      theta = theta - lr*momentum

      # monitor losses
      y_pred = model(x,theta)
      losses[lidx] = mse(y, y_pred)

    # monitor thetas
    thetas[:,epoch+1] = theta[:,0]

  # Return output
  return thetas, losses


# TODO 6: complete the code below
def ADAM(x,y, theta, lr, beta, gamma, bs, grad_func, model, n_epochs=1):
  """ADAM for theta parameters.
  
  Args:
    x: (n, 1) torch.tensor with inputs
    y: (2, 1) torch.tensor with outputs
    theta: (2, 1) torch tensor with initial parameters (intercept, slope)
    lr: (scalar) learning rate
    beta: momentum smoothing parameter
    gamma: momentum for variance
    bs: batch size
    grad_func: function to calculate gradients
    model: model for predictions
    n_epochs: (scalar) number of epochs to run the GD for
  """
  
  # monitor thetas
  n = x.size(0)
  n_batches = n // bs if n % bs == 0 else n // bs + 1
  thetas = torch.empty(2, n_epochs+1)
  thetas[:,0] = theta[:,0]

  # monitor losses
  losses = torch.empty((n_epochs*n_batches+1, 1))
  
  # initial prediction and loss
  y_pred = model(x,theta)
  losses[0] = mse(y, y_pred)

  # grad descent
  lidx = 0
  momentum = torch.zeros_like(theta)
  v = torch.ones_like(theta)
  eps = 1e-6
  for epoch in list(range(n_epochs)):
    idx = torch.randperm(n)
    for i in range(0, n, bs):
      lidx += 1
      batch_end = i+bs
      batch_idx = idx[i:batch_end]
      bx = x[batch_idx,:]
      by = y[batch_idx,:]

      # update theta
      grad = grad_func(bx,by,theta)
      momentum = beta * momentum + (1-beta)*grad
      momentum = momentum / (1-beta**lidx)
      v = gamma * v + (1-gamma)*grad**2
      v = v / (1-gamma**lidx)
      theta = theta - lr*(momentum/(v**0.5 +eps))

      # monitor losses
      y_pred = model(x,theta)
      losses[lidx] = mse(y, y_pred)

    # monitor thetas
    thetas[:,epoch+1] = theta[:,0]

  # Return output
  return thetas, losses