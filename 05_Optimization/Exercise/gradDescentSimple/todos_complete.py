"""3.2.1 Gradient descent - TODOs.

Complete the code here below.
"""

# import libraries - do not use numpy!
import torch


# TODO 1: complete the code below
def linreg_closed(data):
  """Closed form solution for parameters of linear model.
  y = theta_0 + theta_1 x
  
  Args:
    data: (n, 2) torch tensor; inputs - 1st col, outputs 2nd col
  """
  
  # inputs / ouputs
  x = data[:,0:1]
  y = data[:,1:2]

  x0 = torch.ones_like(x)
  x = torch.cat((x0,x), dim=1)
  
  th = torch.pinverse(x.T @ x) 
  th = th @ x.T @ y
  
  # parameters
  theta_0 = th[0,0]
  theta_1 = th[1,0]

  # Return output
  return theta_0, theta_1


# TODO 2: complete the code below
def mse(y, y_predict):
  """Calculate the mean square error between targets and predictions.
   
  Args:
    y, y_predict: targets and predictions
  """

  mse = torch.mean((y - y_predict)**2)

  return mse


# TODO 3: complete the code below
def linreg_gd(data, lr, n_epochs=1000):
  """Gradient descent solution for parameters of linear model.
  y = theta_0 + theta_1 x
  
  Args:
    data: (n, 2) torch tensor; inputs - 1st col, outputs 2nd col
    lr: (scalar) learning rate
    n_epochs: (scalar) number of epochs to run the GD for
  """
  
  # inputs / ouputs
  x = data[:,0:1]
  y = data[:,1:2]

  x0 = torch.ones_like(x)
  x = torch.cat((x0,x), dim=1)
  n = x.shape[0]

  # initialize theta
  th = torch.randn(2,1)
  theta_0 = th[0,0]
  theta_1 = th[1,0]
  
  # monitor losses
  losses = torch.empty((n_epochs, 1))
  
  # initial prediction an loss
  y_pred = theta_0 + theta_1*x
  losses[0] = mse(y, y_pred)
  
  # grad descent
  for epoch in list(range(n_epochs)):
    grad = 2/n*x.T@(x@th - y)
    th = th - lr*grad

    # monitor losses
    theta_0 = th[0,0]
    theta_1 = th[1,0]
    y_pred = theta_0 + theta_1*x
    losses[epoch] = mse(y, y_pred)
  
  # final parameters
  theta_0 = th[0,0]
  theta_1 = th[1,0]

  # Return output
  return theta_0, theta_1, losses


# TODO 3: complete the code below
def linreg_sgd(data, bs, lr, n_epochs=1000):
  """Stochastic gradient descent solution for parameters of linear model.
  y = theta_0 + theta_1 x
  
  Args:
    data: (n, 2) torch tensor; inputs - 1st col, outputs 2nd col
    bs: batch size
    lr: (scalar) learning rate
    n_epochs: (scalar) number of epochs to run the GD for
  """
  
  # inputs / ouputs
  x = data[:,0:1]
  y = data[:,1:2]

  x0 = torch.ones_like(x)
  x = torch.cat((x0,x), dim=1)

  # initialize theta
  th = torch.randn(2,1)
  theta_0 = th[0,0]
  theta_1 = th[1,0]
  
  # monitor losses
  n = x.size(0)
  n_batches = n // bs if n % bs == 0 else n // bs + 1
  losses = torch.empty((n_epochs*n_batches+1, 1))
  
  # initial prediction an loss
  y_pred = theta_0 + theta_1*x
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
      grad = 2/bs*bx.T@(bx@th - by)        
      th = th - lr*grad

      # monitor losses
      theta_0 = th[0,0]
      theta_1 = th[1,0]
      y_pred = theta_0 + theta_1*x
      losses[lidx] = mse(y, y_pred)
  
  # final parameters
  theta_0 = th[0,0]
  theta_1 = th[1,0]

  # Return output
  return theta_0, theta_1, losses

