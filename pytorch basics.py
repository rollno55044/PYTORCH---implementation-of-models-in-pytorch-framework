import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic autograd example 1               (Line 25 to 39)
# 2. Basic autograd example 2               (Line 46 to 83)
# 3. Loading data from numpy                (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189)

# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

# Create tensors.


 # x = torch.tensor(1 , requires_grad=True)
 # w = torch.tensor(2, requires_grad=True)
 # b = torch.tensor(3, requires_grad=True)

# Build a computational graph.
 # y = w * x + b
#
#
# compute gradient
#  y.backward()

# Print out the gradients.
#  print(x.grad)
#  print(w.grad)
#  print(b.grad)

# ================================================================== #
#                    2. Basic autograd example 2                     #
# ================================================================== #

# Create tensors of shape (10, 3) and (10, 2).
'''''
x = torch.randn(10,3)
y = torch.randn(10,2)

# Build a fully connected layer.
linear = nn.Linear(3,2)
print( 'w:',linear.weight)
print( 'b:' , linear.bias )


# Build loss function and optimizer.

criteration = nn.MSELoss()
optimizer  = torch.optim.SGD(linear.parameters() , lr=0.01)


# Forward pass.
pred = linear(x)

# Compute loss.

loss = criteration(pred,y)
print( 'loss:', loss.item)

# Backward pass.

loss.backward()

# Print out the gradients.
print ('dL/dw: ', linear.weight.grad)
print ('dL/db: ', linear.bias.grad)

# 1-step gradient descent.
optimizer.step()

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criteration(pred,y)
print('loss after 1 step optimization: ', loss.item())
'''''

# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #


# x = np.array([[1,5] , [2,4]])
#
# print(x)
#
# Convert the numpy array to a torch tensor.
# y = torch.from_numpy(x)
# print(y)
# z = y.numpy()


# ================================================================== #
#                         4. Input pipline                           #
# ================================================================== #


# Download and construct CIFAR-10 dataset.
train__dataset = torchvision.datasets.CIFAR10(root='./pytorch/dataset',train = True ,transform=transforms.ToTensor,download=True)

# Fetch one data pair (read data from disk).
images ,label = train__dataset[0]
print(images.size)
print(label)

# Data loader (this provides queues and threads in a very simple way).

train_loader = torch.utils.data.DataLoader(train__dataset , batch_size = 64 , shuffle  = true)


# When iteration starts, queue and thread star
# 
# t to load data from files.
data_iter = iter(train_loader)
images , label = data_iter.next()

# When iteration starts, queue and thread start to load data from files.

fj