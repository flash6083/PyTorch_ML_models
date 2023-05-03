# Converting everything to tensors and using AutoGrad 
# from PyTorch to implement the gradient function

# Implementing Linear Regression using manual functions

# Function we will use to build and test our model 
# F(x) = 2 * X
# F'(x) = w * x. We need to find the value of w(weight)

import torch

X = torch.tensor([1,2,3,4,5,6,7,8,9,10], dtype=torch.float32) # Training examples
Y = 2 * X # Output Function that is to be simulated
print(X)
print(Y)

# Initialising our weight variable w
w = torch.tensor(0.0, dtype=torch.float32 , requires_grad=True) 

# Model Prediction

# Simulating the forward pass in a Neural Network based model
def forward(x):
    return w*x

# Loss function = MSE (Mean Squared Error)
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# Gradient function
# Formula of MSE = 1/N * (w*x - y)**2
# Gradient dj/dw = 1/N 2x (w*x - y)

print(f'Prediction before training. Eg input = f(5). It gives {forward(15):.3f}')

# Training

learning_rate = 0.01
epochs = 10

for epoch in range(epochs):
    # prediction = forward pass
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradient 
    l.backward() # dl/dw
    
    # update weights
    with torch.no_grad():
        w-= learning_rate * w.grad
        
    w.grad.zero_()
    
    if epoch%2 == 0:
        print(f'Epoch {epoch+1}: \n w = {w:.3f}, loss = {l:.8f}')
        
print(f'Prediction after training. Eg input = f(5). It gives {forward(5):.3f}')