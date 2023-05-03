# Converting everything to tensors and using AutoGrad 
# from PyTorch to implement the gradient function, using torch.nn to implement
# loss and torch.optim() for optimizer

# Implementing Linear Regression using manual functions

# Function we will use to build and test our model 
# F(x) = 2 * X
# F'(x) = w * x. We need to find the value of w(weight)

import torch
import torch.nn as nn

X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32) # Training examples
Y = 2 * X # Output Function that is to be simulated
print(X)
print(Y)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

# Model Prediction
print(f'Prediction before training for x = 5. It gives {model(X_test).item():.3f}')

# Training

learning_rate = 0.01
epochs = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(epochs):
    # prediction = forward pass
    y_pred = model(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradient 
    l.backward() # dl/dw
    
    # update weights
    optimizer.step()
    
    # zero gradient
    optimizer.zero_grad()
    
    if epoch%10 == 0:
        w,b = model.parameters()
        print(f'Epoch {epoch+1}: \n w = {w[0][0].item():.3f}, loss = {l:.8f}')
        
print(f'Prediction after training for x =5 .It gives {model(X_test).item():.3f}')