import numpy as np

# Implementing Linear Regression using manual functions

# Function we will use to build and test our model 
# F(x) = 2 * x

# F'(x) = w * x. We need to find the value of w(weight)

X = np.array([1,2,3,4], dtype=np.float32) # Training examples
Y = 2*X # Output Function that is to be simulated

# Initialising our weight variable w
w = 0.0 

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

def gradient(x,y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f'Prediction before training. Eg input = f(5). It gives {forward(5):.3f}')

# Training

learning_rate = 0.01
epochs = 20

for epoch in range(epochs):
    # prediction = forward pass
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradient = backward pass
    dw = gradient(X,Y,y_pred)
    
    # update weights
    w-= learning_rate * dw
    
    if epoch%2 == 0:
        print(f'Epoch {epoch+1}: \n w = {w:.3f}, loss = {l:.8f}')
        
print(f'Prediction after training. Eg input = f(5). It gives {forward(5):.3f}')