import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Setting device configuration for GPU enabled devices

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setting all the hyperparameters for the CNN model
num_epochs = 20
batch_size = 10
learning_rate = 0.001

#Performing normalization to convert images in range[0,1] to normalized tensors in range[-1,1]

transform = transforms.Compose( [transforms.ToTensor(), 
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# Getting the CIFAR10 dataset from torchvision

train_dataset = torchvision.datasets.CIFAR10(root='./data', train = True, 
                                             download=True, transform=transform) 

test_dataset = torchvision.datasets.CIFAR10(root='./data', train = False, 
                                            download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Implementing the ConvNet Model

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3,32,3) # Input Channel, Output Channel, Kernel(Filter) Size
        self.conv2 = nn.Conv2d(32,32,3)
        self.pool1 = nn.MaxPool2d(2,2) # Kernel Size, Stride
        
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool2 = nn.MaxPool2d(2,2) # Kernel Size, Stride
        
        self.fc1 = nn.Linear(1600, 128) # 1600 as input to a ANN for flatenning the output from the conv layer
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(F.relu(self.conv2(x)))
        
        x = F.relu(self.conv3(x))
        x = self.pool2(F.relu(self.conv4(x)))

        x = x.reshape(x.size(0),-1) # Flatenning the layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    
model = ConvNet().to(device = device)

criterion = nn.CrossEntropyLoss() # Loss Function for multi-class classification
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) # SGD optimizer

total_steps = len(train_loader)

# Iterating over the dataset for Training our model

for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        # Original image shape: [10, 3, 32, 32 ](batch_size, colour_channel, width, height) = (4,3,1024)
        # Input Layer: 3 input channels, 6 output channels, 5 kernel size
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass 
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.5f}')
        
print("Finished Training")

# Evaluating our model

with torch.no_grad():
    n_correct = 0 
    n_samples = 0
    n_correct_class = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        
        # max returns (value,index)
        _, predicted = torch.max(outputs,1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_correct_class[label]+=1
            n_class_samples[label]+=1
            
    acc = 100.00 * n_correct/n_samples
    print(f'Accuracy of the network: {acc:.3f}%')
    
    # Calculating accuracy of individual classes
    for i in range(10):
        acc = 100.0 * n_correct_class[i]/n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc}%')
