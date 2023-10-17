# --------------------------------------------------------------------------------
# Petros Polydorou - 2023 ETH Zurich 
# Semester Project - Domain Adaptation
# Description - Demo Network implementation
# --------------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cpu')

# Dummy data: Assume you have grayscale images of size 28x28 and you have 3 classes

num_classes = 3
batch_size = 16
dummy_data = torch.randn(batch_size, 1, 28, 28)               # generates random numbers from a standard normal distribution (m=0, std=1). batch_size: number of samples, 1: number of channels (grey), 28x28 resolution
dummy_labels = torch.randint(0, num_classes, (batch_size,))   # generates random integers within a range (from 0 to num_classes). (batch_size,) specifies the shape of a tensor. 
                                                              # when you see (batch_size,), it means you are defining a tuple with a single element, where batch_size is that element

# Define a simple CNN model

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):                          # it's a constructor, is used to initialize instances of the class when you create objects. 
        super(SimpleCNN, self).__init__()                     # calles the constructor of the parent class (nn.Module). Super() function is used to call a method of a parent class
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1) 
        self.relu = nn.ReLU()                                 # out_channels=16 -> there are 16 filtes, padding 1 adds a pixel to the input to maintain spatial dimensions.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)     # Max-pooling is a downsampling operation that reduces the spatial dimensions of the feature maps
        self.fc1 = nn.Linear(16 * 14 * 14, num_classes) # Adjust input size based on your image dimensions
                                                              # input is: 16 channels, 14 = 28 / 2 due to pooling, output is: the number of classes
    
    def forward(self, x):                                   
        x = self.conv1(x)                                     # x = (16, 16, 14, 14) due to padding remains the same 
        x = self.relu(x)                                      # x = (16, 16, 28, 28)
        x = self.pool(x)                                      # x = (16, 16, 14, 14) = ((batch_size = number of images, channels, resolution(dim = 0), resolution(dim=1))
        x = x.view(x.size(0), -1)                             # x = (16, 3126) to reshape x before passing through fully connected. x before is 4D vector  
        x = self.fc1(x)                                       # x = (16, 3) reshape in a flatter version. So we keep the number of images per batch 16 and -1 is a placeholder for the remaining dims and 
        return x                                              # Pytorch will calculate so that the total number of elements remains the same. x will become 1D vector.
                                                              
        
# Instantiate the model

model = SimpleCNN(num_classes)

# Define loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop

num_epochs = 10

for epoch in range(num_epochs):                               # This loop will iterate over the entire dataset 10 times.
    running_loss = 0.0                                        # initialization of the variable.
    for i in range(0, len(dummy_data), int(batch_size/2)):    # size of each batch
        inputs = dummy_data[i:i+batch_size]
        labels = dummy_labels[i:i+batch_size]
        
        optimizer.zero_grad()                                 # zero the gradients of the model's parameters. necessary because PyTorch accumulates gradients by default, and you want to reset them for each batch
        
        outputs = model(inputs)                               # pass the batch of inputs through the model to obtain prediction
        loss = criterion(outputs, labels)                     # calculate the loss between the model's predictions (outputs) and the ground truth labels (labels) 
        loss.backward()                                       # perform backpropagation to calculate the gradients by computing the derivative of the loss with respect to each model parameter. gradients indicate how the loss would change with small adjustments to each parameter
        optimizer.step()                                      # After computing the gradients, this line updates the model's parameters using an optimization algorithm. It adjusts the model's weights and biases in the direction that reduces the loss. The specific optimization algorithm (SGD, Adam, etc.) determines how these adjustments are made.
        running_loss += loss.item()                           # accumulate the batch loss. Btw loss.item() extracts the loss value from the tensor loss. item() is a built method of tensors
                                                              # loss.backward() - Location: find it as a method of a tensor, so you call it like loss.backward() where loss is a PyTorch tensor.
                                                              # 
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

print("Training finished!")