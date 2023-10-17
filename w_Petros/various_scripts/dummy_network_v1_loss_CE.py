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
dummy_data = torch.randn(batch_size, 1, 28, 28)               
dummy_labels = torch.randint(0, num_classes, (batch_size,))   
                                                              
# Define a simple CNN model

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):                           
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 14 * 14, num_classes) # Adjust input size based on your image dimensions
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
        
# Instantiate the model

model = SimpleCNN(num_classes)

# Define loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop

num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(0, len(dummy_data), batch_size):
        inputs = dummy_data[i:i+batch_size]
        labels = dummy_labels[i:i+batch_size]
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")

print("Training finished!")