# --------------------------------------------------------------------------------
# Petros Polydorou - 2023 ETH Zurich 
# Semester Project - Domain Adaptation
# Description - Demo Contrastive Loss implementation
# --------------------------------------------------------------------------------

import numpy as np
from matplotlib import pyplot as plt
import torch
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_metric_learning import losses

device = torch.device('cpu')

# 1. Create Dummy Images

batch_size, embedding_size = 32, 128
query = torch.randn(batch_size, embedding_size)
image_size = (3, 64, 64) # Example size for colored images of 64x64
dummy_images = torch.randn(batch_size, *image_size) # Random data to act as our images
                                                    # placing the asterisk (*) before image_size, you're telling Python to unpack the elements of the image_size tuple and pass them as separate arguments to 

# 2. Define a simple CNN encoder

class CNNEncoder(nn.Module):
   
    def __init__(self, out_features=embedding_size):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)    # (in_channels=3, out_channels=32, kernels_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)   # 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.fc = nn.Linear(128 * 8 * 8, out_features)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return F.normalize(x, dim=1) # Normalize the output
        
encoder = CNNEncoder()

# 3. Compute InfoNCE loss

def infonce_loss(anchors, positives, temperature=0.1):
    batch_size = anchors.size(0)
    negatives = positives[torch.randperm(batch_size)]
    positive_similarity = (anchors * positives).sum(dim=1) # After the element-wise multiplication, you sum the elements along dimension 1 of the resulting tensor. sum the products of each row
    negative_similarity = (anchors * negatives).sum(dim=1)
    
    loss = - F.log_softmax(positive_similarity / temperature - negative_similarity / temperature, dim=0).mean()
    
    return loss


# 4. Train the model using the InfoNCE loss

optimizer = optim.Adam(encoder.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):

    optimizer.zero_grad()

    # Forward pass
    z = encoder(dummy_images)

    # Positive samples (just a permutation for simplicity)
    z_positives = z[torch.randperm(batch_size)] # a random permutation is a random order of elements, and it's a fundamental concept in probability, statistics, and computer science.
                                                # it's a random rearrangemnt of elements 
    loss = infonce_loss(z, z_positives)

    # Backward pass
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
