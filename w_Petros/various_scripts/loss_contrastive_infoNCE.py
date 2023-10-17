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

from info_nce import InfoNCE

device = torch.device('cpu')

# 1. Create Dummy Images

batch_size, embedding_size = 32, 128
image_size = (3, 64, 64) # Example size for colored images of 64x64
dummy_images = torch.randn(batch_size, *image_size) # Random data to act as our images
                                                    # placing the asterisk (*) before image_size, you're telling Python to unpack the elements of the image_size tuple and pass them as separate arguments to 

# 2. Define a simple CNN encoder

class CNNEncoder(nn.Module):
   
    def __init__(self, out_features=embedding_size):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)  # (in_channels=3, out_channels=32) output (32,32,32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2) # output (64, 16 ,16)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)# output (128, 8 ,8)
        self.fc = nn.Linear(128 * 8 * 8, out_features)                     # output (128)
                                                    
    def forward(self, x):                          # using batch_size = 32 # x.size(32, 3, 64, 64)
        x = F.relu(self.conv1(x))                                          # x.size(32, 32, 32, 32)
        x = F.relu(self.conv2(x))                                          # x.size(32, 64, 16, 16)
        x = F.relu(self.conv3(x))                                          # x.size(32, 128, 8, 8)
        x = x.view(x.size(0), -1)                                          # x.size(0) = 32
        x = self.fc(x)                                                     # x.size(32, 128)     
        
        return F.normalize(x, dim=1) # Normalize the output                # dim =1 per column
        
encoder = CNNEncoder()

# 3. Compute InfoNCE loss

# Can be used with negative keys, whereby every combination between query and negative key is compared.
loss_nce= InfoNCE(negative_mode='unpaired') # negative_mode='unpaired' is the default value
batch_size, num_negative, embedding_size = 32, 48, 128

# 4. Train the model using the InfoNCE loss

optimizer = optim.Adam(encoder.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):

    optimizer.zero_grad()

    # Forward pass
    z = encoder(dummy_images)

    # Positive samples (just a permutation for simplicity)
    z_positives = z[torch.randperm(batch_size)]
    z_negatives = torch.randn(num_negative, embedding_size)
    
    # Loss function
    loss = loss_nce(z, z_positives, z_negatives)

    # Backward pass
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")



"""
# Can be used without explicit negative keys, whereby each sample is compared with the other samples in the batch.
loss = InfoNCE()
batch_size, embedding_size = 32, 128
query = torch.randn(batch_size, embedding_size)
positive_key = torch.randn(batch_size, embedding_size)
output = loss(query, positive_key)
"""

"""
# Can be used with negative keys, whereby every combination between query and negative key is compared.
loss_nce= InfoNCE(negative_mode='unpaired') # negative_mode='unpaired' is the default value
batch_size, num_negative, embedding_size = 32, 48, 128
query = torch.randn(batch_size, embedding_size)
positive_key = torch.randn(batch_size, embedding_size)
negative_keys = torch.randn(num_negative, embedding_size)
output = loss(query, positive_key, negative_keys)
"""

"""
# Can be used with negative keys, whereby each query sample is compared with only the negative keys it is paired with.
loss = InfoNCE(negative_mode='paired')
batch_size, num_negative, embedding_size = 32, 6, 128
query = torch.randn(batch_size, embedding_size)
positive_key = torch.randn(batch_size, embedding_size)
negative_keys = torch.randn(batch_size, num_negative, embedding_size)
output = loss(query, positive_key, negative_keys)
"""