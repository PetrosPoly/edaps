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

from tools.panoptic_deeplab.utils import rgb2id
from mmseg.utils.visualize_pred import prep_gt_pan_for_vis, subplotimgV2

###################################################
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Define a custom Siamese network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128*4*4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)  # Output 2-dimensional embeddings
        )

    def forward_one(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

# Create a dataset and data loaders (pairs of similar and dissimilar samples)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Initialize the Siamese network and NTXentLoss
model = SiameseNetwork()
criterion = nn.CrossEntropyLoss()  # NTXentLoss is often used with CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        input1, input2 = inputs[0], inputs[1]  # Sample two inputs from the batch
        output1, output2 = model(input1, input2)

        # Calculate the similarity (cosine similarity) between the embeddings
        similarity = torch.nn.functional.cosine_similarity(output1, output2)
        
        # Create labels for the loss (1 for similar, 0 for dissimilar)
        labels = torch.ones(labels.size(0)).long().to(similarity.device)

        # Compute the NT-Xent loss
        loss = criterion(similarity.unsqueeze(1), labels)

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# After training, you can use the model to obtain embeddings for new data.
# You can also visualize the embeddings if needed.
