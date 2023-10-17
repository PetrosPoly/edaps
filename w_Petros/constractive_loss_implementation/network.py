# --------------------------------------------------------------------------------
# Petros Polydorou - 2023 ETH Zurich 
# Semester Project - Domain Adaptation
# Description - Define simple class neural network - encoder
# --------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a simple CNN encoder

class CNNEncoder(nn.Module):
   
    def __init__(self, out_features=32):
        super(CNNEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)   # (in_channels=3, out_channels=32) output (32,32,32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)  # output (64, 16 ,16)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2) # output (128, 8 ,8)
        self.fc = nn.Linear(128 * 8 * 8, out_features)                      # output (128)
                                                    
    def forward(self, x):                          # using batch_size = 32  # x.size(32, 3, 64, 64)
        x = F.relu(self.conv1(x))                                           # x.size(32, 32, 32, 32)
        x = F.relu(self.conv2(x))                                           # x.size(32, 64, 16, 16)
        x = F.relu(self.conv3(x))                                           # x.size(32, 128, 8, 8)
        x = x.view(x.size(0), -1)                                           # x.size(0) = 32
        x = self.fc(x)                                                      # x.size(32, 128)     
        
        return F.normalize(x, dim=1) # Normalize the output                 # dim =1 per column