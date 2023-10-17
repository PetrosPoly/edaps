# --------------------------------------------------------------------------------
# Petros Polydorou - 2023 ETH Zurich 
# Semester Project - Domain Adaptation
# Description - Demo Contrastive Loss implementation
# --------------------------------------------------------------------------------

import torch
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import random 
from collections import defaultdict

from w_Petros.constractive_loss_implementation.dataset import ContrastiveDataset
from w_Petros.constractive_loss_implementation.network import CNNEncoder
from w_Petros.constractive_loss_implementation.visualization_embeddings import visualize_embeddings

from info_nce import InfoNCE

device = torch.device('cpu')
batch_size, num_negative, embedding_size = 32 , 48, 128

##  1. Load the ImageNet dataset  ## 

# Define the arguments 
dataset_directory = "data_dummy/IMagenet/tiny-imagenet-200/train"
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# Load the dataset 
imagenet_dataset = datasets.ImageFolder(root=dataset_directory, transform=transform)

# Create a dictionary to store images by class
images_by_class = defaultdict(list) 
for image, label in imagenet_dataset:
    images_by_class[label].append(image)

# Define the dataset & the dataloader
num_negative_samples = 6
dataset = ContrastiveDataset(images_by_class, query_equal_positive = True, num_negative_samples = 6)  # this object is a list
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

##  2. Define a simple CNN encoder  ## 
encoder = CNNEncoder(out_features = embedding_size)

##  3. Compute InfoNCE loss  ## 
loss_nce = InfoNCE(negative_mode='paired') # negative_mode='unpaired' is the default value

# Before your training loop
best_loss = float('inf')  # start with a very high loss
save_path = "w_Petros/constractive_loss_implementation/best_model.pth"

##  4. Train the model using the InfoNCE loss  ##
optimizer = optim.Adam(encoder.parameters(), lr=0.001)
num_epochs = 100

for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):

        # queries, positives, negatives_list = batch
        queries, positives, negatives = batch
        
        # Convert lists to tensors
        queries = queries.to(device)  # Move tensors to device (e.g., GPU)
        positives = positives.to(device)
        # negatives = negatives_list_tensor[:, 0].to(device)
        negatives = negatives.to(device)
    
        # Reshape negatives to fit the encoder's expected input shape (batch_size, img_size=(3, 64, 64)
        negatives_reshape = negatives.view(-1, 3, 64, 64)
        
        # Forward Pass
        anchor_embeddings = encoder(queries)
        positive_embeddings = encoder(positives)
        negative_embeddings_reshaped = encoder(negatives_reshape)
        negative_embeddings = negative_embeddings_reshaped.view(-1, num_negative_samples, embedding_size) 
        
        # Compute Loss
        loss = loss_nce(anchor_embeddings, positive_embeddings, negative_embeddings)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Assume `loss` is your computed loss for the current epoch
        if loss < best_loss:
          best_loss = loss
        torch.save(encoder.state_dict(), save_path)
        
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
print("Training complete!")

##  5. Extract the Embeddings and Visualize ## 

# Build the data for visualization 
random_labels = random.sample(images_by_class.keys(), 3)
samples = []
labels = []

for label in random_labels:
    imgs = random.sample(images_by_class[label], 30)
    samples.extend(imgs)
    labels.extend([label] * 10) # creates a new list containing ten copies of the value label

samples_tensor = torch.stack(samples)

# Extract from untrained model
encoder_untrained = CNNEncoder(out_features=embedding_size)
with torch.no_grad():
    embeddings_untrained = encoder_untrained(samples_tensor).cpu().numpy()

path_to_output = 'w_Petros/constractive_loss_implementation/visualizations/vis_out_untrained.png'
visualize_embeddings(embeddings_untrained, labels, path_to_output)

# Extract from trained model
best_model_path = save_path
encoder.load_state_dict(torch.load(best_model_path))
encoder.eval()  # Set the model to evaluation mode

with torch.no_grad():
    embeddings_trained = encoder(samples_tensor).cpu().numpy()

path_to_output = 'w_Petros/constractive_loss_implementation/visualizations/vis_out_trained.png'
visualize_embeddings(embeddings_trained, labels, path_to_output)



