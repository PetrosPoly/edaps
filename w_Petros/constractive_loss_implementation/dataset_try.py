import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import random

from collections import defaultdict

# Define the batch size
batch_size = 32

# Load the ImageNet dataset
dataset_directory = "data_dummy/IMagenet/tiny-imagenet-200/train"
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

imagenet_dataset = datasets.ImageFolder(root=dataset_directory, transform=transform)
dataloader = DataLoader(imagenet_dataset, batch_size=batch_size, shuffle=True)

# Create a dictionary to store images by class
"""
Defaultdict overrides one method to provide a default value for a 
nonexistent key, which helps to simplify the code when you deal with 
keys that might not exist in the dictionary.

In the context of defaultdict(list), list is the default factory function. 
It means that if you try to access a key that doesn't exist, defaultdict will 
use the list function to create a default value for that key, which is an empty list []
"""

images_by_class = defaultdict(list) 
# Iterate over the dataset and store images by class
for image, label in imagenet_dataset:
    images_by_class[label].append(image)
""" 
# Create a dictionary to organize the dataset by class labels
class_to_images = {}
for i in range(len(imagenet_dataset)):
    image, label = imagenet_dataset[i]
    if label not in class_to_images:
        class_to_images[label] = []
    class_to_images[label].append(image)
"""

# Create a list to store positive and negative samples
query_samples = []
positive_samples = []
negative_samples = []

for _ in range(batch_size):
    # Sample a random class label for the positive sample
    positive_label = random.choice(list(images_by_class.keys()))
    
    # Separate one image as the query and the rest as positives
    query_image, *positive_images_for_this_query = images_by_class[positive_label]
    query_samples.append(query_image)
    positive_samples.extend(positive_images_for_this_query)

    # Sample a different class label for the negative sample
    negative_label = random.choice(list(images_by_class.keys()))
    while negative_label == positive_label:
        negative_label = random.choice(list(images_by_class.keys()))
    
    # Add all images from the negative class
    negative_samples.extend(images_by_class[negative_label])

# Create DataLoader for positive and negative samples
positive_loader = DataLoader(positive_samples, batch_size=batch_size, shuffle=True)
negative_loader = DataLoader(negative_samples, batch_size=batch_size, shuffle=True)

def get_triplet(images_by_class):
    # Step 1: Randomly select a class for the query and positive sample
    pos_class = random.choice(list(images_by_class.keys()))
    
    # Ensure there's at least 2 images in the chosen class to get a distinct positive pair
    while len(images_by_class[pos_class]) < 2:
        pos_class = random.choice(list(images_by_class.keys()))

    # Step 2: Randomly select two images from this class
    query_img, positive_img = random.sample(images_by_class[pos_class], 2)
    
    # Step 3: Randomly select a different class for the negative sample
    neg_class = random.choice(list(images_by_class.keys()))
    while neg_class == pos_class:
        neg_class = random.choice(list(images_by_class.keys()))
    
    negative_img = random.choice(images_by_class[neg_class])
    
    return query_img, positive_img, negative_img

