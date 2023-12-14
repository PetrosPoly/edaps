# --------------------------------------------------------------------------------
# Petros Polydorou - 2023 ETH Zurich 
# Semester Project - Domain Adaptation
# Description - Define simple class dataset
# --------------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset
import random

class ContrastiveDataset(Dataset):
    def __init__(self, images_by_class, query_equal_positive, num_negative_samples):
        self.data = []
        for _ in range(len(images_by_class)):  # Adjust this depending on how many samples you want
            query, positives, negatives = get_extended_triplet(images_by_class, query_equal_positive, num_negative_samples)
            self.data.append((query, positives, negatives))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_extended_triplet(images_per_class, query_equal_positive, num_negative_samples):
    if  query_equal_positive == True:
        
        # Select a class for the query and positive samples
        pos_class = random.choice(list(images_per_class.keys()))
       
        # Ensure there's at least 2 images in the chosen class
        while len(images_per_class[pos_class]) < 2:
            pos_class = random.choice(list(images_per_class.keys()))
        
        # Select one image as the query
        query_img = random.choice(images_per_class[pos_class])
        
        # Select one positive image different the query 
        positive_img = random.choice(images_per_class[pos_class])
        while not torch.equal(positive_img, query_img):
            positive_img = random.choice(images_per_class[pos_class])            
        
        # Select a different class for the negative samples
        neg_class = random.choice(list(images_per_class.keys()))
        while neg_class == pos_class:
            neg_class = random.choice(list(images_per_class.keys()))        
        
        # Ensure the negative class has enough samples
        while len(images_per_class[neg_class]) < num_negative_samples:
            neg_class = random.choice(list(images_per_class.keys()))
            while neg_class == pos_class:
                neg_class = random.choice(list(images_per_class.keys()))      
        
        # Select 6 images from the negative class
        negative_imgs = random.sample(images_per_class[neg_class], num_negative_samples)
        negative_imgs_tensor = torch.stack(negative_imgs, dim=0) # This will be of shape (6, 3, 64, 64)
        return query_img, positive_img, negative_imgs_tensor
        # return query_img, positive_img, negative_imgs
    else:
        # Select a class for the query and positive samples
        pos_class = random.choice(list(images_per_class.keys()))
   
        # Ensure there's at least 2 images in the chosen class
        while len(images_per_class[pos_class]) < 2:
            pos_class = random.choice(list(images_per_class.keys()))

        # Select one image as the query
        query_img = random.choice(images_per_class[pos_class])
        # All other images from that class become the positives
        positive_imgs = [img for img in images_per_class[pos_class] if not torch.equal(img, query_img)]
    
        # Select a different class for the negative samples
        neg_class = random.choice(list(images_per_class.keys()))
        while neg_class == pos_class:
            neg_class = random.choice(list(images_per_class.keys()))
        # All images from the negative class
        negative_imgs = images_per_class[neg_class]
        
        return query_img, positive_imgs, negative_imgs




