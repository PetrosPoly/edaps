import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from mmdet.core import BitmapMasks
from w_Petros.bbox_visualization_per_instance.functions import isValidBox, get_bbox_coord
from typing import Optional

# Create random images 
device = torch.device('cpu')
batch_size, embeddings = 2, 256
image_size = (3, 64, 64)
dummy_images = torch.randn(batch_size, *image_size)

# Encoder - Decoder 

class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        
        # Initial layers (smaller kernels and strides to maintain more spatial resolution)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Deeper layers
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # Initial layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Deeper layers
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        
        return x  # Output shape will be (batch_size, 256, 128, 128)

backbone = CNNEncoder()

# Extra functions that I need # 
def divide_box(box):
    x1, y1, x2, y2 = box
    width = x2 - x1 + 1
    height = y2 - y1 + 1
    # No overlapping of sub - boxes
    roi1 = (x1, y1, x1 + width // 2 - 1, y1 + height // 2 - 1)
    roi2 = (x1 + width // 2, y1, x2, y1 + height // 2 - 1)
    roi3 = (x1, y1 + height // 2 , x1 + width // 2 - 1, y2)
    roi4 = (x1 + width // 2, y1 + height // 2, x2, y2)
    return roi1, roi2, roi3, roi4

# Scatter Mean implementation # 
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
       # out.div_(count, rounding_mode='floor') # I am using Pytorch 1.7.1 which does not support this argument
        out = out // count
    return out

# The definition of the function # 
def call(panoptic, segments, thing_list, X_n):
    print('call function is being called')
    results = {}
    height, width = panoptic.shape[0], panoptic.shape[1]
    gt_masks = []
    gt_bboxes = []
    selected_labels = []   # Creating lists to store labels and features that start with 24000 for example
    selected_features = []
    for seg in segments:
        cat_id = seg["category_id"]
        mask = (panoptic == seg["id"])
        if not mask.sum() == 0:
            if cat_id in thing_list:
                box = get_bbox_coord(mask)
                # added by Petros - Start# 
                subregion_labels = np.full_like(panoptic, fill_value=-100, dtype=np.uint8)
                contrast_label = np.zeros_like(panoptic, dtype=np.uint8)
                subboxes = [None, None, None, None]
                if isValidBox(box):
                    subboxes[0], subboxes[1], subboxes[2], subboxes[3] = divide_box(box)
                    for i, subbox in enumerate(subboxes):
                        x1, y1, x2, y2 = subbox
                        subregion_labels[y1:y2+1, x1:x2+1] = i
                    contrast_label = panoptic * 10 + subregion_labels
                    contrast_label_flatten_tensor = torch.from_numpy(contrast_label.flatten())
                    unique_values, indicies = torch.unique(contrast_label_flatten_tensor,sorted = True, return_inverse=True)
                    mean_features = scatter_mean(X_n, indicies, dim=0)
                    print('unique values =', unique_values)
                    print('mean_features =', mean_features)
                    # Loop through each label and feature
                    for label, feature in zip(unique_values, mean_features):
                        # Check if string representation of label starts with '24000'
                        if str(label.item()).startswith(str(seg["id"])):
                            selected_labels.append(label.item())
                            selected_features.append(feature.tolist())  # Convert tensor to list
                    # added by Petros - Finish # 
                    gt_masks.append(mask.astype(np.uint8))
                    gt_bboxes.append(box)
    gt_masks = BitmapMasks(gt_masks, height, width)
    # Create a dictionary with two keys: "labels" and "features"
    results['subregion_labels_contrastive'] =  selected_labels
    #results['panoptic_labels_contrastive'] = selected_labels // 10
    results['features'] =  selected_features
    results['gt_masks'] = gt_masks
    results['gt_bboxes'] = np.asarray(gt_bboxes).astype(np.float32)
    # adding the fields
    results['bbox_fields'] = ['gt_bboxes_ignore', 'gt_bboxes']
    results['mask_fields'] = ['gt_masks']
    results['seg_fields'] = ['gt_semantic_seg']
    results['pan_fields'] = ['gt_panoptic_only_thing_classes']
    results['maxinst_fields'] = ['max_inst_per_class']
    return results, unique_values, indicies, contrast_label_flatten_tensor

if __name__ == '__main__':

    # Filenames
    data_root = 'data/synthia'
    ann_dir = 'panoptic-labels-crowdth-0-for-daformer/synthia_panoptic'
    img_dir = 'RGB'
    image_filename = '0000000_panoptic.png'
    input_filename = os.path.join(data_root, ann_dir, image_filename) 

    # Create the output directory & check if the directory exists, and raise an AssertionError if not
    thing_list = [11, 12, 13, 14, 15, 16, 17, 18]
    random_panoptic = np.array([[24001, 24001, 24001, 24001, 7, 8],
                [24001, 24001, 24000, 24000, 24000, 7],
                [33000, 33000, 24000, 24000, 24000, 7],
                [33000, 33000, 24000, 24000, 10, 24000],
                [33000, 33000, 24000, 24000, 10, 24000]])
    random_segments = [{'category_id': 0, 'id': 7}, 
                {'category_id': 1, 'id': 8},
                {'category_id': 2, 'id': 10},
                {'category_id': 11, 'id': 24000},
                {'category_id': 11, 'id': 24001},
                {'category_id': 15, 'id': 33000}]
   # take the results
    img = torch.randint(1, 10, (3,5,6))
    image_flat = img.view(3, -1)
    X_n = image_flat.t()

   # produce the results
    results, unique_values, indicies, contrast = call(random_panoptic, random_segments, thing_list, X_n)
    print('Sub labels = ', results['sub_labels'])
    print('features = ',  results['features'])