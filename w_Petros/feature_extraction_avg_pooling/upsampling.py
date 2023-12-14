import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from w_Petros.bbox_visualization_per_instance.functions import isValidBox, get_bbox_coord, divide_bounding_box_v1

# the input of the network
img = torch.rand(2,3,512,512)

# the output of the network
x_1 = torch.randn(2,64,128,128)
x_2 = torch.randn(2,128,64,64)
x_3 = torch.randn(2,320,32,32)
x_4 = torch.randn(2,512,16,16)

# create a binary masks
mask_1 = np.array((x_1 > 0.5).float())
mask_2 = (x_2 > 0.5).float().numpy()
mask_3 = (x_3 > 0.5).float().numpy()
mask_4 = (x_4 > 0.5).float().numpy()

# keep the 2nd image and 3rd channel
mask_1 = np.array((x_1 > -6e-01).float())[1,2,:,:]
mask_2 = (x_2 > -6e-01).float().numpy()[1,2,:,:]
mask_3 = (x_3 > -6e-01).float().numpy()[1,2,:,:]
mask_4 = (x_4 > -6e-01).float().numpy()[1,2,:,:]

# creates the bboxes  
bbox_1 = get_bbox_coord(mask_1)
bbox_2 = get_bbox_coord(mask_2)
bbox_3 = get_bbox_coord(mask_3)
bbox_4 = get_bbox_coord(mask_4)

list_of_valid_bboxes = [isValidBox(bbox_1), isValidBox(bbox_2),isValidBox(bbox_3),isValidBox(bbox_4)]

# modes of interpolation
modes_of_interpolations = ['nearest','bilinear','bicubic']

# upsamlping of the output
def upsample_the_image(x, mode, scale_factor = 0.25):
       scaled_x = F.interpolate(x, scale_factor = scale_factor, mode = mode)
       return scaled_x

# for mode in modes_of_interpolations:
x_1_upscaled = upsample_the_image(x_1, modes_of_interpolations[0], scale_factor = 4)
x_2_upscaled = upsample_the_image(x_2, modes_of_interpolations[0], scale_factor = 8)
x_3_upscaled = upsample_the_image(x_3, modes_of_interpolations[0], scale_factor = 16)
x_4_upscaled = upsample_the_image(x_4, modes_of_interpolations[0], scale_factor = 32)

# create a binary masks
mask_1_upscaled = (x_1_upscaled > 0.5).float().numpy()[1,2,:,:]
mask_2_upscaled = (x_2_upscaled > 0.5).float().numpy()[1,2,:,:]
mask_3_upscaled = (x_3_upscaled > 0.5).float().numpy()[1,2,:,:]
mask_4_upscaled = (x_4_upscaled > 0.5).float().numpy()[1,2,:,:]

# creates the bboxes
bbox_1_upscaled = get_bbox_coord(mask_1_upscaled)
bbox_2_upscaled = get_bbox_coord(mask_2_upscaled)
bbox_3_upscaled = get_bbox_coord(mask_3_upscaled)
bbox_4_upscaled = get_bbox_coord(mask_4_upscaled)

# create the subrois
roi1_1, roi1_2, roi1_3, roi1_4 = divide_bounding_box_v1(bbox_1_upscaled)
roi2_1, roi2_2, roi2_3, roi2_4 = divide_bounding_box_v1(bbox_2_upscaled)
roi3_1, roi3_2, roi3_3, roi3_4 = divide_bounding_box_v1(bbox_3_upscaled)
roi4_1, roi4_2, roi4_3, roi4_4 = divide_bounding_box_v1(bbox_4_upscaled)


       
