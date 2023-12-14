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



def nt_xent_loss(out_1, out_2, temperature):
    
    out = torch.cat([out_1, out_2], dim=0)                          # concatenates two tensors, out_1 and out_2, along dimension 0. This means the tensors are stacked on top of each other to create a single 
    n_samples = len(out)                                            # tensor out / dimension 0 corresponds to the rows, and dimension 1 corresponds to the columns.+
                                                                    # number of samples are 4

    # Full similarity matrix                                        # (contintiguous : we want after the transpose to have a contiguous memory layout)
    cov = torch.mm(out, out.t().contiguous())                       # computes the matrix product (dot product). This operation calculates the covariance matrix (cov) for the data in out
    sim = torch.exp(cov / temperature)                              # exponentiates the elements Is used to compute the similarity matrix sim. The temperature controls the spread or concentration of the similarity scores.

                                                                    # n_samples represents the number of rows (and columns) in the matrix. device ensures that matrix is created on the same device as the sim tensor. 
                                                                    # Bool() converts to a boolean matrix
    mask = ~torch.eye(n_samples, device=sim.device).bool()          # Creates identity matrix with the same number of rows and columns as sim.~ inverts the values in the matrix, so you get a matrix with zeros 
    neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)   # on the diagonal and ones elsewhere..bool() converts the resulting matrix to a boolean type.
                                                                    # extracts the similarity values from the similarity matrix sim based on the mask
                                                                    # view(n_samples, -1) : reshapes the extracted values into a tensor with shape / n_samples : rows and -1 : So, .view(n_samples, -1) is a convenient way
                                                                    # to reshape a tensor when you know the desired number of rows (n_samples) but want PyTorch to automatically determine the number of columns to maintain the total element count.
                                                                    # sum(dim=-1) computes the sum along the last dimension to calculate the total negative similarity for each sample.  
    # Positive similarity
    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature) # out_1 * out_2 perfors element_wise multiplication, torch.sum(..., dim=-1) calculates the sum along the last dimension. 
    pos = torch.cat([pos, pos], dim=0)                              # The shape here (2,1) and (2,1) so we are summing along the last dimension which the columns 1. 
                                                                    # The last dimension refers to the dimension with the highest index. It's the rightmost dimension when you represent a tensor shape.
    loss = -torch.log(pos / neg).mean()                             
    return loss

if __name__ == '__main__':
    device = torch.device('cpu')
    out_1 = torch.randn([2,1])
    out_2 = torch.randn([2,1])
    out_1 = torch.randint(1,10, size = (2,1))                       # generates a tensor with integer values
    out_2 = torch.randint(1,10, size = (2,1))
    temperature = 0.3

    cal_the_loss = nt_xent_loss(out_1, out_2, temperature)

