
import numpy as np
from scipy.sparse import csr_matrix
#from torch_scatter import scatter_mean

# Create a sparse matrix as an example
data = np.array([1, 2, 3, 4, 5])
row = np.array([0, 1, 2, 3, 4])
col = np.array([0, 1, 3, 4, 5])
sparse_mat = csr_matrix((data, (row, col)), shape=(6, 6))

print("Original Sparse Matrix:")
print(sparse_mat)

# Step 1: Flatten the sparse matrix
flattened_array = sparse_mat.toarray().ravel()

print("\nFlattened Array:")
print(flattened_array)

# Step 2: Resize it to the desired fixed size (e.g., 20 elements)
fixed_size = 20
resized_array = np.resize(flattened_array, fixed_size)

print("\nResized Array to Fixed Size:")
print(resized_array)


import torch
from w_Petros.constractive_implementation.Extact_features_average_pooling import scatter_mean
# from torch_scatter import scatter_mean            # could not be installed

# Example features
features_patch1 = torch.randn(20)
features_patch2 = torch.randn(15)
features_patch3 = torch.randn(15)

def resample_to_fixed_size(features, size=10):
    num_features = len(features)
    # Create bins
    bins = torch.linspace(0, num_features, steps=size+1, dtype=torch.long)
    # Create index tensor
    index = torch.cat([torch.full((bins[i+1]-bins[i],), i) for i in range(size)])
    # Use scatter mean to aggregate
    return scatter_mean(features, index)

result_patch1 = resample_to_fixed_size(features_patch1)
result_patch2 = resample_to_fixed_size(features_patch2)
result_patch3 = resample_to_fixed_size(features_patch3)

print(result_patch1)  # 10-element vector
print(result_patch2)  # 10-element vector
print(result_patch3)  # 10-element vector


import torch

example = torch.tensor([0, 0, 2, 3, 5, 4, 3, 6, 2, 4, 3, 2, 5 ,3, 2, 5, 2, 4, 1, 3, 0, 2, 4, 5,])
unique , indicies , counts= torch.unique(example, sorted = False, return_inverse = True, return_counts = True, dim = 0)
print('example =', example)
print('indicies =', indicies)
print('unique =', unique)
print('counts =', counts)
