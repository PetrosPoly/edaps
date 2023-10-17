# --------------------------------------------------------------------------------
# Petros Polydorou - 2023 ETH Zurich 
# Semester Project - Domain Adaptation
# Description - Sample
# --------------------------------------------------------------------------------

import torch
import torch
import torch.nn.functional as F

# Constants
device = torch.device('cpu')
temperature  = 0.3 
negative_mode = 'paired'
reduction = 'mean'

#Loss arguments 
batch_size, num_negative, embedding_size = 32, 6, 128
query = torch.randn(batch_size, embedding_size)
positive_key = torch.randn(batch_size, embedding_size)
negative_keys = torch.randn(batch_size, num_negative, embedding_size)

# Transpose function
def transpose(x):
    return x.transpose(-2, -1)

if negative_keys is not None:
    # Explicit negative keys

    # Cosine between positive pairs
    positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

    if negative_mode == 'unpaired':
        # Cosine between all query-negative combinations
        negative_logits = query @ transpose(negative_keys)

    elif negative_mode == 'paired':
        query = query.unsqueeze(1) 
        negative_logits = query @ transpose(negative_keys)
        negative_logits = negative_logits.squeeze(1)

    # First index in last dimension are the positive samples
    logits = torch.cat([positive_logit, negative_logits], dim=1)
    labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
else:
    # Negative keys are implicitly off-diagonal positive keys.

    # Cosine between all combinations
    logits = query @ transpose(positive_key)

    # Positive keys are the entries on the diagonal
    labels = torch.arange(len(query), device=query.device)

softmax = F.log_softmax(logits / temperature, 1)

Loss = F.cross_entropy(logits / temperature, labels, reduction=reduction)

input = torch.randn(3, 5, requires_grad=True)
target = target = torch.randint(5, (3,), dtype=torch.int64)

print('labels', labels)
print('size', labels.dim())