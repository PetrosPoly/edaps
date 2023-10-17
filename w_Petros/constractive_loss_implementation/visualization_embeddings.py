# --------------------------------------------------------------------------------
# Petros Polydorou - 2023 ETH Zurich 
# Semester Project - Domain Adaptation
# Description - Demo Visualizations of the embeddings
# --------------------------------------------------------------------------------

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

"""
def visualize_embeddings(embeddings, labels, path_to_output):
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 10))
    for i, label in enumerate(set(labels)):
        plt.scatter(embeddings_2d[labels == label, 0], embeddings_2d[labels == label, 1], label=str(label))
    plt.legend()
    plt.savefig(path_to_output)
    plt.show()
    plt.close()
"""

def visualize_embeddings(embeddings, labels, path_to_output):
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    unique_labels = set(labels)
    for label in unique_labels:
        indices = [i for i, x in enumerate(labels) if x == label]
        ax.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=str(label))
     
    plt.legend()
    fig.savefig(path_to_output)  # Save the figure object directly
    plt.show()
    plt.close()