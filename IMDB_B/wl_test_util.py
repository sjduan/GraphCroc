import torch

def initialize_labels(adj_matrix):
    return torch.sum(adj_matrix, dim=1).tolist()

def relabel(adj_matrix, labels):
    new_labels = []
    for i in range(adj_matrix.size(0)):

        neighbors = torch.nonzero(adj_matrix[i]).squeeze().tolist()
        if not isinstance(neighbors, list):
            neighbors = [neighbors]
        neighbor_labels = sorted([labels[j] for j in neighbors])
        new_labels.append((labels[i], tuple(neighbor_labels)))
    return new_labels

def compress_labels(labels):
    label_map = {}
    compressed_labels = []
    for label in labels:
        if label not in label_map:
            label_map[label] = len(label_map) + 1
        compressed_labels.append(label_map[label])
    return compressed_labels

def wl_algorithm(adj_matrix1, adj_matrix2, iterations):
    labels1 = initialize_labels(adj_matrix1)
    labels2 = initialize_labels(adj_matrix2)

    for _ in range(iterations):
        new_labels1 = relabel(adj_matrix1, labels1)
        new_labels2 = relabel(adj_matrix2, labels2)

        compressed_labels1 = compress_labels(new_labels1)
        compressed_labels2 = compress_labels(new_labels2)

        if sorted(compressed_labels1) != sorted(compressed_labels2):
            return False

        labels1 = compressed_labels1
        labels2 = compressed_labels2

    return True