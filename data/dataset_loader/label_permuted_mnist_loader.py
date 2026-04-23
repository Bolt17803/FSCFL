import torchvision.transforms as transforms
import numpy as np
import torchvision
import torch
from data.dataset_loader.utils import divide_array, LocalDataset
import os
import os.path
from torch.utils.data import Dataset
from global_config import ROOT_DIRECTORY, PROJECTS_DIRECTORY
from pathlib import Path


class MNISTDataset(Dataset):
    def __init__(self, dataset_path=None, type='train'):
        if dataset_path is None:
            dataset_path = os.path.join(ROOT_DIRECTORY, "data", "dataset", "MNIST")
            Path(dataset_path).mkdir(parents=True, exist_ok=True)

        dataset = torchvision.datasets.MNIST(root=dataset_path, train=type == 'train',
                                             download=True)
        self.dataset_length = len(dataset)
        # dataset.data is (N, 28, 28) uint8 — normalize to [0,1] and add channel dim -> (N, 1, 28, 28)
        self.imgs = torch.from_numpy((dataset.data.numpy() / 255.).astype(np.float32)).unsqueeze(1)
        self.labels = dataset.targets.clone().type(torch.LongTensor)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

def get_label_permuted_mnist_local_datasets_custom_clusters(config):
    mnist_dataset = MNISTDataset()
    dataset_length = mnist_dataset.dataset_length
    subarrays = []

    N_CLUSTERS = config['n_models']  # Use the n_models parameter directly
    N_CLIENT = config['n_clients']

    assert N_CLIENT % N_CLUSTERS == 0, \
        f"n_clients ({N_CLIENT}) must be divisible by n_models ({N_CLUSTERS})"

    labels = mnist_dataset.labels.detach().numpy()

    indices = np.arange(dataset_length)
    np.random.shuffle(indices)
    
    # Split indices into N_CLUSTERS parts
    cluster_indices_list = np.array_split(indices, N_CLUSTERS)
    entire_cluster_subarrays = cluster_indices_list

    # Apply different label permutations to each cluster
    new_label = mnist_dataset.labels.clone()
    # label_shifts = [0, 5]  # Default for 2 clusters, add more as needed
    
    # For 3 clusters: [0, 3, 6] or [0, 5, 2], etc.
    # For 4 clusters: [0, 3, 6, 2] or [0, 2, 5, 7], etc.
    
    # if N_CLUSTERS == 3:
    #     label_shifts = [0, 3, 6]  # Example shifts for 3 clusters
    # elif N_CLUSTERS == 4:
    #     label_shifts = [0, 2, 5, 7]  # Example shifts for 4 clusters
    label_shifts = [(i * (10 // N_CLUSTERS)) % 10 for i in range(N_CLUSTERS)]

    for k in range(N_CLUSTERS):
        bt = torch.zeros(dataset_length, dtype=torch.bool)
        bt[cluster_indices_list[k]] = True
        new_label = torch.where(bt, (mnist_dataset.labels + label_shifts[k]) % 10, new_label)
    
    mnist_dataset.labels = new_label

    # Distribute clusters to clients
    for k in range(N_CLUSTERS):
        cluster_subarrays = np.array_split(entire_cluster_subarrays[k], N_CLIENT // N_CLUSTERS)
        for c in range(len(cluster_subarrays)):
            subarrays.append(cluster_subarrays[c])
    
    local_datasets = [LocalDataset(mnist_dataset, idx, task=config["task"]) for idx in subarrays]

    for i, local_dataset in enumerate(local_datasets):
        local_dataset.transformation_function = transforms.Compose(
            [transforms.Normalize((0.5,), (0.5,))])
    
    return local_datasets, [None for _ in range(config["n_clients"])]

def get_label_permuted_mnist_local_datasets(config):
    mnist_dataset = MNISTDataset()
    dataset_length = mnist_dataset.dataset_length
    subarrays = []

    N_CLUSTERS = 2  # config['n_models']
    N_CLIENT = config['n_clients']
    labels = mnist_dataset.labels.detach().numpy()

    indices = np.arange(dataset_length)
    np.random.shuffle(indices)
    cluster_1_indices, cluster_2_indices = np.split(indices, 2)
    entire_cluster_subarrays = [cluster_1_indices, cluster_2_indices]

    bt = torch.zeros(dataset_length, dtype=torch.bool)
    bt[cluster_1_indices] = True

    # Permute labels for cluster 1: shift by 5 mod 10 (same strategy as CIFAR10)
    new_label = torch.where(bt, (mnist_dataset.labels + 5) % 10, mnist_dataset.labels)
    mnist_dataset.labels = new_label

    for k in range(N_CLUSTERS):
        cluster_subarrays = np.array_split(entire_cluster_subarrays[k], N_CLIENT // N_CLUSTERS)
        for c in range(len(cluster_subarrays)):
            subarrays.append(cluster_subarrays[c])

    local_datasets = [LocalDataset(mnist_dataset, indices, task=config["task"]) for indices in subarrays]

    for i, local_dataset in enumerate(local_datasets):
        # MNIST is grayscale: single channel mean and std
        local_dataset.transformation_function = transforms.Compose(
            [transforms.Normalize((0.5,), (0.5,))])

    return local_datasets, [None for _ in range(config["n_clients"])]


if __name__ == "__main__":
    # random data
    dataset = MNISTDataset()