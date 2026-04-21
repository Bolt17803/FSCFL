# References:
# Code from https://github.com/felisat/clustered-federated-learning/blob/master/clustered_federated_learning.ipynb
import numpy as np
from torchvision import datasets, transforms
from data.dataset_loader.utils import divide_array, LocalDataset


def split_noniid(train_idcs, train_labels, alpha, n_clients):
    '''
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    '''
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels[train_idcs] == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]
    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [train_idcs[np.concatenate(idcs)] for idcs in client_idcs]

    return client_idcs


def get_dirichlet_rotated_EMNIST_local_datasets(config):
    DIRICHLET_ALPHA = 1.0
    data = datasets.EMNIST(root="./data/datasets", split="byclass", download=True)

    idcs = np.random.permutation(len(data))
    train_idcs, test_idcs = idcs[:200000], idcs[10000:20000]
    train_labels = data.train_labels.numpy()

    client_idcs = split_noniid(train_idcs, train_labels, alpha=DIRICHLET_ALPHA, n_clients=config["n_clients"])

    # Build client datasets with transforms at construction time
    client_data = []
    for i, c_idcs in enumerate(client_idcs):
        if i < config["n_clients"] // 2:
            tfm = transforms.Compose([
                transforms.RandomRotation((90, 90)),
                transforms.ToTensor()
            ])
        else:
            tfm = transforms.Compose([transforms.ToTensor()])
        client_data.append(LocalDataset(data, c_idcs, transformation_function=tfm, task=config["task"]))

    # Fix: use keyword argument so transform isn't swallowed by `indices`
    test_data = LocalDataset(
        data,
        test_idcs,
        transformation_function=transforms.Compose([transforms.ToTensor()])
    )

    info = [None for _ in range(config["n_clients"])]
    return client_data, info


if __name__ == "__main__":
    print("hello world!")
