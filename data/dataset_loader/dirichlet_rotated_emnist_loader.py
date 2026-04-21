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

    # Set ToTensor at the dataset level — the standard torchvision pattern.
    # This guarantees EMNIST.__getitem__ always returns a tensor, so PIL images
    # can never reach the collator regardless of any wrapper, random_split, or
    # missing transformation_function.
    data = datasets.EMNIST(root="./data/datasets", split="byclass", download=True,
                           transform=transforms.ToTensor())

    idcs = np.random.permutation(len(data))
    train_idcs, test_idcs = idcs[:200000], idcs[10000:20000]
    train_labels = data.targets.numpy()  # train_labels is deprecated, renamed to targets

    client_idcs = split_noniid(train_idcs, train_labels, alpha=DIRICHLET_ALPHA, n_clients=config["n_clients"])

    # Per-client rotation is applied on top of the already-tensor output.
    # RandomRotation works on both PIL Images and tensors (torchvision >= 0.8).
    client_data = []
    for i, c_idcs in enumerate(client_idcs):
        if i < config["n_clients"] // 2:
            tfm = transforms.RandomRotation((90, 90))
        else:
            tfm = None
        client_data.append(LocalDataset(data, c_idcs, transformation_function=tfm, task=config["task"]))

    # EMNIST already applies ToTensor via dataset.transform, so no extra transform needed.
    test_data = LocalDataset(data, test_idcs)

    # Return test_data so client.py never hits the random_split path, which returns
    # a plain Subset and strips transformation_function.
    info = [test_data for _ in range(config["n_clients"])]
    return client_data, info


if __name__ == "__main__":
    print("hello world!")