# Custom Algorithm Commands

The new custom algorithm is exposed as `CFLGP_SOFT`.

It follows the design from your notes:
- warmup with local client training,
- functional-similarity clustering from shared Gaussian noise,
- temporal smoothing plus thresholded graph construction,
- Louvain clustering,
- soft assignment from cluster-average models back to each client.

Shared custom flags used below:
- `--custom_warmup_rounds`: warmup rounds before clustering starts
- `--custom_similarity_interval`: clustering refresh frequency
- `--custom_similarity_momentum`: smoothing factor `lambda`
- `--custom_noise_samples`: number of shared Gaussian samples
- `--custom_threshold_std_scale`: threshold scale in `mean + scale * std`
- `--custom_top_k_clusters`: number of cluster models blended per client

Install dependencies first if needed:

```bash
pip install torch torchvision torchaudio numpy scipy scikit-learn matplotlib tqdm networkx
```

## Rotated MNIST

```bash
python main.py --algorithm CFLGP_SOFT --dataset rotated_mnist_8_angles --model SimpleLinear --n_clients 128 --n_models 4 --batch_size 100 --learning_rate 0.1 --n_communication_rounds 200 --protocol model_averaging --local_update_epoch 1 --random_seed 0 --custom_warmup_rounds 10 --custom_similarity_interval 2 --custom_similarity_momentum 0.8 --custom_noise_samples 64 --custom_threshold_std_scale 0.5 --custom_top_k_clusters 2
```

## Label Permuted MNIST
```bash
python main.py --algorithm CFLGP_SOFT --dataset label_permuted_mnist --model SimpleLinear --n_clients 128 --n_models 4 --batch_size 100 --learning_rate 0.1 --n_communication_rounds 200 --protocol model_averaging --local_update_epoch 1 --random_seed 0 --custom_warmup_rounds 10 --custom_similarity_interval 2 --custom_similarity_momentum 0.8 --custom_noise_samples 64 --custom_threshold_std_scale 0.5 --custom_top_k_clusters 2
```

## CIFAR-10

```bash
python main.py --algorithm CFLGP_SOFT --dataset label_permuted_cifar10 --model Resnet18 --n_clients 80 --n_models 2 --batch_size 100 --learning_rate 0.01 --optimizer adam --n_communication_rounds 300 --protocol model_averaging --local_update_epoch 1 --random_seed 0 --custom_warmup_rounds 10 --custom_similarity_interval 2 --custom_similarity_momentum 0.8 --custom_noise_samples 64 --custom_threshold_std_scale 0.5 --custom_top_k_clusters 2
```

## EMNIST

```bash
python main.py --algorithm CFLGP_SOFT --dataset dirichlet_rotated_emnist --model MADMOConvNet --n_clients 80 --n_models 2 --batch_size 200 --learning_rate 0.1 --n_communication_rounds 100 --protocol model_averaging --local_update_epoch 1 --random_seed 0 --custom_warmup_rounds 10 --custom_similarity_interval 2 --custom_similarity_momentum 0.8 --custom_noise_samples 64 --custom_threshold_std_scale 0.5 --custom_top_k_clusters 2
```

## GPU / CPU

- GPU example: add `--gpu 0`
- CPU example: add `--gpu -1`
