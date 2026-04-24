

# packages to install
!pip install -q torch torchvision torchaudio tensorboard
!pip install -q numpy scipy scikit-learn matplotlib seaborn tqdm networkx

# Experiment execution code for the dataset setting - CIFAR-10 and ResNet-18

### 1. CFLGP 
Command : python main.py --algorithm CFLGP --dataset label_permuted_cifar10 --model Resnet18 --n_clients 80 --n_models 2 --batch_size 100 --learning_rate 0.01 --optimizer adam --n_communication_rounds 300 --clustering_period 2 --protocol model_averaging --local_update_epoch 1 --clustering_termination_threshold 30 --use_last_layer_only True --random_seed 0

### 2. IFCA
Command : python main.py --algorithm IFCA --dataset label_permuted_cifar10 --model Resnet18 --n_clients 80 --n_models 2 --batch_size 100 --learning_rate 0.01 --optimizer adam --n_communication_rounds 300 --clustering_period 2 --protocol model_averaging --local_update_epoch 1 --random_seed 0

### 3. MADMO
Command : python main.py --algorithm MADMO --dataset label_permuted_cifar10 --model Resnet18 --n_clients 80 --batch_size 100 --learning_rate 0.01 --optimizer adam --n_communication_rounds 300 --protocol model_averaging --local_update_epoch 1 --rho 0.1 --random_seed 0

### 4 . Fedavg
Command : python main.py --algorithm FEDAVG --dataset label_permuted_cifar10 --model Resnet18 --n_clients 80 --n_models 1 --batch_size 100 --learning_rate 0.01 --optimizer adam --n_communication_rounds 300 --protocol model_averaging --local_update_epoch 1 --random_seed 0


# Experiment execution code for the dataset setting - label permuted MNIST

### 1. CFLGP 
Command : python main.py --algorithm CFLGP --dataset label_permuted_mnist --model SimpleLinear --n_clients 128 --n_models 2 --batch_size 100 --learning_rate 0.1 --n_communication_rounds 200 --clustering_period 2 --protocol model_averaging --clustering_termination_threshold 20 --random_seed 0

### 2. IFCA
Command : python main.py --algorithm IFCA --dataset label_permuted_mnist --model SimpleLinear --n_clients 128 --n_models 2 --batch_size 100 --learning_rate 0.1 --n_communication_rounds 200 --clustering_period 2 --protocol model_averaging --random_seed 0

### 3. MADMO
Command : python main.py --algorithm MADMO --dataset label_permuted_mnist --model SimpleLinear --n_clients 128 --batch_size 100 --learning_rate 0.1 --n_communication_rounds 200 --protocol model_averaging --local_update_epoch 1 --rho 0.1 --random_seed 0

### 4. Fedavg
Command : python main.py --algorithm FEDAVG --dataset label_permuted_mnist --model SimpleLinear --n_clients 128 --n_models 1 --batch_size 100 --learning_rate 0.1 --n_communication_rounds 200 --protocol model_averaging --random_seed 0

# Experiment execution code for the dataset setting - Rotated MNIST

### 1. CFLGP 
Command : python main.py --algorithm CFLGP --dataset rotated_mnist_8_angles --model SimpleLinear --n_clients 128 --n_models 4 --batch_size 100 --learning_rate 0.1 --n_communication_rounds 200 --clustering_period 2 --protocol model_averaging --clustering_termination_threshold 20 --random_seed 0

### 2. IFCA
Command : python main.py --algorithm IFCA --dataset rotated_mnist_8_angles --model SimpleLinear --n_clients 128 --n_models 4 --batch_size 100 --learning_rate 0.1 --n_communication_rounds 200 --clustering_period 2 --protocol model_averaging --random_seed 0

### 3. MADMO
Command : python main.py --algorithm MADMO --dataset rotated_mnist_8_angles --model SimpleLinear --n_clients 128 --batch_size 100 --learning_rate 0.1 --n_communication_rounds 200 --protocol model_averaging --local_update_epoch 1 --rho 0.1 --random_seed 0

### 4. Fedavg
Command : python main.py --algorithm FEDAVG --dataset rotated_mnist_8_angles --model SimpleLinear --n_clients 128 --n_models 1 --batch_size 100 --learning_rate 0.1 --n_communication_rounds 200 --protocol model_averaging --random_seed 0



# Experiment execution code for the dataset setting - EMNIST

### 1. CFLGP 
Command : python main.py --algorithm CFLGP --dataset dirichlet_rotated_emnist --model MADMOConvNet --n_clients 80 --n_models 2 --batch_size 200 --learning_rate 0.1 --n_communication_rounds 100 --clustering_period 2 --protocol model_averaging --clustering_termination_threshold 10 --random_seed 0

### 2. IFCA
Command : python main.py --algorithm IFCA --dataset dirichlet_rotated_emnist --model MADMOConvNet --n_clients 80 --n_models 2 --batch_size 200 --learning_rate 0.1 --n_communication_rounds 100 --clustering_period 2 --protocol model_averaging --random_seed 0

### 3. MADMO
Command : python main.py --algorithm MADMO --dataset dirichlet_rotated_emnist --model MADMOConvNet --n_clients 80 --batch_size 200 --learning_rate 0.1 --n_communication_rounds 100 --protocol model_averaging --local_update_epoch 1 --rho 0.1 --random_seed 0

### 4. Fedavg
Command : python main.py --algorithm FEDAVG --dataset dirichlet_rotated_emnist --model MADMOConvNet --n_clients 80 --n_models 1 --batch_size 200 --learning_rate 0.1 --n_communication_rounds 100 --protocol model_averaging --random_seed 0

## IMPORTANT: to use cpu use parameter "--gpu -1", to use gpu use the index of the gpu core, for eg: "--gpu 0"