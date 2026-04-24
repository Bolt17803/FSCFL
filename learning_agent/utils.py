import numpy as np
from itertools import product, permutations
import time
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_samples, silhouette_score
import random
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import AgglomerativeClustering
from learning_agent.system_functions import flatten_tensor

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def spectral_clustering_and_matching(gradient_profile_matrix, n_centers, n_clients, estimated_cluster_ids_old=None,
                                     clustering_algorithm='KMeans'):
    start_time = time.time()
    P, singular_values, Q = np.linalg.svd(gradient_profile_matrix, full_matrices=False)
    reduced_G = np.matmul(np.transpose(P[:, :n_centers]), gradient_profile_matrix)
    SVD_time = time.time() - start_time
    # print("SVD takes ", time.time()-start_time, " sec.")

    start_time = time.time()
    estimated_part_ids = None
    if clustering_algorithm == "KMeans":
        kmeans = KMeans(n_clusters=n_centers, init="k-means++", random_state=42).fit(np.transpose(reduced_G))
        cluster_centers = kmeans.cluster_centers_
        estimated_part_ids = kmeans.labels_
    elif clustering_algorithm == "DBSCAN":
        dbscan = DBSCAN(eps=0.7, min_samples=2, leaf_size=10).fit(np.transpose(reduced_G))
        estimated_part_ids = dbscan.labels_
    elif clustering_algorithm == "AgglomerativeClustering":
        ac = AgglomerativeClustering(n_clusters=n_centers).fit(np.transpose(reduced_G))
        estimated_part_ids = ac.labels_

    K_means_clustering_time = time.time() - start_time

    start_time = time.time()
    if estimated_cluster_ids_old is not None:
        best_estimated_part_ids = estimated_part_ids
        best_n_consistency = 0
        for model_order in list(permutations([m_idx for m_idx in range(n_centers)])):
            temp_estimated_part_ids = np.array(model_order)[estimated_part_ids]
            n_consistency = np.sum((estimated_cluster_ids_old - temp_estimated_part_ids) == 0)
            if best_n_consistency < n_consistency:
                best_estimated_part_ids = temp_estimated_part_ids
                best_n_consistency = n_consistency
    else:
        best_estimated_part_ids = estimated_part_ids
    index_matching_time = time.time() - start_time

    info = {
        "estimated_cluster_ids": best_estimated_part_ids,
        "reduced_gradient_profile_matrix": reduced_G,
        "singular_values": singular_values,
        "SVD_time": SVD_time,
        "K_means_clustering_time": K_means_clustering_time,
        "index_matching_time": index_matching_time
    }
    return info


def get_num_cluster(gradient_profile_matrix, n_centers, n_clients, estimated_cluster_ids_old=None):
    start_time = time.time()

    P, singular_values, Q = np.linalg.svd(gradient_profile_matrix, full_matrices=False)
    print("singular_values:", singular_values)
    singular_gaps = singular_values[:-1] - singular_values[1:]
    print("singular_gaps:", singular_gaps)
    num_of_leading_sv = 1 + np.argmax(singular_gaps)
    clipped_num_of_leading_sv = np.clip(num_of_leading_sv, a_min=3, a_max=n_clients)
    print("num of leading singular values (min clipping 3) / clipped: ", num_of_leading_sv, " / ",
          clipped_num_of_leading_sv)

    reduced_G = np.matmul(np.transpose(P[:, :clipped_num_of_leading_sv]), gradient_profile_matrix)
    silhouette_avg_list = []
    for k in np.arange(2, n_centers + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)  # .fit(np.transpose(reduced_G))
        cluster_labels = kmeans.fit_predict(np.transpose(reduced_G))
        silhouette_avg = silhouette_score(np.transpose(reduced_G), cluster_labels)
        silhouette_avg_list.append(silhouette_avg)

    print("silhouette_avg_list:", silhouette_avg_list)
    proposed_k = np.argmax(silhouette_avg_list) + 2
    return proposed_k


def get_sparse_random_projection_matrix(n_features, compressed_n_features):
    positive_matrix = np.zeros(shape=(compressed_n_features, n_features), dtype=np.bool)
    negative_matrix = np.zeros(shape=(compressed_n_features, n_features), dtype=np.bool)
    zero_matrix = np.zeros(shape=(compressed_n_features, n_features), dtype=np.bool)

    s = 3

    for i in range(compressed_n_features):
        #
        r = np.random.choice([1, 0, -1], size=n_features, p=[1 / (2 * s), 1 - 1 / s, 1 / (2 * s)])
        for j in range(n_features):
            if r[j] == 1:
                positive_matrix[i, j] = True
            elif r[j] == -1:
                negative_matrix[i, j] = True
            elif r[j] == 0:
                zero_matrix[i, j] = True
            else:
                raise AssertionError

    return positive_matrix, zero_matrix, negative_matrix


def get_sparse_projected_vector(positive_matrix, zero_matrix, negative_matrix, source):
    n_compressed_features, n_features = positive_matrix.shape
    target = np.zeros(shape=n_compressed_features)
    for i in range(n_compressed_features):
        target[i] = np.sum(source[positive_matrix[i]]) - np.sum(source[negative_matrix[i]])

    return target


# https://github.com/felisat/clustered-federated-learning
def pairwise_angles(sources):
    angles = torch.zeros([len(sources), len(sources)])
    for i, source1 in enumerate(sources):
        for j, source2 in enumerate(sources):
            s1 = flatten_tensor(source1)
            s2 = flatten_tensor(source2)
            angles[i, j] = torch.sum(s1 * s2) / (torch.norm(s1) * torch.norm(s2) + 1e-12)

    return angles.numpy()


def compute_pairwise_similarities(clients):
    # dW = model_size_storage on model averaging scheme
    return pairwise_angles([client.model_size_storage for client in clients])


def cluster_clients(S):
    clustering = AgglomerativeClustering(affinity="precomputed", linkage="complete").fit(-S)
    c1 = np.argwhere(clustering.labels_ == 0).flatten()
    c2 = np.argwhere(clustering.labels_ == 1).flatten()
    return c1, c2


def compute_max_update_norm(cluster):
    return np.max([torch.norm(flatten_tensor(client.model_size_storage)).item() for client in cluster])


def compute_mean_update_norm(cluster):
    return torch.norm(torch.mean(torch.stack([flatten_tensor(client.model_size_storage) for client in cluster]),
                                 dim=0)).item()


def cosine_similarity_matrix(embeddings):
    embeddings = np.array(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    normalized_embeddings = embeddings / norms
    similarities = normalized_embeddings @ normalized_embeddings.T
    similarities = np.clip(similarities, -1.0, 1.0)
    return similarities


def smooth_similarity_matrix(current_similarity_matrix, previous_similarity_matrix, momentum):
    if previous_similarity_matrix is None:
        return current_similarity_matrix
    return momentum * current_similarity_matrix + (1.0 - momentum) * previous_similarity_matrix


def build_similarity_graph(similarity_matrix, threshold_std_scale=0.5):
    import networkx as nx

    n_clients = similarity_matrix.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_clients))

    upper_triangle = similarity_matrix[np.triu_indices(n_clients, k=1)]
    if upper_triangle.size == 0:
        threshold = 1.0
    else:
        threshold = float(np.mean(upper_triangle) + threshold_std_scale * np.std(upper_triangle))

    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            weight = float(similarity_matrix[i, j])
            if weight > threshold:
                graph.add_edge(i, j, weight=weight)

    return graph, threshold


def louvain_cluster_clients(graph, n_clients, random_seed=0):
    if graph.number_of_edges() == 0:
        return np.arange(n_clients, dtype=int)

    try:
        import networkx as nx
        communities = nx.community.louvain_communities(graph, weight="weight", seed=random_seed)
    except AttributeError:
        print("AttributeError: louvain_communities not found, using greedy_modularity_communities")
        import networkx as nx
        communities = nx.community.greedy_modularity_communities(graph, weight="weight")

    estimated_cluster_ids = np.zeros(n_clients, dtype=int)
    for cluster_idx, community in enumerate(communities):
        for client_idx in community:
            estimated_cluster_ids[client_idx] = cluster_idx
    return estimated_cluster_ids


def visualize_clustering(similarity_matrix, cluster_ids, round_num, save_dir):
    try:
        import matplotlib.pyplot as plt
        import os
        try:
            import seaborn as sns
            has_sns = True
        except ImportError:
            has_sns = False

        n_clients = similarity_matrix.shape[0]
        
        # Sort clients by their cluster ID
        sorted_indices = np.argsort(cluster_ids)
        sorted_cluster_ids = cluster_ids[sorted_indices]
        
        # Reorder the similarity matrix
        reordered_matrix = similarity_matrix[sorted_indices][:, sorted_indices]
        
        plt.figure(figsize=(10, 8))
        
        if has_sns:
            ax = sns.heatmap(reordered_matrix, cmap='viridis', vmin=-1.0, vmax=1.0)
        else:
            plt.imshow(reordered_matrix, cmap='viridis', vmin=-1.0, vmax=1.0, aspect='auto')
            plt.colorbar()
            ax = plt.gca()
            
        plt.title(f"Similarity Matrix - Round {round_num}")
        plt.xlabel("Clients (sorted by cluster)")
        plt.ylabel("Clients (sorted by cluster)")
        
        # Add lines to delineate clusters
        unique_clusters, counts = np.unique(sorted_cluster_ids, return_counts=True)
        current_pos = 0
        for count in counts:
            current_pos += count
            if current_pos < n_clients:
                if has_sns:
                    ax.axhline(current_pos, color='red', linestyle='--', linewidth=1)
                    ax.axvline(current_pos, color='red', linestyle='--', linewidth=1)
                else:
                    plt.axhline(current_pos - 0.5, color='red', linestyle='--', linewidth=1)
                    plt.axvline(current_pos - 0.5, color='red', linestyle='--', linewidth=1)
                    
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"similarity_heatmap_round_{round_num}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
    except ImportError:
        print("matplotlib is required for visualize_clustering. Skipping visualization.")


if __name__ == "__main__":
    # random data
    gradient_profile_matrix = np.random.random_sample(size=(128, 8))
    n_centers = 4
    n_clients = 8
    estimated_cluster_ids_old = None

    results = spectral_clustering_and_matching(gradient_profile_matrix,
                                               n_centers,
                                               n_clients,
                                               estimated_cluster_ids_old,
                                               clustering_algorithm="DBSCAN")
