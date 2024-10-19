from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import time
from sklearn.decomposition import PCA


torch.set_num_threads(16)
torch.manual_seed(0)
device = 'cuda'

main_path = '/workspace/lsh/imagenet/'

local_dir = "/workspace/lsh/imagenet/data"

processor = AutoImageProcessor.from_pretrained("google/vit-large-patch32-384",use_fast=True)
model = AutoModelForImageClassification.from_pretrained("google/vit-large-patch32-384")


model.eval()

transform = transforms.Compose([
        transforms.Resize((384, 384)),  # Resize images to a consistent size
        transforms.ToTensor(),           # Convert PIL images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_npz_files_from_directory(directory):
    data = []
    # Loop through all files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.npz'):
            file_path = os.path.join(directory, file_name)
            npz_file = np.load(file_path)
            # Extract the data from each .npz file (you might need to adapt this depending on the structure inside the .npz)
            data.append({key: npz_file[key] for key in npz_file})
    return data

output_dir = main_path
# Define paths for each split
ds_dir = os.path.join(output_dir, "vit_features")

# train_dir = os.path.join(output_dir, "vit_features/train")
# test_dir = os.path.join(output_dir, "vit_features/test")
# valid_dir = os.path.join(output_dir, "vit_features/val")
# Load the data
ds = load_npz_files_from_directory(ds_dir)

train_data = ds[1]
test_data = ds[2]
val_data = ds[2]

# train_data = load_npz_files_from_directory(train_dir)
# test_data = load_npz_files_from_directory(test_dir)
# valid_data = load_npz_files_from_directory(valid_dir)

# Now you can access and combine the data as needed
print("Train data loaded:", len(train_data))
print("Test data loaded:", len(test_data))
print("Valid data loaded:", len(val_data))

# Example: accessing the first file's content in train data
print(train_data[0])

class CustomDataset():
    def __init__(self, data):
        self.features = data['features']
        self.labels = data['labels']
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return {'features': feature, 'label': label}

# Custom collate function
def custom_collate(batch):
    features = [item['features'] for item in batch]  # Extract features
    labels = [item['label'] for item in batch]  # Extract labels
    
    # Convert lists to tensors
    features = torch.tensor(features)
    labels = torch.tensor(labels)
    
    return {'features': features, 'label': labels}


train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False,collate_fn=custom_collate)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False,collate_fn=custom_collate)

train_features = []
train_labels = []

for batch in train_dataloader:
    train_features.append(batch['features'].numpy())
    train_labels.append(batch['label'].numpy())

# Concatenating all batches into final NumPy arrays
train_features_np = np.concatenate(train_features, axis=0)
train_labels_np = np.concatenate(train_labels, axis=0)

# Collecting test features and labels
test_features = []
test_labels = []

for batch in test_dataloader:
    test_features.append(batch['features'].numpy())
    test_labels.append(batch['label'].numpy())

# Concatenating all batches into final NumPy arrays
test_features_np = np.concatenate(test_features, axis=0)
test_labels_np = np.concatenate(test_labels, axis=0)

n_components = 100  

# Initialize PCA and fit it on the train features
pca = PCA(n_components=n_components)
train_features_reduced = pca.fit_transform(train_features_np)

# Apply PCA transformation on test features
test_features_reduced = pca.transform(test_features_np)



def mean_average_precision_at_k(true_labels, sorted_indices, k):
    """Calculate Mean Average Precision at k for all queries"""
    average_precisions = []
    sorted_indices = np.array(sorted_indices) 
    for i in range(len(true_labels)):
        relevant = (train_labels_np[sorted_indices[i, :k].astype(int)] == true_labels[i]).float()  # Ensure integer indexing
        precision_at_i = torch.cumsum(relevant, dim=0) / torch.arange(1, k + 1, dtype=torch.float)
        if relevant.sum() > 0:
            average_precisions.append((precision_at_i * relevant).sum() / relevant.sum())
        else:
            average_precisions.append(torch.tensor(0.0))
    return torch.mean(torch.stack(average_precisions))

def precision_at_k(true_labels, sorted_indices, k):
    """Calculate Precision at k"""
    precision_at_k = []

    for i in range(len(true_labels)):
        # Ensure sorted_indices[i, :k] is a numpy array of integers
        relevant_indices = sorted_indices[i][:k]
        relevant_indices = relevant_indices[relevant_indices != -1]  # Assuming -1 is the padding value
        
        if len(relevant_indices) > 0:
            relevant = (train_labels_np[relevant_indices.astype(int)] == true_labels[i]).float()
            precision_at_k.append(relevant.sum() / k)
        else:
            precision_at_k.append(torch.tensor(0.0))
    return torch.mean(torch.stack(precision_at_k))

k_values = [1000, 1200, 1500, 500]

def cluster_based_retrieval(test_features_np, train_features_np, train_labels, test_labels_pred, train_cluster_labels, k=50):
    sorted_indices = []
    for i, test_feature in enumerate(test_features_np):
        test_cluster = test_labels_pred[i]
        
        # Get train features that are in the same cluster as the test feature's predicted cluster
        same_cluster_indices = np.where(train_cluster_labels == test_cluster)[0]
        same_cluster_train_features = train_features_np[same_cluster_indices]
        
        # Calculate pairwise distances within the same cluster
        distances = pairwise_distances(test_feature.reshape(1, -1), same_cluster_train_features).flatten()
        
        # Sort by distances to get the closest neighbors within the cluster
        sorted_cluster_indices = np.argsort(distances)[:k]
        if len(sorted_cluster_indices) < k:
            sorted_indices.append(np.pad(same_cluster_indices[sorted_cluster_indices], (0, k - len(sorted_cluster_indices)), constant_values=-1))
        else:
            sorted_indices.append(same_cluster_indices[sorted_cluster_indices])
    
    return np.array(sorted_indices, dtype=object)


for i in k_values:
    print(f"Evaluating for k={i} clusters")

    kmeans = KMeans(n_clusters=i)
    train_features_reduced = train_features_reduced.reshape(-1, train_features_reduced.shape[-1])

    kmeans.fit(train_features_reduced)
    test_features_reduced = test_features_reduced.reshape(-1, test_features_reduced.shape[-1])
    # print(test_features_np)
    # Predict cluster labels for the test set
    test_labels_pred = kmeans.predict(test_features_reduced)
    

    train_cluster_labels = kmeans.labels_

    start_time = time.time()
    # Sort the distances to get the nearest clusters for each test sample
    sorted_indices = cluster_based_retrieval(test_features_reduced, train_features_reduced, train_labels, test_labels_pred, train_cluster_labels)    # print(sorted_indices)
    end_time = time.time()
    retrieval_time = end_time - start_time

    # Convert train and test labels to tensors for metric calculation
    train_labels_tensor = torch.tensor(train_labels_np)
    test_labels_tensor = torch.tensor(test_labels_np)
    map_score = mean_average_precision_at_k(test_labels_tensor, sorted_indices, k=50)
    precision_at_10 = precision_at_k(test_labels_tensor, sorted_indices, k=10)
    precision_at_50 = precision_at_k(test_labels_tensor, sorted_indices, k=50)

    # Output the results
    print(f"For k={i} clusters:") 
    print(f"retrieval-time: {retrieval_time}")
    print(f"Mean Average Precision (mAP)@50: {map_score:.4f}")
    print(f"Precision@10: {precision_at_10:.4f}")
    print(f"Precision@50: {precision_at_50:.4f}")