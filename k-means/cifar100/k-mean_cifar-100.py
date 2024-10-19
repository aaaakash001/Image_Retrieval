from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset, load_from_disk
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from torchmetrics.retrieval import RetrievalMAP, RetrievalPrecision
import time

torch.set_num_threads(16)
torch.manual_seed(0)
device = 'cpu'

main_path = '/workspace/lsh/cifar-100/'

local_dir = "/workspace/lsh/cifar-100/data"

processor = AutoImageProcessor.from_pretrained("google/vit-large-patch32-384",use_fast=True)
model = AutoModelForImageClassification.from_pretrained("google/vit-large-patch32-384")
# Load the dataset and save it in the specified directory
ds = load_dataset("uoft-cs/cifar100", cache_dir=local_dir)

model.eval()

transform = transforms.Compose([
        transforms.Resize((384, 384)),  # Resize images to a consistent size
        transforms.ToTensor(),           # Convert PIL images to tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def extract_features(batch):
    imgs = batch['img']  # This will now be a list of images in a batch
    img_tensors = [transform(img).unsqueeze(0).to('cpu') for img in imgs]  # Add batch dimension and move to device
    img_tensors = torch.cat(img_tensors, dim=0)  # Concatenate to create a batch tensor

    inputs = {'pixel_values': img_tensors}  # Adjust for model input (assuming using Hugging Face models)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states  # List of all hidden states

    final_hidden_state = hidden_states[-1]  # Shape: [batch_size, num_patches, hidden_size]
    
    pooled_features = final_hidden_state.mean(dim=1)  # Shape: [batch_size, hidden_size]
    
    return {"features": pooled_features.cpu().numpy()}  #

output_dir = main_path+ "features/cifar100/"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "vit_large/") 


# Check if features already exist
if os.path.exists(output_file):
    print(f"Features file already exists at {output_file}. Loading features...")
    # Load existing features
    ds_feature = load_from_disk(output_file)
else:
    print("Features file does not exist. Extracting features...")

    # Use `map` to apply the feature extraction across the entire dataset
    ds_feature = ds.map(extract_features, batched=True, batch_size=64, num_proc=16)

    # Save the dataset with the new features to disk
    ds_feature.save_to_disk(output_file)  # Saving in the Arrow format
    print("Features saved successfully!")

def custom_collate(batch):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to a consistent size
        transforms.ToTensor(),           # Convert PIL images to tensors
    ])

    images = [transform(item['img']) for item in batch]  # Load and transform images
    labels = torch.tensor([item['fine_label'] for item in batch])          # Convert labels to a tensor
    features = [torch.tensor(item['features']) for item in batch]     # Convert features to tensors

    return {
        'img': torch.stack(images),    # Stack image tensors into a single tensor
        'label': labels,
        'features': torch.stack(features)  # Stack feature tensors into a single tensor
    }



train_dataloader = DataLoader(ds_feature['train'], batch_size=64, shuffle=False,collate_fn=custom_collate)
test_dataloader = DataLoader(ds_feature['test'], batch_size=64, shuffle=False,collate_fn=custom_collate)

train_features = []
train_labels = []
test_features = []
test_labels = []

for batch in train_dataloader:
    train_features.append(batch['features'].numpy())
    train_labels.append(batch['label'].numpy())
train_features_np = np.concatenate(train_features)
train_labels_np = np.concatenate(train_labels)

for batch in test_dataloader:
    test_features.append(batch['features'].numpy())
    test_labels.append(batch['label'].numpy())
test_features_np = np.concatenate(test_features)
test_labels_np = np.concatenate(test_labels)



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

k_values = [5, 10, 25, 50, 100, 200, 300, 400, 500]


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
    kmeans.fit(train_features_np)

    # Predict cluster labels for the test set
    test_labels_pred = kmeans.predict(test_features_np)
    
    train_cluster_labels = kmeans.labels_

    start_time = time.time()
    # Sort the distances to get the nearest clusters for each test sample
    sorted_indices = cluster_based_retrieval(test_features_np, train_features_np, train_labels, test_labels_pred, train_cluster_labels)    # print(sorted_indices)
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