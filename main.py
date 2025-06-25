import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torchvision.datasets import CIFAR10
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


CIFAR10_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck')

class Config:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NORMAL_CLASS = 1  # 1 is 'automobile'
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 128
    LATENT_DIM = 128  
    SAVE_DIR = "results"

print(f"Using device: {Config.DEVICE}")
print(f"Normal Class: '{CIFAR10_CLASSES[Config.NORMAL_CLASS]}' (Index: {Config.NORMAL_CLASS})")
os.makedirs(Config.SAVE_DIR, exist_ok=True)



class DeepSVDD(nn.Module):
    def __init__(self, latent_dim):
        super(DeepSVDD, self).__init__()
        weights = ResNet50_Weights.DEFAULT
        backbone = resnet50(weights=weights)
        
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        self.feature_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone(x)
        x = self.feature_head(x)
        return x


def get_cifar10_dataloaders(normal_class_idx):
    """Prepares CIFAR-10 dataloaders for the one-class classification task."""
    # ResNet50 expects specific transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    # --- Create Normal Training Set (only contains the normal class) ---
    train_indices = [i for i, (_, label) in enumerate(train_dataset) if label == normal_class_idx]
    train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=Config.BATCH_SIZE, shuffle=True)


    test_targets = [0 if label == normal_class_idx else 1 for _, label in test_dataset]
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, test_targets

def initialize_center(model, train_loader, device):
    """Initializes the hypersphere center c as the mean of the network outputs."""
    print("Initializing hypersphere center c...")
    model.eval()
    all_outputs = []
    with torch.no_grad():
        for inputs, _ in tqdm(train_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs.detach().cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    center = torch.mean(all_outputs, dim=0)
    return center.to(device)

def train_and_evaluate_svdd():
    """Main function to run the Deep SVDD experiment."""
    train_loader, test_loader, test_labels = get_cifar10_dataloaders(Config.NORMAL_CLASS)
    model = DeepSVDD(latent_dim=Config.LATENT_DIM).to(Config.DEVICE)
    optimizer = optim.Adam(model.feature_head.parameters(), lr=Config.LEARNING_RATE) # Only train the head

    center = initialize_center(model, train_loader, Config.DEVICE)
    
    print("\n--- Starting Deep SVDD Training ---")
    model.train()
    for epoch in range(Config.NUM_EPOCHS):
        total_loss = 0
        for inputs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS}"):
            inputs = inputs.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            dist = torch.sum((outputs - center) ** 2, dim=1)
            loss = torch.mean(dist)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}, Average Loss: {total_loss / len(train_loader):.6f}")

    print("\n--- Evaluating Deep SVDD Model ---")
    model.eval()
    anomaly_scores = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(Config.DEVICE)
            outputs = model(inputs)
            dist = torch.sum((outputs - center) ** 2, dim=1)
            anomaly_scores.extend(dist.cpu().numpy())

    auc = roc_auc_score(test_labels, anomaly_scores)
    print(f"\nDeep SVDD Final AUC: {auc:.4f}")

    visualize_results(test_loader.dataset, anomaly_scores, 'svdd')

    return model, auc

def run_ocsvm_baseline(svdd_model, train_loader, test_loader, test_labels):
    """Runs the One-Class SVM baseline on features extracted by the SVDD model."""
    print("\n--- Running One-Class SVM Baseline ---")
    svdd_model.eval()

    print("Extracting features from the normal training set...")
    train_features = []
    with torch.no_grad():
        for inputs, _ in tqdm(train_loader):
            inputs = inputs.to(Config.DEVICE)
            outputs = svdd_model(inputs)
            train_features.append(outputs.cpu().numpy())
    train_features = np.concatenate(train_features, axis=0)

    print("Training One-Class SVM...")
    ocsvm = OneClassSVM(gamma='auto', nu=0.1) # nu is an important hyperparameter
    ocsvm.fit(train_features)

    print("Extracting features from the test set...")
    test_features = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader):
            inputs = inputs.to(Config.DEVICE)
            outputs = svdd_model(inputs)
            test_features.append(outputs.cpu().numpy())
    test_features = np.concatenate(test_features, axis=0)
    

    anomaly_scores_ocsvm = -ocsvm.decision_function(test_features)
    auc_ocsvm = roc_auc_score(test_labels, anomaly_scores_ocsvm)

    print(f"\nOne-Class SVM Final AUC: {auc_ocsvm:.4f}")

    visualize_results(test_loader.dataset, anomaly_scores_ocsvm, 'ocsvm')
    
    return auc_ocsvm

def visualize_results(test_dataset, scores, method_name):
    """Saves a plot of images with their anomaly scores."""
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    normal_indices = [i for i, (_, label) in enumerate(test_dataset) if label == Config.NORMAL_CLASS]
    anomalous_indices = [i for i, (_, label) in enumerate(test_dataset) if label != Config.NORMAL_CLASS]

    best_normal = min(normal_indices, key=lambda i: scores[i])
    worst_normal = max(normal_indices, key=lambda i: scores[i])
    best_anomaly = max(anomalous_indices, key=lambda i: scores[i])
    worst_anomaly = min(anomalous_indices, key=lambda i: scores[i])

    indices_to_plot = [best_normal, worst_normal, best_anomaly, worst_anomaly]
    titles = [
        f"Correct Normal\nScore: {scores[best_normal]:.2f}",
        f"Incorrect Normal (High Score)\nScore: {scores[worst_normal]:.2f}",
        f"Correct Anomaly\nScore: {scores[best_anomaly]:.2f}",
        f"Incorrect Anomaly (Low Score)\nScore: {scores[worst_anomaly]:.2f}"
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for i, idx in enumerate(indices_to_plot):
        img, _ = test_dataset[idx]
        img = inv_normalize(img).permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    plt.suptitle(f"Visualization for {method_name.upper()}", fontsize=16)
    plt.savefig(os.path.join(Config.SAVE_DIR, f"visualization_{method_name}.png"))
    plt.close()
    print(f"Saved visualization to {os.path.join(Config.SAVE_DIR, f'visualization_{method_name}.png')}")



if __name__ == '__main__':
    trained_svdd_model, svdd_auc = train_and_evaluate_svdd()
    
    train_loader, test_loader, test_labels = get_cifar10_dataloaders(Config.NORMAL_CLASS)
    ocsvm_auc = run_ocsvm_baseline(trained_svdd_model, train_loader, test_loader, test_labels)

    print("\n--- FINAL RESULTS ---")
    print(f"Normal Class: {CIFAR10_CLASSES[Config.NORMAL_CLASS]}")
    print(f"Deep SVDD AUC:     {svdd_auc:.4f}")
    print(f"One-Class SVM AUC: {ocsvm_auc:.4f}")