"""
Step 1: Build and Train PyTorch MLP for MNIST
==============================================

This script:
1. Loads the MNIST dataset
2. Defines a Multi-Layer Perceptron (MLP) architecture
3. Trains the model
4. Saves the trained weights for later conversion to TT-NN

Architecture:
- Input: 784 (28x28 flattened images)
- Hidden Layer 1: 512 neurons with ReLU
- Hidden Layer 2: 256 neurons with ReLU
- Output: 10 classes (digits 0-9) with Softmax
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


class SimpleMLP(nn.Module):
    """Multi-Layer Perceptron for MNIST digit classification"""
    
    def __init__(self, input_size=784, hidden1=512, hidden2=256, num_classes=10):
        super(SimpleMLP, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)
        
        self.layer_dims = {
            'input': input_size,
            'hidden1': hidden1,
            'hidden2': hidden2,
            'output': num_classes
        }
    
    def forward(self, x):
        # Flatten input if needed
        x = x.view(-1, 784)
        
        # Forward pass
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        
        return x


def load_mnist_data(batch_size=64):
    """Load and preprocess MNIST dataset"""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Create data directory
    os.makedirs('./data', exist_ok=True)
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        './data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        './data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_model(model, train_loader, test_loader, epochs=5, learning_rate=0.001):
    """Train the MLP model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    print(f"Training on {device}")
    print(f"Model architecture: {model.layer_dims}")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop with progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100 * correct / total:.2f}%'})
        
        # Calculate epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader, device)
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    return history


def evaluate_model(model, test_loader, device):
    """Evaluate model accuracy on test set"""
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def save_model(model, filepath='models/mnist_mlp.pth'):
    """Save trained model weights"""
    
    os.makedirs('models', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'layer_dims': model.layer_dims
    }, filepath)
    print(f"\nModel saved to {filepath}")


def visualize_predictions(model, test_loader, num_samples=10):
    """Visualize some predictions"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    outputs = model(images)
    _, predictions = torch.max(outputs, 1)
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < num_samples:
            img = images[i].cpu().squeeze()
            ax.imshow(img, cmap='gray')
            ax.set_title(f'True: {labels[i].item()}\nPred: {predictions[i].item()}',
                        color='green' if labels[i] == predictions[i] else 'red')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('models/predictions.png')
    print("Sample predictions saved to models/predictions.png")


if __name__ == '__main__':
    print("=" * 60)
    print("MNIST MLP Training with PyTorch")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size=64)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\n[2/4] Creating MLP model...")
    model = SimpleMLP(input_size=784, hidden1=512, hidden2=256, num_classes=10)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    print("\n[3/4] Training model...")
    history = train_model(model, train_loader, test_loader, epochs=3, learning_rate=0.001)
    
    # Save model
    print("\n[4/4] Saving model...")
    save_model(model)
    
    # Visualize predictions
    visualize_predictions(model, test_loader)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    print("=" * 60)
