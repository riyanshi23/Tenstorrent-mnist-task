"""
Step 3: TT-NN Inference Implementation
=======================================

This script demonstrates how to implement the MNIST MLP inference using TT-NN.
"""

import numpy as np
import os

# import ttnn
# import torch


class TTNNModelSimulator:
    
    def __init__(self, weights_dir='models/ttnn_weights'):
        """Load converted weights"""
        self.weights_dir = weights_dir
        self.weights = {}
        self.biases = {}
        
        # Load all converted weights
        self._load_parameters()
    
    def _load_parameters(self):
        """Load weights and biases from saved numpy files"""
        
        print("Loading converted TT-NN weights...")
        
        for filename in os.listdir(self.weights_dir):
            if filename.endswith('.npy'):
                filepath = os.path.join(self.weights_dir, filename)
                param = np.load(filepath)
                
                # Parse filename: e.g., "fc1_weight.npy"
                parts = filename.replace('.npy', '').split('_')
                layer_name = parts[0]
                param_type = parts[1]
                
                if param_type == 'weight':
                    self.weights[layer_name] = param
                elif param_type == 'bias':
                    self.biases[layer_name] = param
                
                print(f"  Loaded {filename}: shape {param.shape}")
    
    def preprocess_input(self, image):
        """
        Preprocess MNIST image for TT-NN
        
        Input: 28x28 image (or batch of images)
        Output: Flattened 784-dim vector (or batch x 784)
        """
        
        # Flatten image
        if len(image.shape) == 2:  # Single image
            image = image.reshape(1, -1)
        else:  # Batch of images
            batch_size = image.shape[0]
            image = image.reshape(batch_size, -1)
        
        # Normalize (MNIST preprocessing)
        image = (image - 0.1307) / 0.3081
        
        return image
    
    def linear(self, x, weight, bias):
        """
        Simulates TT-NN linear layer: y = x @ weight + bias
        
        In actual TT-NN, this would be:
        y = ttnn.linear(x, weight, bias=bias)
        """
        y = np.matmul(x, weight) + bias
        return y
    
    def relu(self, x):
        """
        Simulates TT-NN ReLU activation
        
        In actual TT-NN, this would be:
        y = ttnn.relu(x)
        """
        return np.maximum(0, x)
    
    def softmax(self, x):
        """
        Softmax for classification probabilities
        
        In actual TT-NN, this would be:
        y = ttnn.softmax(x, dim=-1)
        """
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def forward(self, x):
        """
        Forward pass through the MLP
        
        Architecture:
        Input (784) -> FC1 (512) -> ReLU -> FC2 (256) -> ReLU -> FC3 (10) -> Softmax
        """
        
        # Layer 1: input -> hidden1
        x = self.linear(x, self.weights['fc1'], self.biases['fc1'])
        x = self.relu(x)
        
        # Layer 2: hidden1 -> hidden2
        x = self.linear(x, self.weights['fc2'], self.biases['fc2'])
        x = self.relu(x)
        
        # Layer 3: hidden2 -> output
        x = self.linear(x, self.weights['fc3'], self.biases['fc3'])
        
        # Softmax for probabilities
        probabilities = self.softmax(x)
        
        return probabilities
    
    def predict(self, image):
        """Make prediction on a single image or batch"""
        
        # Preprocess
        x = self.preprocess_input(image)
        
        # Forward pass
        probabilities = self.forward(x)
        
        # Get predicted class
        predictions = np.argmax(probabilities, axis=-1)
        
        return predictions, probabilities


def demonstrate_ttnn_inference():
    """Demonstrate TT-NN inference with sample data"""
    
    print("=" * 60)
    print("TT-NN MNIST Inference (Simulated)")
    print("=" * 60)
    
    # Initialize model
    print("\n[1/3] Initializing TT-NN model...")
    model = TTNNModelSimulator()
    
    # Load a sample MNIST image
    print("\n[2/3] Loading test data...")
    try:
        from torchvision import datasets, transforms
        
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
        
        # Get a few test samples
        num_samples = 5
        images = []
        labels = []
        
        for i in range(num_samples):
            img, label = test_dataset[i]
            images.append(img.numpy().squeeze())
            labels.append(label)
        
        images = np.array(images)
        
        print(f"Loaded {num_samples} test images")
        
        # Make predictions
        print("\n[3/3] Running TT-NN inference...")
        predictions, probabilities = model.predict(images)
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        for i in range(num_samples):
            true_label = labels[i]
            pred_label = predictions[i]
            confidence = probabilities[i][pred_label] * 100
            
            status = "[PASS]" if true_label == pred_label else "[FAIL]"
            print(f"{status} Sample {i+1}: True={true_label}, Predicted={pred_label}, "
                  f"Confidence={confidence:.2f}%")
        
        accuracy = np.mean(predictions == np.array(labels)) * 100
        print(f"\nAccuracy on {num_samples} samples: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"Could not load test data: {e}")
        print("Please run 1_pytorch_mlp.py first to download MNIST dataset")


def explain_actual_ttnn_implementation():
    
    print("\n" + "=" * 60)
    print("=" * 60)


if __name__ == '__main__':
    demonstrate_ttnn_inference()
    
    explain_actual_ttnn_implementation()
    
    print("\n" + "=" * 60)
    print("TT-NN Inference Script Complete!")
    print("=" * 60)
