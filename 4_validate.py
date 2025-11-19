"""
Step 4: Validation - Compare PyTorch vs TT-NN Outputs
======================================================

This script:
1. Loads the original PyTorch model
2. Loads the TT-NN converted weights
3. Runs the same input through both models
4. Compares outputs layer by layer
5. Validates numerical precision and correctness
"""

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import os


class SimpleMLP(nn.Module):
    """Same architecture as in training script"""
    
    def __init__(self, input_size=784, hidden1=512, hidden2=256, num_classes=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class TTNNSimulator:
    """Simulates TT-NN inference using converted weights"""
    
    def __init__(self, weights_dir='models/ttnn_weights'):
        self.weights = {}
        self.biases = {}
        self._load_parameters(weights_dir)
    
    def _load_parameters(self, weights_dir):
        """Load converted TT-NN weights"""
        for filename in os.listdir(weights_dir):
            if filename.endswith('.npy'):
                filepath = os.path.join(weights_dir, filename)
                param = np.load(filepath)
                parts = filename.replace('.npy', '').split('_')
                layer_name = parts[0]
                param_type = parts[1]
                
                if param_type == 'weight':
                    self.weights[layer_name] = param
                elif param_type == 'bias':
                    self.biases[layer_name] = param
    
    def forward(self, x):
        """Forward pass through converted model"""
        # Convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        
        # Flatten
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        
        # Layer 1
        x = np.matmul(x, self.weights['fc1']) + self.biases['fc1']
        x = np.maximum(0, x)  # ReLU
        
        # Layer 2
        x = np.matmul(x, self.weights['fc2']) + self.biases['fc2']
        x = np.maximum(0, x)  # ReLU
        
        # Layer 3
        x = np.matmul(x, self.weights['fc3']) + self.biases['fc3']
        
        return x


def load_pytorch_model(model_path='models/mnist_mlp.pth'):
    """Load trained PyTorch model"""
    checkpoint = torch.load(model_path, map_location='cpu')
    model = SimpleMLP()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def get_test_samples(num_samples=10):
    """Get test samples from MNIST"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    images = []
    labels = []
    for i in range(num_samples):
        img, label = test_dataset[i]
        images.append(img)
        labels.append(label)
    
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    return images, labels


def compare_outputs(pytorch_output, ttnn_output, tolerance=1e-5):
    """Compare PyTorch and TT-NN outputs"""
    
    # Convert to numpy for comparison
    if isinstance(pytorch_output, torch.Tensor):
        pytorch_output = pytorch_output.detach().numpy()
    
    # Calculate differences
    abs_diff = np.abs(pytorch_output - ttnn_output)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    # Check if within tolerance
    is_close = np.allclose(pytorch_output, ttnn_output, atol=tolerance, rtol=tolerance)
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'is_close': is_close,
        'tolerance': tolerance
    }


def validate_layer_by_layer(pytorch_model, ttnn_model, test_input):
    """Compare outputs layer by layer"""
    
    print("\n" + "=" * 60)
    print("LAYER-BY-LAYER VALIDATION")
    print("=" * 60)
    
    # Flatten input
    x_pytorch = test_input.view(-1, 784)
    x_ttnn = test_input.numpy().reshape(-1, 784)
    
    # Layer 1
    print("\n[Layer 1: fc1 + relu]")
    x_pytorch = pytorch_model.fc1(x_pytorch)
    x_ttnn = np.matmul(x_ttnn, ttnn_model.weights['fc1']) + ttnn_model.biases['fc1']
    
    result = compare_outputs(x_pytorch, x_ttnn)
    print(f"  Max difference: {result['max_diff']:.2e}")
    print(f"  Mean difference: {result['mean_diff']:.2e}")
    print(f"  Match: {result['is_close']} [PASS]" if result['is_close'] else f"  Match: {result['is_close']} [FAIL]")
    
    x_pytorch = pytorch_model.relu1(x_pytorch)
    x_ttnn = np.maximum(0, x_ttnn)
    
    # Layer 2
    print("\n[Layer 2: fc2 + relu]")
    x_pytorch = pytorch_model.fc2(x_pytorch)
    x_ttnn = np.matmul(x_ttnn, ttnn_model.weights['fc2']) + ttnn_model.biases['fc2']
    
    result = compare_outputs(x_pytorch, x_ttnn)
    print(f"  Max difference: {result['max_diff']:.2e}")
    print(f"  Mean difference: {result['mean_diff']:.2e}")
    print(f"  Match: {result['is_close']} [PASS]" if result['is_close'] else f"  Match: {result['is_close']} [FAIL]")
    
    x_pytorch = pytorch_model.relu2(x_pytorch)
    x_ttnn = np.maximum(0, x_ttnn)
    
    # Layer 3
    print("\n[Layer 3: fc3]")
    x_pytorch = pytorch_model.fc3(x_pytorch)
    x_ttnn = np.matmul(x_ttnn, ttnn_model.weights['fc3']) + ttnn_model.biases['fc3']
    
    result = compare_outputs(x_pytorch, x_ttnn)
    print(f"  Max difference: {result['max_diff']:.2e}")
    print(f"  Mean difference: {result['mean_diff']:.2e}")
    print(f"  Match: {result['is_close']} [PASS]" if result['is_close'] else f"  Match: {result['is_close']} [FAIL]")
    
    return x_pytorch, x_ttnn


def validate_predictions(images, labels):
    """Validate predictions match between PyTorch and TT-NN"""
    
    print("\n" + "=" * 60)
    print("PREDICTION VALIDATION")
    print("=" * 60)
    
    # Load models
    pytorch_model = load_pytorch_model()
    ttnn_model = TTNNSimulator()
    
    # Get predictions from both models
    with torch.no_grad():
        pytorch_output = pytorch_model(images)
        pytorch_pred = torch.argmax(pytorch_output, dim=1)
    
    ttnn_output = ttnn_model.forward(images)
    ttnn_pred = np.argmax(ttnn_output, axis=1)
    
    # Compare
    print(f"\nTesting on {len(images)} samples:")
    print("-" * 60)
    
    matches = 0
    correct_pytorch = 0
    correct_ttnn = 0
    
    for i in range(len(images)):
        true_label = labels[i].item()
        pt_pred = pytorch_pred[i].item()
        tt_pred = ttnn_pred[i]
        
        pred_match = pt_pred == tt_pred
        pt_correct = pt_pred == true_label
        tt_correct = tt_pred == true_label
        
        if pred_match:
            matches += 1
        if pt_correct:
            correct_pytorch += 1
        if tt_correct:
            correct_ttnn += 1
        
        status = "[PASS]" if pred_match else "[FAIL]"
        print(f"{status} Sample {i+1}: True={true_label}, "
              f"PyTorch={pt_pred}, TT-NN={tt_pred}")
    
    print("-" * 60)
    print(f"\nPrediction Agreement: {matches}/{len(images)} ({100*matches/len(images):.1f}%)")
    print(f"PyTorch Accuracy: {correct_pytorch}/{len(images)} ({100*correct_pytorch/len(images):.1f}%)")
    print(f"TT-NN Accuracy: {correct_ttnn}/{len(images)} ({100*correct_ttnn/len(images):.1f}%)")
    
    # Numerical comparison of outputs
    result = compare_outputs(pytorch_output, ttnn_output)
    print(f"\nNumerical Precision:")
    print(f"  Max output difference: {result['max_diff']:.2e}")
    print(f"  Mean output difference: {result['mean_diff']:.2e}")
    
    return matches == len(images)


if __name__ == '__main__':
    print("=" * 60)
    print("PyTorch vs TT-NN Validation")
    print("=" * 60)
    
    try:
        # Load test data
        print("\n[1/3] Loading test samples...")
        images, labels = get_test_samples(num_samples=10)
        print(f"Loaded {len(images)} test samples")
        
        # Load models
        print("\n[2/3] Loading models...")
        pytorch_model = load_pytorch_model()
        ttnn_model = TTNNSimulator()
        print("Models loaded successfully")
        
        # Validate layer by layer
        print("\n[3/3] Running validation...")
        validate_layer_by_layer(pytorch_model, ttnn_model, images[0:1])
        
        # Validate predictions
        all_match = validate_predictions(images, labels)
        
        # Final result
        print("\n" + "=" * 60)
        if all_match:
            print("[SUCCESS] All predictions match!")
            print("TT-NN conversion is CORRECT")
        else:
            print("[WARNING] Some predictions differ")
            print("Check numerical precision settings")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("\nPlease run the previous scripts first:")
        print("  1. python 1_pytorch_mlp.py")
        print("  2. python 2_convert_weights.py")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
