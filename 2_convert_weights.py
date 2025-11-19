"""
Step 2: Convert PyTorch Weights to TT-NN Format
================================================

This script:
1. Loads the trained PyTorch model
2. Extracts weights and biases from each layer
3. Converts them to the format required by TT-NN
4. Saves converted weights for TT-NN inference

Key conversions:
- Weight matrices: Need proper transposition for TT-NN linear operations
- Bias vectors: Convert to appropriate format
- Data type: Ensure compatibility with TT-NN (float32/bfloat16)
"""

import torch
import numpy as np
import os
import json


def load_pytorch_model(model_path='models/mnist_mlp.pth'):
    """Load the trained PyTorch model"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    layer_dims = checkpoint['layer_dims']
    
    print(f"Loaded model from {model_path}")
    print(f"Architecture: {layer_dims}")
    
    return state_dict, layer_dims


def extract_layer_parameters(state_dict):
    """Extract weights and biases from PyTorch state dict"""
    
    # PyTorch MLP has 3 linear layers: fc1, fc2, fc3
    layers = {}
    
    for key, value in state_dict.items():
        # Extract layer name and parameter type (weight or bias)
        parts = key.split('.')
        layer_name = parts[0]  # e.g., 'fc1', 'fc2', 'fc3'
        param_type = parts[1]   # e.g., 'weight', 'bias'
        
        if layer_name not in layers:
            layers[layer_name] = {}
        
        # Convert to numpy for easier manipulation
        layers[layer_name][param_type] = value.numpy()
        
        print(f"  {key}: shape {value.shape}")
    
    return layers


def convert_weights_for_ttnn(layers):
    """
    Convert PyTorch weights to TT-NN format
    
    Important notes:
    - PyTorch Linear: output = input @ weight.T + bias
      where weight shape is [out_features, in_features]
    
    - TT-NN Linear: output = input @ weight + bias
      where weight should be [in_features, out_features]
    
    So we need to transpose PyTorch weights!
    """
    
    converted_layers = {}
    
    for layer_name, params in layers.items():
        converted_layers[layer_name] = {}
        
        # Transpose weights for TT-NN
        if 'weight' in params:
            pytorch_weight = params['weight']  # Shape: [out, in]
            ttnn_weight = pytorch_weight.T      # Shape: [in, out]
            converted_layers[layer_name]['weight'] = ttnn_weight
            
            print(f"\n{layer_name} weight conversion:")
            print(f"  PyTorch shape: {pytorch_weight.shape} (out x in)")
            print(f"  TT-NN shape: {ttnn_weight.shape} (in x out)")
        
        # Biases remain the same
        if 'bias' in params:
            converted_layers[layer_name]['bias'] = params['bias']
            print(f"  Bias shape: {params['bias'].shape}")
    
    return converted_layers


def save_converted_weights(converted_layers, output_dir='models/ttnn_weights'):
    """Save converted weights in numpy format"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each layer's parameters
    for layer_name, params in converted_layers.items():
        for param_type, param_value in params.items():
            filename = f"{layer_name}_{param_type}.npy"
            filepath = os.path.join(output_dir, filename)
            np.save(filepath, param_value)
            print(f"Saved: {filepath}")
    
    # Save metadata
    metadata = {
        'layers': list(converted_layers.keys()),
        'conversion_notes': {
            'weight_format': 'Transposed for TT-NN (in_features x out_features)',
            'bias_format': 'Same as PyTorch',
            'data_type': 'float32'
        }
    }
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved: {metadata_path}")


def verify_conversion(original_layers, converted_layers):
    """Verify that conversion was done correctly"""
    
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    
    for layer_name in original_layers.keys():
        if 'weight' in original_layers[layer_name]:
            orig_weight = original_layers[layer_name]['weight']
            conv_weight = converted_layers[layer_name]['weight']
            
            # Check if transpose was done correctly
            assert orig_weight.shape[0] == conv_weight.shape[1], \
                f"{layer_name}: Dimension mismatch after transpose"
            assert orig_weight.shape[1] == conv_weight.shape[0], \
                f"{layer_name}: Dimension mismatch after transpose"
            
            # Check if transpose is correct
            assert np.allclose(orig_weight.T, conv_weight), \
                f"{layer_name}: Transpose not done correctly"
            
            print(f"[OK] {layer_name}: Weights correctly transposed")
        
        if 'bias' in original_layers[layer_name]:
            orig_bias = original_layers[layer_name]['bias']
            conv_bias = converted_layers[layer_name]['bias']
            
            assert np.array_equal(orig_bias, conv_bias), \
                f"{layer_name}: Bias values changed unexpectedly"
            
            print(f"[OK] {layer_name}: Bias correctly preserved")
    
    print("\n[SUCCESS] All conversions verified successfully!")


if __name__ == '__main__':
    print("=" * 60)
    print("Converting PyTorch Weights to TT-NN Format")
    print("=" * 60)
    
    # Load PyTorch model
    print("\n[1/4] Loading PyTorch model...")
    state_dict, layer_dims = load_pytorch_model()
    
    # Extract parameters
    print("\n[2/4] Extracting layer parameters...")
    layers = extract_layer_parameters(state_dict)
    
    # Convert to TT-NN format
    print("\n[3/4] Converting weights for TT-NN...")
    converted_layers = convert_weights_for_ttnn(layers)
    
    # Save converted weights
    print("\n[4/4] Saving converted weights...")
    save_converted_weights(converted_layers)
    
    # Verify conversion
    verify_conversion(layers, converted_layers)
    
    print("\n" + "=" * 60)
    print("Weight Conversion Complete!")
    print("=" * 60)
    print("\nConverted weights saved in: models/ttnn_weights/")
    print("Ready for TT-NN inference!")
