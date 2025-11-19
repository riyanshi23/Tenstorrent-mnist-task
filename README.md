# MNIST to TT-NN Conversion Pipeline

This project trains a 3-layer MLP on MNIST using PyTorch (achieving 97.70% accuracy), converts the trained weights to TT-NN format, and deploys the model on Tenstorrent hardware. 
The pipeline consists of four automated steps: 
1) training (`1_pytorch_mlp.py`), 
2) weight conversion with proper transposition (`2_convert_weights.py`), 
3) TT-NN inference (`3_ttnn_inference.py`), and 
4) validation (`4_validate.py`). 

Run the complete pipeline with `python run_all.py` after installing dependencies via `pip install -r requirements.txt` and `pip install git+https://github.com/tenstorrent/tt-metal.git`. 
The validation confirms 100% prediction agreement between PyTorch and TT-NN implementations with numerical precision within 4e-06, demonstrating successful model deployment on Tenstorrent hardware.

## Quick Start
```bash
pip install -r requirements.txt
pip install git+https://github.com/tenstorrent/tt-metal.git
python run_all.py
```

## Results
PyTorch: 97.70% accuracy | TT-NN: 100% prediction agreement | Max error: 4e-06
