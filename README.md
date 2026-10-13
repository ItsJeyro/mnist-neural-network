# MNIST Neural Network from Scratch

A handwritten digit classifier built from scratch using **Python and CuPy**, with manually implemented forward propagation, backpropagation, and momentum-based gradient descent.

This project is part of my journey into artificial intelligence and neural networks. My goal is to understand how neural networks work internally by implementing their core mechanisms instead of relying on high-level deep learning frameworks.

## Architecture

The network consists of four layers:

```text
Input (784)
    ↓
Dense Layer (286 neurons, ReLU)
    ↓
Dense Layer (286 neurons, ReLU)
    ↓
Output Layer (10 neurons, Softmax)
```

- **Input:** 784 features representing a 28 × 28 grayscale image.
- **Hidden layers:** Two dense layers with 286 neurons each.
- **Output:** 10 classes, corresponding to digits 0–9.
- **Trainable parameters:** Approximately 309,000.

## Technologies

- Python
- CuPy for GPU-accelerated numerical computations
- NumPy-compatible array operations
- Matplotlib for training visualization
- MNIST handwritten digit dataset

## Training

The network uses:

- ReLU activation for hidden layers
- Softmax activation for output probabilities
- Backpropagation implemented manually
- Stochastic gradient descent with momentum
- Mini-batch training
- Separate training, validation, and test datasets

## Results

The model has achieved approximately **97% validation accuracy** in my experiments.

Results may vary depending on initialization, training configuration, and evaluation procedure.

## What I Learned

- How forward propagation works
- How to calculate gradients and implement backpropagation
- How momentum affects optimization
- How to train a neural network using mini-batches
- How to use CuPy for GPU computation
- How to evaluate a model using validation and test data

## Current Status

This is an educational project that I am developing to deepen my understanding of machine learning.

Future experiments may include network architecture improvements, model quantization, and eventually building a small language model from scratch.

## About

I'm documenting my progress as I learn to build AI systems from the ground up.

**This repository is a learning journey, not a finished production model.**

## Installation

### Requirements

- Python
- A compatible NVIDIA GPU and CUDA environment for GPU acceleration with CuPy
- Git

### Setup

Clone the repository:

```bash
git clone https://github.com/jeromeguilm-create/mnist-neural-network.git
cd mnist-neural-network
```

Create a virtual environment:

```bash
python -m venv .venv
```

Bypass PowerShell security:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Running the project
Running the training script
```bash
python training.py
```

Running the test script
```bash
python test.py
```

Running the visualisation script (Vibe coded because I couldn't be bothered to do it)
```bash
python visualisation.py
```
