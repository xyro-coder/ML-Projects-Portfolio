# 🧠 Deep Neural Network from Scratch for MNIST Digit Classification

This project implements a fully vectorized deep feedforward neural network built entirely from first principles using NumPy, and trained to classify handwritten digits from the MNIST dataset. The model achieves an accuracy of ~91% on the training set without the use of any high-level deep learning libraries, serving both as a powerful educational tool and a demonstration of fundamental machine learning principles.

## 🔍 Overview

The implementation includes:

- Full support for arbitrary-layer neural networks
- Sigmoid activation functions in hidden layers
- Softmax output layer for multi-class probability prediction
- Categorical cross-entropy loss computation
- A custom implementation of the Adam optimizer with bias correction
- Layer-wise backpropagation using efficient matrix operations

## 📊 Performance

After training for 200 epochs on normalized MNIST images, the model achieves:

- Training Accuracy: ~91%
- Loss Function: Categorical Cross-Entropy
- Optimization: Custom Adam Optimizer with momentum and RMS scaling

## 📚 Dataset

- Source: `fetch_openml('mnist_784')` via scikit-learn
- Samples: 70,000 grayscale images of handwritten digits (28×28 pixels)
- Preprocessing:
  - Pixel values scaled to the [0, 1] range
  - Labels one-hot encoded

## 🧠 Network Architecture

```
Input Layer       : 784 units (28x28 flattened)
Hidden Layer 1    : 256 neurons, Sigmoid activation
Hidden Layer 2    : 128 neurons, Sigmoid activation
Output Layer      : 10 neurons, Softmax activation (digit classes 0–9)
```

Architecture is modular and defined via the `sizes` parameter passed during initialization.

## 🛠 Technologies

- Python 3.10+
- NumPy – For matrix operations and numerical computing
- Scikit-learn – For dataset loading and one-hot encoding

Note: torch is imported but not utilized in the actual model logic.

## ⚙️ Core Components

### Network.__init__(sizes, X, y)
Initializes weight matrices and bias vectors using Gaussian sampling for each layer.

### Forward_propagation()
Performs layer-wise forward passes:
- Linear transformation: z = W⋅a + b
- Non-linear activation: Sigmoid for hidden layers, Softmax for output

### Cost()
Computes numerically stable cross-entropy loss between predictions and labels.

### backwards()
Implements backpropagation to compute gradients for each layer:
- Supports matrix-based gradient calculation
- Accumulates weight and bias gradients in reverse order

### optimizer(...)
Custom implementation of the Adam optimization algorithm:
- Momentum estimation (β₁) and RMS scaling (β₂)
- Bias correction to stabilize initial updates
- Efficient parameter updates using adaptive learning rates

### train()
Executes the full training loop across a configurable number of epochs.

### accuracy()
Computes classification accuracy by comparing predicted and actual labels.

## 🚀 Example Usage

```python
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Load and preprocess data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y.reshape(-1, 1)).toarray()
X = X / 255.0  # Normalize to [0, 1]

# Initialize and train the model
nn = Network([784, 256, 128, 10], X, y_encoded)
nn.train()

# Evaluate accuracy
print("Training Accuracy:", nn.accuracy())
```

## 📈 Training Log (Sample Output)

```
Epoch 0/200 | Iteration 0 | Cost: 2.302513
...
Epoch 199/200 | Iteration 199 | Cost: 0.129874
Training complete!
Training Accuracy: 0.9112
```

## 🧪 Future Improvements

- Implement support for mini-batch gradient descent
- Integrate L2 regularization or dropout to reduce overfitting
- Add early stopping and validation set tracking
- Incorporate training curves for loss and accuracy visualization

## 📄 License

This project is open-source under the MIT License.

## 🧠 Final Notes

This project demonstrates a ground-up approach to understanding neural networks through code. It is ideal for learners, educators, and practitioners who want to demystify the internal mechanics of deep learning by building every component manually. The achieved 91% accuracy proves that with fundamental tools and mathematical principles, one can build powerful machine learning models without any black-box abstractions.