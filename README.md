# Regularization Techniques with Dropout in PyTorch (Fashion-MNIST)

This project documents my practical exploration of **regularization techniques in deep learning**, with a strong focus on **Dropout**, applied to an image classification task using **Fashion-MNIST** and PyTorch. The work combines architectural regularization, data augmentation, optimizer-level regularization, early stopping, and uncertainty estimation.

---
## Objective

The objective of this work is to:
- Reduce overfitting in deep neural networks
- Study the impact of Dropout at different stages of a model
- Apply optimizer-based regularization (L2 / weight decay)
- Explore uncertainty estimation using Monte Carlo Dropout
- Combine multiple regularization strategies in a single training pipeline

---

## Dataset – Fashion-MNIST

- Dataset: Fashion-MNIST
- Images: 28 × 28 grayscale clothing images
- Classes: 10 fashion categories
- Training and validation datasets loaded separately

---

## Data Augmentation and Preprocessing

### Training Transformations
To improve generalization, the following augmentations are applied:
- Random horizontal flip
- Random rotation
- Random affine transformations (translation)
- Normalization

These augmentations simulate variations such as noise, slight shifts, and distortions.

### Validation Transformations
- Only tensor conversion and normalization
- No augmentation applied to validation data

---

## Model Architecture – Regularized CNN

A convolutional neural network is implemented with built-in regularization:

### Feature Extractor
- Convolution → BatchNorm → ReLU → MaxPooling (2 blocks)
- Feature maps expanded from 32 → 64 channels
- Spatial downsampling via max pooling
- Flattened output for classification

### Classifier
- Fully connected layer (64 × 7 × 7 → 512)
- ReLU activation
- Dropout applied to randomly deactivate neurons
- Final linear layer with 10 outputs

Dropout is used only in the classifier to reduce co-adaptation.

---

## Loss Function and Regularization Strategy

### Loss Function
- CrossEntropyLoss with **label smoothing**
- Reduces overconfidence in predictions

### L2 Regularization (Weight Decay)
- Applied using AdamW optimizer
- Weight decay applied **only to weights**
- Biases and BatchNorm parameters are excluded

This avoids over-penalizing parameters that should not be regularized.

### L1 Regularization (Experimented)
- L1 regularization was tested
- Disabled due to reduced accuracy and underfitting

---

## Training Strategy

- Optimizer: AdamW
- Learning Rate Scheduler: ReduceLROnPlateau
- Scheduler reduces LR when validation loss stagnates
- Early stopping implemented with configurable patience
- Best model weights saved and restored based on validation loss

This ensures stable training and prevents overfitting.

---

## Evaluation Metrics

- Training loss
- Validation loss
- Validation accuracy
- Learning rate changes logged during training

---

## Alternative Dropout-Heavy Architecture

An alternative fully connected architecture is defined:
- Dropout applied before every linear layer
- Demonstrates that excessive dropout can lead to underfitting
- Used for comparison and experimentation

---

## Monte Carlo Dropout for Uncertainty Estimation

To estimate prediction uncertainty:

- Dropout layers are explicitly enabled during inference
- Multiple stochastic forward passes are performed
- Mean and standard deviation of class probabilities are computed

For each image:
- Mean probability represents confidence
- Standard deviation represents uncertainty

Predictions with high uncertainty are flagged as unreliable.

---

## Custom Monte Carlo Dropout Wrapper

A reusable `MonteCarloClassifier` class is implemented:
- Automates repeated forward passes
- Returns mean probabilities and uncertainty estimates
- Encapsulates dropout-based uncertainty logic

---

## Max-Norm Regularization (Experimental)

- A custom max-norm constraint function is implemented
- Limits the L2 norm of weights to a fixed threshold
- Prevents individual neurons from dominating training
- Intended to complement dropout-based regularization

---

## Key Observations

- Data augmentation significantly improves generalization
- Dropout reduces overfitting when applied selectively
- Excessive dropout can cause underfitting
- Weight decay is more effective when decoupled from optimization
- Monte Carlo Dropout provides meaningful uncertainty estimates
- Combining multiple regularization techniques leads to more stable models

---

## Conclusion

This project demonstrates a comprehensive approach to regularization in deep learning using PyTorch. By combining architectural regularization, optimizer-level penalties, data augmentation, early stopping, and uncertainty estimation, the work provides a deep practical understanding of how overfitting can be controlled in real-world image classification tasks.

---
