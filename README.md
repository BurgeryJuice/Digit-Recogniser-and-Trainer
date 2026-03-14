# MNIST from Scratch: A Deep Dive into Vectorized Backpropagation

This repository contains a manual implementation of a 3L Neural Network (3 layers) to solve the MNIST digit recognition challenge. Built using only **NumPy**, this project focuses on the underlying mathematics of gradient descent and weight optimization without the use of high-level frameworks like PyTorch or TensorFlow.

## 🧠 Architectural Overview
The model follows a $(784 \rightarrow 16 \rightarrow 16 \rightarrow 10)$ architecture:
* **Input Layer:** 784 neurons (28x28 flattened pixels).
* **Hidden Layers:** 2 layers with 16 neurons each, utilizing **ReLU** activation to mitigate the vanishing gradient problem.
* **Output Layer:** 10 neurons with **Sigmoid** activation (for multi-class probability mapping).
* **Initialization:** He Initialization ($W = \text{randn} \cdot \sqrt{2/n}$) to maintain variance across layers and ensure stable signal flow.

## 📉 Mathematical Implementation
I implemented the full backward pass using vectorized calculus to handle the dataset efficiently. The core update logic follows the chain rule for partial derivatives:

$$DZ_3 = A_3 - Y$$
$$DW_3 = \frac{1}{m} \cdot \text{np.dot}(DZ_3, A_2^T)$$
$$DZ_2 = \text{np.dot}(W_3^T, DZ_3) \odot \sigma'(Z_2)$$

> **Note:** The backpropagation engine explicitly handles the derivative of the ReLU function ($Z > 0$) to propagate errors back to the initial weights.

## 📊 Performance
* **Accuracy:** Currently achieving **~84%**.
* **Limitation:** Accuracy is constrained by the low hidden layer dimension (16 neurons) and the use of Sigmoid on the output layer instead of Softmax. 
* **Optimization:** Stochastic Gradient Descent (SGD) with a learning rate ($\alpha$) of $0.1$.

## 🛠️ How to Run
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/BurgeryJuice/](https://github.com/BurgeryJuice/)[YOUR_REPO_NAME]
