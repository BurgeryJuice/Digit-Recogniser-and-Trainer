# MNIST from Scratch: A Deep Dive into Vectorized Backpropagation

This repository contains the manual implementation of a 3L Neural Network (3 layers) to solve the MNIST digit recognition challenge. Built using only **NumPy**, this project focuses on the underlying mathematics of gradient descent and weight optimization without the use of high-level frameworks like PyTorch or TensorFlow.



##Mathematical Implementation
I implemented the full backward pass using vectorized calculus to handle the dataset efficiently. The core update logic follows the chain rule for partial derivatives.


##  Performance
* **Accuracy:** Currently  **~84%**.
* **Limitation:** Accuracy is constrained by the low hidden layer dimension (16 neurons) and the use of Sigmoid on the output layer instead of other functions like Softmax. 


## How to Run
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/BurgeryJuice/Digit-Recogniser-and-Trainer.git](https://github.com/BurgeryJuice/Digit-Recogniser-and-Trainer.git)
