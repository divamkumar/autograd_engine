# Neural Network from Scratch with Custom Autograd Engine

This is a personal project for me to dive deep into the fundamentals of deep learning by building a complete Multi-Layer Perceptron (MLP) and its underlying automatic differentiation (autograd) engine entirely from basic Python operations.

This project is heavily inspired by [Andrej Karpathy's "micrograd" video series](https://www.youtube.com/watch?v=VMj-3S1tku0), as well as his [micrograd GitHub repository](https://github.com/karpathy/micrograd/tree/master). I essentially re-implemented his project to bolster my own understanding by adding my notes and coding this in my own style.

## Features

* **Custom Autograd Engine (`ZeroDimTensor`):**
    * Implements a custom `ZeroDimTensor` class to wrap scalar numerical values.
    * Automatically builds a dynamic computational graph by tracking operations (addition, multiplication, subtraction, division, power, tanh).
    * Calculates and stores gradients for all participating tensors via the `backward()` method.

* **Neural Network Components:**
    * **`Perceptron` Class:** A single neuron implementation with learnable weights and bias, and `tanh` activation.
    * **`Layer` Class:** A collection of `Perceptron`s, representing a single dense layer in a neural network.
    * **`MultiLayerPerceptron` (MLP) Class:** Stacks multiple `Layer`s to form a complete feedforward neural network.

* **End-to-End Training:**
    * Demonstrates the full training loop: Forward Pass, Loss Calculation (Mean Squared Error), Backward Pass (Backpropagation), and Parameter Updates (Batch Gradient Descent).

## How It Works

At its heart, the project revolves around the `ZeroDimTensor` class. Every scalar value (input data, weights, biases, intermediate results) is wrapped in a `ZeroDimTensor` object.

1.  **Computational Graph Construction:** As mathematical operations are performed on `ZeroDimTensor` objects (e.g., `a + b`, `x * w`), a directed acyclic graph (DAG) is implicitly built. Each `ZeroDimTensor` stores references to its immediate parents and the operation that created it.

2.  **Forward Pass:** When `model(x_input)` is called, data flows from the initial input `ZeroDimTensor`s through all layers, computing intermediate `ZeroDimTensor` values until the final loss `ZeroDimTensor` is produced at the root of the graph.

3.  **Backward Pass (Backpropagation):** Calling `.backward()` on the final loss `ZeroDimTensor` initiates the magic. It traverses the computational graph in reverse topological order, applying the chain rule at each node to compute and accumulate the gradient of the loss with respect to every contributing `ZeroDimTensor`.

4.  **Gradient Descent Optimization:** The computed gradients for the model's learnable parameters (weights and biases) are then used to iteratively adjust their values, taking small steps in the direction that minimizes the overall loss.

### Understanding Gradient Flow in This Model
The neural network in this project uses `tanh` activation functions for 2 of its hidden layers. `tanh` can contribute to the [Vanishing Gradient Problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) in deep neural networks, since the backward pass can cause the gradients to become small when neurons generate extreme values in the forward pass. Outputting logits creates a linear output layer, ensuring that the initial gradient entering the network during backpropagation does not immediately get reduced. Combined with small initial weights preventing immediate saturataion, this setup allows the model to still learn properly despite the possibility of vanishing gradients.

## Getting Started

This project is designed to be straightforward to run.

**Prerequisites:**

* Python 3.x
* Windows Subsystem for Linux (WSL) with Ubuntu (or any Linux distribution) installed.
* A web browser on Windows (Chrome, Edge, Firefox, etc.).

**Running the Code with Jupyter Notebook (Recommended):**

This project is best explored interactively using Jupyter Notebook.

1.  **Clone the repository:**
    Open your WSL Ubuntu terminal and clone the repository:
    ```
    git clone [https://github.com/divamkumar/nn_and_autograd_from_scratch](https://github.com/divamkumar/nn_and_autograd_from_scratch)
    cd nn_and_autograd_from_scratch
    ```

2.  **Set up a Python Virtual Environment (Recommended):**
    A virtual environment keeps your project dependencies isolated.
    ```
    # Create the virtual environment (replace 'venv_name' with a name you prefer)
    python3 -m venv venv_name
    
    # Activate the virtual environment
    source venv_name/bin/activate
    ```
    (You'll need to run `source venv_nn/bin/activate` every time you open a new WSL terminal and want to work on this project.)

3.  **Install Jupyter Notebook:**
    With your virtual environment activated, install Jupyter:
    ```
    pip install jupyter
    ```

4.  **Launch Jupyter Notebook from WSL:**
    Navigate to the project directory (if you're not already there after cloning) and launch Jupyter.
    ```
    cd ~/nn_and_autograd_from_scratch # Make sure you are in the project root
    jupyter notebook --no-browser --port 8888
    ```
    * `--no-browser`: Prevents Jupyter from trying to open a browser within WSL.
    * `--port 8888`: Specifies the port. You can change this if 8888 is in use (e.g., `--port 8889`).

5.  **Access Jupyter in Your Windows Browser:**
    After running the command, you will see output in your WSL terminal similar to this:
    ```
    [I 20XX-YY-ZZ HH:MM:SS.ms NotebookApp] Serving notebooks from local directory: /home/youruser/nn_and_autograd_from_scratch
    [I 20XX-YY-ZZ HH:MM:SS.ms NotebookApp] The Jupyter Notebook is running at:
    [I 20XX-YY-ZZ HH:MM:SS.ms NotebookApp] http://localhost:8888/?token=YOUR_UNIQUE_TOKEN_STRING_HERE
    [I 20XX-YY-ZZ HH:MM:SS.ms NotebookApp]  or [http://127.0.0.1:8888/?token=YOUR_UNIQUE_TOKEN_STRING_HERE](http://127.0.0.1:8888/?token=YOUR_UNIQUE_TOKEN_STRING_HERE)
    ```
    **Copy the entire URL** (including the `token=...` part) and paste it into any web browser on your Windows machine.
    The first time, you might be prompted to set a password using the token.

6.  **Open `main.py` (or a `.ipynb` file):**
    Once in the Jupyter interface, navigate to `main.py` (or create a new notebook) and copy/paste your project code into a Jupyter cell to run it interactively.

## Project Structure

* `ZeroDimTensor` Class: The core autograd engine for scalar values.
* `Perceptron` Class: Represents a single neuron.
* `Layer` Class: Groups multiple `Perceptron`s.
* `MultiLayerPerceptron` Class: Orchestrates multiple `Layer`s to form the full neural network.
* Training Loop: Demonstrates how to use these classes to train an MLP on a simple binary classification dataset.

## Concepts Demonstrated

This project provides a hands-on understanding of:

* **Automatic Differentiation (Autograd):** The principle behind modern ML frameworks.
* **Computational Graphs:** How operations are chained and tracked.
* **Forward Pass:** Computing predictions.
* **Backward Pass / Backpropagation:** Efficiently calculating gradients.
* **Gradients and the Chain Rule:** The mathematical backbone of learning.
* **Learnable Parameters:** Weights and Biases.
* **Loss Functions:** Quantifying model error (e.g., Mean Squared Error).
* **Gradient Descent:** The optimization algorithm.
* **Activation Functions:** Their role in introducing non-linearity (`tanh`).
* **Saturation and Vanishing Gradients:** Understanding why `tanh` can cause issues in deep networks and how linear output layers help.
* **Logits:** The raw outputs of the final layer before final non-linear transformations.

## Future Enhancements (Ideas for Expansion)

* **Multi-Dimensional Tensors:** Generalize `ZeroDimTensor` to handle `numpy.ndarray` for efficient vector/matrix operations (like a true PyTorch/TensorFlow `Tensor` class).
* **More Activation Functions:** Implement ReLU, Sigmoid, Leaky ReLU, etc.
* **Optimizers:** Add more sophisticated optimizers (e.g., SGD with momentum, Adam).
* **Batching:** Implement mini-batch training for larger datasets.
* **Model Saving/Loading:** Add functionality to save and load trained model parameters.
* **Visualization:** Graph the computational tree or plot loss curves.
* **Different Loss Functions:** Implement Cross-Entropy for classification.
