# Gradient Descent Optimization

From-scratch implementations and numerical comparisons of three fundamental optimization algorithms used in machine learning:

- Batch Gradient Descent (BGD)
- Stochastic Gradient Descent (SGD)
- Mini-Batch Gradient Descent (MBGD)

## Overview

Gradient descent is an iterative optimization method for minimizing differentiable objective functions. This project investigates how different gradient estimation strategies affect convergence when fitting a linear regression model.

The algorithms are implemented using NumPy and evaluated on synthetic data with known regression parameters.

## Mathematical Formulation

Consider the linear regression model

y = Xθ + ε,

where X is the design matrix, θ is the parameter vector, and ε represents random noise.

We minimize the mean squared error objective:

J(θ) = (1 / 2n) ||Xθ - y||².

Its gradient is:

∇J(θ) = (1 / n) Xᵀ(Xθ - y).

The general gradient descent update is:

θₖ₊₁ = θₖ - α∇J(θₖ),

where α is the learning rate.

## Implemented Algorithms

### Batch Gradient Descent

Computes the gradient using the entire training dataset at every update. This produces deterministic updates for a fixed dataset and initialization.

### Stochastic Gradient Descent

Uses one randomly selected training observation per update. This reduces the cost of individual updates but introduces variability into the optimization trajectory.

### Mini-Batch Gradient Descent

Uses a subset of training observations for each update, balancing computational efficiency with gradient stability.

## Numerical Experiments

The notebook includes:

1. Synthetic linear regression data generated from a known model.
2. Training and comparison of all three optimizers.
3. Convergence curves showing objective values during training.
4. Visual comparisons of the fitted regression lines.
5. Comparison against NumPy's least-squares solution.
6. Automated checks of parameter recovery, gradient correctness, and reproducibility.

## Running the Project

Install the required Python packages:

```bash
python -m pip install numpy matplotlib pandas jupyter
```

Open the notebook:

```bash
jupyter notebook gradient-descent-optimization.ipynb
```

Run the notebook cells in order to reproduce the experiments.

## Project Structure

```text
gradient-descent-optimization/
├── README.md
└── gradient-descent-optimization.ipynb
```

## Key Learning Objectives

- Understand gradient-based optimization from first principles.
- Compare full-gradient and stochastic parameter updates.
- Investigate convergence and the effect of gradient noise.
- Validate numerical implementations against an analytical benchmark.
- Develop reproducible optimization experiments.

## Notes

The experiments use synthetic linear regression data to demonstrate and validate the algorithms. Their performance on this example does not establish convergence guarantees for arbitrary nonconvex objectives.
