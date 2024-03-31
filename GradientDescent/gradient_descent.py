import numpy as np


# Function to compute the gradient
def gradient(x):
    x1, x2 = x
    grad_x1 = 2 * (x1 - x2 ** 2 - 3) + 2 * (x1 - x2 - 1)
    grad_x2 = 2 * (x1 - x2 ** 2 - 3) * (-2 * x2) + 2 * (x1 - x2 - 1) * (-1)
    return np.array([grad_x1, grad_x2])


# Gradient descent parameters
lambda_ = 0.2  # leaning rate
x_0 = np.array([-0.5, -0.5])  # starting vector

# Perform gradient descent for k = 1, 2
x_1 = x_0 - lambda_ * gradient(x_0)
x_2 = x_1 - lambda_ * gradient(x_1)

print(x_1, x_2)
