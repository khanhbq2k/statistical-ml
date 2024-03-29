from sympy import symbols, diff, solve, hessian, Matrix
import numpy as np
import matplotlib.pyplot as plt

# # Define symbols
# x1, x2 = symbols('x1 x2')
#
# # Define the function
# f = (x1 - x2 ** 2 - 3) ** 2 + (x1 - x2 - 1) ** 2
#
# # Calculate partial derivatives
# f_x1 = diff(f, x1)
# f_x2 = diff(f, x2)
#
# # Solve the system of equations for critical points
# critical_points = solve((f_x1, f_x2), (x1, x2))
#
# # Show the partial derivatives and critical points
# f_x1, f_x2, critical_points
#
# # Calculate the Hessian matrix
# H = hessian(f, (x1, x2))
#
# # Evaluate the Hessian matrix at the real critical point
# H_real_critical_point = H.subs({x1: 19 / 8, x2: 1 / 2})
#
# # Calculate the determinant of the Hessian matrix at the critical point
# det_H = H_real_critical_point.det()
#
# # Show the Hessian matrix at the critical point and its determinant
# H_real_critical_point, det_H


###
def f(x1, x2):
    return (x1 - x2 ** 2 - 3) ** 2 + (x1 - x2 - 1) ** 2


# Define the gradient of the function
def grad_f(x1, x2):
    df_x1 = 4 * x1 - 2 * x2 ** 2 - 2 * x2 - 8
    df_x2 = -2 * x1 - 4 * x2 * (x1 - x2 ** 2 - 3) + 2 * x2 + 2
    return np.array([df_x1, df_x2])


# Gradient descent function
def gradient_descent(x_start, learning_rate, n_steps):
    x_path = [x_start]
    x_current = x_start

    for _ in range(n_steps):
        grad = grad_f(x_current[0], x_current[1])
        x_current = x_current - learning_rate * grad
        x_path.append(x_current)

    return np.array(x_path)


# Starting point
x_start = np.array([-0.5, -0.5])

# Learning rates and steps
learning_rates = [0.01, 0.05]
n_steps = 10

# Calculate paths
paths = [gradient_descent(x_start, lr, n_steps) for lr in learning_rates]

print(paths)

# Plotting
x1_values = np.linspace(-1, 4, 900)
x2_values = np.linspace(-1, 4, 900)
X1, X2 = np.meshgrid(x1_values, x2_values)
F = f(X1, X2)

plt.figure(figsize=(10, 6))
contour = plt.contour(X1, X2, F, levels=np.logspace(0, 5, 35), cmap='viridis')
plt.clabel(contour, inline=True, fontsize=8)

# Plot paths
for path, lr in zip(paths, learning_rates):
    plt.plot(path[:, 0], path[:, 1], marker='o', label=f'λ = {lr}')

plt.title('Contour Plot of f(x) and Gradient Descent Paths')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.legend()
plt.show()
