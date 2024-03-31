import numpy as np
import matplotlib.pyplot as plt


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

# Another way to plot
# x1_values = np.linspace(-1, 4, 400)
# x2_values = np.linspace(-1, 4, 400)
# X1, X2 = np.meshgrid(x1_values, x2_values)
# F = f(X1, X2)
#
# plt.figure(figsize=(8, 5))
# contour = plt.contourf(X1, X2, F, levels=np.linspace(F.min(), F.max(), 50), cmap='coolwarm')
# plt.colorbar(contour)
# plt.clabel(contour, inline=True, fontsize=8, fmt='%1.0f')
#
# # Plot paths with modified style
# for path, lr in zip(paths, learning_rates):
#     plt.scatter(path[:, 0], path[:, 1], marker='x', s=100, label=f'LR = {lr}')
#
# plt.title('Modified Contour Plot and Gradient Descent Trajectories')
# plt.xlabel('X-axis ($x_1$)')
# plt.ylabel('Y-axis ($x_2$)')
# plt.legend(title='Learning Rates')
# plt.grid(True, linestyle='--', color='grey', alpha=0.5)
# plt.show()
