from sympy import symbols, diff, solve, hessian, Matrix
import numpy as np
import matplotlib.pyplot as plt

# Define symbols
x1, x2 = symbols('x1 x2')

# Define the function
f = (x1 - x2 ** 2 - 3) ** 2 + (x1 - x2 - 1) ** 2

# Calculate partial derivatives
f_x1 = diff(f, x1)
f_x2 = diff(f, x2)

# Solve the system of equations for critical points
critical_points = solve((f_x1, f_x2), (x1, x2))

# Show the partial derivatives and critical points
print("f_x1: ", f_x1)
print("f_x2: ", f_x2)
print("Critical_points: ", critical_points)

# Calculate the Hessian matrix
H = hessian(f, (x1, x2))

# Evaluate the Hessian matrix at the real critical point
H_real_critical_point = H.subs({x1: 19 / 8, x2: 1 / 2})

# Calculate the determinant of the Hessian matrix at the critical point
det_H = H_real_critical_point.det()

# Show the Hessian matrix at the critical point and its determinant
print("Hessian matrix: ", H_real_critical_point, ", det_H:", det_H)
