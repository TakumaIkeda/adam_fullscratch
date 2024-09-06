import numpy as np
import matplotlib.pyplot as plt

# Griewank function definition
def griewank(x1, x2):
    term1 = 1 + (x1**2 + x2**2) / 4000
    term2 = np.cos(x1 / np.sqrt(1)) * np.cos(x2 / np.sqrt(2))
    return term1 - term2

# Define grid
x1 = np.linspace(-5, 5, 400)
x2 = np.linspace(-5, 5, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = griewank(X1, X2)

# Plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.title('2D Griewank Function')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.savefig('results/griewank.png')
