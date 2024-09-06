import math
import random
import matplotlib.pyplot as plt
import numpy as np

def rosenbrock(x, y):
  return (x - 1) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(x, y):
  dldx = 2 * (x - 1) - 400 * x * (y - x ** 2)
  dldy = 200 * (y - x ** 2)
  return dldx, dldy

def himmelblau(x, y):
  return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

def himmelblau_grad(x, y):
  dldx = 2 * (2 * x * (x ** 2 + y - 11) + x + y ** 2 - 7)
  dldy = 2 * (x ** 2 + 2 * y * (x + y ** 2 - 7) + y - 11)
  return dldx, dldy

def ackley(x, y):
  return -20 * math.exp(-0.2 * math.sqrt(0.5 * (x ** 2 + y ** 2))) - math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))) + 20 + math.e

def ackley_grad(x, y):
  dldx = 2 * math.pi * math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))) * math.sin(2 * math.pi * x) + 2 * x * math.exp(-0.2 * math.sqrt(0.5 * (x ** 2 + y ** 2))) / math.sqrt(0.5 * (x ** 2 + y ** 2))
  dldy = 2 * math.pi * math.exp(0.5 * (math.cos(2 * math.pi * x) + math.cos(2 * math.pi * y))) * math.sin(2 * math.pi * y) + 2 * y * math.exp(-0.2 * math.sqrt(0.5 * (x ** 2 + y ** 2))) / math.sqrt(0.5 * (x ** 2 + y ** 2))
  return dldx, dldy

def adam(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-12, x=0, y=0):
  i = 0
  history = []
  nu_x = 0
  nu_y = 0
  s_x = 0
  s_y = 0
  while True:
    # dldx, dldy = rosenbrock_grad(x, y)
    dldx, dldy = himmelblau_grad(x, y)
    # dldx, dldy = ackley_grad(x, y)
    nu_x = beta1 * nu_x + (1 - beta1) * dldx
    nu_y = beta1 * nu_y + (1 - beta1) * dldy
    s_x = beta2 * s_x + (1 - beta2) * dldx ** 2
    s_y = beta2 * s_y + (1 - beta2) * dldy ** 2
    x -= lr * nu_x / (s_x + epsilon) ** 0.5
    y -= lr * nu_y / (s_y + epsilon) ** 0.5

    history.append((x, y))

    if i % 1000 == 0:
      # print(f'iter: {i}, x: {x}, y: {y}, dldx: {dldx}, dldy: {dldy}, loss: {rosenbrock(x, y)}')
      print(f'iter: {i}, x: {x}, y: {y}, dldx: {dldx}, dldy: {dldy}, loss: {himmelblau(x, y)}')
      # print(f'iter: {i}, x: {x}, y: {y}, dldx: {dldx}, dldy: {dldy}, loss: {ackley(x, y)}')
    i += 1

    if math.sqrt(dldx ** 2 + dldy ** 2) < 1e-5:
      break

    if i > 1e6:
      break

  return history

results = []
random.seed(42)
for _ in range(10):
  x_init = random.uniform(-6, 6)
  y_init = random.uniform(-6, 6)
  results.append(adam(lr=0.01, x=x_init, y=y_init))
  print(f"iter: {len(results[-1])}, x: {results[-1][-1][0]}, y: {results[-1][-1][1]}, loss: {himmelblau(results[-1][-1][0], results[-1][-1][1])}")
# print(f"iter: {len(result)}, x: {result[-1][0]}, y: {result[-1][1]}, loss: {rosenbrock(result[-1][0], result[-1][1])}")
# print(f"iter: {len(result)}, x: {result[-1][0]}, y: {result[-1][1]}, loss: {ackley(result[-1][0], result[-1][1])}")

# plt.figure()
# plt.plot([math.log(rosenbrock(x, y)) for x, y in result])
# plt.xlabel('iteration')
# plt.ylabel('log(loss)')
# plt.savefig('results/adam.png')

# for rosenbrock
# Define grid
# x1 = np.linspace(-2, 2, 400)
# x2 = np.linspace(-2, 2, 400)
# X1, X2 = np.meshgrid(x1, x2)
# Z = rosenbrock(X1, X2)

# Plot
# plt.figure(figsize=(8, 6))
# contour = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
# plt.colorbar(contour)
# plt.title('2D Rosenbrock Function')
# plt.title('2D Himmelblau Function')
# plt.xlabel('$x_1$')
# plt.ylabel('$x_2$')
# x = [x for x, y in result]
# y = [y for x, y in result]
# plt.plot(x, y, color='red')
# plt.savefig('results/adam_path.png')

# for himmelblau
# Define grid
x1 = np.linspace(-6, 6, 400)
x2 = np.linspace(-6, 6, 400)
X1, X2 = np.meshgrid(x1, x2)
Z = himmelblau(X1, X2)

# Plot
plt.figure(figsize=(8, 6))
contour = plt.contourf(X1, X2, Z, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.title('2D Himmelblau Function')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
for result in results:
  x = [x for x, y in result]
  y = [y for x, y in result]
  plt.plot(x, y, color='red')
plt.savefig('results/adam_path.png')
