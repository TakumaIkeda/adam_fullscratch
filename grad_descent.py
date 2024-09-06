import math
import random
import matplotlib.pyplot as plt
import math

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

def grad_descent(lr=0.01, x=0, y=0):
  i = 0
  history = []
  while True:
    # dldx, dldy = rosenbrock_grad(x, y)
    # dldx, dldy = himmelblau_grad(x, y)
    dldx, dldy = ackley_grad(x, y)
    x -= lr * dldx
    y -= lr * dldy

    history.append((x, y))

    if i % 1000 == 0:
      # print(f'iter: {i}, x: {x}, y: {y}, dldx: {dldx}, dldy: {dldy}, loss: {himmelblau(x, y)}')
      print(f'iter: {i}, x: {x}, y: {y}, dldx: {dldx}, dldy: {dldy}, loss: {ackley(x, y)}')

    i += 1

    if math.sqrt(dldx ** 2 + dldy ** 2) < 1e-6:
      break

    if i > 1e6:
      break

  return history

random.seed(42)
x_init = random.uniform(-2, 2)
y_init = random.uniform(-2, 2)

result = grad_descent(lr=0.001, x=x_init, y=y_init)

# print(f"iter: {len(result)}, x: {result[-1][0]}, y: {result[-1][1]}, loss: {himmelblau(result[-1][0], result[-1][1])}")
print(f"iter: {len(result)}, x: {result[-1][0]}, y: {result[-1][1]}, loss: {ackley(result[-1][0], result[-1][1])}")

# plt.figure()
# plt.plot([math.log(himmelblau(x, y)) for x, y in result])
# plt.xlabel('iteration')
# plt.ylabel('log(loss)')
# plt.savefig('results/grad_descent.png')

plt.figure()
x = [x for x, y in result]
y = [y for x, y in result]
# マーカーはなし、線で結ぶ
plt.plot(x, y, marker='', linestyle='-')
X = [i / 10 for i in range(-30, 30)]
Y = [i / 10 for i in range(-30, 30)]
# Z = [[himmelblau(x, y) for x in X] for y in Y]
Z = [[ackley(x, y) for x in X] for y in Y]
plt.contour(X, Y, Z, levels=20)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('results/grad_descent_path.png')