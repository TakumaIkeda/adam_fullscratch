import math
import random
import matplotlib.pyplot as plt

def rosenbrock(x, y):
  return (x - 1) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(x, y):
  dldx = 2 * (x - 1) - 400 * x * (y - x ** 2)
  dldy = 200 * (y - x ** 2)
  return dldx, dldy

def momentum(lr=0.01, beta=0.9, x=0, y=0):
  i = 0
  history = []
  nu_x, nu_y = 0, 0

  while True:
    dldx, dldy = rosenbrock_grad(x, y)
    nu_x = beta * nu_x + (1 - beta) * dldx
    nu_y = beta * nu_y + (1 - beta) * dldy

    x -= lr * nu_x
    y -= lr * nu_y

    history.append((x, y))

    if i % 1000 == 0:
      print(f'iter: {i}, x: {x}, y: {y}, dldx: {dldx}, dldy: {dldy}, loss: {rosenbrock(x, y)}')

    i += 1

    if math.sqrt(dldx ** 2 + dldy ** 2) < 1e-6:
      break

    if i > 1e6:
      break

  return history

random.seed(0)
x_init = random.uniform(-2, 2)
y_init = random.uniform(-2, 2)

result = momentum(lr=0.001, x=x_init, y=y_init)

plt.figure()
plt.plot([math.log(rosenbrock(x, y)) for x, y in result])
plt.xlabel('iteration')
plt.ylabel('log(loss)')
plt.savefig('results/momentum.png')
