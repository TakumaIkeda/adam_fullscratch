import math
import random
import matplotlib.pyplot as plt

def rosenbrock(x, y):
  return (x - 1) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(x, y):
  dldx = 2 * (x - 1) - 400 * x * (y - x ** 2)
  dldy = 200 * (y - x ** 2)
  return dldx, dldy

def rmsprop(lr=0.01, beta2=0.999, epsilon=1e-12, x=0, y=0):
  i = 0
  history = []
  s_x, s_y = 0, 0
  while True:
    dldx, dldy = rosenbrock_grad(x, y)
    s_x = beta2 * s_x + (1 - beta2) * dldx ** 2
    s_y = beta2 * s_y + (1 - beta2) * dldy ** 2
    x -= lr * dldx / (s_x + epsilon) ** 0.5
    y -= lr * dldy / (s_y + epsilon) ** 0.5

    history.append((x, y))

    if i % 1000 == 0:
      print(f'iter: {i}, x: {x}, y: {y}, dldx: {dldx}, dldy: {dldy}, loss: {rosenbrock(x, y)}')


    i += 1
    if math.sqrt(dldx ** 2 + dldy ** 2) < 1e-5:
      break
    if i > 1e6:
      break

  return history

random.seed(0)
x_init = random.uniform(-2, 2)
y_init = random.uniform(-2, 2)

result = rmsprop(lr=0.001, beta2=0.99999, epsilon=1e-8, x=x_init, y=y_init)

plt.figure()
plt.plot([math.log(rosenbrock(x, y)) for x, y in result])
plt.xlabel('iteration')
plt.ylabel('log(loss)')
plt.savefig('results/rmsprop.png')
