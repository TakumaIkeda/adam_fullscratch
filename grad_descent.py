import math

def rosenbrock(x, y):
  return (x - 1) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(x, y):
  dldx = 2 * (x - 1) - 400 * x * (y - x ** 2)
  dldy = 200 * (y - x ** 2)
  return dldx, dldy

def grad_descent(lr=0.01, x=0, y=0):
  i = 0
  while True:
    dldx, dldy = rosenbrock_grad(x, y)
    x -= lr * dldx
    y -= lr * dldy

    print(f'iter: {i}, x: {x}, y: {y}, dldx: {dldx}, dldy: {dldy}, loss: {rosenbrock(x, y)}')
    if math.sqrt(dldx ** 2 + dldy ** 2) < 1e-5:
      break
    i += 1

grad_descent(lr=0.002, x=0, y=0)