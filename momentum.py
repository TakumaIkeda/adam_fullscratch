import math

def rosenbrock(x, y):
  return (x - 1) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(x, y):
  dldx = 2 * (x - 1) - 400 * x * (y - x ** 2)
  dldy = 200 * (y - x ** 2)
  return dldx, dldy

def momentum(lr=0.01, beta=0.9, x=0, y=0):
  nu_x_prev = 0
  nu_y_prev = 0

  i = 0
  while True:
    dldx, dldy = rosenbrock_grad(x, y)
    nu_x = beta * nu_x_prev + (1 - beta) * dldx
    nu_y = beta * nu_y_prev + (1 - beta) * dldy
    x -= lr * nu_x
    y -= lr * nu_y
    nu_x_prev = nu_x
    nu_y_prev = nu_y
    print(f'iter: {i}, x: {x}, y: {y}, dldx: {dldx}, dldy: {dldy}, loss: {rosenbrock(x, y)}')
    if math.sqrt(dldx ** 2 + dldy ** 2) < 1e-5:
      break
    i += 1

momentum(lr=0.03, beta=0.9, x=0, y=0)