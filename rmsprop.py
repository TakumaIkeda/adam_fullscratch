import math

def rosenbrock(x, y):
  return (x - 1) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(x, y):
  dldx = 2 * (x - 1) - 400 * x * (y - x ** 2)
  dldy = 200 * (y - x ** 2)
  return dldx, dldy

def rmsprop(lr=0.01, beta2=0.999, epsilon=1e-12, x=0, y=0):
  s_x_prev = 0
  s_y_prev = 0

  i = 0
  while True:
    dldx, dldy = rosenbrock_grad(x, y)
    s_x = beta2 * s_x_prev + (1 - beta2) * dldx ** 2
    s_y = beta2 * s_y_prev + (1 - beta2) * dldy ** 2
    x -= lr * dldx / (s_x + epsilon) ** 0.5
    y -= lr * dldy / (s_y + epsilon) ** 0.5
    s_x_prev = s_x
    s_y_prev = s_y
    print(f'iter: {i}, x: {x}, y: {y}, dldx: {dldx}, dldy: {dldy}, loss: {rosenbrock(x, y)}')
    if math.sqrt(dldx ** 2 + dldy ** 2) < 1e-5:
      break
    i += 1

rmsprop(lr=1e-5, beta2=0.999, epsilon=1e-12, x=0, y=0)