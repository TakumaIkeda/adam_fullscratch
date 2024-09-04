import math

def rosenbrock(x, y):
  return (x - 1) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_grad(x, y):
  dldx = 2 * (x - 1) - 400 * x * (y - x ** 2)
  dldy = 200 * (y - x ** 2)
  return dldx, dldy

def adam(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-12, x=0, y=0):
  nu_x = 0
  nu_y = 0
  s_x = 0
  s_y = 0
  nu_x_prev = 0
  nu_y_prev = 0
  s_x_prev = 0
  s_y_prev = 0

  i = 0
  while True:
    dldx, dldy = rosenbrock_grad(x, y)
    nu_x = beta1 * nu_x_prev + (1 - beta1) * dldx
    nu_y = beta1 * nu_y_prev + (1 - beta1) * dldy
    s_x = beta2 * s_x_prev + (1 - beta2) * dldx ** 2
    s_y = beta2 * s_y_prev + (1 - beta2) * dldy ** 2
    x -= lr * nu_x / (s_x + epsilon) ** 0.5
    y -= lr * nu_y / (s_y + epsilon) ** 0.5
    nu_x_prev = nu_x
    nu_y_prev = nu_y
    s_x_prev = s_x
    s_y_prev = s_y
    print(f'iter: {i}, x: {x}, y: {y}, dldx: {dldx}, dldy: {dldy}, loss: {rosenbrock(x, y)}')
    if math.sqrt(dldx ** 2 + dldy ** 2) < 1e-5:
      break
    i += 1

adam(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, x=0, y=0)