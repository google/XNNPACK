# %%
# pylint: disable=all

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import differential_evolution


def poly_eval(c, x):
  # Ensure we use the dtype of the coefficients for calculation
  res = np.zeros_like(x, dtype=c.dtype)
  for i, v in enumerate(c):
    res += v * (x**i).astype(c.dtype)
  return res

def rational_approximation(
    f, x_min, x_max, p_degree, q_degree, dtype=np.float64, force_zero=False
):
  """Implementation of the Remez exchange algorithm.

  If force_zero=True, ensures the approximation is exact at x=0 by fixing p0 = f(0).
  """
  # If forced, we have one less degree of freedom to solve for
  n_points = p_degree + q_degree + (1 if force_zero else 2)

  # Initial guess for exchange points
  nodes = (
      0.5 * (x_min + x_max)
      + 0.5
      * (x_max - x_min)
      * np.cos(np.pi * np.arange(n_points - 1, -1, -1) / (n_points - 1))
  ).astype(dtype)

  best_p, best_q = None, None
  last_max_err = np.inf
  f0 = dtype(f(0))

  for iteration in range(25):
    A = np.zeros((n_points, n_points), dtype=dtype)
    b = np.zeros(n_points, dtype=dtype)
    y_nodes = f(nodes).astype(dtype)

    for i in range(n_points):
      if force_zero:
        # p0 is fixed to f0. System solves for p1...pn
        # P(x) - f(x)Q(x) = +/- E * Q(x)
        # (f0 + p1*x + ...) - f(x)(1 + q1*x + ...) = +/- E * (1 + q1*x + ...)
        # p1*x + ... - f(x)(q1*x + ...) - +/- E * (1 + q1*x + ...) = f(x) - f0
        for j in range(1, p_degree + 1):
          A[i, j-1] = nodes[i] ** j
        for j in range(1, q_degree + 1):
          A[i, p_degree + j - 1] = -y_nodes[i] * (nodes[i] ** j)
        A[i, -1] = -((-1) ** i)
        b[i] = y_nodes[i] - f0
      else:
        for j in range(p_degree + 1):
          A[i, j] = nodes[i] ** j
        for j in range(1, q_degree + 1):
          A[i, p_degree + j] = -y_nodes[i] * (nodes[i] ** j)
        A[i, -1] = -((-1) ** i)
        b[i] = y_nodes[i]

    try:
      sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
      sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    sol = sol.astype(dtype)
    if force_zero:
        p_coeffs = np.concatenate(([f0], sol[:p_degree]))
        q_coeffs = np.concatenate(([dtype(1.0)], sol[p_degree : -1]))
    else:
        p_coeffs = sol[: p_degree + 1]
        q_coeffs = np.concatenate(([dtype(1.0)], sol[p_degree + 1 : -1]))

    x_fine = np.linspace(x_min, x_max, 20000).astype(dtype)
    f_fine = f(x_fine).astype(dtype)
    approx_fine = poly_eval(p_coeffs, x_fine) / poly_eval(q_coeffs, x_fine)
    err_fine = approx_fine - f_fine
    abs_err = np.abs(err_fine)
    current_max_err = np.max(abs_err)

    if current_max_err >= last_max_err and iteration > 0:
      break

    last_max_err = current_max_err
    best_p, best_q = p_coeffs, q_coeffs

    new_nodes = []
    if abs_err[0] > abs_err[1]: new_nodes.append(x_fine[0])
    for i in range(1, len(x_fine) - 1):
      if abs_err[i] > abs_err[i - 1] and abs_err[i] > abs_err[i + 1]:
        new_nodes.append(x_fine[i])
    if abs_err[-1] > abs_err[-2]: new_nodes.append(x_fine[-1])

    if len(new_nodes) >= n_points:
      idx = np.argsort(np.abs(err_fine[np.searchsorted(x_fine, new_nodes)]))[-n_points:]
      nodes = np.sort([new_nodes[i] for i in idx])
    else:
      max_err_x = x_fine[np.argmax(abs_err)]
      closest_node_idx = np.argmin(np.abs(nodes - max_err_x))
      nodes[closest_node_idx] = max_err_x
      nodes = np.sort(nodes)

  return best_p, best_q


def print_polynomial(name, coeffs):
  # Determine precision and wrapper based on dtype
  if coeffs.dtype == np.float32:
    precision = 10
    suffix = "f"
    ty = "float"
  else:
    precision = 18
    suffix = ""
    ty = "double"

  # Reversing to put constant coefficient last (p_n, ..., p_1, p_0)
  rev_coeffs = coeffs[::-1]

  print(f"  std::array<{ty}, {len(rev_coeffs + 1)}> {name} = {{")
  for i, val in enumerate(rev_coeffs):
    print(f"      {val:.{precision}e}{suffix},")
  print("  };")


def plot_error(f, x, approx, title="Relative Error"):
  target = f(x)
  # Use a small epsilon for division safety if target is zero
  with np.errstate(divide="ignore", invalid="ignore"):
    error = (approx - target) / target

  # Mask out regions where target is zero if necessary, or use absolute error there
  error = np.where(target == 0, approx - target, error)

  max_err = np.max(np.abs(error))
  print(f"\nMaximum relative error: {max_err:.2e}")
  plt.figure(figsize=(10, 5))
  plt.plot(x, error)
  plt.axhline(0, color="black", lw=1, alpha=0.5)
  plt.title(f"{title} (Max Rel Error: {max_err:.2e})")
  plt.grid(True)
  plt.show()


# %%
# fp32 expm1
f = lambda x: np.expm1(x)
p_degree, q_degree = 3, 3
x_min, x_max = -0.5, 0.5
p, q = rational_approximation(
    f, x_min, x_max, p_degree, q_degree, dtype=np.float32, force_zero=True
)

print_polynomial("p", p)
print_polynomial("q", q)

# Evaluate final error
x_test = np.linspace(x_min, x_max, 5000)
approx = poly_eval(p, x_test) / poly_eval(q, x_test)
plot_error(f, x_test, approx)
# %%
# fp64 expm1
f = np.expm1
p_degree, q_degree = 6, 6
x_min, x_max = -0.5, 0.5
p, q = rational_approximation(
    f, x_min, x_max, p_degree, q_degree, dtype=np.float64, force_zero=True
)

print_polynomial("p", p)
print_polynomial("q", q)

# Evaluate final error
x_test = np.linspace(x_min, x_max, 5000)
approx = poly_eval(p, x_test) / poly_eval(q, x_test)
plot_error(f, x_test, approx)
# %%
import math

# fp64 log2(x + 1) ~= x + x^2*P(x)/Q(x)
f = lambda x: (np.log1p(x) - x) / x**2
p_degree, q_degree = 2, 2
x_min, x_max = np.sqrt(2) / 2 - 1, np.sqrt(2) - 1
p, q = rational_approximation(
    f, x_min, x_max, p_degree, q_degree, dtype=np.float32
)

print_polynomial("P", p)
print_polynomial("Q", q)

# Evaluate final error
x_test = np.linspace(x_min, x_max, 5000)
approx = x_test**2 * poly_eval(p, x_test) / poly_eval(q, x_test) + x_test
plot_error(lambda x: np.log1p(x), x_test, approx)
# %%
import math

# fp64 log2(x + 1) ~= x + x^2*P(x)/Q(x)
f = lambda x: (np.log1p(x) - x) / x**2
p_degree, q_degree = 5, 6
x_min, x_max = np.sqrt(2) / 2 - 1, np.sqrt(2) - 1
p, q = rational_approximation(
    f, x_min, x_max, p_degree, q_degree, dtype=np.float64
)

print_polynomial("P", p)
print_polynomial("Q", q)

# Evaluate final error
x_test = np.linspace(x_min, x_max, 5000)
approx = x_test**2 * poly_eval(p, x_test) / poly_eval(q, x_test) + x_test
plot_error(lambda x: np.log1p(x), x_test, approx)
# %%
