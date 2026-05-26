# %%
# pylint: disable=all

# %%
import matplotlib.pyplot as plt
import numpy as np


def poly_eval(c, x):
  # Ensure we use the dtype of the coefficients for calculation
  res = np.zeros_like(x, dtype=c.dtype)
  for i, v in enumerate(c):
    res += v * (x**i).astype(c.dtype)
  return res


def rational_approximation(
    f, x_min, x_max, p_degree, q_degree, dtype=np.float64
):
  """Implementation of the Remez exchange algorithm.
  """
  n_points = p_degree + q_degree + 2

  # Initial guess for exchange points
  nodes = (
      0.5 * (x_min + x_max)
      + 0.5
      * (x_max - x_min)
      * np.cos(np.pi * np.arange(n_points - 1, -1, -1) / (n_points - 1))
  ).astype(dtype)

  best_p, best_q = None, None
  last_max_err = np.inf

  for iteration in range(25):
    A = np.zeros((n_points, n_points), dtype=dtype)
    b = np.zeros(n_points, dtype=dtype)
    y_nodes = f(nodes).astype(dtype)

    for i in range(n_points):
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
# We approximate (expm1(x) - x) / x^2 to model the higher order terms
f = lambda x: np.where(x == 0, 0, (np.expm1(x) - x) / x**2)
p_degree, q_degree = 6, 0
x_min, x_max = -0.5, 0.5
p, q = rational_approximation(
    f, x_min, x_max, p_degree, q_degree, dtype=np.float32
)

print_polynomial("p", p)

# Evaluate final error
# expm1(x) ~= x + x^2 * (P(x)/Q(x))
x_test = np.linspace(x_min, x_max, 5000)
approx = x_test**2 * poly_eval(p, x_test) + x_test
plot_error(np.expm1, x_test, approx)
# %%
# fp64 expm1
# We approximate (expm1(x) - x) / x^2 to model the higher order terms
f = lambda x: np.where(x == 0, 0, (np.expm1(x) - x) / x**2)
p_degree, q_degree = 10, 0
x_min, x_max = -0.5, 0.5
p, q = rational_approximation(
    f, x_min, x_max, p_degree, q_degree, dtype=np.float64
)

print_polynomial("p", p)
print_polynomial("q", q)

# Evaluate final error
# expm1(x) ~= x + x^2 * (P(x)/Q(x))
x_test = np.linspace(x_min, x_max, 5000)
approx = x_test**2 * poly_eval(p, x_test) / poly_eval(q, x_test) + x_test
plot_error(np.expm1, x_test, approx)
# %%
import math

# fp32 log2(x + 1) ~= x + x^2*P(x)/Q(x)
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
import scipy.special

# fp32 erf(x) ~= x*P(x^2)/Q(x^2)
f = lambda t: scipy.special.erf(np.sqrt(t)) / np.sqrt(t)

p_degree, q_degree = 5, 5
x_min, x_max = 1e-7, 3.8

p, q = rational_approximation(
    f, x_min, x_max**2, p_degree, q_degree, dtype=np.float32
)

print_polynomial("P", p)
print_polynomial("Q", q)

# Evaluate and plot relative error for erf(x)
x_test = np.linspace(x_min, x_max, 5000)
t_test = x_test**2
approx = x_test * (poly_eval(p, t_test) / poly_eval(q, t_test))
plot_error(
    scipy.special.erf,
    x_test,
    approx,
    title="Relative Error for erf(x) (Odd Approximation)",
)
# %%
# Manual Exploration: 3-Piece erf Approximation
x_min, x_limit = 1e-7, 3.8
dtype = np.float32

# These specific thresholds and degrees achieved 2.22e-07 in the search
x1, x2 = 0.8, 1.6
p_deg, q_deg, r_deg = 6, 6, 6

# Generate Approximations
f = lambda t: scipy.special.erf(np.sqrt(t)) / np.sqrt(t)
g = lambda t: scipy.special.erfcx(t)

# Regenerate coefficients
p, _ = rational_approximation(f, x_min**2, x1**2, p_deg, 0, dtype=dtype)
q, _ = rational_approximation(g, x1, x2, q_deg, 0, dtype=dtype)
r, _ = rational_approximation(g, x2, x_limit, r_deg, 0, dtype=dtype)

# Verify on a high-density grid
x_test = np.linspace(x_min, x_limit, 10000)
t_test = x_test**2
approx1 = x_test * poly_eval(p, t_test)
approx2 = 1 - np.exp(-t_test) * poly_eval(q, x_test)
approx3 = 1 - np.exp(-t_test) * poly_eval(r, x_test)

approx = np.where(x_test < x1, approx1, np.where(x_test < x2, approx2, approx3))

# Plot and Print
plot_error(
    scipy.special.erf,
    x_test,
    approx,
    title=f"Refined 3-Piece erf (P={p_deg}, Q={q_deg}, R={r_deg})",
)

print("Final Optimized 3-Piece Coefficients:")
print_polynomial("P", p)
print_polynomial("Q", q)
print_polynomial("R", r)
# %%
import scipy.special

# fp64 erf(x) ~= x*P(x^2)/Q(x^2) for small x
f = lambda t: scipy.special.erf(np.sqrt(t)) / np.sqrt(t)
# For large x, we approximate erf(x) = 1 - exp(-x^2)*erfcx(x)
g = lambda t: scipy.special.erfcx(t)

p_degree, q_degree = 5, 5
r_degree, s_degree = 8, 8
x_min, x_max, x_max2 = 1e-16, 0.9, 5.9

p, q = rational_approximation(
    f, x_min**2, x_max**2, p_degree, q_degree, dtype=np.float64
)

r, s = rational_approximation(
    g, x_max, x_max2, r_degree, s_degree, dtype=np.float64
)

# Evaluate and plot relative error for erf(x)
x_test = np.linspace(x_min, x_max2, 5000)
t_test = x_test**2
approx1 = x_test * (poly_eval(p, t_test) / poly_eval(q, t_test))
approx2 = 1 - np.exp(-t_test) * poly_eval(r, x_test) / poly_eval(s, x_test)
approx = np.where(x_test < x_max, approx1, approx2)
plot_error(
    scipy.special.erf,
    x_test,
    approx,
    title="Relative Error for erf(x) (Combined Approximation)",
)

print_polynomial("P", p)
print_polynomial("Q", q)
print_polynomial("R", r)
print_polynomial("S", s)
