# pylint: disable=all

# %%
# This notebook contains the code used to generate the coefficients and
# constants used in the implementation of these kernels
import numpy as np


def rational_approximation(
    f, x_min, x_max, numerator_degree, denominator_degree
):
  """Iteratively Reweighted Least Squares (Lawson's algorithm) using Power Basis."""
  n_samples = 4000
  i = np.arange(1, n_samples + 1)

  # Generate nodes
  x = 0.5 * (x_min + x_max) + 0.5 * (x_max - x_min) * np.cos(
      (2 * i - 1) * np.pi / (2 * n_samples)
  )
  y = f(x)

  weights = np.ones(n_samples)

  best_p, best_q = None, None
  min_max_err = np.inf

  for _ in range(200):
    # A * coeff = b
    # coeff = [p_0, ..., p_p, q_1, ..., q_q]
    A = np.zeros((n_samples, numerator_degree + 1 + denominator_degree))
    for j in range(numerator_degree + 1):
      A[:, j] = x**j
    for j in range(1, denominator_degree + 1):
      A[:, numerator_degree + j] = -y * (x**j)

    W = np.sqrt(weights)[:, None]
    Aw = W * A
    bw = W[:, 0] * y

    # Solve in float64
    sol, _, _, _ = np.linalg.lstsq(Aw, bw, rcond=None)

    p_coeffs = sol[: numerator_degree + 1]
    q_coeffs = np.concatenate(([1.0], sol[numerator_degree + 1 :]))

    p_val = sum(p_coeffs[j] * x**j for j in range(numerator_degree + 1))
    q_val = sum(q_coeffs[j] * x**j for j in range(denominator_degree + 1))

    errs = np.abs(p_val / q_val - y)
    current_max_err = np.max(errs)

    if current_max_err < min_max_err:
      min_max_err = current_max_err
      best_p, best_q = p_coeffs.copy(), q_coeffs.copy()

    # Update weights (Lawson's algorithm)
    weights = weights * errs
    weights /= np.mean(weights)

  return best_p, best_q


def poly_eval(c, x):
  res = 0
  for i, v in enumerate(c):
    res += v * (x**i)
  return res


# %%
# fp64 exp2
f = np.exp2
numerator_degree, denominator_degree = 7, 5
x_min, x_max = -0.5, 0.5
p, q = rational_approximation(
    f, x_min, x_max, numerator_degree, denominator_degree
)

for i, val in enumerate(p):
  print(f"valpha_{i} = f64({val:.16e}) * output_multiplier")

for i, val in enumerate(q):
  print(f"vbeta_{i} = f64({val:.16e})")

# Evaluate final error
x_test = np.linspace(x_min, x_max, 5000)
approx = poly_eval(p, x_test) / poly_eval(q, x_test)
max_err = np.max(np.abs(approx - f(x_test)))
print(f"\nMaximum absolute error (float64): {max_err:.2e}")


# %%
import math

# fp32 log2(x + 1)
f = lambda x: np.where(x == 0, 1.0/np.log(2), np.log1p(x) / (x*np.log(2)))
numerator_degree, denominator_degree = 2, 3
x_min, x_max = 0, 1
p, q = rational_approximation(
    f, x_min, x_max, numerator_degree, denominator_degree
)

for i, val in enumerate(p):
  print(f"  valpha_{i+1} = {val:.10e}")

for i, val in enumerate(q):
  print(f"  vbeta_{i} = {val:.10e}")

# Evaluate final error
x_test = np.linspace(x_min, x_max, 5000)
approx = x_test * poly_eval(p, x_test) / poly_eval(q, x_test)
max_err = np.max(np.abs(approx - np.log2(x_test + 1)))
print(f"\nMaximum absolute error (float64): {max_err:.2e}")
