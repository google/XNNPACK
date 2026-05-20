"""Helpers for unary kernels."""

# pylint: disable=wildcard-import
# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
from ynnpack.kernels.elementwise.compiler import *


def eval_polynomial(x, coeffs):
  """Evaluate a polynomial of x with an array of `coeffs`.

  Args:
    x: The input value.
    coeffs: The coefficients of the polynomial, from highest degree term to
      lowest.

  Returns:
    The value of the polynomial.
  """
  y = coeffs[0]
  for i in range(1, len(coeffs)):
    if i == len(coeffs) - 1 and coeffs[i] == 0:
      y = x * y
    else:
      y = multiply_add(x, y, coeffs[i])
  return y


def round_small_fp32(a):
  """Round a float in [2^-22, 2^22) to the nearest whole number."""
  vmagic = 1.5 * (2**23)
  return (vmagic + a) - vmagic


def round_small_fp64(a):
  """Round a float in [2^-51, 2^51) to the nearest whole number."""
  vmagic = 1.5 * (2**52)
  return (vmagic + a) - vmagic


def round_small(a):
  """Round a float to the nearest whole number."""
  if a.ty == Float(32):
    return round_small_fp32(a)
  elif a.ty == Float(64):
    return round_small_fp64(a)
  else:
    raise ValueError("Unsupported dtype: %s" % a.ty)
