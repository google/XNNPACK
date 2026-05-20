"""Definition of exp kernel."""

# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.util import *  # pylint: disable=wildcard-import


# exp(a) = exp(ln(2)*b + a')
#        = 2^b*exp(a')
#
# Where a' is in [-0.5, 0.5].
# exp(a') is given by rational polynomials p(a')/q(a')
# ln2_hi = ln(2), ln2_lo = ln(2) - float(ln2_hi)
def exp_impl(a, p, q, log2e, ln2_hi, ln2_lo, max_a, min_a):
  a_clamped = min(max(a, min_a), max_a)
  b = round_small(a_clamped * log2e)
  a_prime = (a_clamped - b * ln2_hi) - b * ln2_lo

  # Compute 2^b.
  exp2b = exp2_round(b)
  exp2b = select(isnan(a), a, exp2b)

  p_a_prime = eval_polynomial(a_prime, p)
  q_a_prime = eval_polynomial(a_prime, q)

  # Divide the numerator by the denominator, obtaining exp(r).
  exp_a_prime = p_a_prime / q_a_prime

  # Compute 2^b * exp(a_prime).
  return exp2b * exp_a_prime


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_multiplier", Float(32)),
    Scalar("input_multiplier", Float(32)),
)
@operator_name("exp")
def exp_fp32(a_buf, x_buf, output_multiplier, input_multiplier):
  p = [
      1.0244545527e-02,
      1.1163228750e-01,
      5.2334904671e-01,
      1.0000000000e+00
  ]
  q = [
      -6.3802185468e-03,
      8.8284827769e-02,
      -4.7665113211e-01,
      1.0000000000e+00
  ]
  log2e = 1.4426950408889634
  ln2_hi = 0.693145751953125
  ln2_lo = 1.4286067653e-06
  max_a = 88.72283935546875
  min_a = -88.02969360351562

  a = load(a_buf) * input_multiplier
  x = exp_impl(a, p, q, log2e, ln2_hi, ln2_lo, max_a, min_a)
  return store(x * output_multiplier, x_buf)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    # These are actually fp32, but we convert them to fp64 when loading them.
    Scalar("output_multiplier", Float(64)),
    Scalar("input_multiplier", Float(64)),
)
@operator_name("exp")
def exp_fp64(a_buf, x_buf, output_multiplier, input_multiplier):
  # Polynomial coefficients
  p = [
      f64(3.239974270508748009e-05),
      f64(9.791151627310745124e-04),
      f64(1.377501781822836231e-02),
      f64(1.105887196698455144e-01),
      f64(4.989688485947128549e-01),
      f64(9.999999999999998890e-01)
  ]
  q = [
      f64(-3.362442950087436857e-05),
      f64(1.003649080773944512e-03),
      f64(-1.399594422100019359e-02),
      f64(1.116198710751181572e-01),
      f64(-5.010311514052862014e-01),
      f64(1.000000000000000000e+00)
  ]
  log2e = f64(1.4426950408889634)
  ln2_hi = f64(6.93147180369123816490e-01)
  ln2_lo = f64(1.90821492927058770002e-10)
  max_a = f64(709.782712893384)
  min_a = f64(-709.0895657128241)

  a = load(a_buf) * input_multiplier
  x = exp_impl(a, p, q, log2e, ln2_hi, ln2_lo, max_a, min_a)
  return store(x * output_multiplier, x_buf)
