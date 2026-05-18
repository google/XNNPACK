"""Definition of exp kernel."""

# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.util import *  # pylint: disable=wildcard-import


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("input_multiplier", Float(32)),
    Scalar("output_multiplier", Float(32)),
)
@operator_name("log")
def log_fp32(a, x, input_multiplier, output_multiplier):
  # Some useful constants.
  vmantissa_mask = i32(0x007FFFFF)

  # Polynomial coefficients
  p = [1.7932410538e-01, 1.2533404827e00, 1.4426950216e00, 0.0000000000e00]
  q = [3.1246891245e-02, 4.7536420822e-01, 1.3687485456e00, 1.0000000000e00]

  vx = load(a) * input_multiplier

  # log2(x) = log2(x'*2^exp) = log2(x') + exp where x' is in [1, 2) and exp is
  # an integer. x' is the mantissa with an exponent of 0, and exp is
  # floor(log2(x)).
  vexp = floor_log2(vx)
  vx_bits = reinterpret_cast(Int(32), vx)
  one_bits = reinterpret_cast(Int(32), f32(1.0))
  vx_norm_bits = (vx_bits & vmantissa_mask) | one_bits
  vx_norm = reinterpret_cast(Float(32), vx_norm_bits)

  # Our polynomial approximates log2(x + 1) on the interval [0, 1]
  vr = vx_norm - 1.0

  vp = eval_polynomial(vr, p)
  vq = eval_polynomial(vr, q)

  # Divide the numerator by the denominator.
  vy = vp / vq

  # log2(x') = vy
  vy = (vexp + vy) * output_multiplier

  return store(vy, x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    Scalar("input_multiplier", Float(64)),
    Scalar("output_multiplier", Float(64)),
)
@operator_name("log")
def log_fp64(a, x, input_multiplier, output_multiplier):
  # Some useful constants.
  vmantissa_mask = i64(0x000FFFFFFFFFFFFF)

  # Polynomial coefficients
  p = [
      f64(-2.527446318229127544e-03),
      f64(-8.135266103629081036e-02),
      f64(-5.867592932264611427e-01),
      f64(-1.284504598302105949e00),
      f64(-1.400574861027075180e-01),
      f64(2.047856959129240373e00),
      f64(1.442695040888963165e00),
      f64(0.0),
  ]
  q = [
      f64(-3.354382895618343314e-04),
      f64(-1.824426552234581847e-02),
      f64(-2.083019139798093777e-01),
      f64(-8.110403419400664671e-01),
      f64(-1.015513156413926588e00),
      f64(5.293193537676974536e-01),
      f64(1.919466277410421862e00),
      f64(1.000000000000000000e00),
  ]

  vx = load(a) * input_multiplier

  # log2(x) = log2(x'*2^exp) = log2(x') + exp where x' is in [1, 2) and exp is
  # an integer. x' is the mantissa with an exponent of 0, and exp is
  # floor(log2(x)).
  vexp = floor_log2(vx)
  vx_bits = reinterpret_cast(Int(64), vx)
  one_bits = reinterpret_cast(Int(64), f64(1.0))
  vx_norm_bits = (vx_bits & vmantissa_mask) | one_bits
  vx_norm = reinterpret_cast(Float(64), vx_norm_bits)

  # Our polynomial approximates log2(x + 1)
  vr = vx_norm - f64(1.0)

  vp = eval_polynomial(vr, p)
  vq = eval_polynomial(vr, q)

  # Divide the numerator by the denominator.
  vy = vp / vq

  # log2(x') = vy
  vy = (vexp + vy) * output_multiplier

  return store(vy, x)
