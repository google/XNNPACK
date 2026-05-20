"""Definition of exp kernel."""

# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import
from ynnpack.kernels.unary.util import *  # pylint: disable=wildcard-import


def log_impl(a, p, q, ln2_hi, ln2_lo):
  # log(a) = log2(m) * ln2 + e * ln2 = (e + log2(m)) * ln2
  vexp = floor_log2(a)

  if a.ty == Float(32):
    vmantissa_mask = i32(0x007FFFFF)
    one_bits = i32(0x3F800000)
    vx_bits = reinterpret_cast(Int(32), a)
    vx_norm = reinterpret_cast(Float(32), (vx_bits & vmantissa_mask) | one_bits)
  else:
    vmantissa_mask = i64(0x000FFFFFFFFFFFFF)
    one_bits = i64(0x3FF0000000000000)
    vx_bits = reinterpret_cast(Int(64), a)
    vx_norm = reinterpret_cast(Float(64), (vx_bits & vmantissa_mask) | one_bits)

  # Adjust the mantissa to the range [1/sqrt(2), sqrt(2)].
  cond = vx_norm > math.sqrt(2.0)
  vx_norm = select(cond, vx_norm * 0.5, vx_norm)
  vexp = select(cond, vexp + 1, vexp)

  if a.ty == Float(64):
    vr = vx_norm - f64(1.0)
  else:
    vr = vx_norm - 1.0

  vp = eval_polynomial(vr, p)
  vq = eval_polynomial(vr, q)
  vy = vp / vq

  # result = (vexp + vy) * ln2
  # order: (vexp + vy) * ln2_hi + (vexp + vy) * ln2_lo
  vlog2 = vexp + vy
  return multiply_add(vlog2, ln2_lo, vlog2 * ln2_hi)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("input_multiplier", Float(32)),
    Scalar("output_multiplier", Float(32)),
)
@operator_name("log")
def log_fp32(a_buf, x_buf, input_multiplier, output_multiplier):
  # Polynomial coefficients for log2(1+x)/x on [-0.3, 0.42]
  p = [
      2.9484698176e-01,
      1.5053815842e00,
      1.4426950216e00,
      0.0000000000e00,
  ]
  q = [
      5.6864939630e-02,
      6.4276754856e-01,
      1.5434517860e00,
      1.0000000000e00,
  ]

  ln2_hi = 0.693145751953125
  ln2_lo = 1.4286067653e-06

  a = load(a_buf) * input_multiplier
  x = log_impl(a, p, q, ln2_hi, ln2_lo)
  return store(x * output_multiplier, x_buf)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    Scalar("input_multiplier", Float(64)),
    Scalar("output_multiplier", Float(64)),
)
@operator_name("log")
def log_fp64(a_buf, x_buf, input_multiplier, output_multiplier):
  # Polynomial coefficients for log2(1+x)/x on [-0.3, 0.42]
  p = [
      f64(7.343492468411156292e-03),
      f64(1.857349351736143628e-01),
      f64(1.229266931139749275e00),
      f64(3.212163638325623349e00),
      f64(3.596265560063666378e00),
      f64(1.442695040888963831e00),
      f64(0.000000000000000000e00),
  ]
  q = [
      f64(1.034012664071056126e-03),
      f64(4.419595181114822913e-02),
      f64(4.467065862797955367e-01),
      f64(1.799252214383866288e00),
      f64(3.389539502820695382e00),
      f64(2.992741333502965340e00),
      f64(1.000000000000000000e00),
  ]
  ln2_hi = f64(6.93147180369123816490e-01)
  ln2_lo = f64(1.90821492927058770002e-10)

  a = load(a_buf) * input_multiplier
  x = log_impl(a, p, q, ln2_hi, ln2_lo)
  return store(x * output_multiplier, x_buf)
