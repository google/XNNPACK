"""Definition of exp kernel."""

# pylint: disable=undefined-variable
# pylint: disable=missing-function-docstring
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


def qd_round_f32(a):
  # If `x` is an floating point value in the range `[2^-22, 2^22)`, then
  # `(x + magic) - magic`` will generate the floating point value corresponding
  # to `round(x)`.
  vmagic = 1.5*(2**23)
  return (vmagic + a) - vmagic


def qd_round_f64(a):
  vmagic = 1.5*(2**52)
  return (vmagic + a) - vmagic


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_multiplier", Float(32)),
    Scalar("input_multiplier", Float(32)),
)
@operator_name("exp")
def exp_fp32(a, x, output_multiplier, input_multiplier):
  # The monomial coefficients of the numerator polynomial.
  valpha_0 = output_multiplier
  valpha_1 = 4.1594290733e-01 * output_multiplier
  valpha_2 = 7.2068706155e-02 * output_multiplier
  valpha_3 = 5.5380910635e-03 * output_multiplier

  # The monomial coefficients of the denominator polynomial (`vbeta_0 = 1.0).
  vbeta_1 = -2.7720427513e-01
  vbeta_2 = 2.3986088112e-02

  va = load(a) * input_multiplier
  # Clamp `vz_prime = x * log2(e)` to the maximum exponents [-127, 128].
  vz_prime = min(max(va, -127.0), 128.0)

  # Decompose x * log2e into `z` (integer part) and `r` (remainder).
  vz = qd_round_f32(vz_prime)
  vr = vz_prime - vz

  # Compute 2^z.
  v2z = exp2_round(vz)

  # Evaluate the numerator polynomial p(f).
  vp = multiply_add(vr, valpha_3, valpha_2)
  vp = multiply_add(vr, vp, valpha_1)
  vp = multiply_add(vr, vp, valpha_0)

  # Evaluate the denominator polynomial q(r).
  vq = multiply_add(vr, vbeta_2, vbeta_1)
  vq = multiply_add(vr, vq, 1.0)

  # Divide the numerator by the denominator, obtaining 2^r.
  v2r = vp / vq

  # Compute 2^z * 2^r.
  vx = v2z * v2r

  return store(vx, x)


@const_buffer("a", Float(64))
@buffer("x", Float(64))
@params(
    # These are actually fp32, but we convert them to fp64 when loading them.
    Scalar("output_multiplier", Float(64)),
    Scalar("input_multiplier", Float(64)),
)
@operator_name("exp")
def exp_fp64(a, x, output_multiplier, input_multiplier):
  # The monomial coefficients of the numerator polynomial.
  valpha_0 = output_multiplier
  valpha_1 = f64(3.4676676602009343e-01) * output_multiplier
  valpha_2 = f64(4.8547260476602683e-03) * output_multiplier
  valpha_3 = f64(-1.6449512448332461e-02) * output_multiplier
  valpha_4 = f64(-3.9542371727433537e-03) * output_multiplier
  valpha_5 = f64(-4.7934022810405645e-04) * output_multiplier
  valpha_6 = f64(-3.3721468729512247e-05) * output_multiplier
  valpha_7 = f64(-1.1751189228021222e-06) * output_multiplier

  # The monomial coefficients of the denominator polynomial (`vbeta_0 = 1.0).
  vbeta_1 = f64(-3.4638041453985308e-01)
  vbeta_2 = f64(4.7208268280255761e-03)
  vbeta_3 = f64(7.9839081451320987e-03)
  vbeta_4 = f64(-1.0149212714116937e-03)
  vbeta_5 = f64(4.2353669794714666e-05)

  va = load(a) * input_multiplier
  # Clamp `vz_prime = x * log2(e)` to the maximum exponents [-1023, 1024].
  vz_prime = min(max(va, -1023.0), 1024.0)

  # Decompose x * log2e into `z` (integer part) and `r` (remainder).
  vz = qd_round_f64(vz_prime)
  vr = vz_prime - vz

  # Compute 2^z.
  v2z = exp2_round(vz)

  # Evaluate the numerator polynomial p(f).
  vp = multiply_add(vr, valpha_7, valpha_6)
  vp = multiply_add(vr, vp, valpha_5)
  vp = multiply_add(vr, vp, valpha_4)
  vp = multiply_add(vr, vp, valpha_3)
  vp = multiply_add(vr, vp, valpha_2)
  vp = multiply_add(vr, vp, valpha_1)
  vp = multiply_add(vr, vp, valpha_0)

  # Evaluate the denominator polynomial q(r).
  vq = multiply_add(vr, vbeta_5, vbeta_4)
  vq = multiply_add(vr, vq, vbeta_3)
  vq = multiply_add(vr, vq, vbeta_2)
  vq = multiply_add(vr, vq, vbeta_1)
  vq = multiply_add(vr, vq, f64(1.0))

  # Divide the numerator by the denominator, obtaining 2^r.
  v2r = vp / vq

  # Compute 2^z * 2^r.
  vx = v2z * v2r

  return store(vx, x)


@const_buffer("a", Float(32))
@buffer("x", Float(32))
@params(
    Scalar("output_offset", Float(32)),
    Scalar("output_multiplier", Float(32)),
    Scalar("input_multiplier", Float(32)),
)
@operator_name("erf")
def erf_fp32(a, x, output_offset, output_multiplier, input_multiplier):
  # Cap the inputs to this value as `erf(x)` will always be `+/-1.0f`
  # beyond this point. This value is chosen roughly as the first floating point
  # number as of which the interpolation returns +/-1.0f.
  vmax_abs_x = 3.8

  # The monomial coefficients of the numerator polynomial (`valpha_0` = 0.0).
  # We distribute the output_multiplier into the coefficients, these multiplies
  # are evaluated outside the loop over the output.
  valpha_1 = 1.1283791149e00 * output_multiplier
  valpha_3 = 1.8942591284e-01 * output_multiplier
  valpha_5 = 5.2645591597e-02 * output_multiplier
  valpha_7 = 3.7304820991e-03 * output_multiplier
  valpha_9 = 2.8532683811e-04 * output_multiplier
  valpha_11 = 2.0742698573e-06 * output_multiplier

  # The monomial coefficients of the denominator polynomial (`vbeta_0 = 1.0).
  vbeta_2 = 5.0120705366e-01
  vbeta_4 = 1.1372791231e-01
  vbeta_6 = 1.4898274094e-02
  vbeta_8 = 1.1562824948e-03
  vbeta_10 = 3.8364178182e-05

  # Clamp the inputs to the interpolation range.
  vx = load(a) * input_multiplier
  vx = min(vmax_abs_x, vx)
  vx = max(-vmax_abs_x, vx)

  # Since the polynomials are odd/even, we need x^2.
  vx2 = vx * vx

  # Evaluate the numerator polynomial p.
  vp = multiply_add(vx2, valpha_11, valpha_9)
  vp = multiply_add(vx2, vp, valpha_7)
  vp = multiply_add(vx2, vp, valpha_5)
  vp = multiply_add(vx2, vp, valpha_3)
  vp = multiply_add(vx2, vp, valpha_1)
  vp = vx * vp

  # Evaluate the denominator polynomial q.
  vq = multiply_add(vx2, vbeta_10, vbeta_8)
  vq = multiply_add(vx2, vq, vbeta_6)
  vq = multiply_add(vx2, vq, vbeta_4)
  vq = multiply_add(vx2, vq, vbeta_2)
  vq = multiply_add(vx2, vq, 1.0)

  # Divide the numerator by the denominator.
  return store((vp / vq) + output_offset, x)


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

  # The monomial coefficients of the numerator polynomial.
  valpha_1 = 1.4426950180e+00
  valpha_2 = 1.2540218881e+00
  valpha_3 = 1.7968840189e-01

  # The monomial coefficients of the denominator polynomial.
  vbeta_1 = 1.3692200200e+00
  vbeta_2 = 4.7585666772e-01
  vbeta_3 = 3.1328686121e-02

  vx = load(a) * input_multiplier

  # log2(x) = log2(x'*2^exp) = log2(x') + exp where x' is in [1, 2) and exp is
  # an integer. x' is the mantissa with an exponent of 0, and exp is
  # floor(log2(x)).
  vexp = floor_log2(vx)
  vx_bits = reinterpret_cast(Int(32), vx)
  one_bits = reinterpret_cast(Int(32), f32(1.0))
  vx_norm_bits = (vx_bits & vmantissa_mask) | one_bits
  vx_norm = reinterpret_cast(Float(32), vx_norm_bits)

  # Our polynomial approximates log2(x + 1)
  vr = vx_norm - 1.0

  # Evaluate the numerator polynomial p.
  vp = multiply_add(vr, valpha_3, valpha_2)
  vp = multiply_add(vr, vp, valpha_1)
  vp = vr * vp

  # Evaluate the denominator polynomial q.
  vq = multiply_add(vr, vbeta_3, vbeta_2)
  vq = multiply_add(vr, vq, vbeta_1)
  vq = multiply_add(vr, vq, 1.0)

  # Divide the numerator by the denominator.
  vy = vp / vq

  # log2(x') = vy
  vy = (vexp + vy) * output_multiplier

  return store(vy, x)
