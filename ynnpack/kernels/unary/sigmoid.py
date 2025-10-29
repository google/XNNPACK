"""Definition of sigmoid kernel."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


@const_buffer('a', Float(32))
@buffer('x', Float(32))
@operator_name('sigmoid')
def sigmoid_fp32(a, x):
  """Polynomial approximation of sigmoid."""
  vmagic_bias = float.fromhex('0x1.8000FEp23')
  vminus_log2e = float.fromhex('-0x1.715476p0')
  vln2_hi = float.fromhex('0x1.62E400p-1')
  vln2_lo = float.fromhex('0x1.7F7D1Cp-20')
  vc5 = float.fromhex('-0x1.0F9F9Cp-7')
  vc4 = float.fromhex('0x1.573A1Ap-5')
  vc3 = float.fromhex('-0x1.555A80p-3')
  vc2 = float.fromhex('0x1.FFFDC6p-2')
  vc1 = float.fromhex('-0x1.FFFFF6p-1')
  vdenorm_cutoff = float.fromhex('0x1.5D589Ep+6')

  vx = load(a)
  vz = abs(vx)

  vn = multiply_add(vz, vminus_log2e, vmagic_bias)

  vs = reinterpret_cast(
      Float(32), logical_shift_left(reinterpret_cast(Int(32), vn), i32(23))
  )
  vn = vn - vmagic_bias

  vt = multiply_add(vn, vln2_hi, vz)
  vt = multiply_add(vn, vln2_lo, vt)

  vp = multiply_add(vt, vc5, vc4)
  vp = multiply_add(vt, vp, vc3)
  vp = multiply_add(vt, vp, vc2)
  vp = multiply_add(vt, vp, vc1)

  vt = vt * vs
  ve = multiply_add(vt, vp, vs)
  vd = ve + 1.0

  vf = ve / vd
  vf = select(vz > vdenorm_cutoff, f32(0.0), vf)
  vf = select(vx > f32(0.0), 1.0 - vf, vf)

  return store(vf, x)
