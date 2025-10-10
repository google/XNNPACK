"""Definition of convert kernels."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


def quantize(result_type, a, scale, zero_point, x):
  """Quantize fp32 to int8."""
  inv_scale = 1 / load(scale)
  zero_point = saturating_cast(Int(16), load(zero_point))
  vx = round(load(a) * inv_scale)
  vx = saturating_add(saturating_cast(Int(16), vx), zero_point)
  return store(saturating_cast(result_type, vx), x)


@const_buffer("a", Float(32), BroadcastMode.LOCAL_VAR)
# We compute the reciprocal of the scale, so using specialization for
# broadcast enables the reciprocal to be lifted out of the inner loop.
@const_buffer("scale", Float(32), BroadcastMode.SPECIALIZE)
@const_buffer("zero_point", Int(32), BroadcastMode.LOCAL_VAR)
@buffer("x", Int(8))
@operator_name("quantize_int8")
def quantize_fp32_to_int8(a, scale, zero_point, x):
  """Quantize fp32 to int8."""
  return quantize(Int(8), a, scale, zero_point, x)


@const_buffer("a", Float(32), BroadcastMode.LOCAL_VAR)
# We compute the reciprocal of the scale, so using specialization for
# broadcast enables the reciprocal to be lifted out of the inner loop.
@const_buffer("scale", Float(32), BroadcastMode.SPECIALIZE)
@const_buffer("zero_point", Int(32), BroadcastMode.LOCAL_VAR)
@buffer("x", UInt(8))
@operator_name("quantize_uint8")
def quantize_fp32_to_uint8(a, scale, zero_point, x):
  return quantize(UInt(8), a, scale, zero_point, x)
