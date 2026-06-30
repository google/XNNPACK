"""Definition of convert kernels."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


# We compute the reciprocal of the scale, so using specialization for
# broadcast enables the reciprocal to be lifted out of the inner loop.
quantize_scale_broadcast_mode = BroadcastMode.SPECIALIZE


def quantize(result_type, a, scale, zero_point, x):
  """Quantize fp32 to int8."""
  inv_scale = 1 / load(scale)
  zero_point = saturating_cast(Int(16), load(zero_point))
  vx = round(cast(Float(32), load(a)) * inv_scale)
  vx = add_sat(saturating_cast(Int(16), vx), zero_point)
  return store(saturating_cast(result_type, vx), x)


@const_buffer("a", Float(32), BroadcastMode.LOCAL_VAR)
@const_buffer("scale", Float(32), quantize_scale_broadcast_mode)
@const_buffer("zero_point", Int(32), BroadcastMode.LOCAL_VAR)
@buffer("x", Int(8))
@operator_name("quantize_int8")
def quantize_fp32_to_int8(a, scale, zero_point, x):
  """Quantize fp32 to int8."""
  return quantize(Int(8), a, scale, zero_point, x)


@const_buffer("a", Float(32), BroadcastMode.LOCAL_VAR)
@const_buffer("scale", Float(32), quantize_scale_broadcast_mode)
@const_buffer("zero_point", Int(32), BroadcastMode.LOCAL_VAR)
@buffer("x", UInt(8))
@operator_name("quantize_uint8")
def quantize_fp32_to_uint8(a, scale, zero_point, x):
  return quantize(UInt(8), a, scale, zero_point, x)


@const_buffer("a", Float(16), BroadcastMode.LOCAL_VAR)
@const_buffer("scale", Float(32), quantize_scale_broadcast_mode)
@const_buffer("zero_point", Int(32), BroadcastMode.LOCAL_VAR)
@buffer("x", Int(8))
@operator_name("quantize_int8")
def quantize_fp16_to_int8(a, scale, zero_point, x):
  return quantize(Int(8), a, scale, zero_point, x)


@const_buffer("a", Float(16), BroadcastMode.LOCAL_VAR)
@const_buffer("scale", Float(32), quantize_scale_broadcast_mode)
@const_buffer("zero_point", Int(32), BroadcastMode.LOCAL_VAR)
@buffer("x", UInt(8))
@operator_name("quantize_uint8")
def quantize_fp16_to_uint8(a, scale, zero_point, x):
  return quantize(UInt(8), a, scale, zero_point, x)


@const_buffer("a", BFloat(16), BroadcastMode.LOCAL_VAR)
@const_buffer("scale", Float(32), quantize_scale_broadcast_mode)
@const_buffer("zero_point", Int(32), BroadcastMode.LOCAL_VAR)
@buffer("x", Int(8))
@operator_name("quantize_int8")
def quantize_bf16_to_int8(a, scale, zero_point, x):
  return quantize(Int(8), a, scale, zero_point, x)


@const_buffer("a", BFloat(16), BroadcastMode.LOCAL_VAR)
@const_buffer("scale", Float(32), quantize_scale_broadcast_mode)
@const_buffer("zero_point", Int(32), BroadcastMode.LOCAL_VAR)
@buffer("x", UInt(8))
@operator_name("quantize_uint8")
def quantize_bf16_to_uint8(a, scale, zero_point, x):
  return quantize(UInt(8), a, scale, zero_point, x)


def dequantize(result_type, a, zero_point, scale, x):
  zeroed = cast(Float(32), load(a)) - cast(Float(32), load(zero_point))
  return store(cast(result_type, zeroed * load(scale)), x)


@const_buffer("a", Int(8), BroadcastMode.LOCAL_VAR)
@const_buffer("zero_point", Int(32), BroadcastMode.LOCAL_VAR)
@const_buffer("scale", Float(32), BroadcastMode.LOCAL_VAR)
@buffer("x", Float(32))
@operator_name("dequantize")
def dequantize_int8_to_fp32(a, zero_point, scale, x):
  return dequantize(Float(32), a, zero_point, scale, x)


@const_buffer("a", Int(8), BroadcastMode.LOCAL_VAR)
@const_buffer("zero_point", Int(32), BroadcastMode.LOCAL_VAR)
@const_buffer("scale", Float(32), BroadcastMode.LOCAL_VAR)
@buffer("x", Float(16))
@operator_name("dequantize")
def dequantize_int8_to_fp16(a, zero_point, scale, x):
  return dequantize(Float(16), a, zero_point, scale, x)


@const_buffer("a", Int(8), BroadcastMode.LOCAL_VAR)
@const_buffer("zero_point", Int(32), BroadcastMode.LOCAL_VAR)
@const_buffer("scale", Float(32), BroadcastMode.LOCAL_VAR)
@buffer("x", BFloat(16))
@operator_name("dequantize")
def dequantize_int8_to_bf16(a, zero_point, scale, x):
  return dequantize(BFloat(16), a, zero_point, scale, x)


@const_buffer("a", UInt(8), BroadcastMode.LOCAL_VAR)
@const_buffer("zero_point", Int(32), BroadcastMode.LOCAL_VAR)
@const_buffer("scale", Float(32), BroadcastMode.LOCAL_VAR)
@buffer("x", Float(32))
@operator_name("dequantize")
def dequantize_uint8_to_fp32(a, zero_point, scale, x):
  return dequantize(Float(32), a, zero_point, scale, x)


@const_buffer("a", UInt(8), BroadcastMode.LOCAL_VAR)
@const_buffer("zero_point", Int(32), BroadcastMode.LOCAL_VAR)
@const_buffer("scale", Float(32), BroadcastMode.LOCAL_VAR)
@buffer("x", Float(16))
@operator_name("dequantize")
def dequantize_uint8_to_fp16(a, zero_point, scale, x):
  return dequantize(Float(16), a, zero_point, scale, x)


@const_buffer("a", UInt(8), BroadcastMode.LOCAL_VAR)
@const_buffer("zero_point", Int(32), BroadcastMode.LOCAL_VAR)
@const_buffer("scale", Float(32), BroadcastMode.LOCAL_VAR)
@buffer("x", BFloat(16))
@operator_name("dequantize")
def dequantize_uint8_to_bf16(a, zero_point, scale, x):
  return dequantize(BFloat(16), a, zero_point, scale, x)
