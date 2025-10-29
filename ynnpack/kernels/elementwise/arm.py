"""ARM NEON target for elementwise kernels compiler."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.common_rules import add_saturating_cast_rules
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


def make_neon_cast_patterns(vector_bits):
  """Adds NEON cast patterns."""
  assert vector_bits == 128
  vf32_a = f32_a.with_lanes(vector_bits // 32)
  vf32_b = f32_b.with_lanes(vector_bits // 32)
  vf16_a = f16_a.with_lanes(vector_bits // 32)

  return (
      [
          i.vectorize(vector_bits)
          for i in [
              Rule(
                  cast(Float(32), i32_a),
                  Op(Float(32), "vcvtq_f32_s32", [i32_a]),
              ),
              Rule(
                  cast(Int(32), round(f32_a)),
                  Op(Int(32), "cast_f32_to_int32", [f32_a]),
              ),
          ]
      ]
      + [
          Rule(
              cast(Float(32, vector_bits // 32), vf16_a),
              Op(Float(32, vector_bits // 32), "vcvt_f32_f16", [vf16_a]),
          ),
          Rule(
              cast(
                  Float(16, vector_bits // 16),
                  combine_vectors([vf32_a, vf32_b]),
              ),
              Op(
                  Float(16, vector_bits // 16),
                  "cast_f32_to_f16",
                  [vf32_a, vf32_b],
              ),
          ),
      ]
      + add_saturating_cast_rules(vector_bits)
  )


def make_neon_reinterpret_cast_patterns(vector_bits):
  assert vector_bits == 128
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              Op(Float(32), "reinterpret_cast", [i32_a]),
              Op(Float(32), "vreinterpretq_f32_s32", [i32_a]),
          ),
          Rule(
              Op(Int(32), "reinterpret_cast", [f32_a]),
              Op(Int(32), "vreinterpretq_s32_f32", [f32_a]),
          ),
          Rule(
              Op(Float(32), "reinterpret_cast", [u32_a]),
              Op(Float(32), "vreinterpretq_f32_u32", [u32_a]),
          ),
          Rule(
              Op(UInt(32), "reinterpret_cast", [f32_a]),
              Op(UInt(32), "vreinterpretq_u32_f32", [f32_a]),
          ),
      ]
  ]


def make_neon_integer_patterns(vector_bits):
  assert vector_bits == 128
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(u8_a + u8_b, Op(UInt(8), "vaddq_u8", [u8_a, u8_b])),
          Rule(i8_a + i8_b, Op(Int(8), "vaddq_s8", [i8_a, i8_b])),
          Rule(u16_a + u16_b, Op(UInt(16), "vaddq_u16", [u16_a, u16_b])),
          Rule(i16_a + i16_b, Op(Int(16), "vaddq_s16", [i16_a, i16_b])),
          Rule(u32_a + u32_b, Op(UInt(32), "vaddq_u32", [u32_a, u32_b])),
          Rule(i32_a + i32_b, Op(Int(32), "vaddq_s32", [i32_a, i32_b])),
          Rule(u8_a - u8_b, Op(UInt(8), "vsubq_u8", [u8_a, u8_b])),
          Rule(i8_a - i8_b, Op(Int(8), "vsubq_s8", [i8_a, i8_b])),
          Rule(u16_a - u16_b, Op(UInt(16), "vsubq_u16", [u16_a, u16_b])),
          Rule(i16_a - i16_b, Op(Int(16), "vsubq_s16", [i16_a, i16_b])),
          Rule(u32_a - u32_b, Op(UInt(32), "vsubq_u32", [u32_a, u32_b])),
          Rule(i32_a - i32_b, Op(Int(32), "vsubq_s32", [i32_a, i32_b])),
          Rule(i16_a * i16_b, Op(Int(16), "vmulq_s16", [i16_a, i16_b])),
          Rule(i32_a * i32_b, Op(Int(32), "vmulq_s32", [i32_a, i32_b])),
          Rule(u32_a & u32_b, Op(UInt(32), "vandq_u32", [u32_a, u32_b])),
          Rule(i32_a & i32_b, Op(Int(32), "vandq_s32", [i32_a, i32_b])),
          Rule(u32_a | u32_b, Op(UInt(32), "vorrq_u32", [u32_a, u32_b])),
          Rule(i32_a | i32_b, Op(Int(32), "vorrq_s32", [i32_a, i32_b])),
          Rule(u32_a ^ u32_b, Op(UInt(32), "veorq_u32", [u32_a, u32_b])),
          Rule(i32_a ^ i32_b, Op(Int(32), "veorq_s32", [i32_a, i32_b])),
          Rule(
              saturating_add(u8_a, u8_b),
              Op(UInt(8), "vqaddq_u8", [u8_a, u8_b]),
          ),
          Rule(
              saturating_add(i8_a, i8_b),
              Op(Int(8), "vqaddq_s8", [i8_a, i8_b]),
          ),
          Rule(
              saturating_add(u16_a, u16_b),
              Op(UInt(16), "vqaddq_u16", [u16_a, u16_b]),
          ),
          Rule(
              saturating_add(i16_a, i16_b),
              Op(Int(16), "vqaddq_s16", [i16_a, i16_b]),
          ),
          Rule(
              saturating_sub(u8_a, u8_b),
              Op(UInt(8), "vqsubq_u8", [u8_a, u8_b]),
          ),
          Rule(
              saturating_sub(i8_a, i8_b),
              Op(Int(8), "vqsubq_s8", [i8_a, i8_b]),
          ),
          Rule(
              saturating_sub(u16_a, u16_b),
              Op(UInt(16), "vqsubq_u16", [u16_a, u16_b]),
          ),
          Rule(
              saturating_sub(i16_a, i16_b),
              Op(Int(16), "vqsubq_s16", [i16_a, i16_b]),
          ),
      ]
  ] + [
      Rule(
          logical_shift_left(i16_a.with_lanes(8), broadcast(i16_b, 8)),
          Op(Int(16, 8), "vshlq_n_s16", [i16_a.with_lanes(8), i16_b]),
      ),
      Rule(
          logical_shift_left(i32_a.with_lanes(4), broadcast(i32_b, 4)),
          Op(Int(32, 4), "vshlq_n_s32", [i32_a.with_lanes(4), i32_b]),
      ),
  ]


def make_neon_broadcast_patterns(vector_bits):
  assert vector_bits == 128
  return [
      Rule(
          broadcast(i32_a, 4),
          Op(Int(32, 4), "vdupq_n_s32", [i32_a]),
      ),
      Rule(
          broadcast(u32_a, 4),
          Op(UInt(32, 4), "vdupq_n_u32", [u32_a]),
      ),
      Rule(
          broadcast(f32_a, 4),
          Op(Float(32, 4), "vdupq_n_f32", [f32_a]),
      ),
  ]


def make_neon_float32_patterns(vector_bits):
  assert vector_bits == 128
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(f32_a + f32_b, Op(Float(32), "vaddq_f32", [f32_a, f32_b])),
          Rule(f32_a - f32_b, Op(Float(32), "vsubq_f32", [f32_a, f32_b])),
          Rule(f32_a * f32_b, Op(Float(32), "vmulq_f32", [f32_a, f32_b])),
          Rule(
              max(f32_a, f32_b),
              Op(Float(32), "vmaxq_f32", [f32_a, f32_b]),
          ),
          Rule(
              min(f32_a, f32_b),
              Op(Float(32), "vminq_f32", [f32_a, f32_b]),
          ),
          Rule(abs(f32_a), Op(Float(32), "vabsq_f32", [f32_a])),
          Rule(round(f32_a), Op(Float(32), "round_f32", [f32_a])),
          Rule(ceil(f32_a), Op(Float(32), "ceil_f32", [f32_a])),
          Rule(floor(f32_a), Op(Float(32), "floor_f32", [f32_a])),
          Rule(sqrt(f32_a), Op(Float(32), "sqrt_f32", [f32_a])),
          Rule(f32_a & f32_b, Op(Float(32), "and_f32", [f32_a, f32_b])),
          Rule(f32_a | f32_b, Op(Float(32), "or_f32", [f32_a, f32_b])),
          Rule(f32_a ^ f32_b, Op(Float(32), "xor_f32", [f32_a, f32_b])),
      ]
  ]


class ARM(Target):
  """NEON target for elementwise kernels compiler."""

  def add_load_intrinsics(self):
    self.load_intrinsics[Int(8, 16)] = "vld1q_s8"
    self.load_intrinsics[UInt(8, 16)] = "vld1q_u8"
    self.load_intrinsics[Int(16, 8)] = "vld1q_s16"
    self.load_intrinsics[UInt(16, 8)] = "vld1q_u16"
    self.load_intrinsics[Int(32, 4)] = "vld1q_s32"
    self.load_intrinsics[UInt(32, 4)] = "vld1q_u32"
    self.load_intrinsics[Float(32, 4)] = "vld1q_f32"
    self.load_intrinsics[Float(16, 8)] = "vld1q_f16"
    self.load_intrinsics[Float(16, 4)] = "vld1_f16"

  def add_store_intrinsics(self):
    self.store_intrinsics[Int(8, 16)] = "vst1q_s8"
    self.store_intrinsics[UInt(8, 16)] = "vst1q_u8"
    self.store_intrinsics[Int(16, 8)] = "vst1q_s16"
    self.store_intrinsics[UInt(16, 8)] = "vst1q_u16"
    self.store_intrinsics[Int(32, 4)] = "vst1q_s32"
    self.store_intrinsics[UInt(32, 4)] = "vst1q_u32"
    self.store_intrinsics[Float(32, 4)] = "vst1q_f32"
    self.store_intrinsics[Float(16, 8)] = "vst1q_f16"
    self.store_intrinsics[Float(16, 4)] = "vst1_f16"

  def legalize_type(self, ty, is_const=True):
    # This is the type which ARM intrinsics expect as argument for pointers.
    if ty.is_float() and ty.size == 16 and ty.lanes == 1:
      return "__fp16"
    return super().legalize_type(ty, is_const)

  def update_for_neon(self):
    self.header += """

namespace {

YNN_INTRINSIC float32x4_t and_f32(float32x4_t a, float32x4_t b) {
  return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

YNN_INTRINSIC float32x4_t or_f32(float32x4_t a, float32x4_t b) {
  return vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

YNN_INTRINSIC float32x4_t xor_f32(float32x4_t a, float32x4_t b) {
  return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b)));
}

YNN_INTRINSIC float32x4_t not_f32(float32x4_t a) {
  return vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(a)));
}

YNN_INTRINSIC float32x4_t round_f32(float32x4_t a) {
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
  float32x4_t vmax_non_int_val = vdupq_n_f32(8388608.0f);
  float32x4_t vfilter = vreinterpretq_f32_u32(vcaltq_f32(a, vmax_non_int_val));
  float32x4_t vhalf = vdupq_n_f32(0.5f);
  float32x4_t vsign_mask = vdupq_n_f32(-0.0f);
  float32x4_t vsigned_half = or_f32(and_f32(a, vsign_mask), vhalf);
  float32x4_t vresult_away = vcvtq_f32_s32(
                                vcvtq_s32_f32(vaddq_f32(a, vsigned_half)));
  // vresult_away is round ties away from zero.
  // We want round ties to even.
  // If vresult_away is odd, and it was a tie, we need to correct by 1 towards 0.
  uint32x4_t tie = vceqq_f32(vabsq_f32(vsubq_f32(vresult_away, a)), vhalf);
  int32x4_t i_result_away = vcvtq_s32_f32(vresult_away);
  uint32x4_t odd = vcgtq_s32(
                      vandq_s32(i_result_away, vdupq_n_s32(1)), vdupq_n_s32(0));
  uint32x4_t correct_mask = vandq_u32(tie, odd);
  float32x4_t correction = and_f32(vreinterpretq_f32_u32(correct_mask),
                            or_f32(and_f32(a, vsign_mask), vdupq_n_f32(1.0f)));
  float32x4_t vresult = vsubq_f32(vresult_away, correction);

  return or_f32(and_f32(vfilter, vresult),
                and_f32(not_f32(vfilter), a));
#else
  return vrndnq_f32(a);
#endif
}

YNN_INTRINSIC int32x4_t cast_f32_to_int32(float32x4_t f) {
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
  return vcvtq_s32_f32(round_f32(f));
#else
  return vcvtnq_s32_f32(f);
#endif
}

YNN_INTRINSIC uint32x4_t cast_f32_to_uint32(float32x4_t f) {
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
  return vcvtq_u32_f32(round_f32(f));
#else
  return vcvtnq_u32_f32(f);
#endif
}

YNN_INTRINSIC int16x8_t saturating_cast_f32_to_int16(float32x4_t f0, float32x4_t f1) {
  return vcombine_s16(vqmovn_s32(cast_f32_to_int32(f0)), vqmovn_s32(cast_f32_to_int32(f1)));
}

YNN_INTRINSIC int8x16_t saturating_cast_f32_to_int8(float32x4_t f0, float32x4_t f1, float32x4_t f2, float32x4_t f3) {
  const int16x8_t i01_16 = vcombine_s16(vqmovn_s32(cast_f32_to_int32(f0)), vqmovn_s32(cast_f32_to_int32(f1)));
  const int16x8_t i23_16 = vcombine_s16(vqmovn_s32(cast_f32_to_int32(f2)), vqmovn_s32(cast_f32_to_int32(f3)));
  return vcombine_s8(vqmovn_s16(i01_16), vqmovn_s16(i23_16));
}

YNN_INTRINSIC uint8x16_t saturating_cast_f32_to_uint8(float32x4_t f0, float32x4_t f1, float32x4_t f2, float32x4_t f3) {
  const uint16x8_t i01_16 = vcombine_u16(vqmovn_u32(cast_f32_to_uint32(f0)), vqmovn_u32(cast_f32_to_uint32(f1)));
  const uint16x8_t i23_16 = vcombine_u16(vqmovn_u32(cast_f32_to_uint32(f2)), vqmovn_u32(cast_f32_to_uint32(f3)));
  return vcombine_u8(vqmovn_u16(i01_16), vqmovn_u16(i23_16));
}

YNN_INTRINSIC int16x8_t saturating_cast_int32_to_int16(int32x4_t a, int32x4_t b) {
  return vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
}

YNN_INTRINSIC int8x16_t saturating_cast_int16_to_int8(int16x8_t a, int16x8_t b) {
  return vcombine_s8(vqmovn_s16(a), vqmovn_s16(b));
}

YNN_INTRINSIC uint8x16_t saturating_cast_int16_to_uint8(int16x8_t a, int16x8_t b) {
  return vcombine_u8(vqmovun_s16(a), vqmovun_s16(b));
}

YNN_INTRINSIC float32x4_t ceil_f32(float32x4_t a) {
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
  float32x4_t vmax_non_int_val = vdupq_n_f32(8388608.0f);
  uint32x4_t vuse_rounding = vcaltq_f32(a, vmax_non_int_val);
  float32x4_t vtrunc = vcvtq_f32_s32(vcvtq_s32_f32(a));
  uint32x4_t vceil_mask = vcltq_f32(vtrunc, a);
  float32x4_t vone = vdupq_n_f32(1.0f);
  float32x4_t vceiled =
      vbslq_f32(vceil_mask, vaddq_f32(vtrunc, vone), vtrunc);
  return vbslq_f32(vuse_rounding, vceiled, a);
#else
  return vrndpq_f32(a);
#endif
}

YNN_INTRINSIC float32x4_t floor_f32(float32x4_t a) {
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
  float32x4_t vmax_non_int_val = vdupq_n_f32(8388608.0f);
  uint32x4_t vuse_rounding = vcaltq_f32(a, vmax_non_int_val);
  float32x4_t vtrunc = vcvtq_f32_s32(vcvtq_s32_f32(a));
  uint32x4_t vfloor_mask = vcgtq_f32(vtrunc, a);
  float32x4_t vone = vdupq_n_f32(1.0f);
  float32x4_t vfloored =
      vbslq_f32(vfloor_mask, vsubq_f32(vtrunc, vone), vtrunc);
  return vbslq_f32(vuse_rounding, vfloored, a);
#else
  return vrndmq_f32(a);
#endif
}

YNN_INTRINSIC float32x4_t sqrt_f32(float32x4_t a) {
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
    // Get the initial low-precision estimate of 1/sqrt(a).
    float32x4_t rsqrt_estimate = vrsqrteq_f32(a);

    // Perform one Newton-Raphson refinement
    rsqrt_estimate = vmulq_f32(vrsqrtsq_f32(vmulq_f32(a, rsqrt_estimate),
                                rsqrt_estimate), rsqrt_estimate);
    // Perform second Newton-Raphson refinement
    rsqrt_estimate = vmulq_f32(vrsqrtsq_f32(vmulq_f32(a, rsqrt_estimate),
                                rsqrt_estimate), rsqrt_estimate);

    float32x4_t sqrt_result = vmulq_f32(a, rsqrt_estimate);

    return sqrt_result;
#else
  return vsqrtq_f32(a);
#endif
}

} // namespace
"""

    self.types.update({
        Int(8, 16): "int8x16_t",
        Int(16, 8): "int16x8_t",
        Int(32, 4): "int32x4_t",
        UInt(8, 16): "uint8x16_t",
        UInt(16, 8): "uint16x8_t",
        UInt(32, 4): "uint32x4_t",
        Float(32, 4): "float32x4_t",
    })

    self.patterns += make_neon_float32_patterns(128)
    self.patterns += make_neon_integer_patterns(128)
    self.patterns += make_neon_cast_patterns(128)
    self.patterns += make_neon_reinterpret_cast_patterns(128)
    self.patterns += make_neon_broadcast_patterns(128)

  def update_for_fp16(self):
    self.header += """
namespace {

YNN_INTRINSIC float16x8_t cast_f32_to_f16(float32x4_t f0, float32x4_t f1) {
  return vcombine_f16(vcvt_f16_f32(f0), vcvt_f16_f32(f1));
}
} // namespace
"""

    self.types.update({
        Float(16, 4): "float16x4_t",
        Float(16, 8): "float16x8_t",
    })

  def __init__(self, features):
    Target.__init__(self)
    self.features = features
    self.vector_bits = 128
    self.tail_strategy = TailStrategy.MEMCPY

    self.header += "#include <arm_neon.h>\n"

    # These are transitive.
    implied_features = {
        "NEONFP16": ["NEON"],
        "FMA": ["NEON"],
    }
    all_features = []
    self.compute_all_features(features, implied_features, all_features)

    known_features = ["NEON", "NEONFP16", "FMA"]
    for feature in all_features:
      if feature not in known_features:
        raise ValueError(f"Unknown feature: {feature}")

    self.add_load_intrinsics()
    self.add_store_intrinsics()

    if "NEON" in all_features:
      self.update_for_neon()
    if "NEONFP16" in all_features:
      self.update_for_fp16()

  def arch_flags(self):
    return "|".join(["arch_flag::" + i.lower() for i in self.features])

  def arch_string(self):
    features_str = "_".join([i.lower() for i in self.features])
    return features_str
