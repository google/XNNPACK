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
              Op(Float(32, vector_bits // 32), "cast_f16_to_f32", [vf16_a]),
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


def make_neon_float32_patterns(vector_bits):
  assert vector_bits == 128
  return []


class ARM(Target):
  """NEON target for elementwise kernels compiler."""

  def update_for_neon(self):
    self.header += """

namespace {

YNN_INTRINSIC int32x4_t cast_f32_to_int32(float32x4_t f) {
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
  return vcvtq_s32_f32(ynn::simd::round(ynn::simd::f32x4{f}).v);
#else
  return vcvtnq_s32_f32(f);
#endif
}

YNN_INTRINSIC uint32x4_t cast_f32_to_uint32(float32x4_t f) {
#if defined(__ARM_ARCH) && __ARM_ARCH < 8
  return vcvtq_u32_f32(ynn::simd::round(ynn::simd::f32x4{f}).v);
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

} // namespace
"""

    self.types.update({
        Int(8, 16): "simd::vec<int8_t, 16>",
        Int(16, 8): "simd::vec<int16_t, 8>",
        Int(32, 4): "simd::vec<int32_t, 4>",
        UInt(8, 16): "simd::vec<uint8_t, 16>",
        UInt(16, 8): "simd::vec<uint16_t, 8>",
        UInt(32, 4): "simd::vec<uint32_t, 4>",
        Float(32, 4): "simd::vec<float, 4>",
    })

    self.patterns += make_neon_float32_patterns(128)
    self.patterns += make_neon_integer_patterns(128)
    self.patterns += make_neon_cast_patterns(128)
    self.patterns += make_neon_reinterpret_cast_patterns(128)

  def update_for_fp16(self):
    self.header += """
namespace {

YNN_INTRINSIC float32x4_t cast_f16_to_f32(uint16x4_t f0) {
  return vcvt_f32_f16(vreinterpret_f16_u16(f0));
}

YNN_INTRINSIC uint16x8_t cast_f32_to_f16(float32x4_t f0, float32x4_t f1) {
  return vreinterpretq_u16_f16(vcombine_f16(vcvt_f16_f32(f0), vcvt_f16_f32(f1)));
}

} // namespace
"""

    self.types.update({
        Float(16, 4): "simd::vec<half, 4>",
        Float(16, 8): "simd::vec<half, 8>",
    })

  def __init__(self, features):
    Target.__init__(self)
    self.features = features
    self.vector_bits = 128
    self.tail_strategy = TailStrategy.VECTOR

    self.header += "#include <arm_neon.h>\n"
    self.header += (
        '#include "ynnpack/base/simd/arm_neon.h"\n'
    )

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

    if "NEON" in all_features:
      self.update_for_neon()
    if "NEONFP16" in all_features:
      self.update_for_fp16()

  def arch_flags(self):
    return "|".join(["arch_flag::" + i.lower() for i in self.features])

  def arch_string(self):
    features_str = "_".join([i.lower() for i in self.features])
    return features_str
