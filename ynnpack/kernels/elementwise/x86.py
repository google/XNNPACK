"""X86 target for elementwise kernels compiler."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.common_rules import add_saturating_cast_rules
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


def make_x86_cast_patterns(vector_bits):
  """Adds x86 cast patterns."""

  return add_saturating_cast_rules(vector_bits)


def make_x86_integer_patterns(vector_bits, prefix):
  return [
      Rule(
          logical_shift_left(
              i16_a.with_lanes(vector_bits // 16),
              broadcast(i16_b, vector_bits // 16),
          ),
          Op(Int(16, vector_bits // 16), prefix + "slli_epi16", [i16_a, i16_b]),
      ),
      Rule(
          logical_shift_left(
              i32_a.with_lanes(vector_bits // 32),
              broadcast(i32_b, vector_bits // 32),
          ),
          Op(
              Int(32, vector_bits // 32),
              prefix + "slli_epi32",
              [i32_a.with_lanes(vector_bits // 32), i32_b],
          ),
      ),
      Rule(
          u32_a.with_lanes(vector_bits // 32)
          >> broadcast(u32_b, vector_bits // 32),
          Op(
              UInt(32, vector_bits // 32),
              prefix + "srli_epi32",
              [u32_a.with_lanes(vector_bits // 32), u32_b],
          ),
      ),
  ]


# TODO(vksnk): These are only correct for SSE2
def make_x86_float_comparison_patterns(vector_bits, prefix):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              equal(f32_a, f32_b),
              Op(Float(32), prefix + "cmpeq_ps", [f32_a, f32_b]),
          ),
          Rule(
              not_equal(f32_a, f32_b),
              Op(Float(32), prefix + "cmpneq_ps", [f32_a, f32_b]),
          ),
          Rule(
              f32_a > f32_b,
              Op(Float(32), prefix + "cmpgt_ps", [f32_a, f32_b]),
          ),
          Rule(
              f32_a < f32_b,
              Op(Float(32), prefix + "cmplt_ps", [f32_a, f32_b]),
          ),
          Rule(
              f32_a >= f32_b,
              Op(Float(32), prefix + "cmpge_ps", [f32_a, f32_b]),
          ),
          Rule(
              f32_a <= f32_b,
              Op(Float(32), prefix + "cmple_ps", [f32_a, f32_b]),
          ),
      ]
  ]


# TODO(vksnk): These are only correct for SSE2
def make_x86_integer_comparison_patterns(vector_bits, prefix):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              equal(i8_a, i8_b), Op(Int(8), prefix + "cmpeq_epi8", [i8_a, i8_b])
          ),
          Rule(
              equal(i16_a, i16_b),
              Op(Int(16), prefix + "cmpeq_epi16", [i16_a, i16_b]),
          ),
          Rule(
              equal(i32_a, i32_b),
              Op(Int(32), prefix + "cmpeq_epi32", [i32_a, i32_b]),
          ),
          Rule(
              i8_a > i8_b,
              Op(Int(8), prefix + "cmpgt_epi8", [i8_a, i8_b]),
          ),
          Rule(
              i16_a > i16_b,
              Op(Int(16), prefix + "cmpgt_epi16", [i16_a, i16_b]),
          ),
          Rule(
              i32_a > i32_b,
              Op(Int(32), prefix + "cmpgt_epi32", [i32_a, i32_b]),
          ),
          Rule(
              i8_a < i8_b,
              Op(Int(8), prefix + "cmpgt_epi8", [i8_b, i8_a]),
          ),
          Rule(
              i16_a < i16_b,
              Op(Int(16), prefix + "cmpgt_epi16", [i16_b, i16_a]),
          ),
          Rule(
              i32_a < i32_b,
              Op(Int(32), prefix + "cmpgt_epi32", [i32_b, i32_a]),
          ),
      ]
  ]


def make_x86_fma_patterns(vector_bits, prefix):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              multiply_add(f32_a, f32_b, f32_c),
              Op(Float(32), prefix + "fmadd_ps", [f32_a, f32_b, f32_c]),
              ["FMA3", "AVX512F"],
          ),
          Rule(
              multiply_sub(f32_a, f32_b, f32_c),
              Op(Float(32), prefix + "fmsub_ps", [f32_a, f32_b, f32_c]),
              ["FMA3", "AVX512F"],
          ),
      ]
  ]


class X86(Target):
  """X86 target for elementwise kernels compiler."""

  def update_for_sse2(self):
    """Updates the target for SSE2 support."""
    self.patterns += make_x86_integer_patterns(128, "_mm_")
    self.patterns += make_x86_cast_patterns(128)
    self.patterns += make_x86_float_comparison_patterns(128, "_mm_")
    self.patterns += make_x86_integer_comparison_patterns(128, "_mm_")

    self.header += """
namespace {

YNN_INTRINSIC __m128i saturating_cast_f32_to_int8(__m128 f0, __m128 f1, __m128 f2, __m128 f3) {
  const __m128 max_int16 = _mm_set1_ps((1 << 15) - 1);
  f0 = _mm_min_ps(f0, max_int16);
  f1 = _mm_min_ps(f1, max_int16);
  f2 = _mm_min_ps(f2, max_int16);
  f3 = _mm_min_ps(f3, max_int16);
  const __m128i i0 = _mm_cvtps_epi32(f0);
  const __m128i i1 = _mm_cvtps_epi32(f1);
  const __m128i i2 = _mm_cvtps_epi32(f2);
  const __m128i i3 = _mm_cvtps_epi32(f3);
  const __m128i i01_16 = _mm_packs_epi32(i0, i1);
  const __m128i i23_16 = _mm_packs_epi32(i2, i3);
  return _mm_packs_epi16(i01_16, i23_16);
}

YNN_INTRINSIC __m128i saturating_cast_f32_to_int16(__m128 f0, __m128 f1) {
  const __m128 max_int16 = _mm_set1_ps((1 << 15) - 1);
  f0 = _mm_min_ps(f0, max_int16);
  f1 = _mm_min_ps(f1, max_int16);
  const __m128i i0 = _mm_cvtps_epi32(f0);
  const __m128i i1 = _mm_cvtps_epi32(f1);
  return _mm_packs_epi32(i0, i1);
}

YNN_INTRINSIC __m128i saturating_cast_int32_to_int16(__m128i a, __m128i b) {
  return _mm_packs_epi32(a, b);
}

YNN_INTRINSIC __m128i saturating_cast_int16_to_int8(__m128i a, __m128i b) {
  return _mm_packs_epi16(a, b);
}

YNN_INTRINSIC __m128i saturating_cast_int16_to_uint8(__m128i a, __m128i b) {
  return _mm_packus_epi16(a, b);
}

YNN_INTRINSIC __m128i saturating_cast_f32_to_uint8(__m128 f0, __m128 f1, __m128 f2, __m128 f3) {
  const __m128 max_uint16 = _mm_set1_ps((1 << 16) - 1);
  f0 = _mm_min_ps(f0, max_uint16);
  f1 = _mm_min_ps(f1, max_uint16);
  const __m128i i0 = _mm_cvtps_epi32(f0);
  const __m128i i1 = _mm_cvtps_epi32(f1);
  const __m128i i2 = _mm_cvtps_epi32(f2);
  const __m128i i3 = _mm_cvtps_epi32(f3);
  const __m128i i01_16 = _mm_packs_epi32(i0, i1);
  const __m128i i23_16 = _mm_packs_epi32(i2, i3);
  return _mm_packus_epi16(i01_16, i23_16);
}

} // namespace
"""

  def update_for_sse41(self):
    """Updates the target for SSE41 support."""

  def update_for_avx(self):
    """Updates the target for AVX support."""
    self.header += """
namespace {

YNN_INTRINSIC __m256 greater_than(__m256 a, __m256 b) {
  return _mm256_cmp_ps(a, b, _CMP_GT_OS);
}

} // namespace

"""

  def update_for_avx2(self):
    """Updates the target for AVX2 support."""
    self.patterns += make_x86_integer_patterns(256, "_mm256_")
    self.patterns += make_x86_cast_patterns(256)

    self.header += """
namespace {

YNN_INTRINSIC __m256i saturating_cast_f32_to_int8(__m256 f0, __m256 f1, __m256 f2, __m256 f3) {
  const __m256 max_int16 = _mm256_set1_ps((1 << 15) - 1);
  f0 = _mm256_min_ps(f0, max_int16);
  f1 = _mm256_min_ps(f1, max_int16);
  f2 = _mm256_min_ps(f2, max_int16);
  f3 = _mm256_min_ps(f3, max_int16);
  const __m256i i0 = _mm256_cvtps_epi32(f0);
  const __m256i i1 = _mm256_cvtps_epi32(f1);
  const __m256i i2 = _mm256_cvtps_epi32(f2);
  const __m256i i3 = _mm256_cvtps_epi32(f3);
  const __m256i i01_16 = _mm256_packs_epi32(i0, i1);
  const __m256i i23_16 = _mm256_packs_epi32(i2, i3);
  const __m256i r = _mm256_packs_epi16(i01_16, i23_16);
  return _mm256_permutevar8x32_epi32(r, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

YNN_INTRINSIC __m256i saturating_cast_f32_to_int16(__m256 f0, __m256 f1) {
  const __m256 max_int16 = _mm256_set1_ps((1 << 15) - 1);
  f0 = _mm256_min_ps(f0, max_int16);
  f1 = _mm256_min_ps(f1, max_int16);
  const __m256i i0 = _mm256_cvtps_epi32(f0);
  const __m256i i1 = _mm256_cvtps_epi32(f1);
  const __m256i i01_16 = _mm256_packs_epi32(i0, i1);
  return _mm256_permute4x64_epi64(i01_16, (0 << 0) + (2 << 2) + (1 << 4) + (3 << 6));
}

YNN_INTRINSIC __m256i saturating_cast_int32_to_int16(__m256i a, __m256i b) {
  const __m256i r = _mm256_packs_epi32(a, b);
  return _mm256_permute4x64_epi64(r, (0 << 0) + (2 << 2) + (1 << 4) + (3 << 6));
}

YNN_INTRINSIC __m256i saturating_cast_int16_to_int8(__m256i a, __m256i b) {
  const __m256i r = _mm256_packs_epi16(a, b);
  return _mm256_permute4x64_epi64(r, (0 << 0) + (2 << 2) + (1 << 4) + (3 << 6));
}

YNN_INTRINSIC __m256i saturating_cast_int16_to_uint8(__m256i a, __m256i b) {
  const __m256i r = _mm256_packus_epi16(a, b);
  return _mm256_permute4x64_epi64(r, (0 << 0) + (2 << 2) + (1 << 4) + (3 << 6));
}

YNN_INTRINSIC __m256i saturating_cast_f32_to_uint8(__m256 f0, __m256 f1, __m256 f2, __m256 f3) {
  const __m256 max_uint16 = _mm256_set1_ps((1 << 16) - 1);
  f0 = _mm256_min_ps(f0, max_uint16);
  f1 = _mm256_min_ps(f1, max_uint16);
  f2 = _mm256_min_ps(f2, max_uint16);
  f3 = _mm256_min_ps(f3, max_uint16);
  const __m256i i0 = _mm256_cvtps_epi32(f0);
  const __m256i i1 = _mm256_cvtps_epi32(f1);
  const __m256i i2 = _mm256_cvtps_epi32(f2);
  const __m256i i3 = _mm256_cvtps_epi32(f3);
  const __m256i i01_16 = _mm256_packs_epi32(i0, i1);
  const __m256i i23_16 = _mm256_packs_epi32(i2, i3);
  const __m256i r = _mm256_packus_epi16(i01_16, i23_16);
  return _mm256_permutevar8x32_epi32(r, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

}  // namespace
"""

  def update_for_fma3(self):
    """Updates the target for FMA3 support."""
    self.patterns += make_x86_fma_patterns(256, "_mm256_")

  def update_for_f16c(self):
    """Updates the target for F16C support."""

  def update_for_avx512f(self):
    """Updates the target for AVX512F support."""
    self.patterns += make_x86_fma_patterns(512, "_mm512_")
    self.patterns += make_x86_integer_patterns(512, "_mm512_")
    self.patterns += make_x86_cast_patterns(512)

  def update_for_avx512bf16(self):
    """Updates the target for AVX512BF16 support."""

  def update_for_avx512bw(self):
    """Updates the target for AVX512BW support."""
    self.header += """
namespace {

YNN_INTRINSIC __m512i saturating_cast_f32_to_int8(__m512 f0, __m512 f1, __m512 f2, __m512 f3) {
  const __m512 max_int16 = _mm512_set1_ps((1 << 15) - 1);
  f0 = _mm512_min_ps(f0, max_int16);
  f1 = _mm512_min_ps(f1, max_int16);
  f2 = _mm512_min_ps(f2, max_int16);
  f3 = _mm512_min_ps(f3, max_int16);
  const __m512i i0 = _mm512_cvtps_epi32(f0);
  const __m512i i1 = _mm512_cvtps_epi32(f1);
  const __m512i i2 = _mm512_cvtps_epi32(f2);
  const __m512i i3 = _mm512_cvtps_epi32(f3);
  const __m512i i01_16 = _mm512_packs_epi32(i0, i1);
  const __m512i i23_16 = _mm512_packs_epi32(i2, i3);
  const __m512i r = _mm512_packs_epi16(i01_16, i23_16);
  return _mm512_permutexvar_epi32(_mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), r);
}

YNN_INTRINSIC __m512i saturating_cast_f32_to_uint8(__m512 f0, __m512 f1, __m512 f2, __m512 f3) {
  const __m512 max_uint16 = _mm512_set1_ps((1 << 16) - 1);
  f0 = _mm512_min_ps(f0, max_uint16);
  f1 = _mm512_min_ps(f1, max_uint16);
  const __m512i i0 = _mm512_cvtps_epi32(f0);
  const __m512i i1 = _mm512_cvtps_epi32(f1);
  const __m512i i2 = _mm512_cvtps_epi32(f2);
  const __m512i i3 = _mm512_cvtps_epi32(f3);
  const __m512i i01_16 = _mm512_packus_epi32(i0, i1);
  const __m512i i23_16 = _mm512_packus_epi32(i2, i3);
  const __m512i r = _mm512_packus_epi16(i01_16, i23_16);
  return _mm512_permutexvar_epi32(_mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), r);
}

YNN_INTRINSIC __m512i saturating_cast_f32_to_int16(__m512 f0, __m512 f1) {
  const __m512 max_int16 = _mm512_set1_ps((1 << 15) - 1);
  f0 = _mm512_min_ps(f0, max_int16);
  f1 = _mm512_min_ps(f1, max_int16);
  const __m512i i0 = _mm512_cvtps_epi32(f0);
  const __m512i i1 = _mm512_cvtps_epi32(f1);
  const __m512i r = _mm512_packs_epi32(i0, i1);
  return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r);
}

YNN_INTRINSIC __m512i saturating_cast_int32_to_int16(__m512i a, __m512i b) {
  const __m512i r = _mm512_packs_epi32(a, b);
  return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r);
}

YNN_INTRINSIC __m512i saturating_cast_int16_to_int8(__m512i a, __m512i b) {
  const __m512i r = _mm512_packs_epi16(a, b);
  return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r);
}

YNN_INTRINSIC __m512i saturating_cast_int16_to_uint8(__m512i a, __m512i b) {
  const __m512i r = _mm512_packus_epi16(a, b);
  return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r);
}

} // namespace

"""

  def get_natural_lanes_num(self, ty):
    """Returns a number of lanes in the native vector type."""
    # TODO(vksnk): this a temporary workaround until we get rid of combine/split
    if ty.type_class == "float" and ty.size == 16:
      return (self.vector_bits // 2) // ty.size
    return self.vector_bits // ty.size

  def __init__(self, features):
    Target.__init__(self)
    self.features = features

    # These are transitive.
    implied_features = {
        "SSE41": ["SSE2"],
        "AVX": ["SSE41"],
        "AVX2": ["AVX"],
        "F16C": ["AVX"],
        "FMA3": ["AVX"],
        "AVX512F": ["AVX2", "FMA3"],
        "AVX512BW": ["AVX512F"],
        "AVX512BF16": ["AVX512BW"],
    }
    all_features = []
    self.compute_all_features(features, implied_features, all_features)

    self.header += "#include <immintrin.h>\n"

    known_features = [
        "SSE2",
        "SSE41",
        "AVX",
        "AVX2",
        "FMA3",
        "F16C",
        "AVX512F",
        "AVX512BW",
        "AVX512BF16",
    ]
    for feature in all_features:
      if feature not in known_features:
        raise ValueError(f"Unknown feature: {feature}")

    simd_header = ""
    if "AVX512F" in all_features:
      simd_header = "x86_avx512.h"
      self.tail_strategy = TailStrategy.VECTOR
      self.vector_bits = 512
    elif "AVX2" in all_features:
      simd_header = "x86_avx2.h"
      self.tail_strategy = TailStrategy.VECTOR
      self.vector_bits = 256
    elif "AVX" in all_features:
      simd_header = "x86_avx.h"
      self.tail_strategy = TailStrategy.VECTOR
      self.vector_bits = 256
    elif "SSE41" in all_features:
      simd_header = "x86_sse41.h"
      self.tail_strategy = TailStrategy.VECTOR
      self.vector_bits = 128
    elif "SSE2" in all_features:
      simd_header = "x86_sse2.h"
      self.tail_strategy = TailStrategy.VECTOR
      self.vector_bits = 128

    self.header += (
        f'#include "ynnpack/base/simd/{simd_header}"\n'
    )

    if "F16C" in all_features:
      self.header += (
          '#include "ynnpack/base/simd/x86_f16c.h"\n'
      )
    if "AVX512BW" in all_features:
      self.update_for_avx512bw()
    if "AVX512BF16" in all_features:
      self.update_for_avx512bf16()
    if "AVX512F" in all_features:
      self.update_for_avx512f()
    if "FMA3" in all_features:
      self.update_for_fma3()
    if "F16C" in all_features:
      self.update_for_f16c()
    if "AVX2" in all_features:
      self.update_for_avx2()
    if "AVX" in all_features:
      self.update_for_avx()
    if "SSE41" in all_features:
      self.update_for_sse41()
    if "SSE2" in all_features:
      self.update_for_sse2()
