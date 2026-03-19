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


def make_x86_fma_patterns(vector_bits):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              multiply_add(f32_a, f32_b, f32_c),
              Op(Float(32), "fma", [f32_a, f32_b, f32_c]),
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

  def update_for_fma3(self):
    """Updates the target for FMA3 support."""
    self.patterns += make_x86_fma_patterns(256)

  def update_for_f16c(self):
    """Updates the target for F16C support."""

  def update_for_avx512f(self):
    """Updates the target for AVX512F support."""
    self.patterns += make_x86_fma_patterns(512)
    self.patterns += make_x86_integer_patterns(512, "_mm512_")
    self.patterns += make_x86_cast_patterns(512)

  def update_for_avx512bf16(self):
    """Updates the target for AVX512BF16 support."""

  def update_for_avx512bw(self):
    """Updates the target for AVX512BW support."""

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
      self.header += (
          '#include "ynnpack/base/simd/x86_fma3.h"\n'
      )
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
