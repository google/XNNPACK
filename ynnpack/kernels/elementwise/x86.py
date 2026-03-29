"""X86 target for elementwise kernels compiler."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.common_rules import *  # pylint: disable=wildcard-import
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


def make_x86_cast_patterns():
  """Adds x86 cast patterns."""

  return add_saturating_cast_rules()


def make_x86_integer_patterns():
  return add_shift_rules()


class X86(Target):
  """X86 target for elementwise kernels compiler."""

  def update_for_sse2(self):
    """Updates the target for SSE2 support."""
    self.patterns += make_x86_integer_patterns()
    self.patterns += make_x86_cast_patterns()
    self.header += """
namespace ynn {
namespace {
template <>
YNN_INTRINSIC ynn::simd::vec<float, 4> select_greater_than(ynn::simd::vec<float, 4> a, ynn::simd::vec<float, 4> b, ynn::simd::vec<float, 4> c, ynn::simd::vec<float, 4> d) {
  __m128 mask = _mm_cmpgt_ps(a.v, b.v);
  return ynn::simd::vec<float, 4>{_mm_or_ps(_mm_and_ps(mask, c.v), _mm_andnot_ps(mask, d.v))};
}
} // namespace
} // namespace ynn
"""

  def update_for_sse41(self):
    """Updates the target for SSE41 support."""

  def update_for_avx(self):
    """Updates the target for AVX support."""
    self.header += """
namespace ynn {
namespace {
template <>
YNN_INTRINSIC ynn::simd::vec<float, 8> select_greater_than(ynn::simd::vec<float, 8> a, ynn::simd::vec<float, 8> b, ynn::simd::vec<float, 8> c, ynn::simd::vec<float, 8> d) {
  __m256 mask = _mm256_cmp_ps(a.v, b.v, _CMP_GT_OS);
  return ynn::simd::vec<float, 8>{_mm256_blendv_ps(d.v, c.v, mask)};
}
} // namespace
} // namespace ynn
"""

  def update_for_avx2(self):
    """Updates the target for AVX2 support."""
    self.patterns += make_x86_integer_patterns()
    self.patterns += make_x86_cast_patterns()

  def update_for_fma3(self):
    """Updates the target for FMA3 support."""
    self.patterns += add_fma_rules()

  def update_for_f16c(self):
    """Updates the target for F16C support."""

  def update_for_avx512f(self):
    """Updates the target for AVX512F support."""
    self.patterns += add_fma_rules()
    self.patterns += make_x86_integer_patterns()
    self.patterns += make_x86_cast_patterns()
    self.header += """
namespace ynn {
namespace {
template <>
YNN_INTRINSIC ynn::simd::vec<float, 16> select_greater_than(ynn::simd::vec<float, 16> a, ynn::simd::vec<float, 16> b, ynn::simd::vec<float, 16> c, ynn::simd::vec<float, 16> d) {
  __mmask16 mask = _mm512_cmp_ps_mask(a.v, b.v, _CMP_GT_OS);
  return ynn::simd::vec<float, 16>{_mm512_mask_blend_ps(mask, d.v, c.v)};
}
} // namespace
} // namespace ynn
"""

  def update_for_avx512bf16(self):
    """Updates the target for AVX512BF16 support."""

  def update_for_avx512bw(self):
    """Updates the target for AVX512BW support."""

  def __init__(self, features):
    Target.__init__(self)
    self.patterns += add_select_rules()
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
