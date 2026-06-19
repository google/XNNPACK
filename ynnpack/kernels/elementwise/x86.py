"""X86 target for elementwise kernels compiler."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import
from ynnpack.kernels.elementwise.rules import *  # pylint: disable=wildcard-import


class X86(Target):
  """X86 target for elementwise kernels compiler."""

  def update_for_sse2(self):
    """Updates the target for SSE2 support."""

  def update_for_sse41(self):
    """Updates the target for SSE41 support."""

  def update_for_avx(self):
    """Updates the target for AVX support."""

  def update_for_avx2(self):
    """Updates the target for AVX2 support."""

  def update_for_fma3(self):
    """Updates the target for FMA3 support."""
    self.patterns += add_fma_rules()

  def update_for_f16c(self):
    """Updates the target for F16C support."""

  def update_for_avx512(self):
    """Updates the target for AVX512 support."""
    self.patterns += add_fma_rules()

  def update_for_avx512bf16(self):
    """Updates the target for AVX512BF16 support."""

  def __init__(self, features):
    Target.__init__(self)
    self.patterns += add_select_rules()
    self.patterns += add_saturating_cast_rules()
    self.patterns += add_shift_rules()

    self.features = features

    # These are transitive.
    implied_features = {
        "SSE41": ["SSE2"],
        "AVX": ["SSE41"],
        "AVX2": ["AVX"],
        "F16C": ["AVX"],
        "FMA3": ["AVX"],
        "AVX512": ["AVX2", "FMA3"],
        "AVX512BF16": ["AVX512"],
    }
    all_features = []
    self.compute_all_features(features, implied_features, all_features)

    self.header += "#include <immintrin.h>\n"

    known_features = [
        "SSE2",
        "SSE2_FMA",
        "SSE41",
        "AVX",
        "AVX2",
        "FMA3",
        "F16C",
        "AVX512",
        "AVX512BF16",
    ]
    for feature in all_features:
      if feature not in known_features:
        raise ValueError(f"Unknown feature: {feature}")

    simd_header = ""
    if "AVX512" in all_features:
      simd_header = "x86_vec512.h"
      self.tail_strategy = TailStrategy.VECTOR
      self.vector_bits = 512
    elif "AVX2" in all_features:
      simd_header = "x86_vec256.h"
      self.tail_strategy = TailStrategy.VECTOR
      self.vector_bits = 256
    elif "AVX" in all_features:
      simd_header = "x86_vec256.h"
      self.tail_strategy = TailStrategy.VECTOR
      self.vector_bits = 256
    elif "SSE41" in all_features:
      simd_header = "x86_vec128.h"
      self.tail_strategy = TailStrategy.VECTOR
      self.vector_bits = 128
    elif "SSE2_FMA" in all_features:
      simd_header = "x86_vec128.h"
      self.tail_strategy = TailStrategy.VECTOR
      self.vector_bits = 128
    elif "SSE2" in all_features:
      simd_header = "x86_vec128.h"
      self.tail_strategy = TailStrategy.VECTOR
      self.vector_bits = 128

    self.header += (
        f'#include "ynnpack/base/simd/{simd_header}"\n'
    )

    if "AVX512BF16" in all_features:
      self.update_for_avx512bf16()
    if "AVX512" in all_features:
      self.update_for_avx512()
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
