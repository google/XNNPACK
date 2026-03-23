"""ARM NEON target for elementwise kernels compiler."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.common_rules import *  # pylint: disable=wildcard-import
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


def make_neon_cast_patterns(vector_bits):
  """Adds NEON cast patterns."""
  assert vector_bits == 128

  return add_saturating_cast_rules(vector_bits)


def make_neon_integer_patterns(vector_bits):
  assert vector_bits == 128
  return [
      Rule(
          logical_shift_left(i16_a.with_lanes(0), broadcast(i16_b, 0)),
          logical_shift_left(i16_a.with_lanes(0), i16_b),
      ),
      Rule(
          logical_shift_left(i32_a.with_lanes(0), broadcast(i32_b, 0)),
          logical_shift_left(i32_a.with_lanes(0), i32_b),
      ),
  ]


def make_neon_float32_patterns(vector_bits):
  assert vector_bits == 128
  return []


def make_neon_fma_patterns(vector_bits):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              multiply_add(f32_a, f32_b, f32_c),
              Op(Float(32), "fma", [f32_a, f32_b, f32_c]),
              ["FMA"],
          ),
      ]
  ]


class ARM(Target):
  """NEON target for elementwise kernels compiler."""

  def update_for_neon(self):
    """Updates the target for NEON support."""
    self.patterns += make_neon_float32_patterns(128)
    self.patterns += make_neon_integer_patterns(128)
    self.patterns += make_neon_cast_patterns(128)
    self.header += """
namespace ynn {
namespace {
template <>
YNN_INTRINSIC simd::vec<float, 4> select_greater_than(simd::vec<float, 4> a, simd::vec<float, 4> b, simd::vec<float, 4> c, simd::vec<float, 4> d) {
  uint32x4_t mask = vcgtq_f32(a.v, b.v);
  return simd::vec<float, 4>{vbslq_f32(mask, c.v, d.v)};
}
}
} // namespace ynn
"""

  def update_for_fp16(self):
    """Updates the target for FP16 support."""

  def update_for_fma(self):
    self.patterns += make_neon_fma_patterns(128)

  def __init__(self, features):
    Target.__init__(self)
    self.patterns += add_select_rules()
    self.features = features
    self.vector_bits = 128
    self.tail_strategy = TailStrategy.VECTOR

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

    self.header += "#include <arm_neon.h>\n"
    self.header += (
        '#include "ynnpack/base/simd/arm_neon.h"\n'
    )

    if "NEON" in all_features:
      self.update_for_neon()
    if "NEONFP16" in all_features:
      self.header += (
          '#include "ynnpack/base/simd/arm_neonfp16.h"\n'
      )
      self.update_for_fp16()
    if "FMA" in all_features:
      self.header += (
          '#include "ynnpack/base/simd/arm_neonfma.h"\n'
      )
      self.update_for_fma()

  def arch_flags(self):
    return "|".join(["arch_flag::" + i.lower() for i in self.features])

  def arch_string(self):
    features_str = "_".join([i.lower() for i in self.features])
    return features_str
