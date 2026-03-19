"""ARM NEON target for elementwise kernels compiler."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.common_rules import add_saturating_cast_rules
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


def make_neon_cast_patterns(vector_bits):
  """Adds NEON cast patterns."""
  assert vector_bits == 128

  return add_saturating_cast_rules(vector_bits)


def make_neon_integer_patterns(vector_bits):
  assert vector_bits == 128
  return [
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

  def update_for_fp16(self):
    """Updates the target for FP16 support."""

  def update_for_fma(self):
    self.patterns += make_neon_fma_patterns(128)

  def __init__(self, features):
    Target.__init__(self)
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
