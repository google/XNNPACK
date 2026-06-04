"""WASM SIMD target for elementwise kernels compiler."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import
from ynnpack.kernels.elementwise.rules import *  # pylint: disable=wildcard-import


class WASM(Target):
  """WASM SIMD target for elementwise kernels compiler."""

  def update_for_simd128(self):
    """Updates the target for WASM SIMD128 support."""

  def __init__(self, features):
    Target.__init__(self)
    self.patterns += add_select_rules()
    self.patterns += add_saturating_cast_rules()
    self.patterns += add_shift_rules()

    self.features = features
    self.vector_bits = 128
    self.tail_strategy = TailStrategy.VECTOR

    # These are transitive.
    implied_features = {
        "SIMD128": [],
    }
    all_features = []
    self.compute_all_features(features, implied_features, all_features)

    known_features = ["SIMD128"]
    for feature in all_features:
      if feature not in known_features:
        raise ValueError(f"Unknown feature: {feature}")

    self.header += (
        '#include "ynnpack/base/simd/wasm_simd128.h"\n'
    )

    if "SIMD128" in all_features:
      self.update_for_simd128()

  def arch_flags(self):
    return "|".join(["arch_flag::" + i.lower() for i in self.features])

  def arch_string(self):
    features_str = "_".join([i.lower() for i in self.features])
    return features_str
