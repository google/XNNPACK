"""Hexagon HVX target for elementwise kernels compiler."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import
from ynnpack.kernels.elementwise.rules import *  # pylint: disable=wildcard-import


class Hexagon(Target):
  """Hexagon HVX target for elementwise kernels compiler."""

  def update_for_hvx(self):
    """Updates the target for HVX support."""
    # HVX has no native blend/select on float vectors, so we implement
    # `select_greater_than` using `Q6_Q_vcmp_gt_VsfVsf` to form a vector
    # predicate and `Q6_V_vmux_QVV` to select between the two operands.
    self.header += """
namespace ynn {
namespace {
template <>
YNN_INTRINSIC simd::vec<float, 32> select_greater_than(simd::vec<float, 32> a, simd::vec<float, 32> b, simd::vec<float, 32> c, simd::vec<float, 32> d) {
  HVX_VectorPred mask = Q6_Q_vcmp_gt_VsfVsf(a.v, b.v);
  return simd::vec<float, 32>{Q6_V_vmux_QVV(mask, c.v, d.v)};
}
}
} // namespace ynn
"""

  def __init__(self, features):
    Target.__init__(self)
    self.patterns += add_select_rules()
    self.patterns += add_saturating_cast_rules()
    self.patterns += add_shift_rules()

    self.features = features
    # HVX vectors are 1024 bits wide (128 bytes).
    self.vector_bits = 1024
    self.tail_strategy = TailStrategy.VECTOR

    # These are transitive.
    implied_features = {
        "HVX": [],
    }
    all_features = []
    self.compute_all_features(features, implied_features, all_features)

    known_features = ["HVX"]
    for feature in all_features:
      if feature not in known_features:
        raise ValueError(f"Unknown feature: {feature}")

    self.header += (
        '#include "ynnpack/base/simd/hexagon_hvx.h"\n'
    )

    if "HVX" in all_features:
      self.update_for_hvx()

  def arch_flags(self):
    return "|".join(["arch_flag::" + i.lower() for i in self.features])

  def arch_string(self):
    features_str = "_".join([i.lower() for i in self.features])
    return features_str