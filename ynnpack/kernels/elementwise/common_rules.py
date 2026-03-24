"""Pattern matching rules which can be shared across backends."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


def add_saturating_cast_rules(vector_bits):
  """Adds saturating cast patterns."""

  vf32_a = f32_a.with_lanes(0)

  return [
      Rule(
          saturating_cast(
              Int(8, 0),
              round(vf32_a),
          ),
          Op(
              Int(8, 0),
              "saturating_rounding_cast",
              [vf32_a],
          ),
      ),
      Rule(
          saturating_cast(
              UInt(8, 0),
              round(vf32_a),
          ),
          Op(
              UInt(8, vector_bits // 8),
              "saturating_rounding_cast",
              [vf32_a],
          ),
      ),
      Rule(
          saturating_cast(
              Int(16, 0),
              round(vf32_a),
          ),
          Op(
              Int(16, 0),
              "saturating_rounding_cast",
              [vf32_a],
          ),
      ),
  ]


def add_select_rules():
  """Adds generic select rewrite patterns."""
  x = WildCard()
  y = WildCard()
  z = WildCard()
  w = WildCard()
  return [Rule(select(x > y, z, w), select_greater_than(x, y, z, w))]
