"""Pattern matching rules which can be shared across backends."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


class Rule:
  """Represents a pattern matching and replacement rule."""

  def __init__(self, pattern, result, features=[], flags=0):  # pylint: disable=dangerous-default-value
    self.pattern = pattern
    self.result = result
    self.features = features
    self.flags = flags

  def __str__(self):
    return f"{self.pattern} -> {self.result} ({', '.join(self.features)})"

  def __repr__(self):
    return str(self)


def add_saturating_cast_rules():
  """Adds saturating cast patterns."""

  vf32_a = Var("a", Float(32, 0))

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
              UInt(8, 0),
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


def add_fma_rules():
  """Adds generic fma rewrite patterns."""
  vf32_a = Var("a", Float(32, 0))
  vf32_b = Var("b", Float(32, 0))
  vf32_c = Var("c", Float(32, 0))
  vf64_a = Var("a", Float(64, 0))
  vf64_b = Var("b", Float(64, 0))
  vf64_c = Var("c", Float(64, 0))
  return [
      Rule(
          multiply_add(vf32_a, vf32_b, vf32_c),
          Op(Float(32), "fma", [vf32_a, vf32_b, vf32_c]),
      ),
      Rule(
          multiply_add(vf64_a, vf64_b, vf64_c),
          Op(Float(64), "fma", [vf64_a, vf64_b, vf64_c]),
      )
  ]


def add_select_rules():
  """Adds generic select rewrite patterns."""
  x = WildCard()
  y = WildCard()
  z = WildCard()
  w = WildCard()
  return [Rule(select(x > y, z, w), select_greater_than(x, y, z, w))]


def add_shift_rules():
  """Adds generic shift rewrite patterns."""
  vi16_a = Var("a", Int(16, 0))
  i16_b = Var("b", Int(16))
  vi32_a = Var("a", Int(32, 0))
  i32_b = Var("b", Int(32))

  return [
      Rule(
          logical_shift_left(
              vi16_a,
              broadcast(i16_b, 0),
          ),
          logical_shift_left(vi16_a, i16_b),
      ),
      Rule(
          logical_shift_left(
              vi32_a,
              broadcast(i32_b, 0),
          ),
          logical_shift_left(vi32_a, i32_b),
      ),
  ]
