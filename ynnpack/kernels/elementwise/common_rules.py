"""Pattern matching rules which can be shared across backends."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


def add_saturating_cast_rules(vector_bits):
  """Adds saturating cast patterns."""

  vf32_a = f32_a.with_lanes(vector_bits // 32)
  vf32_b = f32_b.with_lanes(vector_bits // 32)
  vf32_c = f32_c.with_lanes(vector_bits // 32)
  vf32_d = f32_d.with_lanes(vector_bits // 32)

  vi32_a = i32_a.with_lanes(vector_bits // 32)
  vi32_b = i32_b.with_lanes(vector_bits // 32)

  vi16_a = i16_a.with_lanes(vector_bits // 16)
  vi16_b = i16_b.with_lanes(vector_bits // 16)

  return [
      Rule(
          saturating_cast(
              Int(8, vector_bits // 8),
              combine_vectors(
                  [round(vf32_a), round(vf32_b), round(vf32_c), round(vf32_d)]
              ),
          ),
          Op(
              Int(8, vector_bits // 8),
              "saturating_cast_f32_to_int8",
              [vf32_a, vf32_b, vf32_c, vf32_d],
          ),
      ),
      Rule(
          saturating_cast(
              UInt(8, vector_bits // 8),
              combine_vectors(
                  [round(vf32_a), round(vf32_b), round(vf32_c), round(vf32_d)]
              ),
          ),
          Op(
              UInt(8, vector_bits // 8),
              "saturating_cast_f32_to_uint8",
              [vf32_a, vf32_b, vf32_c, vf32_d],
          ),
      ),
      Rule(
          saturating_cast(
              Int(16, vector_bits // 16),
              combine_vectors([round(vf32_a), round(vf32_b)]),
          ),
          Op(
              Int(16, vector_bits // 16),
              "saturating_cast_f32_to_int16",
              [vf32_a, vf32_b],
          ),
      ),
      Rule(
          saturating_cast(
              Int(16, vector_bits // 16), combine_vectors([vi32_a, vi32_b])
          ),
          Op(
              Int(16, vector_bits // 16),
              "saturating_cast_int32_to_int16",
              [vi32_a, vi32_b],
          ),
      ),
      Rule(
          saturating_cast(
              Int(8, vector_bits // 8), combine_vectors([vi16_a, vi16_b])
          ),
          Op(
              Int(8, vector_bits // 8),
              "saturating_cast_int16_to_int8",
              [vi16_a, vi16_b],
          ),
      ),
      Rule(
          saturating_cast(
              UInt(8, vector_bits // 8), combine_vectors([vi16_a, vi16_b])
          ),
          Op(
              UInt(8, vector_bits // 8),
              "saturating_cast_int16_to_uint8",
              [vi16_a, vi16_b],
          ),
      ),
  ]
