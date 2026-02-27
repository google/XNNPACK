# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base class for arm dot kernel generators.

Handles accumulating C tiles to the output.
"""

from ynnpack.kernels.dot.generator.dot_base import dot_base, indent


class arm_neon(dot_base):
  def __init__(self, arch, dot_type, c_type, tile_shape):
    super().__init__(arch, dot_type)
    self.tile_shape = tile_shape
    self.c_type = c_type

  def header(self):
    return super().header() + """
#include <arm_neon.h>

"""

  def init_c_tile(self, i, j):
    if self.c_type == "float":
      return f"float32x4_t c_{i}_{j} = vdupq_n_f32(0.0f);\n"
    elif self.c_type == "int32_t":
      return f"int32x4_t c_{i}_{j} = vdupq_n_s32(0);\n"
    else:
      raise ValueError(f"Unsupported c_type: {self.c_type}")

  def add_c_tile(self, i, j):
    if self.c_type == "float":
      return (
          f"c_{i}_{j} = vaddq_f32(c_{i}_{j},"
          f" vld1q_f32({self.c_in_ptr(i, j)}));\n"
      )
    elif self.c_type == "int32_t":
      return (
          f"c_{i}_{j} = vaddq_s32(c_{i}_{j},"
          f" vld1q_s32({self.c_in_ptr(i, j)}));\n"
      )
    else:
      raise ValueError(f"Unsupported c_type: {self.c_type}")

  def store_c_tile(self, i, j):
    if self.c_type == "float":
      return f"vst1q_f32({self.c_out_ptr(i, j)}, c_{i}_{j});\n"
    elif self.c_type == "int32_t":
      return f"vst1q_s32({self.c_out_ptr(i, j)}, c_{i}_{j});\n"
    else:
      raise ValueError(f"Unsupported c_type: {self.c_type}")

  def shift_c_tiles(self, n):
    assert n % self.tile_shape[1] == 0
    result = ""

    # Shift the accumulator registers down and the pointers up.
    for i in range(0, self.block_shape[0]):
      for j in range(0, n, self.tile_shape[1]):
        result += f"c_{i}_{j} = c_{i}_{j + n};\n"

    result += f"N -= {n};\n"
    result += (
        f"C_in = C_in ? offset_bytes(C_in, {n} * sizeof({self.c_type})) :"
        " nullptr;\n"
    )
    result += f"C_out = offset_bytes(C_out, {n} * sizeof({self.c_type}));\n"
    return result

  def add_c_block_vectors(self, n):
    assert(n % self.tile_shape[1] == 0)
    result = self.add_c_tiles(n)
    result += self.shift_c_tiles(n)
    return result

  def add_c_block_vector_tail(self):
    result = ""
    result += f"assert(N < {self.tile_shape[1]});\n"

    for i in range(0, self.block_shape[0]):
      if self.c_type == "float":
        result += f"float32x2_t c_{i}_0_lo = vget_low_f32(c_{i}_0);\n"
      elif self.c_type == "int32_t":
        result += f"int32x2_t c_{i}_0_lo = vget_low_s32(c_{i}_0);\n"
      else:
        raise ValueError(f"Unsupported c_type: {self.c_type}")

    result += "if (N & 2) {\n"

    add_c_tiles = ""
    # Load all of the output and add it, before writing anything.
    for i in range(0, self.block_shape[0]):
      if self.c_type == "float":
        add_c_tiles += (
            f"c_{i}_0_lo = vadd_f32(c_{i}_0_lo,"
            f" vld1_f32({self.c_in_ptr(i, 0)}));\n"
        )
      elif self.c_type == "int32_t":
        add_c_tiles += (
            f"c_{i}_0_lo = vadd_s32(c_{i}_0_lo,"
            f" vld1_s32({self.c_in_ptr(i, 0)}));\n"
        )
      else:
        raise ValueError(f"Unsupported c_type: {self.c_type}")

    result += "  if (C_in) {\n"
    result += indent(add_c_tiles, "    ") + "\n"
    result += "  }\n"

    for i in reversed(range(0, self.block_shape[0])):
      if self.c_type == "float":
        result += f"  vst1_f32({self.c_out_ptr(i, 0)}, c_{i}_0_lo);\n"
        result += f"  c_{i}_0_lo = vget_high_f32(c_{i}_0);\n"
      elif self.c_type == "int32_t":
        result += f"  vst1_s32({self.c_out_ptr(i, 0)}, c_{i}_0_lo);\n"
        result += f"  c_{i}_0_lo = vget_high_s32(c_{i}_0);\n"
      else:
        raise ValueError(f"Unsupported c_type: {self.c_type}")

    result += (
        f"  C_in = C_in ? offset_bytes(C_in, 2 * sizeof({self.c_type})) :"
        " nullptr;\n"
    )
    result += f"  C_out = offset_bytes(C_out, 2 * sizeof({self.c_type}));\n"
    result += "}\n"
    result += "if (N & 1) {\n"

    add_c_tiles = ""
    # Load all of the output and add it, before writing anything.
    for i in range(0, self.block_shape[0]):
      if self.c_type == "float":
        add_c_tiles += (
            f"c_{i}_0_lo = vadd_f32(c_{i}_0_lo,"
            f" vdup_n_f32(*{self.c_in_ptr(i, 0)}));\n"
        )
      elif self.c_type == "int32_t":
        add_c_tiles += (
            f"c_{i}_0_lo = vadd_s32(c_{i}_0_lo,"
            f" vdup_n_s32(*{self.c_in_ptr(i, 0)}));\n"
        )
      else:
        raise ValueError(f"Unsupported c_type: {self.c_type}")

    result += "  if (C_in) {\n"
    result += indent(add_c_tiles, "    ") + "\n"
    result += "  }\n"

    for i in reversed(range(0, self.block_shape[0])):
      if self.c_type == "float":
        result += f"  vst1_lane_f32({self.c_out_ptr(i, 0)}, c_{i}_0_lo, 0);\n"
      elif self.c_type == "int32_t":
        result += f"  vst1_lane_s32({self.c_out_ptr(i, 0)}, c_{i}_0_lo, 0);\n"
      else:
        raise ValueError(f"Unsupported c_type: {self.c_type}")

    result += "}\n"

    result += "N = 0;\n"
    return result

  def add_c_block_tail(self):
    result = ""
    n = self.block_shape[1] // 2
    while n >= self.tile_shape[1]:
      # This might be a whole vector.
      result += f"if (N & {n}) {{\n"
      result += indent(self.add_c_block_vectors(n), "  ")
      result += "\n}\n"
      n //= 2
    result += "if (N > 0) {\n"
    result += indent(self.add_c_block_vector_tail(), "  ")
    result += "\n}\n"

    return result
