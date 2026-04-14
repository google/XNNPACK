# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=invalid-name

"""Base class for wasm dot kernel generators.

Handles accumulating C tiles to the output.
"""

from ynnpack.kernels.dot.generator.dot_base import dot_base, indent


class wasm(dot_base):
  """WASM dot kernel generator."""

  def __init__(self, arch, dot_type, c_type, tile_shape):
    super().__init__(arch, dot_type)
    self.tile_shape = tile_shape
    self.c_type = c_type

  def header(self):
    return super().header() + """
#include <wasm_simd128.h>

"""

  def c_bits(self):
    if self.c_type == "float":
      return 32
    elif self.c_type == "double":
      return 64
    elif self.c_type == "int32_t":
      return 32
    else:
      raise ValueError(f"Unsupported c_type: {self.c_type}")

  def init_c_tile(self, i, j):
    if self.c_type == "float":
      return f"v128_t c_{i}_{j} = wasm_f32x4_const_splat(0.0f);\n"
    elif self.c_type == "double":
      return f"v128_t c_{i}_{j} = wasm_f64x2_splat(0.0);\n"
    elif self.c_type == "int32_t":
      return f"v128_t c_{i}_{j} = wasm_i32x4_const_splat(0);\n"
    else:
      raise ValueError(f"Unsupported c_type: {self.c_type}")

  def add_c_tile(self, i, j):
    if self.c_type == "float":
      return (
          f"c_{i}_{j} = wasm_f32x4_add(c_{i}_{j},"
          f" wasm_v128_load({self.c_in_ptr(i, j)}));\n"
      )
    elif self.c_type == "double":
      return (
          f"c_{i}_{j} = wasm_f64x2_add(c_{i}_{j},"
          f" wasm_v128_load({self.c_in_ptr(i, j)}));\n"
      )
    elif self.c_type == "int32_t":
      return (
          f"c_{i}_{j} = wasm_i32x4_add(c_{i}_{j},"
          f" wasm_v128_load({self.c_in_ptr(i, j)}));\n"
      )
    else:
      raise ValueError(f"Unsupported c_type: {self.c_type}")

  def store_c_tile(self, i, j):
    if self.c_type in ("float", "double", "int32_t"):
      return f"wasm_v128_store({self.c_out_ptr(i, j)}, c_{i}_{j});\n"
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

    if self.c_bits() == 32:
      # We only need the half-vector case for 32-bit types.

      result += "if (N & 2) {\n"

      add_c_tiles = ""
      # Load all of the output and add it, before writing anything.
      for i in range(0, self.block_shape[0]):
        if self.c_type == "float":
          add_c_tiles += (
              f"c_{i}_0 = wasm_f32x4_add(c_{i}_0,"
              f" wasm_v128_load64_zero({self.c_in_ptr(i, 0)}));\n"
          )
        elif self.c_type == "int32_t":
          add_c_tiles += (
              f"c_{i}_0 = wasm_i32x4_add(c_{i}_0,"
              f" wasm_v128_load64_zero({self.c_in_ptr(i, 0)}));\n"
          )
        else:
          raise ValueError(f"Unsupported c_type: {self.c_type}")

      result += "  if (C_in) {\n"
      result += indent(add_c_tiles, "    ") + "\n"
      result += "  }\n"

      for i in reversed(range(0, self.block_shape[0])):
        result += (
            f"  wasm_v128_store64_lane({self.c_out_ptr(i, 0)}, c_{i}_0, 0);\n"
        )
        result += f"  c_{i}_0 = wasm_i64x2_shuffle(c_{i}_0, c_{i}_0, 1, 1);\n"

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
            f"c_{i}_0 = wasm_f32x4_add(c_{i}_0,"
            f" wasm_v128_load32_zero({self.c_in_ptr(i, 0)}));\n"
        )
      elif self.c_type == "double":
        add_c_tiles += (
            f"c_{i}_0 = wasm_f64x2_add(c_{i}_0,"
            f" wasm_v128_load64_zero({self.c_in_ptr(i, 0)}));\n"
        )
      elif self.c_type == "int32_t":
        add_c_tiles += (
            f"c_{i}_0 = wasm_i32x4_add(c_{i}_0,"
            f" wasm_v128_load32_zero({self.c_in_ptr(i, 0)}));\n"
        )
      else:
        raise ValueError(f"Unsupported c_type: {self.c_type}")

    result += "  if (C_in) {\n"
    result += indent(add_c_tiles, "    ") + "\n"
    result += "  }\n"

    for i in reversed(range(0, self.block_shape[0])):
      if self.c_bits() == 32:
        result += (
            f"  wasm_v128_store32_lane({self.c_out_ptr(i, 0)}, c_{i}_0, 0);\n"
        )
      elif self.c_type == "double":
        result += (
            f"  wasm_v128_store64_lane({self.c_out_ptr(i, 0)}, c_{i}_0, 0);\n"
        )
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
    if self.block_shape[1] > self.tile_shape[1]:
      result += "if (N > 0) {\n"
      result += indent(self.add_c_block_vector_tail(), "  ")
      result += "\n}\n"
    else:
      result += "assert(N > 0);\n"
      result += self.add_c_block_vector_tail()

    return result
