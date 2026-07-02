# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base class for Hexagon HVX dot kernel generators."""

# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import dot_base
from ynnpack.kernels.dot.generator.dot_base import indent


class hexagon_hvx(dot_base):
  """Hexagon HVX dot kernel generator."""

  def __init__(self, arch, dot_type, c_type, tile_shape):
    super().__init__(arch, dot_type)
    self.tile_shape = tile_shape
    self.c_type = c_type

  def header(self):
    return """
#include "ynnpack/base/simd/hexagon_hvx.h"
using ynn::simd::f32x32;
""" + super().header()

  def init_c_tile(self, i, j):
    if self.c_type == "float":
      return f"f32x32 c_{i}_{j}(0.0f);\n"
    else:
      raise ValueError(f"Unsupported c_type: {self.c_type}")

  def add_c_tile(self, i, j):
    if self.c_type == "float":
      return (
          f"c_{i}_{j} = c_{i}_{j} +"
          f" simd::load({self.c_in_ptr(i, j)}, f32x32::N);\n"
      )
    else:
      raise ValueError(f"Unsupported c_type: {self.c_type}")

  def store_c_tile(self, i, j):
    if self.c_type == "float":
      return f"simd::store({self.c_out_ptr(i, j)}, c_{i}_{j});\n"
    else:
      raise ValueError(f"Unsupported c_type: {self.c_type}")

  def shift_c_tiles(self, n):
    assert n % self.tile_shape[1] == 0
    result = ""
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
    assert n % self.tile_shape[1] == 0
    result = self.add_c_tiles(n)
    result += self.shift_c_tiles(n)
    return result

  def product(self, i, j, k):
    c_ij = f"c_{i}_{j}"
    return f"{c_ij} = {c_ij} + a_{i}_{k} * b_{k}_{j};\n"

  def add_c_block_vector_tail(self):
    result = ""
    result += f"assert(N < {self.tile_shape[1]});\n"

    add_c_tiles = ""
    for i in range(0, self.block_shape[0]):
      add_c_tiles += (
          f"c_{i}_0 = c_{i}_0 + simd::load({self.c_in_ptr(i, 0)}, N,"
          f" simd::undef<{self.tile_shape[1]}>());\n"
      )

    result += "if (C_in) {\n"
    result += indent(add_c_tiles, "  ") + "\n"
    result += "}\n"

    for i in reversed(range(0, self.block_shape[0])):
      result += f"simd::store({self.c_out_ptr(i, 0)}, c_{i}_0, N);\n"

    result += "N = 0;\n"
    return result

  def add_c_block_tail(self):
    result = ""
    n = self.block_shape[1] // 2
    while n >= self.tile_shape[1]:
      result += f"if (N >= {n}) {{\n"
      result += indent(self.add_c_block_vectors(n), "  ")
      result += "\n}\n"
      n //= 2
    if self.block_shape[1] > self.tile_shape[1]:
      result += "if (N > 0) {\n"
      result += indent(self.add_c_block_vector_tail(), "  ")
      result += "\n}\n"
    else:
      result += "if (N > 0) {\n"
      result += indent(self.add_c_block_vector_tail(), "  ")
      result += "\n}\n"

    return result
