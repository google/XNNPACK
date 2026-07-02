# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp32 Hexagon HVX dot kernel generators."""

# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring

from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.hexagon import hexagon_hvx


class hexagon_hvx_fp32(hexagon_hvx):

  def __init__(self, arch, tile_shape):
    super().__init__(arch, "fp32", "float", tile_shape)
    self.a_type = "float"
    self.b_type = "float"

  def load_a_tile(self, i, k):
    a_ptr = self.a_ptr(i, k)
    return f"f32x32 a_{i}_{k}(*{a_ptr});\n"

  def load_b_tile(self, k, j):
    b_ptr = self.b_ptr(k, j)
    return f"f32x32 b_{k}_{j} = ynn::simd::load_aligned({b_ptr}, f32x32::N);\n"


class hexagon_hvx_fp32_32x1(hexagon_hvx_fp32):

  def __init__(self):
    super().__init__("hvx", (1, 32, 1))


generate_dot_kernels(
    hexagon_hvx_fp32_32x1(),
    [
        (1, 32, 1),
        (1, 64, 1),
        (1, 128, 1),
        (2, 64, 1),
        (4, 32, 1),
        (4, 64, 1),
        (6, 64, 1),
        (8, 32, 1),
    ],
)
