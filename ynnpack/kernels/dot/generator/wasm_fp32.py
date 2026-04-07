# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp32 Wasm dot kernel generators."""

# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring

from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.wasm import wasm


class wasm_fp32(wasm):

  def __init__(self, arch, tile_shape):
    super().__init__(arch, "fp32", "float", tile_shape)
    self.a_type = "float"
    self.b_type = "float"

  def load_a_tile(self, i, k):
    a_ptr = self.a_ptr(i, k)
    return f"v128_t a_{i}_{k} = wasm_f32x4_splat(*{a_ptr});\n"

  def load_b_tile(self, k, j):
    b_ptr = self.b_ptr(k, j)
    return f"v128_t b_{k}_{j} = wasm_v128_load({b_ptr});\n"

  def product(self, i, j, k):
    c_ij = f"c_{i}_{j}"
    return (
        f"{c_ij} = wasm_f32x4_add({c_ij}, wasm_f32x4_mul(a_{i}_{k},"
        f" b_{k}_{j}));\n"
    )


class wasm_simd128_fp32(wasm_fp32):
  def __init__(self):
    super().__init__("simd128", (1, 4, 1))


generate_dot_kernels(
    wasm_simd128_fp32(),
    [
        (1, 16, 1),
        (2, 16, 1),
        (3, 16, 1),
        (1, 8, 1),
        (2, 8, 1),
        (3, 8, 1),
        (4, 8, 1),
        (4, 4, 1),
        (6, 4, 1),
        (8, 4, 1),
    ],
)
