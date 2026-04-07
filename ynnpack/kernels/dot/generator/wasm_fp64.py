# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp64 arm dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.wasm import wasm


class wasm_fp64(wasm):

  def __init__(self, arch, tile_shape):
    super().__init__(arch, "fp64", "double", tile_shape)
    self.a_type = "double"
    self.b_type = "double"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def load_a_tile(self, i, k):
    a_ptr = self.a_ptr(i, k)
    return f"v128_t a_{i}_{k} = wasm_f64x2_splat(*{a_ptr});\n"

  def load_b_tile(self, k, j):
    b_ptr = self.b_ptr(k, j)
    return f"v128_t b_{k}_{j} = wasm_v128_load({b_ptr});\n"

  def product(self, i, j, k):
    c_ij = f"c_{i}_{j}"
    return (
        f"{c_ij} = wasm_f64x2_add({c_ij}, wasm_f64x2_mul(a_{i}_{k},"
        f" b_{k}_{j}));\n"
    )


class wasm_simd128_fp64(wasm_fp64):

  def __init__(self):
    super().__init__("simd128", (1, 2, 1))


generate_dot_kernels(
    wasm_simd128_fp64(),
    [
        (1, 16, 2),
        (2, 16, 2),
        (3, 16, 2),
        (2, 8, 2),
        (3, 8, 2),
        (4, 8, 2),
        (5, 8, 2),
        (4, 4, 2),
        (6, 4, 2),
        (8, 4, 2),
        (8, 2, 2),
    ],
)
