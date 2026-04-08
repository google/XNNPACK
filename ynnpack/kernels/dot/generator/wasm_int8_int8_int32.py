# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for int8 wasm dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.wasm import wasm


class wasm_int8_int8_int32(wasm):
  def __init__(self, arch, tile_shape):
    super().__init__(arch, "int8_int8_int32", "int32_t", tile_shape)
    self.a_type = "int8_t"
    self.b_type = "int8_t"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def b_alignment_required(self):
    return self.tile_shape[1] * self.tile_shape[2] // 4

  def load_a_tile(self, i, k):
    return (
        f"int32_t a_val_{i}_{k} = (int32_t)*{self.a_ptr(i, k)};\n"
        f"v128_t a_{i}_{k} = wasm_i16x8_splat(a_val_{i}_{k});\n"
    )

  def load_b_tile(self, k, j):
    load32 = f"wasm_v128_load32_zero({self.b_ptr(k, j, 'int32_t')})"
    return f"v128_t b_{k}_{j} = wasm_i16x8_extend_low_i8x16({load32});\n"

  def product(self, i, j, k):
    prod = f"wasm_i32x4_extmul_low_i16x8(a_{i}_{k}, b_{k}_{j})"
    return f"c_{i}_{j} = wasm_i32x4_add(c_{i}_{j}, {prod});\n"


class wasm_simd128_int8_int8_int32(wasm_int8_int8_int32):

  def __init__(self, arch="simd128", tile_shape=(1, 4, 1)):
    super().__init__(arch, tile_shape)


generate_dot_kernels(
    wasm_simd128_int8_int8_int32(),
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
