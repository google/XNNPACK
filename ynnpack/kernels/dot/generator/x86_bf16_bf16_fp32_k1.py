# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for bf16 x86 dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.x86 import x86
from ynnpack.kernels.dot.generator.x86 import x86_avx
from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_bf16_bf16_fp32_k1(x86):
  def __init__(self, arch, bits, tile_shape):
    super().__init__(arch, "bf16_bf16_fp32", "float", bits, tile_shape)
    self.a_type = "bfloat16"
    self.b_type = "bfloat16"

  def header(self):
    return super().header() + """

namespace {

using bfloat16 = uint16_t;

}  // namespace
"""

  def load_a_tile(self, i, k):
    bits = self.bits
    cast = f"{self._mm()}_castsi{bits}_ps"
    bitwise_and = f"{self._mm()}_and_si{bits}"
    # We want to broadcast a single bf16 to a vector of f32. We implement this
    # with a 16-bit broadcast, and then masking off the low 16 bits of each
    # 32-bit value.
    a = f"{self._mm()}_set1_epi16(*{self.a_ptr(i, k)})"
    mask = f"{self._mm()}_set1_epi32(0xffff0000)"
    return f"__m{bits} a_{i}_{k} = {cast}({bitwise_and}({a}, {mask}));\n"

  def load_b_tile(self, k, j):
    bits = self.bits
    mm = self._mm()
    cast = f"{mm}_castsi{bits}_ps"
    b_ptr = self.b_ptr(k, j, f"__m{bits//2}i")
    b = f"{self._mm(bits//2)}_loadu_si{bits//2}({b_ptr})"
    b = f"{mm}_cvtepi16_epi32({b})"
    return f"__m{bits} b_{k}_{j} = {cast}({mm}_slli_epi32({b}, 16));\n"


class x86_avx2_bf16_bf16_fp32_k1(x86_bf16_bf16_fp32_k1, x86_avx):
  def __init__(self, arch="avx2", bits=256, tile_shape=(1, 8, 1)):
    super().__init__(arch, bits, tile_shape)

  def product(self, i, j, k):
    c_ij = f"c_{i}_{j}"
    a_ik = f"a_{i}_{k}"
    b_kj = f"b_{k}_{j}"
    mm = self._mm()
    return f"{c_ij} = {mm}_add_ps({c_ij}, {mm}_mul_ps({a_ik}, {b_kj}));\n"


class x86_avx2_fma3_bf16_bf16_fp32_k1(x86_avx2_bf16_bf16_fp32_k1):
  def __init__(self, arch="avx2_fma3", bits=256, tile_shape=(1, 8, 1)):
    super().__init__(arch, bits, tile_shape)

  def product(self, i, j, k):
    c_ij = f"c_{i}_{j}"
    mm = self._mm()
    return f"{c_ij} = {mm}_fmadd_ps(a_{i}_{k}, b_{k}_{j}, {c_ij});\n"


class x86_avx512_bf16_bf16_fp32_k1(x86_bf16_bf16_fp32_k1, x86_avx512):

  def __init__(self, arch="avx512", bits=512, tile_shape=(1, 16, 1)):
    super().__init__(arch, bits, tile_shape)

  def product(self, i, j, k):
    c_ij = f"c_{i}_{j}"
    mm = self._mm()
    return f"{c_ij} = {mm}_fmadd_ps(a_{i}_{k}, b_{k}_{j}, {c_ij});\n"


generate_dot_kernels(
    x86_avx2_fma3_bf16_bf16_fp32_k1(),
    [
        (1, 32, 1),
        (2, 32, 1),
        (1, 16, 1),
        (2, 16, 1),
        (3, 16, 1),
        (4, 16, 1),
        (5, 16, 1),
        (8, 8, 1),
        (10, 8, 1),
    ],
)

generate_dot_kernels(
    x86_avx512_bf16_bf16_fp32_k1(),
    [
        (1, 64, 1),
        (2, 64, 1),
        (3, 64, 1),
        (4, 64, 1),
        (5, 64, 1),
        (1, 32, 1),
        (2, 32, 1),
        (3, 32, 1),
        (4, 32, 1),
        (5, 32, 1),
        (6, 32, 1),
        (8, 32, 1),
        (10, 32, 1),
        (12, 32, 1),
        (16, 16, 1),
    ],
)
