# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp16 x86 dot kernel generators."""

# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring

from ynnpack.kernels.dot.generator.x86 import x86
from ynnpack.kernels.dot.generator.x86 import x86_avx
from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_fp16_fp16_fp32(x86):
  def __init__(self, arch, bits, tile_shape):
    super().__init__(arch, "fp16_fp16_fp32", "float", bits, tile_shape)
    self.a_type = "half"
    self.b_type = "half"

  def header(self):
    return super().header() + f"""

namespace {{

struct half {{
  std::int16_t value;
}};

YNN_INTRINSIC __m{self.bits} unaligned_load_broadcast(const half* ptr) {{
    __m{self.bits // 2}i v = {self._mm(self.bits // 2)}_set1_epi16(ptr->value);
    return {self._mm()}_cvtph_ps(v);
}}

}}  // namespace
"""

  def load_a_tile(self, i, k):
    return (
        f"__m{self.bits} a_{i}_{k} ="
        f" unaligned_load_broadcast({self.a_ptr(i, k)});\n"
    )

  def load_b_tile(self, k, j):
    # Since we are converting f16 (16-bit) to f32 (32-bit), the input data
    # occupies exactly half the width of the output vector.
    input_bits = self.bits // 2
    load_instruction = f"{self._mm(input_bits)}_loadu_si{input_bits}"

    return (
        f"__m{self.bits} b_{k}_{j} = {self._mm()}_cvtph_ps("
        f"{load_instruction}({self.b_ptr(k, j, f'__m{input_bits}i')}));\n"
    )

  def product(self, i, j, k):
    mul = f"{self._mm()}_mul_ps(a_{i}_{k}, b_{k}_{j})"
    return f"c_{i}_{j} = {self._mm()}_add_ps(c_{i}_{j}, {mul});\n"


class x86_f16c_fp16_fp16_fp32(x86_fp16_fp16_fp32, x86_avx):
  def __init__(self):
    super().__init__(arch="f16c", bits=256, tile_shape=(1, 8, 1))
    self.flags += ["dot_flag::consistent_arithmetic"]


class x86_f16c_fma3_fp16_fp16_fp32(x86_fp16_fp16_fp32, x86_avx):

  def __init__(self):
    super().__init__(arch="fma3", bits=256, tile_shape=(1, 8, 1))
    self.flags += ["dot_flag::consistent_arithmetic"]

  def product(self, i, j, k):
    return (
        f"c_{i}_{j} = {self._mm()}_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});\n"
    )


class x86_avx512_fp16_fp16_fp32(x86_fp16_fp16_fp32, x86_avx512):

  def __init__(self):
    super().__init__(arch="avx512", bits=512, tile_shape=(1, 16, 1))
    self.flags += ["dot_flag::consistent_arithmetic"]

  def product(self, i, j, k):
    return (
        f"c_{i}_{j} = {self._mm()}_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});\n"
    )
