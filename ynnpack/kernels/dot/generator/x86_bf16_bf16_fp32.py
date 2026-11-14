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


class x86_bf16_bf16_fp32(x86):
  def __init__(self, arch, bits, tile_shape):
    super().__init__(arch, "bf16_bf16_fp32", "float", bits, tile_shape)
    self.a_type = "bfloat16"
    self.a_load_type = "std::uint16_t"
    self.b_type = "bfloat16"
    self.b_load_type = "std::uint16_t"
    # Very interestingly, fma doesn't seem to matter for bf16!
    self.flags += ["dot_flag::consistent_arithmetic"]

  def header(self):
    return super().header() + f"""

namespace {{

YNN_INTRINSIC __m{self.bits}i unaligned_load_broadcast_bf16x2(const std::uint16_t* ptr) {{
    int32_t value;
    memcpy(&value, ptr, sizeof(int32_t));
    return {self._mm()}_set1_epi32(value);
}}

}}  // namespace
"""

  # These load tile functions produce a value for both k = 0, and k = 1,
  # assuming that the `product` implementation will add these two partial tile
  # results together.
  def load_a_tile(self, i, k):
    bits = self.bits
    ptr = self.a_ptr(i, k)
    mask = f"{self._mm()}_set1_epi32(0xffff0000)"
    cast = f"{self._mm()}_castsi{bits}_ps"
    a_ik = f"a_{i}_{k}_k2"
    return f"""
__m{bits}i {a_ik} = unaligned_load_broadcast_bf16x2({ptr});
__m{bits} a_{i}_{k+0} = {cast}({self._mm()}_slli_epi32({a_ik}, 16));
__m{bits} a_{i}_{k+1} = {cast}({self._mm()}_and_si{bits}({a_ik}, {mask}));
"""

  def load_b_tile(self, k, j):
    bits = self.bits
    ptr = self.b_ptr(k, j, f"__m{bits}i")
    mask = f"{self._mm()}_set1_epi32(0xffff0000)"
    cast = f"{self._mm()}_castsi{bits}_ps"
    b_kj = f"b_{k}_{j}_k2"
    return f"""
__m{bits}i {b_kj} = {self._mm(bits)}_load_si{bits}({ptr});
__m{bits} b_{k+0}_{j} = {cast}({self._mm()}_slli_epi32({b_kj}, 16));
__m{bits} b_{k+1}_{j} = {cast}({self._mm()}_and_si{bits}({b_kj}, {mask}));
"""

  def generate_block(self, a_tail=False, b_tail=False):
    # Each unpacked bf16x2 tile produces 2 vector registers. When all C, A, and
    # B tiles exceed the vector register file, stream the larger of M or N
    # tiles one tile at a time (or split into two sub-k passes when even
    # streaming still overflows) to avoid register spills.
    m_tiles = self.block_shape[0] // self.tile_shape[0]
    n_tiles = self.block_shape[1] // self.tile_shape[1]
    num_regs = 32 if self.bits == 512 else 16
    num_c = m_tiles * n_tiles
    if num_c + 2 * (m_tiles + n_tiles) + 2 <= num_regs:
      return super().generate_block(a_tail=a_tail, b_tail=b_tail)
    # When m_tiles > n_tiles > 1 and 1-pass streaming still overflows (e.g.
    # 5x16x2 AVX2), neither 1-pass streaming nor 2-pass B re-unpacking beats
    # the default schedule.
    if (
        m_tiles > n_tiles
        and n_tiles > 1
        and num_c + 2 * n_tiles + 3 > num_regs
    ):
      return super().generate_block(a_tail=a_tail, b_tail=b_tail)
    result = ""
    load_a = self.load_a_tile_tail if a_tail else self.load_a_tile
    load_b = self.load_b_tile_tail if b_tail else self.load_b_tile
    if (
        m_tiles > n_tiles
        and n_tiles == 1
        and num_c + 2 * n_tiles + 4 > num_regs
    ) or (m_tiles <= n_tiles and num_c + 2 * m_tiles + 4 >= num_regs):
      # Register pressure is too tight to keep unpacked k+0 and k+1 tiles live
      # simultaneously; run k+1 then k+0 in two passes so only one k-slice is
      # live at a time (preserving the exact k+1 -> k+0 FMA reduction order).
      bits = self.bits
      mm = self._mm()
      cast = f"{mm}_castsi{bits}_ps"
      mask = f"{mm}_set1_epi32(0xffff0000)"
      for k in range(0, self.block_shape[2], self.tile_shape[2]):
        for sub_k in (1, 0):
          if m_tiles > n_tiles:
            for j in range(0, self.block_shape[1], self.tile_shape[1]):
              b_raw = f"{mm}_load_si{bits}({self.b_ptr(k, j, f'__m{bits}i')})"
              if sub_k == 1:
                b_val = f"{cast}({mm}_and_si{bits}({b_raw}, {mask}))"
              else:
                b_val = f"{cast}({mm}_slli_epi32({b_raw}, 16))"
              result += f"__m{bits} b_{k+sub_k}_{j} = {b_val};\n"
            for i in range(0, self.block_shape[0], self.tile_shape[0]):
              a_raw = f"{mm}_set1_epi16(*{self.a_ptr(i, k + sub_k)})"
              result += (
                  f"__m{bits} a_{i}_{k+sub_k} ="
                  f" {cast}({mm}_slli_epi32({a_raw}, 16));\n"
              )
              for j in range(0, self.block_shape[1], self.tile_shape[1]):
                result += self.product_sub_k(i, j, k + sub_k)
          else:
            for i in range(0, self.block_shape[0], self.tile_shape[0]):
              a_raw = f"{mm}_set1_epi16(*{self.a_ptr(i, k + sub_k)})"
              result += (
                  f"__m{bits} a_{i}_{k+sub_k} ="
                  f" {cast}({mm}_slli_epi32({a_raw}, 16));\n"
              )
            for j in range(0, self.block_shape[1], self.tile_shape[1]):
              b_raw = f"{mm}_load_si{bits}({self.b_ptr(k, j, f'__m{bits}i')})"
              if sub_k == 1:
                b_val = f"{cast}({mm}_and_si{bits}({b_raw}, {mask}))"
              else:
                b_val = f"{cast}({mm}_slli_epi32({b_raw}, 16))"
              result += f"__m{bits} b_{k+sub_k}_{j} = {b_val};\n"
              for i in range(0, self.block_shape[0], self.tile_shape[0]):
                result += self.product_sub_k(i, j, k + sub_k)
      result += "\n"
      return result

    for k in range(0, self.block_shape[2], self.tile_shape[2]):
      if m_tiles > n_tiles:
        for j in range(0, self.block_shape[1], self.tile_shape[1]):
          result += load_b(k, j)
        for i in range(0, self.block_shape[0], self.tile_shape[0]):
          result += load_a(i, k)
          for j in range(0, self.block_shape[1], self.tile_shape[1]):
            result += self.product(i, j, k)
      else:
        for i in range(0, self.block_shape[0], self.tile_shape[0]):
          result += load_a(i, k)
        for j in range(0, self.block_shape[1], self.tile_shape[1]):
          result += load_b(k, j)
          for i in range(0, self.block_shape[0], self.tile_shape[0]):
            result += self.product(i, j, k)
    result += "\n"
    return result


class x86_avx2_bf16_bf16_fp32(x86_bf16_bf16_fp32, x86_avx):
  def __init__(self, arch="avx2", bits=256, tile_shape=(1, 8, 2)):
    super().__init__(arch, bits, tile_shape)

  def product_sub_k(self, i, j, k_idx):
    add = f"{self._mm()}_add_ps"
    mul = f"{self._mm()}_mul_ps"
    return (
        f"c_{i}_{j} = {add}(c_{i}_{j}, {mul}(a_{i}_{k_idx}, b_{k_idx}_{j}));\n"
    )

  def product(self, i, j, k):
    # This "backwards" reduction order (1 first, then 0) is used to match the
    # behavior of dpbf16ps, which is documented to do the reduction in this
    # order, enabling all of these kernels to be numerically consistent.
    return self.product_sub_k(i, j, k + 1) + self.product_sub_k(i, j, k + 0)


class x86_avx2_fma3_bf16_bf16_fp32(x86_avx2_bf16_bf16_fp32):
  def __init__(self, arch="avx2_fma3", bits=256, tile_shape=(1, 8, 2)):
    super().__init__(arch, bits, tile_shape)

  def product_sub_k(self, i, j, k_idx):
    return (
        f"c_{i}_{j} = {self._mm()}_fmadd_ps("
        f"a_{i}_{k_idx}, b_{k_idx}_{j}, c_{i}_{j});\n"
    )


class x86_avx512_bf16_bf16_fp32(x86_bf16_bf16_fp32, x86_avx512):

  def __init__(self, arch="avx512", bits=512, tile_shape=(1, 16, 2)):
    super().__init__(arch, bits, tile_shape)

  def product_sub_k(self, i, j, k_idx):
    return (
        f"c_{i}_{j} = {self._mm()}_fmadd_ps("
        f"a_{i}_{k_idx}, b_{k_idx}_{j}, c_{i}_{j});\n"
    )

  def product(self, i, j, k):
    return self.product_sub_k(i, j, k + 1) + self.product_sub_k(i, j, k + 0)


class x86_avx512bf16_bf16_bf16_fp32(x86_bf16_bf16_fp32, x86_avx512):
  def __init__(self, arch="avx512bf16", bits=512, tile_shape=(1, 16, 2)):
    super().__init__(arch, bits, tile_shape)

  def header(self):
    return super().header() + f"""

namespace {{

YNN_INTRINSIC __m{self.bits} unaligned_load_broadcast_2xbf16(const std::uint16_t* ptr) {{
    float value;
    memcpy(&value, ptr, sizeof(float));
    return {self._mm()}_set1_ps(value);
}}

}}  // namespace
"""

  def load_a_tile(self, i, k):
    assert self.block_shape[2] % 2 == 0
    a = f"unaligned_load_broadcast_2xbf16({self.a_ptr(i, k)})"
    return f"""\
__m{self.bits}bh a_{i}_{k} = reinterpret_cast<__m{self.bits}bh>({a});
"""

  def load_b_tile(self, k, j):
    return (
        f"__m{self.bits}bh b_{k}_{j} ="
        f" reinterpret_cast<__m{self.bits}bh>({self._mm()}_load_ps({self.b_ptr(k, j)}));\n"
    )

  def product(self, i, j, k):
    return (
        f"c_{i}_{j} = {self._mm()}_dpbf16_ps(c_{i}_{j}, a_{i}_{k},"
        f" b_{k}_{j});\n"
    )

  def generate_block(self, a_tail=False, b_tail=False):
    return x86.generate_block(self, a_tail=a_tail, b_tail=b_tail)


generate_dot_kernels(
    x86_avx2_fma3_bf16_bf16_fp32(),
    [
        (1, 32, 2),
        (2, 32, 2),
        (1, 16, 2),
        (2, 16, 2),
        (3, 16, 2),
        (4, 16, 2),
        (5, 16, 2),
        (8, 8, 2),
        (10, 8, 2),
        (12, 8, 2),
    ],
)

generate_dot_kernels(
    x86_avx512_bf16_bf16_fp32(),
    [
        (1, 64, 2),
        (2, 64, 2),
        (3, 64, 2),
        (4, 64, 2),
        (5, 64, 2),
        (1, 32, 2),
        (2, 32, 2),
        (3, 32, 2),
        (4, 32, 2),
        (5, 32, 2),
        (6, 32, 2),
        (8, 32, 2),
        (10, 32, 2),
        (12, 32, 2),
        (16, 16, 2),
    ],
)

generate_dot_kernels(
    x86_avx512bf16_bf16_bf16_fp32(),
    [
        (1, 64, 2),
        (2, 64, 2),
        (3, 64, 2),
        (4, 64, 2),
        (5, 64, 2),
        (2, 32, 2),
        (3, 32, 2),
        (4, 32, 2),
        (5, 32, 2),
        (6, 32, 2),
        (8, 32, 2),
        (10, 32, 2),
        (12, 32, 2),
        (16, 16, 2),
    ],
)
