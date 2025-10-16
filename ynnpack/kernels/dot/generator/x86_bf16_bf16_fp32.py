"""Specializations for bf16 x86 dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.x86 import x86
from ynnpack.kernels.dot.generator.x86 import x86_avx
from ynnpack.kernels.dot.generator.x86 import x86_avx512f


class x86_bf16_bf16_fp32(x86):
  def __init__(self, arch, bits, tile_shape):
    super().__init__(arch, "bf16_bf16_fp32", "float", bits, tile_shape)
    self.a_type = "bfloat16"
    self.b_type = "bfloat16"

  def header(self):
    return super().header() + f"""

namespace {{

struct bfloat16 {{
  std::uint16_t value;
}};

YNN_INTRINSIC __m{self.bits}i unaligned_load_broadcast_bf32x2(const bfloat16* ptr) {{
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
__m{bits}i {a_ik} = unaligned_load_broadcast_bf32x2({ptr});
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


class x86_avx2_bf16_bf16_fp32(x86_bf16_bf16_fp32, x86_avx):
  def __init__(self, arch="avx2", bits=256, tile_shape=(1, 8, 2)):
    super().__init__(arch, bits, tile_shape)

  def product(self, i, j, k):
    add = f"{self._mm()}_add_ps"
    mul = f"{self._mm()}_mul_ps"
    return f"""
c_{i}_{j} = {add}(c_{i}_{j}, {mul}(a_{i}_{k+0}, b_{k+0}_{j}));
c_{i}_{j} = {add}(c_{i}_{j}, {mul}(a_{i}_{k+1}, b_{k+1}_{j}));
"""


class x86_avx2_fma3_bf16_bf16_fp32(x86_avx2_bf16_bf16_fp32):
  def __init__(self, arch="avx2_fma3", bits=256, tile_shape=(1, 8, 2)):
    super().__init__(arch, bits, tile_shape)

  def product(self, i, j, k):
    return f"""
c_{i}_{j} = {self._mm()}_fmadd_ps(a_{i}_{k+0}, b_{k+0}_{j}, c_{i}_{j});
c_{i}_{j} = {self._mm()}_fmadd_ps(a_{i}_{k+1}, b_{k+1}_{j}, c_{i}_{j});
"""


class x86_avx512f_bf16_bf16_fp32(x86_bf16_bf16_fp32, x86_avx512f):
  def __init__(self, arch="avx512f", bits=512, tile_shape=(1, 16, 2)):
    super().__init__(arch, bits, tile_shape)

  def product(self, i, j, k):
    return f"""
c_{i}_{j} = {self._mm()}_fmadd_ps(a_{i}_{k+0}, b_{k+0}_{j}, c_{i}_{j});
c_{i}_{j} = {self._mm()}_fmadd_ps(a_{i}_{k+1}, b_{k+1}_{j}, c_{i}_{j});
"""


class x86_avx512bf16_bf16_bf16_fp32(x86_bf16_bf16_fp32, x86_avx512f):
  def __init__(self, arch="avx512bf16", bits=512, tile_shape=(1, 16, 2)):
    super().__init__(arch, bits, tile_shape)

  def header(self):
    return super().header() + f"""

namespace {{

YNN_INTRINSIC __m{self.bits} unaligned_load_broadcast_2xbf16(const bfloat16* ptr) {{
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
