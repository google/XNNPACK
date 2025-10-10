"""Specializations for bf16 x86 dot kernel generators."""

from ynnpack.kernels.dot.generator.x86 import indent, x86, x86_avx, x86_avx512f

class x86_bf16_bf16_fp32(x86):
  def __init__(self, arch, bits, tile_shape):
    super().__init__(arch, "bf16_bf16_fp32", "float", bits, tile_shape)
    self.a_type = "bfloat16"
    self.b_type = "bfloat16"

  def header(self):
    return super().header() + """

namespace {

struct bfloat16 {
  std::uint16_t value;
};

}  // namespace
"""

class x86_avx2_bf16_bf16_fp32(x86_bf16_bf16_fp32, x86_avx):
  def __init__(self, arch="avx2", bits=256, tile_shape=(1, 8, 1)):
    super().__init__(arch, bits, tile_shape)

  def header(self):
    return super().header() + f"""

namespace {{

__m{self.bits} unaligned_load_broadcast_bf16_fp32(const bfloat16* ptr) {{
    uint16_t value;
    memcpy(&value, ptr, sizeof(bfloat16));
    return {self._mm()}_castsi{self.bits}_ps({self._mm()}_slli_epi32({self._mm()}_set1_epi16(value), 16));
}}

}}  // namespace
"""

  def load_a_tile(self, i, k):
    return (
        f"__m{self.bits} a_{i}_{k} ="
        f" unaligned_load_broadcast_bf16_fp32({self.a_ptr(i, k)});\n"
    )

  def load_b_tile(self, k, j):
    u16 = (
        f"{self._mm(self.bits//2)}_load_si{self.bits//2}({self.b_ptr(k, j, f'__m{self.bits//2}i')})"
    )
    u32 = f"{self._mm()}_slli_epi32({self._mm()}_cvtepi16_epi32({u16}), 16)"
    return (
        f"__m{self.bits} b_{k}_{j} ="
        f" {self._mm()}_castsi{self.bits}_ps({u32});\n"
    )

  def product(self, i, j, k):
    mul = f"{self._mm()}_mul_ps(a_{i}_{k}, b_{k}_{j})"
    return f"c_{i}_{j} = {self._mm()}_add_ps(c_{i}_{j}, {mul});\n"


class x86_avx2_fma3_bf16_bf16_fp32(x86_avx2_bf16_bf16_fp32):
  def __init__(self, arch="avx2_fma3", bits=256, tile_shape=(1, 8, 1)):
    super().__init__(arch, bits, tile_shape)

  def product(self, i, j, k):
    return (
        f"c_{i}_{j} = {self._mm()}_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});\n"
    )


class x86_avx512f_bf16_bf16_fp32(x86_bf16_bf16_fp32, x86_avx512f):
  def __init__(self, arch="avx512f", bits=512, tile_shape=(1, 16, 1)):
    super().__init__(arch, bits, tile_shape)

  def header(self):
    return super().header() + f"""

namespace {{

__m{self.bits} unaligned_load_broadcast_bf16_fp32(const bfloat16* ptr) {{
    uint16_t value;
    memcpy(&value, ptr, sizeof(bfloat16));
    return {self._mm()}_castsi{self.bits}_ps({self._mm()}_slli_epi32({self._mm()}_set1_epi16(value), 16));
}}

}}  // namespace
"""

  def load_a_tile(self, i, k):
    return (
        f"__m{self.bits} a_{i}_{k} ="
        f" unaligned_load_broadcast_bf16_fp32({self.a_ptr(i, k)});\n"
    )

  def load_b_tile(self, k, j):
    u16 = (
        f"{self._mm(self.bits//2)}_load_si{self.bits//2}({self.b_ptr(k, j, f'__m{self.bits//2}i')})"
    )
    u32 = f"{self._mm()}_slli_epi32({self._mm()}_cvtepi16_epi32({u16}), 16)"
    return (
        f"__m{self.bits} b_{k}_{j} ="
        f" {self._mm()}_castsi{self.bits}_ps({u32});\n"
    )

  def product(self, i, j, k):
    return (
        f"c_{i}_{j} = {self._mm()}_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});\n"
    )


class x86_avx512bf16_bf16_bf16_fp32(x86_bf16_bf16_fp32, x86_avx512f):
  def __init__(self, arch="avx512bf16", bits=512, tile_shape=(1, 16, 2)):
    super().__init__(arch, bits, tile_shape)

  def header(self):
    return super().header() + f"""

namespace {{

__m{self.bits} unaligned_load_broadcast_2xbf16(const bfloat16* ptr) {{
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
