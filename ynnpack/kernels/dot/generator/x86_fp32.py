"""Specializations for fp32 x86 dot kernel generators."""

from ynnpack.kernels.dot.generator.x86 import x86, x86_avx, x86_avx512f, x86_sse2


class x86_fp32(x86):
  def __init__(self, arch, bits, tile_shape):
    super().__init__(arch, "fp32", "float", bits, tile_shape)
    self.a_type = "float"
    self.b_type = "float"

  def header(self):
    return super().header() + f"""

namespace {{

YNN_INTRINSIC __m{self.bits} unaligned_load_broadcast(const float* ptr) {{
    float value;
    memcpy(&value, ptr, sizeof(float));
    return {self._mm()}_set1_ps(value);
}}

}}  // namespace
"""

  def load_a_tile(self, i, k):
    return (
        f"__m{self.bits} a_{i}_{k} = unaligned_load_broadcast({self.a_ptr(i, k)});\n"
    )

  def load_b_tile(self, k, j):
    return (
        f"__m{self.bits} b_{k}_{j} = {self._mm()}_load_ps({self.b_ptr(k, j)});\n"
    )

  def product(self, i, j, k):
    mul = f"{self._mm()}_mul_ps(a_{i}_{k}, b_{k}_{j})"
    return f"c_{i}_{j} = {self._mm()}_add_ps(c_{i}_{j}, {mul});\n"


class x86_sse2_fp32(x86_fp32, x86_sse2):
  def __init__(self):
    super().__init__("sse2", 128, (1, 4, 1))


class x86_avx_fp32(x86_fp32, x86_avx):
  def __init__(self):
    super().__init__("avx", 256, (1, 8, 1))


class x86_fma3_fp32(x86_fp32, x86_avx):
  def __init__(self):
    super().__init__("fma3", 256, (1, 8, 1))

  def product(self, i, j, k):
    return f"c_{i}_{j} = {self._mm()}_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});\n"


class x86_avx512f_fp32(x86_fp32, x86_avx512f):
  def __init__(self):
    super().__init__("avx512f", 512, (1, 16, 1))

  def product(self, i, j, k):
    return f"c_{i}_{j} = {self._mm()}_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});\n"
