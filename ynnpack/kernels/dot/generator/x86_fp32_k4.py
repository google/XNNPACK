"""Specializations for fp32 x86 dot kernel generators."""

from ynnpack.kernels.dot.generator.x86 import x86_avx


class x86_avx512f_fp32_k4(x86_avx):
  def __init__(self, arch = "avx512f", bits = 128, tile_shape = (1, 4, 4)):
    super().__init__(arch, "fp32", "float", bits, tile_shape)
    self.a_type = "float"
    self.b_type = "float"
    # This kernel already has 4 accumulators per tile in m.
    self.min_tiles = max(1, self.min_tiles // 4)

  def header(self):
    return super().header() + """
namespace {

__m512 unaligned_load_broadcast_x4(const float* ptr) {
    __m128 value = _mm_loadu_ps(ptr);
    return _mm512_broadcast_f32x4(value);
}

__m512 _mm512_hadd_ps(__m512 a, __m512 b) {
    return _mm512_add_ps(_mm512_unpacklo_ps(a, b), _mm512_unpackhi_ps(a, b));
}

}  // namespace
"""

  def init_c_tile(self, i, j):
    if i % 4 != 0: return ""
    result = f"""
__m512 c4_{i+0}_{j} = _mm512_setzero_ps();
__m512 c4_{i+1}_{j} = _mm512_setzero_ps();
__m512 c4_{i+2}_{j} = _mm512_setzero_ps();
__m512 c4_{i+3}_{j} = _mm512_setzero_ps();
"""
    return result

  # In this generator, we tell the base class that we are generating 128-bit
  # tiles, but we accumulate in 512-bit vectors. Here, we process 4 rows of
  # 512-bit accumulators, and turn them into 4 rows of 128-bit outputs.
  def finalize_c_tile(self, i, j):
    if i % 4 != 0: return ""
    return f"""
c4_{i+0}_{j} = _mm512_hadd_ps(c4_{i+0}_{j}, c4_{i+1}_{j});
c4_{i+2}_{j} = _mm512_hadd_ps(c4_{i+2}_{j}, c4_{i+3}_{j});
c4_{i}_{j} = _mm512_hadd_ps(c4_{i}_{j}, c4_{i+2}_{j});
c4_{i}_{j} = _mm512_permutexvar_ps(_mm512_set_epi32(15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0), c4_{i}_{j});
__m128 c_{i+0}_{j} = _mm512_extractf32x4_ps(c4_{i}_{j}, 0);
__m128 c_{i+1}_{j} = _mm512_extractf32x4_ps(c4_{i}_{j}, 1);
__m128 c_{i+2}_{j} = _mm512_extractf32x4_ps(c4_{i}_{j}, 2);
__m128 c_{i+3}_{j} = _mm512_extractf32x4_ps(c4_{i}_{j}, 3);
"""

  def load_a_tile(self, i, k):
    return f"""\
__m512 a_{i}_{k} = unaligned_load_broadcast_x4({self.a_ptr(i, k)});
"""

  def load_b_tile(self, k, j):
    return f"__m512 b_{k}_{j} = _mm512_load_ps({self.b_ptr(k, j)});\n"

  def product(self, i, j, k):
    return f"c4_{i}_{j} = _mm512_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c4_{i}_{j});\n"
