"""X86 target for elementwise kernels compiler."""

# pylint: disable=undefined-variable
from ynnpack.kernels.elementwise.common_rules import add_saturating_cast_rules
from ynnpack.kernels.elementwise.compiler import *  # pylint: disable=wildcard-import


def make_x86_cast_patterns(vector_bits, prefix):
  """Adds x86 cast patterns."""

  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              cast(Float(32), i32_a),
              Op(Float(32), prefix + "cvtepi32_ps", [i32_a]),
          ),
          Rule(
              cast(Int(32), round(f32_a)),
              Op(Int(32), prefix + "cvtps_epi32", [f32_a]),
          ),
      ]
  ] + add_saturating_cast_rules(vector_bits)


def make_x86_reinterpret_cast_patterns(vector_bits, prefix):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              Op(Float(32), "reinterpret_cast", [i32_a]),
              Op(
                  Float(32),
                  prefix + "castsi" + str(vector_bits) + "_ps",
                  [i32_a],
              ),
          ),
          Rule(
              Op(Int(32), "reinterpret_cast", [f32_a]),
              Op(Int(32), prefix + "castps_si" + str(vector_bits), [f32_a]),
          ),
          Rule(
              Op(Float(32), "reinterpret_cast", [u32_a]),
              Op(
                  Float(32),
                  prefix + "castsi" + str(vector_bits) + "_ps",
                  [u32_a],
              ),
          ),
          Rule(
              Op(UInt(32), "reinterpret_cast", [f32_a]),
              Op(UInt(32), prefix + "castps_si" + str(vector_bits), [f32_a]),
          ),
      ]
  ]


def make_x86_integer_patterns(vector_bits, prefix):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(u8_a + u8_b, Op(UInt(8), prefix + "add_epi8", [u8_a, u8_b])),
          Rule(i8_a + i8_b, Op(Int(8), prefix + "add_epi8", [i8_a, i8_b])),
          Rule(
              u16_a + u16_b, Op(UInt(16), prefix + "add_epi16", [u16_a, u16_b])
          ),
          Rule(
              i16_a + i16_b, Op(Int(16), prefix + "add_epi16", [i16_a, i16_b])
          ),
          Rule(
              u32_a + u32_b, Op(UInt(32), prefix + "add_epi32", [u32_a, u32_b])
          ),
          Rule(
              i32_a + i32_b, Op(Int(32), prefix + "add_epi32", [i32_a, i32_b])
          ),
          Rule(u8_a - u8_b, Op(UInt(8), prefix + "sub_epi8", [u8_a, u8_b])),
          Rule(i8_a - i8_b, Op(Int(8), prefix + "sub_epi8", [i8_a, i8_b])),
          Rule(
              u16_a - u16_b, Op(UInt(16), prefix + "sub_epi16", [u16_a, u16_b])
          ),
          Rule(
              i16_a - i16_b, Op(Int(16), prefix + "sub_epi16", [i16_a, i16_b])
          ),
          Rule(
              u32_a - u32_b, Op(UInt(32), prefix + "sub_epi32", [u32_a, u32_b])
          ),
          Rule(
              i32_a - i32_b, Op(Int(32), prefix + "sub_epi32", [i32_a, i32_b])
          ),
          Rule(
              i32_a * i32_b, Op(Int(32), prefix + "mullo_epi32", [i32_a, i32_b])
          ),
          Rule(min(u8_a, u8_b), Op(UInt(8), prefix + "min_epu8", [u8_a, u8_b])),
          Rule(max(u8_a, u8_b), Op(UInt(8), prefix + "max_epu8", [u8_a, u8_b])),
          Rule(
              min(i16_a, i16_b),
              Op(Int(16), prefix + "min_epi16", [i16_a, i16_b]),
          ),
          Rule(
              max(i16_a, i16_b),
              Op(Int(16), prefix + "max_epi16", [i16_a, i16_b]),
          ),
          Rule(
              u32_a & u32_b,
              Op(
                  UInt(32),
                  prefix + "and_si" + str(vector_bits),
                  [u32_a, u32_b],
              ),
          ),
          Rule(
              i32_a & i32_b,
              Op(Int(32), prefix + "and_si" + str(vector_bits), [i32_a, i32_b]),
          ),
          Rule(
              u32_a | u32_b,
              Op(UInt(32), prefix + "or_si" + str(vector_bits), [u32_a, u32_b]),
          ),
          Rule(
              i32_a | i32_b,
              Op(Int(32), prefix + "or_si" + str(vector_bits), [i32_a, i32_b]),
          ),
          Rule(
              u32_a ^ u32_b,
              Op(
                  UInt(32), prefix + "xor_si" + str(vector_bits), [u32_a, u32_b]
              ),
          ),
          Rule(
              i32_a ^ i32_b,
              Op(Int(32), prefix + "xor_si" + str(vector_bits), [i32_a, i32_b]),
          ),
          Rule(
              saturating_add(u8_a, u8_b),
              Op(UInt(8), prefix + "adds_epu8", [u8_a, u8_b]),
          ),
          Rule(
              saturating_add(i8_a, i8_b),
              Op(Int(8), prefix + "adds_epi8", [i8_a, i8_b]),
          ),
          Rule(
              saturating_add(u16_a, u16_b),
              Op(UInt(16), prefix + "adds_epu16", [u16_a, u16_b]),
          ),
          Rule(
              saturating_add(i16_a, i16_b),
              Op(Int(16), prefix + "adds_epi16", [i16_a, i16_b]),
          ),
          Rule(
              saturating_sub(u8_a, u8_b),
              Op(UInt(8), prefix + "subs_epu8", [u8_a, u8_b]),
          ),
          Rule(
              saturating_sub(i8_a, i8_b),
              Op(Int(8), prefix + "subs_epi8", [i8_a, i8_b]),
          ),
          Rule(
              saturating_sub(u16_a, u16_b),
              Op(UInt(16), prefix + "subs_epu16", [u16_a, u16_b]),
          ),
          Rule(
              saturating_sub(i16_a, i16_b),
              Op(Int(16), prefix + "subs_epi16", [i16_a, i16_b]),
          ),
      ]
  ] + [
      Rule(
          logical_shift_left(
              i16_a.with_lanes(vector_bits // 16),
              broadcast(i16_b, vector_bits // 16),
          ),
          Op(Int(16, vector_bits // 16), prefix + "slli_epi16", [i16_a, i16_b]),
      ),
      Rule(
          logical_shift_left(
              i32_a.with_lanes(vector_bits // 32),
              broadcast(i32_b, vector_bits // 32),
          ),
          Op(
              Int(32, vector_bits // 32),
              prefix + "slli_epi32",
              [i32_a.with_lanes(vector_bits // 32), i32_b],
          ),
      ),
  ]


def make_x86_integer_min_max_patterns(vector_bits, prefix):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              min(i8_a, i8_b),
              Op(Int(8), prefix + "min_epi8", [i8_a, i8_b]),
          ),
          Rule(
              max(i8_a, i8_b),
              Op(Int(8), prefix + "max_epi8", [i8_a, i8_b]),
          ),
          Rule(
              min(u16_a, u16_b),
              Op(UInt(16), prefix + "min_epu16", [u16_a, u16_b]),
          ),
          Rule(
              max(u16_a, u16_b),
              Op(UInt(16), prefix + "max_epu16", [u16_a, u16_b]),
          ),
          Rule(
              min(i32_a, i32_b),
              Op(Int(32), prefix + "min_epi32", [i32_a, i32_b]),
          ),
          Rule(
              max(i32_a, i32_b),
              Op(Int(32), prefix + "max_epi32", [i32_a, i32_b]),
          ),
          Rule(
              min(u32_a, u32_b),
              Op(UInt(32), prefix + "min_epu32", [u32_a, u32_b]),
          ),
          Rule(
              max(u32_a, u32_b),
              Op(UInt(32), prefix + "max_epu32", [u32_a, u32_b]),
          ),
      ]
  ]


def make_x86_broadcast_patterns(vector_bits, prefix):
  return [
      Rule(
          broadcast(i8_a, vector_bits // 8),
          Op(Int(8, vector_bits // 8), prefix + "set1_epi8", [i8_a]),
      ),
      Rule(
          broadcast(u8_a, vector_bits // 8),
          Op(UInt(8, vector_bits // 8), prefix + "set1_epi8", [u8_a]),
      ),
      Rule(
          broadcast(i16_a, vector_bits // 16),
          Op(Int(16, vector_bits // 16), prefix + "set1_epi16", [i16_a]),
      ),
      Rule(
          broadcast(u16_a, vector_bits // 16),
          Op(UInt(16, vector_bits // 16), prefix + "set1_epi16", [u16_a]),
      ),
      Rule(
          broadcast(i32_a, vector_bits // 32),
          Op(Int(32, vector_bits // 32), prefix + "set1_epi32", [i32_a]),
      ),
      Rule(
          broadcast(u32_a, vector_bits // 32),
          Op(UInt(32, vector_bits // 32), prefix + "set1_epi32", [u32_a]),
      ),
      Rule(
          broadcast(f32_a, vector_bits // 32),
          Op(Float(32, vector_bits // 32), prefix + "set1_ps", [f32_a]),
      ),
  ]


def make_x86_float32_patterns(vector_bits, prefix):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(f32_a + f32_b, Op(Float(32), prefix + "add_ps", [f32_a, f32_b])),
          Rule(f32_a - f32_b, Op(Float(32), prefix + "sub_ps", [f32_a, f32_b])),
          Rule(f32_a * f32_b, Op(Float(32), prefix + "mul_ps", [f32_a, f32_b])),
          Rule(f32_a / f32_b, Op(Float(32), prefix + "div_ps", [f32_a, f32_b])),
          Rule(
              max(f32_a, f32_b),
              Op(Float(32), prefix + "max_ps", [f32_a, f32_b]),
          ),
          Rule(
              min(f32_a, f32_b),
              Op(Float(32), prefix + "min_ps", [f32_a, f32_b]),
          ),
          Rule(
              ceil(f32_a),
              Op(Float(32), prefix + "ceil_ps", [f32_a]),
              features=["SSE41"],
          ),
          Rule(
              floor(f32_a),
              Op(Float(32), prefix + "floor_ps", [f32_a]),
              features=["SSE41"],
          ),
          Rule(
              sqrt(f32_a),
              Op(Float(32), prefix + "sqrt_ps", [f32_a]),
          ),
          Rule(
              f32_a & f32_b,
              Op(Float(32), prefix + "and_ps", [f32_a, f32_b]),
          ),
          Rule(
              f32_a | f32_b,
              Op(Float(32), prefix + "or_ps", [f32_a, f32_b]),
          ),
          Rule(
              f32_a ^ f32_b,
              Op(Float(32), prefix + "xor_ps", [f32_a, f32_b]),
          ),
      ]
  ]


def make_x86_fma_patterns(vector_bits, prefix):
  return [
      i.vectorize(vector_bits)
      for i in [
          Rule(
              f32_a * f32_b + f32_c,
              Op(Float(32), prefix + "fmadd_ps", [f32_a, f32_b, f32_c]),
              ["FMA3", "AVX512F"],
          ),
          Rule(
              f32_a * f32_b - f32_c,
              Op(Float(32), prefix + "fmsub_ps", [f32_a, f32_b, f32_c]),
              ["FMA3", "AVX512F"],
          ),
      ]
  ]


class X86(Target):
  """X86 target for elementwise kernels compiler."""

  def add_load_intrinsics(self, vector_bits, prefix):
    self.load_intrinsics[Int(8, vector_bits // 8)] = (
        "wrapper" + prefix + "loadu_si" + str(vector_bits)
    )
    self.load_intrinsics[UInt(8, vector_bits // 8)] = (
        "wrapper" + prefix + "loadu_si" + str(vector_bits)
    )
    self.load_intrinsics[Int(16, vector_bits // 16)] = (
        "wrapper" + prefix + "loadu_si" + str(vector_bits)
    )
    self.load_intrinsics[UInt(16, vector_bits // 16)] = (
        "wrapper" + prefix + "loadu_si" + str(vector_bits)
    )
    self.load_intrinsics[Int(32, vector_bits // 32)] = (
        "wrapper" + prefix + "loadu_si" + str(vector_bits)
    )
    self.load_intrinsics[UInt(32, vector_bits // 32)] = (
        "wrapper" + prefix + "loadu_si" + str(vector_bits)
    )
    self.load_intrinsics[Float(32, vector_bits // 32)] = prefix + "loadu_ps"

  def add_store_intrinsics(self, vector_bits, prefix):
    self.store_intrinsics[Int(8, vector_bits // 8)] = (
        "wrapper" + prefix + "storeu_si" + str(vector_bits)
    )
    self.store_intrinsics[UInt(8, vector_bits // 8)] = (
        "wrapper" + prefix + "storeu_si" + str(vector_bits)
    )
    self.store_intrinsics[Int(32, vector_bits // 32)] = (
        "wrapper" + prefix + "storeu_si" + str(vector_bits)
    )
    self.store_intrinsics[Float(32, vector_bits // 32)] = prefix + "storeu_ps"

  def update_for_sse2(self):
    """Updates the target for SSE2 support."""
    self.types.update({
        Int(8, 16): "__m128i",
        Int(16, 8): "__m128i",
        Int(32, 4): "__m128i",
        UInt(8, 16): "__m128i",
        UInt(16, 8): "__m128i",
        UInt(32, 4): "__m128i",
        Float(32, 4): "__m128",
    })

    self.add_load_intrinsics(128, "_mm_")
    self.add_store_intrinsics(128, "_mm_")
    self.patterns += make_x86_float32_patterns(128, "_mm_")
    self.patterns += make_x86_integer_patterns(128, "_mm_")
    self.patterns += make_x86_cast_patterns(128, "_mm_")
    self.patterns += make_x86_reinterpret_cast_patterns(128, "_mm_")
    self.patterns += make_x86_broadcast_patterns(128, "_mm_")

    self.header += """
namespace {

template <typename T>
static inline __m128i wrapper_mm_loadu_si128(const T* ptr) {
  return _mm_loadu_si128((const __m128i*)ptr);
}

template <typename T>
static inline void wrapper_mm_storeu_si128(T* ptr, __m128i v) {
  _mm_storeu_si128((__m128i*)ptr, v);
}

static inline __m128i saturating_cast_f32_to_int8(__m128 f0, __m128 f1, __m128 f2, __m128 f3) {
  const __m128 max_int16 = _mm_set1_ps((1 << 15) - 1);
  f0 = _mm_min_ps(f0, max_int16);
  f1 = _mm_min_ps(f1, max_int16);
  f2 = _mm_min_ps(f2, max_int16);
  f3 = _mm_min_ps(f3, max_int16);
  const __m128i i0 = _mm_cvtps_epi32(f0);
  const __m128i i1 = _mm_cvtps_epi32(f1);
  const __m128i i2 = _mm_cvtps_epi32(f2);
  const __m128i i3 = _mm_cvtps_epi32(f3);
  const __m128i i01_16 = _mm_packs_epi32(i0, i1);
  const __m128i i23_16 = _mm_packs_epi32(i2, i3);
  return _mm_packs_epi16(i01_16, i23_16);
}

static inline __m128i saturating_cast_f32_to_int16(__m128 f0, __m128 f1) {
  const __m128 max_int16 = _mm_set1_ps((1 << 15) - 1);
  f0 = _mm_min_ps(f0, max_int16);
  f1 = _mm_min_ps(f1, max_int16);
  const __m128i i0 = _mm_cvtps_epi32(f0);
  const __m128i i1 = _mm_cvtps_epi32(f1);
  return _mm_packs_epi32(i0, i1);
}

static inline __m128i saturating_cast_int32_to_int16(__m128i a, __m128i b) {
  return _mm_packs_epi32(a, b);
}

static inline __m128i saturating_cast_int16_to_int8(__m128i a, __m128i b) {
  return _mm_packs_epi16(a, b);
}

static inline __m128i saturating_cast_int16_to_uint8(__m128i a, __m128i b) {
  return _mm_packus_epi16(a, b);
}

static inline __m128i saturating_cast_f32_to_uint8(__m128 f0, __m128 f1, __m128 f2, __m128 f3) {
  const __m128 max_uint16 = _mm_set1_ps((1 << 16) - 1);
  f0 = _mm_min_ps(f0, max_uint16);
  f1 = _mm_min_ps(f1, max_uint16);
  const __m128i i0 = _mm_cvtps_epi32(f0);
  const __m128i i1 = _mm_cvtps_epi32(f1);
  const __m128i i2 = _mm_cvtps_epi32(f2);
  const __m128i i3 = _mm_cvtps_epi32(f3);
  const __m128i i01_16 = _mm_packus_epi32(i0, i1);
  const __m128i i23_16 = _mm_packus_epi32(i2, i3);
  return _mm_packus_epi16(i01_16, i23_16);
}

} // namespace
"""

  def update_for_sse41(self):
    """Updates the target for SSE41 support."""
    self.header += """
namespace {

static inline __m128 round(__m128 x) {
  return _mm_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

} // namespace

"""
    self.patterns += make_x86_float32_patterns(128, "_mm_")
    self.patterns += make_x86_integer_min_max_patterns(128, "_mm_")

  def update_for_avx(self):
    """Updates the target for AVX support."""
    self.header += """
namespace {

// Mask table used for partial load/store operations.
static const int32_t mask_table_avx_f32[16] = {-1, -1, -1, -1, -1, -1, -1, -1,
                                               0,  0,  0,  0,  0,  0,  0, 0};

static inline __m256 partial_load_8x(const float* ptr, size_t num_elements) {
  __m256i mask = _mm256_loadu_si256(
            (const __m256i*)(&mask_table_avx_f32[8] - num_elements));
  return _mm256_maskload_ps(ptr, mask);
}

static inline void partial_store_8x(float* output, size_t num_elements, __m256 v) {
  __m256i mask = _mm256_loadu_si256(
            (const __m256i*)(&mask_table_avx_f32[8] - num_elements));
  _mm256_maskstore_ps(output, mask, v);
}

static inline __m256i partial_load_8x(const int32_t* ptr, size_t num_elements) {
  __m256i mask = _mm256_loadu_si256(
            (const __m256i*)(&mask_table_avx_f32[8] - num_elements));
  return _mm256_maskload_epi32(ptr, mask);
}

static inline void partial_store_8x(int32_t* output, size_t num_elements, __m256i v) {
  __m256i mask = _mm256_loadu_si256(
            (const __m256i*)(&mask_table_avx_f32[8] - num_elements));
  _mm256_maskstore_epi32(output, mask, v);
}

template <typename T>
static inline __m256i wrapper_mm256_loadu_si256(const T* ptr) {
  return _mm256_loadu_si256((const __m256i*)ptr);
}

template <typename T>
static inline void wrapper_mm256_storeu_si256(T* ptr, __m256i v) {
  _mm256_storeu_si256((__m256i*)ptr, v);
}

static inline __m256 round(__m256 x) {
  return _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static inline __m256i saturating_cast_f32_to_int8(__m256 f0, __m256 f1, __m256 f2, __m256 f3) {
  const __m256 max_int16 = _mm256_set1_ps((1 << 15) - 1);
  f0 = _mm256_min_ps(f0, max_int16);
  f1 = _mm256_min_ps(f1, max_int16);
  f2 = _mm256_min_ps(f2, max_int16);
  f3 = _mm256_min_ps(f3, max_int16);
  const __m256i i0 = _mm256_cvtps_epi32(f0);
  const __m256i i1 = _mm256_cvtps_epi32(f1);
  const __m256i i2 = _mm256_cvtps_epi32(f2);
  const __m256i i3 = _mm256_cvtps_epi32(f3);
  const __m256i i01_16 = _mm256_packs_epi32(i0, i1);
  const __m256i i23_16 = _mm256_packs_epi32(i2, i3);
  const __m256i r = _mm256_packs_epi16(i01_16, i23_16);
  return _mm256_permutevar8x32_epi32(r, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

static inline __m256i saturating_cast_f32_to_int16(__m256 f0, __m256 f1) {
  const __m256 max_int16 = _mm256_set1_ps((1 << 15) - 1);
  f0 = _mm256_min_ps(f0, max_int16);
  f1 = _mm256_min_ps(f1, max_int16);
  const __m256i i0 = _mm256_cvtps_epi32(f0);
  const __m256i i1 = _mm256_cvtps_epi32(f1);
  const __m256i i01_16 = _mm256_packs_epi32(i0, i1);
  return _mm256_permute4x64_epi64(i01_16, (0 << 0) + (2 << 2) + (1 << 4) + (3 << 6));
}

static inline __m256i saturating_cast_int32_to_int16(__m256i a, __m256i b) {
  const __m256i r = _mm256_packs_epi32(a, b);
  return _mm256_permute4x64_epi64(r, (0 << 0) + (2 << 2) + (1 << 4) + (3 << 6));
}

static inline __m256i saturating_cast_int16_to_int8(__m256i a, __m256i b) {
  const __m256i r = _mm256_packs_epi16(a, b);
  return _mm256_permute4x64_epi64(r, (0 << 0) + (2 << 2) + (1 << 4) + (3 << 6));
}

static inline __m256i saturating_cast_int16_to_uint8(__m256i a, __m256i b) {
  const __m256i r = _mm256_packus_epi16(a, b);
  return _mm256_permute4x64_epi64(r, (0 << 0) + (2 << 2) + (1 << 4) + (3 << 6));
}

static inline __m256i saturating_cast_f32_to_uint8(__m256 f0, __m256 f1, __m256 f2, __m256 f3) {
  const __m256 max_uint16 = _mm256_set1_ps((1 << 16) - 1);
  f0 = _mm256_min_ps(f0, max_uint16);
  f1 = _mm256_min_ps(f1, max_uint16);
  f2 = _mm256_min_ps(f2, max_uint16);
  f3 = _mm256_min_ps(f3, max_uint16);
  const __m256i i0 = _mm256_cvtps_epi32(f0);
  const __m256i i1 = _mm256_cvtps_epi32(f1);
  const __m256i i2 = _mm256_cvtps_epi32(f2);
  const __m256i i3 = _mm256_cvtps_epi32(f3);
  const __m256i i01_16 = _mm256_packus_epi32(i0, i1);
  const __m256i i23_16 = _mm256_packus_epi32(i2, i3);
  const __m256i r = _mm256_packus_epi16(i01_16, i23_16);
  return _mm256_permutevar8x32_epi32(r, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}


} // namespace

"""
    self.types.update({
        Float(32, 8): "__m256",
        Int(8, 32): "__m256i",
        Int(16, 16): "__m256i",
        Int(32, 8): "__m256i",
        UInt(8, 32): "__m256i",
        UInt(16, 16): "__m256i",
        UInt(32, 8): "__m256i",
    })
    self.add_load_intrinsics(256, "_mm256_")
    self.add_store_intrinsics(256, "_mm256_")
    self.patterns += make_x86_float32_patterns(256, "_mm256_")
    self.patterns += make_x86_reinterpret_cast_patterns(256, "_mm256_")
    self.patterns += make_x86_broadcast_patterns(256, "_mm256_")

  def update_for_avx2(self):
    """Updates the target for AVX2 support."""
    self.patterns += make_x86_integer_patterns(256, "_mm256_")
    self.patterns += make_x86_integer_min_max_patterns(256, "_mm256_")
    self.patterns += make_x86_cast_patterns(256, "_mm256_")

  def update_for_fma3(self):
    """Updates the target for FMA3 support."""
    self.patterns += make_x86_fma_patterns(256, "_mm256_")

  def update_for_avx512f(self):
    """Updates the target for AVX512F support."""
    self.header += """
namespace {

static inline __m512 partial_load_16x(const float* ptr, size_t num_elements) {
  __mmask16 mask = _cvtu32_mask16((uint32_t)((1U << num_elements) - 1U));
  return _mm512_maskz_loadu_ps(mask, ptr);
}

static inline void partial_store_16x(float* output, size_t num_elements, __m512 v) {
  __mmask16 mask = _cvtu32_mask16((uint32_t)((1U << num_elements) - 1U));
  _mm512_mask_storeu_ps(output, mask, v);
}

static inline __m512i partial_load_16x(const int32_t* ptr, size_t num_elements) {
  __mmask16 mask = _cvtu32_mask16((uint32_t)((1U << num_elements) - 1U));
  return _mm512_maskz_loadu_epi32(mask, ptr);
}

static inline void partial_store_16x(int32_t* output, size_t num_elements, __m512i v) {
  __mmask16 mask = _cvtu32_mask16((uint32_t)((1U << num_elements) - 1U));
  _mm512_mask_storeu_epi32(output, mask, v);
}

template <typename T>
static inline __m512i wrapper_mm512_loadu_si512(const T* ptr) {
  return _mm512_loadu_si512((const __m512i*)ptr);
}

template <typename T>
static inline void wrapper_mm512_storeu_si512(T* ptr, __m512i v) {
  _mm512_storeu_si512((__m512i*)ptr, v);
}

static inline __m512 round(__m512 x) {
  return _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

} // namespace

"""
    self.types.update({
        Int(8, 64): "__m512i",
        Int(16, 32): "__m512i",
        Int(32, 16): "__m512i",
        UInt(8, 64): "__m512i",
        UInt(16, 32): "__m512i",
        UInt(32, 16): "__m512i",
        Float(32, 16): "__m512",
    })
    self.add_load_intrinsics(512, "_mm512_")
    self.add_store_intrinsics(512, "_mm512_")
    self.patterns += make_x86_fma_patterns(512, "_mm512_")
    self.patterns += make_x86_float32_patterns(512, "_mm512_")
    self.patterns += make_x86_reinterpret_cast_patterns(512, "_mm512_")
    self.patterns += make_x86_broadcast_patterns(512, "_mm512_")
    self.patterns += make_x86_integer_patterns(512, "_mm512_")
    self.patterns += make_x86_integer_min_max_patterns(512, "_mm512_")
    self.patterns += make_x86_cast_patterns(512, "_mm512_")

  def update_for_avx512bw(self):
    """Updates the target for AVX512BW support."""
    self.header += """
namespace {

static inline __m512i partial_load_32x(const int16_t* ptr, size_t num_elements) {
  __mmask32 mask = _cvtu32_mask32((uint32_t)((1ULL << num_elements) - 1U));
  return _mm512_maskz_loadu_epi16(mask, ptr);
}

static inline void partial_store_32x(int16_t* output, size_t num_elements, __m512i v) {
  __mmask32 mask = _cvtu32_mask32((uint32_t)((1ULL << num_elements) - 1U));
  _mm512_mask_storeu_epi16(output, mask, v);
}

static inline void partial_store_64x(int8_t* output, size_t num_elements, __m512i v) {
  __mmask64 mask = _cvtu64_mask64((1ULL << num_elements) - 1ULL);
  _mm512_mask_storeu_epi8(output, mask, v);
}

static inline void partial_store_64x(uint8_t* output, size_t num_elements, __m512i v) {
  __mmask64 mask = _cvtu64_mask64((1ULL << num_elements) - 1ULL);
  _mm512_mask_storeu_epi8(output, mask, v);
}

static inline __m512i saturating_cast_f32_to_int8(__m512 f0, __m512 f1, __m512 f2, __m512 f3) {
  const __m512 max_int16 = _mm512_set1_ps((1 << 15) - 1);
  f0 = _mm512_min_ps(f0, max_int16);
  f1 = _mm512_min_ps(f1, max_int16);
  f2 = _mm512_min_ps(f2, max_int16);
  f3 = _mm512_min_ps(f3, max_int16);
  const __m512i i0 = _mm512_cvtps_epi32(f0);
  const __m512i i1 = _mm512_cvtps_epi32(f1);
  const __m512i i2 = _mm512_cvtps_epi32(f2);
  const __m512i i3 = _mm512_cvtps_epi32(f3);
  const __m512i i01_16 = _mm512_packs_epi32(i0, i1);
  const __m512i i23_16 = _mm512_packs_epi32(i2, i3);
  const __m512i r = _mm512_packs_epi16(i01_16, i23_16);
  return _mm512_permutexvar_epi32(_mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), r);
}

static inline __m512i saturating_cast_f32_to_uint8(__m512 f0, __m512 f1, __m512 f2, __m512 f3) {
  const __m512 max_uint16 = _mm512_set1_ps((1 << 16) - 1);
  f0 = _mm512_min_ps(f0, max_uint16);
  f1 = _mm512_min_ps(f1, max_uint16);
  const __m512i i0 = _mm512_cvtps_epi32(f0);
  const __m512i i1 = _mm512_cvtps_epi32(f1);
  const __m512i i2 = _mm512_cvtps_epi32(f2);
  const __m512i i3 = _mm512_cvtps_epi32(f3);
  const __m512i i01_16 = _mm512_packus_epi32(i0, i1);
  const __m512i i23_16 = _mm512_packus_epi32(i2, i3);
  const __m512i r = _mm512_packus_epi16(i01_16, i23_16);
  return _mm512_permutexvar_epi32(_mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), r);
}

static inline __m512i saturating_cast_f32_to_int16(__m512 f0, __m512 f1) {
  const __m512 max_int16 = _mm512_set1_ps((1 << 15) - 1);
  f0 = _mm512_min_ps(f0, max_int16);
  f1 = _mm512_min_ps(f1, max_int16);
  const __m512i i0 = _mm512_cvtps_epi32(f0);
  const __m512i i1 = _mm512_cvtps_epi32(f1);
  const __m512i r = _mm512_packs_epi32(i0, i1);
  return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r);
}

static inline __m512i saturating_cast_int32_to_int16(__m512i a, __m512i b) {
  const __m512i r = _mm512_packs_epi32(a, b);
  return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r);
}

static inline __m512i saturating_cast_int16_to_int8(__m512i a, __m512i b) {
  const __m512i r = _mm512_packs_epi16(a, b);
  return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r);
}

static inline __m512i saturating_cast_int16_to_uint8(__m512i a, __m512i b) {
  const __m512i r = _mm512_packus_epi16(a, b);
  return _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r);
}

} // namespace

"""

  def __init__(self, features):
    Target.__init__(self)
    self.features = features
    self.load_intrinsics = {}
    self.store_intrinsics = {}
    # These are transitive.
    implied_features = {
        "SSE41": ["SSE2"],
        "AVX": ["SSE41"],
        "AVX2": ["AVX"],
        "FMA3": ["AVX"],
        "AVX512F": ["AVX2", "FMA3"],
        "AVX512BW": ["AVX512F"],
    }
    all_features = []
    self.compute_all_features(features, implied_features, all_features)

    self.header += "#include <immintrin.h>\n"

    known_features = [
        "SSE2",
        "SSE41",
        "AVX",
        "AVX2",
        "FMA3",
        "AVX512F",
        "AVX512BW",
    ]
    for feature in all_features:
      if feature not in known_features:
        raise ValueError(f"Unknown feature: {feature}")

    if "AVX512F" in all_features:
      self.tail_strategy = TailStrategy.MASK
      self.vector_bits = 512
    elif "AVX" in all_features:
      self.tail_strategy = TailStrategy.MEMCPY
      self.vector_bits = 256
    elif "SSE2" in all_features:
      self.tail_strategy = TailStrategy.MEMCPY
      self.vector_bits = 128

    if "AVX512BW" in all_features:
      self.update_for_avx512bw()
    if "AVX512F" in all_features:
      self.update_for_avx512f()
    if "FMA3" in all_features:
      self.update_for_fma3()
    if "AVX2" in all_features:
      self.update_for_avx2()
    if "AVX" in all_features:
      self.update_for_avx()
    if "SSE41" in all_features:
      self.update_for_sse41()
    if "SSE2" in all_features:
      self.update_for_sse2()
