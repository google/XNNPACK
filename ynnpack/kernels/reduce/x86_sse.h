#ifndef XNNPACK_YNNPACK_KERNELS_REDUCE_X86_SSE_H_
#define XNNPACK_YNNPACK_KERNELS_REDUCE_X86_SSE_H_

#include <immintrin.h>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_sse.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"

namespace ynn {

namespace simd {

template <>
struct vec<int32_t, 8> {
  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 8> N = {};

  vec() = default;
  explicit vec(int32_t x) : v{x, x} {}

  YNN_ALWAYS_INLINE vec operator+(vec a) const {
    vec res;
    res.v[0] = this->v[0] + a.v[0];
    res.v[1] = this->v[1] + a.v[1];
    return res;
  }

  s32x4 v[2];
};

// TODO(b/441600173): we could accumulate to s16x16 first and later convert to
// s32 when needed.
template <>
struct vec<int32_t, 16> {
  using value_type = int32_t;
  static constexpr std::integral_constant<size_t, 16> N = {};

  vec() = default;
  explicit vec(int32_t x) : v{vec<int32_t, 8>(x), vec<int32_t, 8>(x)} {}

  YNN_ALWAYS_INLINE vec operator+(vec a) const {
    vec res;
    res.v[0] = this->v[0] + a.v[0];
    res.v[1] = this->v[1] + a.v[1];
    return res;
  }

  vec<int32_t, 8> v[2];
};

using s32x8 = vec<int32_t, 8>;
using s32x16 = vec<int32_t, 16>;

YNN_ALWAYS_INLINE s32x8 load(const int32_t* ptr, s32x8, decltype(s32x8::N)) {
  s32x8 x;

  x.v[0] = load(ptr, s32x4{}, s32x4::N);
  x.v[1] = load(ptr + s32x4::N, s32x4{});

  return x;
}

YNN_ALWAYS_INLINE s32x8 load(const int32_t* ptr, s32x8, size_t n) {
  s32x8 x;

  if (n < s32x4::N) {
    x.v[0] = load(ptr, s32x4{}, n);
    x.v[1] = 0;
  } else {
    x.v[0] = load(ptr, s32x4{}, s32x4::N);
    x.v[1] = load(ptr + s32x4::N, s32x4{}, n - s32x4::N);
  }

  return x;
}

YNN_ALWAYS_INLINE s32x16 load(const int32_t* ptr, s32x16, decltype(s32x16::N)) {
  s32x16 x;

  x.v[0] = load(ptr, s32x8{}, s32x8::N);
  x.v[1] = load(ptr + s32x8::N, s32x8{}, s32x8::N);

  return x;
}

YNN_ALWAYS_INLINE s32x16 load(const int32_t* ptr, s32x16, size_t n) {
  s32x16 x;

  if (n < s32x8::N) {
    x.v[0] = load(ptr, s32x8{}, n);
    x.v[1] = s32x8(0);
  } else {
    x.v[0] = load(ptr, s32x8{}, s32x8::N);
    x.v[1] = load(ptr + s32x8::N, s32x8{}, n - s32x8::N);
  }

  return x;
}

YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x8 b, decltype(s32x8::N)) {
  store(ptr, b.v[0], decltype(s32x4::N){});
  store(ptr + s32x4::N, b.v[1], decltype(s32x4::N){});
}

YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x8 b, size_t n) {
  if (n < s32x4::N) {
    store(ptr, b.v[0], n);
  } else {
    store(ptr, b.v[0], s32x4::N);
    store(ptr + s32x4::N, b.v[1], n - s32x4::N);
  }
}

YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x16 b, decltype(s32x16::N)) {
  store(ptr, b.v[0], decltype(s32x8::N){});
  store(ptr + s32x8::N, b.v[1], decltype(s32x8::N){});
}

YNN_ALWAYS_INLINE void store(int32_t* ptr, s32x16 b, size_t n) {
  if (n < s32x8::N) {
    store(ptr, b.v[0], n);
  } else {
    store(ptr, b.v[0], s32x8::N);
    store(ptr + s32x8::N, b.v[1], n - s32x8::N);
  }
}

}  // namespace simd

using simd::s32x4;
using simd::s8x16;
using simd::u8x16;

// Use psadbw to compute the absolute difference of a and 0, summing 8 of them
// and producing an int64 in their place. We reinterpret the result to be 4
// int32s, which is only correct because we will do a horizontal total reduction
// later.
YNN_ALWAYS_INLINE s32x4 horizontal_add_8x(u8x16 a) {
  return s32x4{_mm_sad_epu8(a.v, _mm_set1_epi8(0))};
}

// psadbw only exists for unsigned values. We can still use it for signed values
// by toggling the most significant bit, which adds 0x80 to the result. We can
// correct the reduction by subtracting that elsewhere.
YNN_ALWAYS_INLINE s32x4 horizontal_add_8x(s8x16 a) {
  return s32x4{
      _mm_sad_epu8(_mm_xor_si128(a.v, _mm_set1_epi8(0x80)), _mm_set1_epi8(0))};
}

template <bool IsSigned>
struct accumulator_int32 {
  static constexpr std::integral_constant<size_t, 4> N = {};
  static constexpr std::integral_constant<size_t, 16> K = {};

  s32x4 acc[N];

  accumulator_int32() = default;

  YNN_ALWAYS_INLINE explicit accumulator_int32(int32_t k) {
    for (size_t i = 0; i < N; ++i) {
      // We rewrite signed int8 as unsigned in this accumulator. To compensate
      // for this, we need to subtract 0x80 for each element of the reduction.
      // Since this value gets reduced by 4x, we want to subtract 0x20 for each
      // element of the reduction (for a total of 0x80).
      acc[i] = IsSigned ? -(k * 0x20) : 0;
    }
  }

  template <typename AT, typename NT, typename KT>
  YNN_ALWAYS_INLINE void reduce(const AT* A, size_t a_stride_n, NT n, KT k) {
    // This value both identifies what we want the padding to be when we load
    // a partial vector of k values, and indicates the type of the load.
    const simd::vec<AT, K> zero(IsSigned ? 0x80 : 0);
    auto a_0 = load(offset_bytes(A, 0 * a_stride_n), zero, k);
    auto a_1 = 1 < n ? load(offset_bytes(A, 1 * a_stride_n), zero, k) : 0;
    auto a_2 = 2 < n ? load(offset_bytes(A, 2 * a_stride_n), zero, k) : 0;
    auto a_3 = 3 < n ? load(offset_bytes(A, 3 * a_stride_n), zero, k) : 0;
    acc[0] = acc[0] + horizontal_add_8x(a_0);
    acc[1] = acc[1] + horizontal_add_8x(a_1);
    acc[2] = acc[2] + horizontal_add_8x(a_2);
    acc[3] = acc[3] + horizontal_add_8x(a_3);
  }

  template <typename NT>
  YNN_ALWAYS_INLINE void accumulate(size_t /*C_stride_m*/,
                                    int32_t* __restrict C, NT n) {
    std::array<s32x4, 4> acc_t =
        simd::transpose<int32_t>({{acc[0], acc[1], acc[2], acc[3]}});
    s32x4 sum = (acc_t[0] + acc_t[1]) + (acc_t[2] + acc_t[3]);
    store(C, load(C, s32x4{}, n) + sum, n);
  }
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_REDUCE_X86_SSE_H_
