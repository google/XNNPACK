// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/base/simd/vec.h"
#include "ynnpack/base/simd/x86_sse2.h"
#include "ynnpack/base/simd/x86_sse2_only.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/reduce.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

using f32x4x8 = multi_vec<f32x4, 8>;
using bf16x8x4 = multi_vec<bf16x8, 4>;

static s32x4x4& operator+=(s32x4x4& a, s8x16 b) {
  a += convert(b, int32_t{});
  return a;
}

static s32x4x4& operator+=(s32x4x4& a, u8x16 b) {
  a += convert(b, int32_t{});
  return a;
}

// Use psadbw to compute the absolute difference of a and 0, summing 8 of them
// and producing an int64 in their place. We reinterpret the result to be 4
// int32s, which is only correct because we will do a horizontal total reduction
// later.
static s32x4 reduce_add(
    s32x4 a, u8x16 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  s32x4 b_s32(_mm_sad_epu8(b.v, _mm_set1_epi8(0)));
  return a += b_s32;
}

// psadbw only exists for unsigned values. We can still use it for signed values
// by toggling the most significant bit, which adds 0x80 to the result. We can
// correct the reduction by subtracting that elsewhere.
static s32x4 reduce_add(
    s32x4 a, s8x16 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 4> /*horizontal_factor*/) {
  s32x4 b_s32(_mm_sad_epu8(_mm_xor_si128(b.v, _mm_set1_epi8(0x80)),
                           _mm_set1_epi8(0)));
  return a += b_s32;
}

static f32x4x8 reduce_add(
    f32x4x8 a, bf16x8x4 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  YNN_UNROLL
  for (int i = 0; i < 4; ++i) {
    f32x4x2 b_f32 = convert(b.v[i], float{});

    a.v[2 * i + 0] += extract<0>(b_f32, f32x4{});
    a.v[2 * i + 1] += extract<1>(b_f32, f32x4{});
  }

  return a;
}

static f32x4 reduce_add(
    f32x4 a, bf16x8 b, Identity /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0xFFFF0000));
  f32x4 evens(_mm_castsi128_ps(_mm_slli_epi32(b.v, 16)));
  f32x4 odds(_mm_and_ps(_mm_castsi128_ps(b.v), mask));

  a += odds;
  a += evens;
  return a;
}

static f32x4x8 reduce_add(
    f32x4x8 a, bf16x8x4 b, Square /*map_fn*/,
    std::integral_constant<size_t, 1> /*horizontal_factor*/) {
  YNN_UNROLL
  for (int i = 0; i < 4; ++i) {
    f32x4x2 b_f32 = convert(b.v[i], float{});
    f32x4 b_lo = extract<0>(b_f32, f32x4{});
    f32x4 b_hi = extract<1>(b_f32, f32x4{});

    a.v[2 * i + 0] += b_lo * b_lo;
    a.v[2 * i + 1] += b_hi * b_hi;
  }

  return a;
}

static f32x4 reduce_add(
    f32x4 a, bf16x8 b, Square /*map_fn*/,
    std::integral_constant<size_t, 2> /*horizontal_factor*/) {
  __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0xFFFF0000));
  f32x4 evens{_mm_castsi128_ps(_mm_slli_epi32(b.v, 16))};
  f32x4 odds{_mm_and_ps(_mm_castsi128_ps(b.v), mask)};

  a += odds * odds;
  a += evens * evens;
  return a;
}

}  // namespace simd

namespace {

using simd::s32x4;
using simd::s32x4x4;
using simd::f32x4;
using simd::f32x4x8;
using simd::bf16x8;
using simd::bf16x8x4;
using simd::f16x8;
using simd::s16x8;
using simd::s8x16;
using simd::u8x16;

using f16x8_rvar = float16_wrapper<f16x8, s16x8>;
using bf16x8_rvar = float16_wrapper<bf16x8, s16x8>;

struct nonzero_identity_sum_accumulator_int32 {
  static constexpr std::integral_constant<size_t, 4> N = {};
  static constexpr std::integral_constant<size_t, 16> K = {};
  static constexpr std::integral_constant<size_t, 4> horizontal_factor = {};

  s32x4 acc[N];

  nonzero_identity_sum_accumulator_int32() = default;

  explicit nonzero_identity_sum_accumulator_int32(size_t k) {
    // id_value is nonzero for uint8 s32 case on x86 sse2 & sse41.
    // We rewrite signed int8 as unsigned in this accumulator. To compensate
    // for this, we need to subtract 0x80 for each element of the reduction.
    // Since this value gets reduced by 4x, we want to subtract 0x20 for each
    // element of the reduction (for a total of 0x80).
    s32x4 zero(-k * 0x20);

    for (size_t i = 0; i < N; ++i) {
      acc[i] = zero;
    }
  }

  template <typename NT, typename KT>
  void reduce(const int8_t* A, size_t A_stride_n, NT n, KT k) {
    // This value both identifies what we want the padding to be when we load
    // a partial vector of k values, and indicates the type of the load.
    const simd::vec<int8_t, K> zero(0x80);
    auto a_0 = load(offset_bytes(A, 0 * A_stride_n), zero, k);
    auto a_1 = 1 < n ? load(offset_bytes(A, 1 * A_stride_n), zero, k) : zero;
    auto a_2 = 2 < n ? load(offset_bytes(A, 2 * A_stride_n), zero, k) : zero;
    auto a_3 = 3 < n ? load(offset_bytes(A, 3 * A_stride_n), zero, k) : zero;

    Identity identity_map;
    acc[0] = reduce_add(acc[0], a_0, identity_map, horizontal_factor);
    acc[1] = reduce_add(acc[1], a_1, identity_map, horizontal_factor);
    acc[2] = reduce_add(acc[2], a_2, identity_map, horizontal_factor);
    acc[3] = reduce_add(acc[3], a_3, identity_map, horizontal_factor);
  }

  template <typename NT>
  void accumulate(size_t /*C_stride_m*/, int32_t* __restrict C, NT n) {
    auto acc_t = simd::transpose<int32_t>({{acc[0], acc[1], acc[2], acc[3]}});
    auto sum = (acc_t[0] + acc_t[1]) + (acc_t[2] + acc_t[3]);
    store(C, load(C, s32x4{}, n) + sum, n);
  }
};

}  // namespace

MIN_MAX_KERNEL(min_max_fp32_4x4_sse2, f32x4, f32x4, float, 4);
MIN_MAX_KERNEL(min_max_bf16_4x8_sse2, bf16x8_rvar, bf16x8_rvar, bfloat16, 8);
MIN_MAX_KERNEL(min_max_fp16_4x8_sse2, f16x8_rvar, f16x8_rvar, half, 8);
MIN_MAX_KERNEL(min_max_uint8_4x16_sse2, u8x16, u8x16, uint8_t, 16);

MIN_MAX_KERNEL(min_fp32_4x4_sse2, f32x4, dummy_t, float, 4);
MIN_MAX_KERNEL(min_bf16_4x8_sse2, bf16x8_rvar, dummy_t, bfloat16, 8);
MIN_MAX_KERNEL(min_fp16_4x8_sse2, f16x8_rvar, dummy_t, half, 8);
MIN_MAX_KERNEL(min_uint8_4x16_sse2, u8x16, dummy_t, uint8_t, 16);

MIN_MAX_KERNEL(max_fp32_4x4_sse2, dummy_t, f32x4, float, 4);
MIN_MAX_KERNEL(max_bf16_4x8_sse2, dummy_t, bf16x8_rvar, bfloat16, 8);
MIN_MAX_KERNEL(max_fp16_4x8_sse2, dummy_t, f16x8_rvar, half, 8);
MIN_MAX_KERNEL(max_uint8_4x16_sse2, dummy_t, u8x16, uint8_t, 16);

void sum_int8_int32_sse2(size_t n, size_t k3, size_t k2, size_t k1,
                         size_t a_stride_n, size_t a_stride_k3,
                         size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    tiled_reduce<sum_accumulator_k1_1<s8x16, s32x4x4>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<nonzero_identity_sum_accumulator_int32, int8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_uint8_int32_sse2(size_t n, size_t k3, size_t k2, size_t k1,
                          size_t a_stride_n, size_t a_stride_k3,
                          size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    tiled_reduce<sum_accumulator_k1_1<u8x16, s32x4x4>, uint8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<s32x4, 16>, uint8_t, int32_t>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a), /*C_stride_m=*/0,
        reinterpret_cast<int32_t*>(c));
  }
}

void sum_bf16_fp32_sse2(size_t n, size_t k3, size_t k2, size_t k1,
                        size_t a_stride_n, size_t a_stride_k3,
                        size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    tiled_reduce<sum_accumulator_k1_1<bf16x8x4, f32x4x8>, bfloat16, float>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x4, 8>, bfloat16, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

void sum_squared_bf16_fp32_sse2(size_t n, size_t k3, size_t k2, size_t k1,
                                size_t a_stride_n, size_t a_stride_k3,
                                size_t a_stride_k2, const void* a, size_t,
                                void* c) {
  if (k1 == 1 && a_stride_n == sizeof(bfloat16)) {
    tiled_reduce<sum_accumulator_k1_1<bf16x8x4, f32x4x8, Square>, bfloat16,
      float>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x4, 8, Square>, bfloat16, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const bfloat16*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

using f32x4x4 = simd::multi_vec<f32x4, 4>;

void sum_fp32_sse2(size_t n, size_t k3, size_t k2, size_t k1,
                   size_t a_stride_n, size_t a_stride_k3, size_t a_stride_k2,
                   const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(float)) {
    tiled_reduce<sum_accumulator_k1_1<f32x4x4, f32x4x4>, float, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const float*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x4, 4>, float, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const float*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

void sum_squared_fp32_sse2(size_t n, size_t k3, size_t k2, size_t k1,
                           size_t a_stride_n, size_t a_stride_k3,
                           size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(float)) {
    tiled_reduce<sum_accumulator_k1_1<f32x4x4, f32x4x4, Square>, float, float>(
        n, k3, k2, a_stride_k3, a_stride_k2, reinterpret_cast<const float*>(a),
        /*C_stride_m=*/0, reinterpret_cast<float*>(c));
  } else {
    tiled_reduce<sum_accumulator_x32<f32x4, 4, Square>, float, float>(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
        reinterpret_cast<const float*>(a), /*C_stride_m=*/0,
        reinterpret_cast<float*>(c));
  }
}

}  // namespace ynn
