// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/reduce/reduce.h"

#include <cassert>
#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/simd/scalar.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/reduce/min_max.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

SUM_FLOAT_K1_KERNEL(sum_k1_fp64, double, double, 0, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_fp64, double, double, 1, identity);
SUM_FLOAT_K1_KERNEL(sum_k1_fp32, float, float, 0, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_fp32, float, float, 1, identity);
SUM_FLOAT_K1_KERNEL(sum_k1_bf16_fp32, bfloat16, float, 0, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_bf16_fp32, bfloat16, float, 1, identity);
SUM_FLOAT_K1_KERNEL(sum_k1_fp16_fp32, half, float, 0, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_fp16_fp32, half, float, 1, identity);
SUM_K1_KERNEL(sum_k1_int32, int32_t, int32_t, 1, 1, identity);
SUM_KN_KERNEL(sum_kn_int32, int32_t, int32_t, 1, identity);
SUM_K1_KERNEL(sum_k1_int8_int32, int8_t, int32_t, 1, 1, identity);
SUM_KN_KERNEL(sum_kn_int8_int32, int8_t, int32_t, 1, identity);
SUM_K1_KERNEL(sum_k1_uint8_int32, uint8_t, int32_t, 1, 1, identity);
SUM_KN_KERNEL(sum_kn_uint8_int32, uint8_t, int32_t, 1, identity);

SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp64, double, double, 0, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp64, double, double, 1, square);
SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp32, float, float, 0, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp32, float, float, 1, square);
SUM_FLOAT_K1_KERNEL(sum_squared_k1_bf16_fp32, bfloat16, float, 0, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_bf16_fp32, bfloat16, float, 1, square);
SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp16_fp32, half, float, 0, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp16_fp32, half, float, 1, square);
SUM_K1_KERNEL(sum_squared_k1_int8_int32, int8_t, int32_t, 1, 1, square);
SUM_KN_KERNEL(sum_squared_kn_int8_int32, int8_t, int32_t, 1, square);
SUM_K1_KERNEL(sum_squared_k1_uint8_int32, uint8_t, int32_t, 1, 1, square);
SUM_KN_KERNEL(sum_squared_kn_uint8_int32, uint8_t, int32_t, 1, square);

// min/max kernels
using f64x1 = simd::vec<double, 1>;
using f32x1 = simd::vec<float, 1>;
using s32x1 = simd::vec<int32_t, 1>;
using s8x1 = simd::vec<int8_t, 1>;
using u8x1 = simd::vec<uint8_t, 1>;
using xf16x1 = sign_magnitude<simd::vec<int16_t, 1>>;
using xf8x1 = sign_magnitude<simd::vec<int8_t, 1>>;

MIN_MAX_K1_KERNEL(min_k1_fp64, f64x1, dummy_t, double, 1);
MIN_MAX_KN_KERNEL(min_kn_fp64, f64x1, dummy_t, double, 1);
MIN_MAX_K1_KERNEL(min_k1_fp32, f32x1, dummy_t, float, 1);
MIN_MAX_KN_KERNEL(min_kn_fp32, f32x1, dummy_t, float, 1);
MIN_MAX_K1_KERNEL(min_k1_xf16, xf16x1, dummy_t, int16_t, 1);
MIN_MAX_KN_KERNEL(min_kn_xf16, xf16x1, dummy_t, int16_t, 1);
MIN_MAX_K1_KERNEL(min_k1_int8, s8x1, dummy_t, int8_t, 1);
MIN_MAX_KN_KERNEL(min_kn_int8, s8x1, dummy_t, int8_t, 1);
MIN_MAX_K1_KERNEL(min_k1_uint8, u8x1, dummy_t, uint8_t, 1);
MIN_MAX_KN_KERNEL(min_kn_uint8, u8x1, dummy_t, uint8_t, 1);
MIN_MAX_K1_KERNEL(min_k1_xf8, xf8x1, dummy_t, int8_t, 1);
MIN_MAX_KN_KERNEL(min_kn_xf8, xf8x1, dummy_t, int8_t, 1);

MIN_MAX_K1_KERNEL(max_k1_fp64, dummy_t, f64x1, double, 1);
MIN_MAX_KN_KERNEL(max_kn_fp64, dummy_t, f64x1, double, 1);
MIN_MAX_K1_KERNEL(max_k1_fp32, dummy_t, f32x1, float, 1);
MIN_MAX_KN_KERNEL(max_kn_fp32, dummy_t, f32x1, float, 1);
MIN_MAX_K1_KERNEL(max_k1_xf16, dummy_t, xf16x1, int16_t, 1);
MIN_MAX_KN_KERNEL(max_kn_xf16, dummy_t, xf16x1, int16_t, 1);
MIN_MAX_K1_KERNEL(max_k1_int8, dummy_t, s8x1, int8_t, 1);
MIN_MAX_KN_KERNEL(max_kn_int8, dummy_t, s8x1, int8_t, 1);
MIN_MAX_K1_KERNEL(max_k1_uint8, dummy_t, u8x1, uint8_t, 1);
MIN_MAX_KN_KERNEL(max_kn_uint8, dummy_t, u8x1, uint8_t, 1);
MIN_MAX_K1_KERNEL(max_k1_xf8, dummy_t, xf8x1, int8_t, 1);
MIN_MAX_KN_KERNEL(max_kn_xf8, dummy_t, xf8x1, int8_t, 1);

MIN_MAX_K1_KERNEL(min_max_k1_fp64, f64x1, f64x1, double, 1);
MIN_MAX_KN_KERNEL(min_max_kn_fp64, f64x1, f64x1, double, 1);
MIN_MAX_K1_KERNEL(min_max_k1_fp32, f32x1, f32x1, float, 1);
MIN_MAX_KN_KERNEL(min_max_kn_fp32, f32x1, f32x1, float, 1);
MIN_MAX_K1_KERNEL(min_max_k1_xf16, xf16x1, xf16x1, int16_t, 1);
MIN_MAX_KN_KERNEL(min_max_kn_xf16, xf16x1, xf16x1, int16_t, 1);
MIN_MAX_K1_KERNEL(min_max_k1_int8, s8x1, s8x1, int8_t, 1);
MIN_MAX_KN_KERNEL(min_max_kn_int8, s8x1, s8x1, int8_t, 1);
MIN_MAX_K1_KERNEL(min_max_k1_uint8, u8x1, u8x1, uint8_t, 1);
MIN_MAX_KN_KERNEL(min_max_kn_uint8, u8x1, u8x1, uint8_t, 1);
MIN_MAX_K1_KERNEL(min_max_k1_xf8, xf8x1, xf8x1, int8_t, 1);
MIN_MAX_KN_KERNEL(min_max_kn_xf8, xf8x1, xf8x1, int8_t, 1);

#define YNN_REDUCE_KERNEL(arch, name, k_dim, A, C)          \
  if (type_of<A>() == a_type && type_of<C>() == c_type) {   \
    if (is_arch_supported(arch)) {                          \
      if (k_dim == reduce_dim::k1 && !res.k1) {             \
        YNN_LOG_DEBUG() << "Using reduce kernel " << #name; \
        res.k1 = name;                                      \
      } else if (k_dim == reduce_dim::kn && !res.kn) {      \
        YNN_LOG_DEBUG() << "Using reduce kernel " << #name; \
        res.kn = name;                                      \
      }                                                     \
    }                                                       \
  }

reduce_kernel get_sum_kernel(ynn_type a_type, ynn_type c_type) {
  reduce_kernel res = {};
#include "ynnpack/kernels/reduce/sum.inc"
  return res;
}

reduce_kernel get_sum_squared_kernel(ynn_type a_type, ynn_type c_type) {
  reduce_kernel res = {};
#include "ynnpack/kernels/reduce/sum_squared.inc"
  return res;
}

#undef YNN_REDUCE_KERNEL

#define YNN_REDUCE_KERNEL(arch, name, k_dim, A, C)          \
  if (type_of<A>() == type) {                               \
    if (is_arch_supported(arch)) {                          \
      if (k_dim == reduce_dim::k1 && !res.k1) {             \
        YNN_LOG_DEBUG() << "Using reduce kernel " << #name; \
        res.k1 = name;                                      \
      } else if (k_dim == reduce_dim::kn && !res.kn) {      \
        YNN_LOG_DEBUG() << "Using reduce kernel " << #name; \
        res.kn = name;                                      \
      }                                                     \
    }                                                       \
  }

reduce_kernel get_min_kernel(ynn_type type) {
  reduce_kernel res = {};
#include "ynnpack/kernels/reduce/min.inc"
  return res;
}

reduce_kernel get_max_kernel(ynn_type type) {
  reduce_kernel res = {};
#include "ynnpack/kernels/reduce/max.inc"
  return res;
}

reduce_kernel get_min_max_kernel(ynn_type type) {
  reduce_kernel res = {};
#include "ynnpack/kernels/reduce/min_max.inc"
  return res;
}

#undef YNN_REDUCE_KERNEL

reduce_kernel get_reduce_kernel(ynn_reduce_operator op, ynn_type a_type,
                                ynn_type c_type) {
  if (op == ynn_reduce_sum) {
    return get_sum_kernel(a_type, c_type);
  } else if (op == ynn_reduce_sum_squared) {
    return get_sum_squared_kernel(a_type, c_type);
  } else if (a_type == c_type) {
    if (op == ynn_reduce_max) {
      return get_max_kernel(c_type);
    } else if (op == ynn_reduce_min) {
      return get_min_kernel(c_type);
    } else if (op == ynn_reduce_min_max) {
      return get_min_max_kernel(c_type);
    }
  }
  return {};
}

}  // namespace ynn
