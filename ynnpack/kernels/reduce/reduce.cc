// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/reduce/reduce.h"

#include <cassert>
#include <cstdint>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/simd/scalar.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/reduce/min_max.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

template <typename T>
using vec1 = simd::vec<T, 1>;

SUM_FLOAT_K1_KERNEL(sum_k1_fp32, float, float, 0, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_fp32, float, float, 1, identity);

SUM_FLOAT_K1_KERNEL(sum_k1_fp64, double, double, 0, 1, identity);
SUM_FLOAT_KN_KERNEL(sum_kn_fp64, double, double, 1, identity);

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

SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp32, float, float, 0, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp32, float, float, 1, square);

SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp64, double, double, 0, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp64, double, double, 1, square);

SUM_FLOAT_K1_KERNEL(sum_squared_k1_bf16_fp32, bfloat16, float, 0, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_bf16_fp32, bfloat16, float, 1, square);

SUM_FLOAT_K1_KERNEL(sum_squared_k1_fp16_fp32, half, float, 0, 1, square);
SUM_FLOAT_KN_KERNEL(sum_squared_kn_fp16_fp32, half, float, 1, square);

SUM_K1_KERNEL(sum_squared_k1_int8_int32, int8_t, int32_t, 1, 1, square);
SUM_KN_KERNEL(sum_squared_kn_int8_int32, int8_t, int32_t, 1, square);

SUM_K1_KERNEL(sum_squared_k1_uint8_int32, uint8_t, int32_t, 1, 1, square);
SUM_KN_KERNEL(sum_squared_kn_uint8_int32, uint8_t, int32_t, 1, square);

// min/max kernels
#define MIN_K1_KERNEL(name, T) MIN_MAX_K1_KERNEL(name, vec1<T>, dummy_t, T, 1)
#define MIN_KN_KERNEL(name, T) MIN_MAX_KN_KERNEL(name, vec1<T>, dummy_t, T, 1)
#define MAX_K1_KERNEL(name, T) MIN_MAX_K1_KERNEL(name, dummy_t, vec1<T>, T, 1)
#define MAX_KN_KERNEL(name, T) MIN_MAX_KN_KERNEL(name, dummy_t, vec1<T>, T, 1)
#define MIN_MAX_K1_KERNEL_SCALAR(name, T) \
  MIN_MAX_K1_KERNEL(name, vec1<T>, vec1<T>, T, 1)
#define MIN_MAX_KN_KERNEL_SCALAR(name, T) \
  MIN_MAX_KN_KERNEL(name, vec1<T>, vec1<T>, T, 1)

MIN_K1_KERNEL(min_k1_fp32, float);
MIN_KN_KERNEL(min_kn_fp32, float);
MIN_K1_KERNEL(min_k1_fp64, double);
MIN_KN_KERNEL(min_kn_fp64, double);
MIN_K1_KERNEL(min_k1_fp16, half);
MIN_KN_KERNEL(min_kn_fp16, half);
MIN_K1_KERNEL(min_k1_bf16, bfloat16);
MIN_KN_KERNEL(min_kn_bf16, bfloat16);
MIN_K1_KERNEL(min_k1_int8, int8_t);
MIN_KN_KERNEL(min_kn_int8, int8_t);
MIN_K1_KERNEL(min_k1_uint8, uint8_t);
MIN_KN_KERNEL(min_kn_uint8, uint8_t);

MAX_K1_KERNEL(max_k1_fp32, float);
MAX_KN_KERNEL(max_kn_fp32, float);
MAX_K1_KERNEL(max_k1_fp64, double);
MAX_KN_KERNEL(max_kn_fp64, double);
MAX_K1_KERNEL(max_k1_fp16, half);
MAX_KN_KERNEL(max_kn_fp16, half);
MAX_K1_KERNEL(max_k1_bf16, bfloat16);
MAX_KN_KERNEL(max_kn_bf16, bfloat16);
MAX_K1_KERNEL(max_k1_int8, int8_t);
MAX_KN_KERNEL(max_kn_int8, int8_t);
MAX_K1_KERNEL(max_k1_uint8, uint8_t);
MAX_KN_KERNEL(max_kn_uint8, uint8_t);

MIN_MAX_K1_KERNEL_SCALAR(min_max_k1_fp32, float);
MIN_MAX_KN_KERNEL_SCALAR(min_max_kn_fp32, float);
MIN_MAX_K1_KERNEL_SCALAR(min_max_k1_fp64, double);
MIN_MAX_KN_KERNEL_SCALAR(min_max_kn_fp64, double);
MIN_MAX_K1_KERNEL_SCALAR(min_max_k1_fp16, half);
MIN_MAX_KN_KERNEL_SCALAR(min_max_kn_fp16, half);
MIN_MAX_K1_KERNEL_SCALAR(min_max_k1_bf16, bfloat16);
MIN_MAX_KN_KERNEL_SCALAR(min_max_kn_bf16, bfloat16);
MIN_MAX_K1_KERNEL_SCALAR(min_max_k1_int8, int8_t);
MIN_MAX_KN_KERNEL_SCALAR(min_max_kn_int8, int8_t);
MIN_MAX_K1_KERNEL_SCALAR(min_max_k1_uint8, uint8_t);
MIN_MAX_KN_KERNEL_SCALAR(min_max_kn_uint8, uint8_t);

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
