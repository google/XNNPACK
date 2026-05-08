// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/reduce/reduce.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/simd/scalar.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/sum.h"

namespace ynn {

namespace {

// This stores accumulators of type `AccT` for a reduction.
// Values of a type later defined are loaded and passed to `F` to produce the
// vaules to combine with the accumulators using a binary operator `ReduceOp`.
template <typename AccT, typename T, size_t N_, size_t K_, typename ReduceOp,
          typename F>
struct accumulator {
  static constexpr std::integral_constant<size_t, N_> N = {};
  static constexpr std::integral_constant<size_t, K_> K = {};

  AccT acc[N][K];

  accumulator() = default;

  YNN_ALWAYS_INLINE explicit accumulator(size_t) {
    for (size_t i = 0; i < N; ++i) {
      std::fill_n(acc[i], K, static_cast<AccT>(ReduceOp::identity));
    }
  }

  template <typename AT, typename N>
  YNN_ALWAYS_INLINE void reduce(const AT* __restrict A, size_t A_stride_n, N n,
                                decltype(K)) {
    ReduceOp op;
    F f;
    for (size_t i = 0; i < n; ++i) {
      const AT* __restrict A_i = offset_bytes(A, i * A_stride_n);
      for (size_t j = 0; j < K; ++j) {
        acc[i][j] = op(acc[i][j], f(A_i[j]));
      }
    }
  }

  template <typename AT, typename N>
  YNN_ALWAYS_INLINE void reduce(const AT* __restrict A, size_t A_stride_n, N n,
                                size_t k) {
    ReduceOp op;
    F f;
    // The compiler is much better at autovectorizing if we make the loop
    // below fixed size. To do that, we make a padded copy of A_i, and use a
    // fixed size loop on that.
    AT tail[K];
    std::fill_n(tail, K, ReduceOp::identity);
    for (size_t i = 0; i < n; ++i) {
      const AT* __restrict A_i = offset_bytes(A, i * A_stride_n);
      memcpy(tail, A_i, k * sizeof(AT));
      for (size_t j = 0; j < K; ++j) {
        acc[i][j] = op(acc[i][j], f(tail[j]));
      }
    }
  }

  YNN_ALWAYS_INLINE void accumulate(size_t /*C_stride_m*/, T* __restrict C,
                                    size_t n) {
    ReduceOp op;
    if (ReduceOp::is_associative) {
      // This seems to autovectorize better
      for (size_t i = 0; i < N; ++i) {
        for (size_t j = 1; j < K; ++j) {
          acc[i][0] = op(acc[i][0], acc[i][j]);
        }
      }
    } else {
      if (K == 16) {
        // Match the reduction order of sum_accumulator_x32 with
        // consistent_tile_k=16.
        for (size_t i = 0; i < N; ++i) {
          for (size_t j = 0; j < 4; ++j) {
            acc[i][j] = op(op(acc[i][j], acc[i][j + 4]),
                           op(acc[i][j + 8], acc[i][j + 12]));
          }
          acc[i][0] = op(op(acc[i][0], acc[i][1]), op(acc[i][2], acc[i][3]));
        }
      } else if (K == 8) {
        // Match the reduction order of sum_accumulator_x64 with
        // consistent_tile_k=8.
        for (size_t i = 0; i < N; ++i) {
          for (size_t j = 0; j < 4; ++j) {
            acc[i][j] = op(acc[i][j], acc[i][j + 4]);
          }
          acc[i][0] = op(op(acc[i][0], acc[i][1]), op(acc[i][2], acc[i][3]));
        }
      } else {
        // For numerical consistency, always do the final k reduction as a
        // binary tree.
        size_t k = K / 2;
        while (k >= 1) {
          for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < k; ++j) {
              acc[i][j] = op(acc[i][j], acc[i][j + k]);
            }
          }
          k /= 2;
        }
      }
    }
    for (size_t i = 0; i < n; ++i) {
      C[i] = op(static_cast<AccT>(C[i]), acc[i][0]);
    }
  }
};

template <typename AccT, typename T, size_t N_, typename ReduceOp, typename F>
struct accumulator_k1_1 {
  static constexpr std::integral_constant<size_t, N_> N = {};
  static constexpr std::integral_constant<size_t, 1> K2 = {};

  template <typename AT, typename NT, typename K2T>
  YNN_ALWAYS_INLINE void reduce_accumulate(const AT* __restrict A, NT n,
                                           size_t /*A_stride_k2*/, K2T,
                                           size_t /*C_stride_m*/,
                                           T* __restrict C) {
    ReduceOp op;
    F f;
    for (size_t i = 0; i < n; ++i) {
      C[i] = op(static_cast<AccT>(C[i]), f(A[i]));
    }
  }
};

// This is the same as above, but computes a min and max reduction at the same
// time.
template <typename AccT, typename T, size_t N_, size_t K_, typename Min,
          typename Max>
struct min_max_accumulator {
  static constexpr std::integral_constant<size_t, N_> N = {};
  static constexpr std::integral_constant<size_t, K_> K = {};

  AccT acc_min[N][K];
  AccT acc_max[N][K];

  min_max_accumulator() = default;

  YNN_ALWAYS_INLINE explicit min_max_accumulator(size_t) {
    for (size_t i = 0; i < N; ++i) {
      std::fill_n(acc_min[i], K, static_cast<AccT>(Min::identity));
      std::fill_n(acc_max[i], K, static_cast<AccT>(Max::identity));
    }
  }

  template <typename AT, typename N>
  YNN_ALWAYS_INLINE void reduce(const AT* __restrict A, size_t A_stride_n, N n,
                                decltype(K)) {
    Min min;
    Max max;
    for (size_t i = 0; i < n; ++i) {
      const AT* __restrict A_i = offset_bytes(A, i * A_stride_n);
      for (size_t j = 0; j < K; ++j) {
        acc_min[i][j] = min(acc_min[i][j], static_cast<AccT>(A_i[j]));
        acc_max[i][j] = max(acc_max[i][j], static_cast<AccT>(A_i[j]));
      }
    }
  }

  template <typename AT, typename N>
  YNN_ALWAYS_INLINE void reduce(const AT* __restrict A, size_t A_stride_n, N n,
                                size_t k) {
    Min min;
    Max max;
    // The compiler is much better at autovectorizing if we make the loop
    // below fixed size. To do that, we make a padded copy of A_i, and use a
    // fixed size loop on that.
    // TODO: It's dumb we have to load this twice in order to use the different
    // identity values.
    AT tail_min[K];
    AT tail_max[K];
    std::fill_n(tail_min, K, Min::identity);
    std::fill_n(tail_max, K, Max::identity);
    for (size_t i = 0; i < n; ++i) {
      const AT* __restrict A_i = offset_bytes(A, i * A_stride_n);
      memcpy(tail_min, A_i, k * sizeof(AT));
      memcpy(tail_max, A_i, k * sizeof(AT));
      for (size_t j = 0; j < K; ++j) {
        acc_min[i][j] = min(acc_min[i][j], static_cast<AccT>(tail_min[j]));
        acc_max[i][j] = max(acc_max[i][j], static_cast<AccT>(tail_max[j]));
      }
    }
  }

  template <typename NT>
  YNN_ALWAYS_INLINE void accumulate(size_t C_stride_m, T* __restrict C, NT n) {
    Min min;
    Max max;
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 1; j < K; ++j) {
        acc_min[i][0] = min(acc_min[i][0], acc_min[i][j]);
        acc_max[i][0] = max(acc_max[i][0], acc_max[i][j]);
      }
    }
    T* __restrict C_min = offset_bytes(C, 0 * C_stride_m);
    T* __restrict C_max = offset_bytes(C, 1 * C_stride_m);
    for (size_t i = 0; i < n; ++i) {
      C_min[i] = min(static_cast<AccT>(C_min[i]), acc_min[i][0]);
      C_max[i] = max(static_cast<AccT>(C_max[i]), acc_max[i][0]);
    }
  }
};

template <typename AccT, typename T, size_t N_, typename Min, typename Max>
struct min_max_accumulator_k1_1 {
  static constexpr std::integral_constant<size_t, N_> N = {};
  static constexpr std::integral_constant<size_t, 1> K2 = {};

  template <typename AT, typename N, typename K2T>
  YNN_ALWAYS_INLINE void reduce_accumulate(const AT* __restrict A, N n,
                                           size_t /*A_stride_k2*/, K2T,
                                           size_t C_stride_m, T* __restrict C) {
    Min min;
    Max max;
    T* __restrict C_min = offset_bytes(C, 0 * C_stride_m);
    T* __restrict C_max = offset_bytes(C, 1 * C_stride_m);
    for (size_t i = 0; i < n; ++i) {
      C_min[i] = min(static_cast<AccT>(C_min[i]), static_cast<AccT>(A[i]));
      C_max[i] = max(static_cast<AccT>(C_max[i]), static_cast<AccT>(A[i]));
    }
  }
};

template <typename T>
struct min_op {
  T operator()(T a, T b) { return min(a, b); }

  static constexpr auto identity = type_info<T>::min_identity();
  static constexpr bool is_associative = true;
};

template <typename T>
struct max_op {
  T operator()(T a, T b) { return max(a, b); }

  static constexpr auto identity = type_info<T>::max_identity();
  static constexpr bool is_associative = true;
};

template <typename T>
struct sum_op {
  T operator()(T a, T b) { return a + b; }

  static constexpr auto identity = type_info<T>::sum_identity();
  static constexpr bool is_associative = is_integral<T>::value;
};

template <typename T, typename A>
struct cast_op {
  T operator()(A a) { return static_cast<T>(a); }
};

template <typename T, typename A>
struct square_op {
  T operator()(A a) { return static_cast<T>(a) * static_cast<T>(a); }
};

// C(j) = ReduceOp(C(j), F(A(j, k3, k2, k1)))
template <typename AccT, typename AT, typename CT, typename ReduceOp,
          typename F = cast_op<CT, AT>>
void reduce(size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_n,
            size_t A_stride_k3, size_t A_stride_k2, const AT* A, CT* C,
            ReduceOp = ReduceOp{}, F = F{}) {
  if (K1 == 1 && A_stride_n == sizeof(AT)) {
    stream_reduce<accumulator_k1_1<AccT, CT, 128 / sizeof(AccT), ReduceOp, F>,
                  AT, CT>(N, K3, K2, A_stride_n, A_stride_k3, A_stride_k2, A,
                          /*C_stride_m=*/0, C);
  } else {
    constexpr size_t K =
        std::is_same<AccT, float>::value    ? consistent_tile_k_fp32
        : std::is_same<AccT, double>::value ? consistent_tile_k_fp64
                                            : 128 / sizeof(AccT);
    tiled_reduce<accumulator<AccT, CT, 1, K, ReduceOp, F>, AT, CT>(
        N, K3, K2, K1, A_stride_n, A_stride_k3, A_stride_k2, A,
        /*C_stride_m=*/0, C);
  }
}

template <typename AccT, typename T, typename Min, typename Max>
void min_max(size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_n,
             size_t A_stride_k3, size_t A_stride_k2, const T* A,
             size_t C_stride_m, T* C, Min, Max) {
  if (K1 == 1 && A_stride_n == sizeof(T)) {
    stream_reduce<
        min_max_accumulator_k1_1<AccT, T, 64 / sizeof(AccT), Min, Max>, T, T>(
        N, K3, K2, A_stride_n, A_stride_k3, A_stride_k2, A, C_stride_m, C);
  } else {
    tiled_reduce<min_max_accumulator<AccT, T, 1, 64 / sizeof(AccT), Min, Max>,
                 T, T>(N, K3, K2, K1, A_stride_n, A_stride_k3, A_stride_k2, A,
                       C_stride_m, C);
  }
}

}  // namespace

SUM_K1_KERNEL(sum_fp32_k1, float, float, consistent_tile_k_fp32, 1, identity);
SUM_KN_KERNEL(sum_fp32_kn, float, float, 1, identity);

SUM_K1_KERNEL(sum_fp64_k1, double, double, consistent_tile_k_fp64, 1, identity);
SUM_KN_KERNEL(sum_fp64_kn, double, double, 1, identity);

SUM_K1_KERNEL(sum_bf16_fp32_k1, bfloat16, float, consistent_tile_k_fp32, 1,
              identity);
SUM_KN_KERNEL(sum_bf16_fp32_kn, bfloat16, float, 1, identity);

SUM_K1_KERNEL(sum_fp16_fp32_k1, half, float, consistent_tile_k_fp32, 1,
              identity);
SUM_KN_KERNEL(sum_fp16_fp32_kn, half, float, 1, identity);

SUM_K1_KERNEL(sum_int32_k1, int32_t, int32_t, 1, 1, identity);
SUM_KN_KERNEL(sum_int32_kn, int32_t, int32_t, 1, identity);

SUM_K1_KERNEL(sum_int8_int32_k1, int8_t, int32_t, 1, 1, identity);
SUM_KN_KERNEL(sum_int8_int32_kn, int8_t, int32_t, 1, identity);

SUM_K1_KERNEL(sum_uint8_int32_k1, uint8_t, int32_t, 1, 1, identity);
SUM_KN_KERNEL(sum_uint8_int32_kn, uint8_t, int32_t, 1, identity);

SUM_K1_KERNEL(sum_squared_fp32_k1, float, float, consistent_tile_k_fp32, 1,
              square);
SUM_KN_KERNEL(sum_squared_fp32_kn, float, float, 1, square);

SUM_K1_KERNEL(sum_squared_fp64_k1, double, double, consistent_tile_k_fp64, 1,
              square);
SUM_KN_KERNEL(sum_squared_fp64_kn, double, double, 1, square);

SUM_K1_KERNEL(sum_squared_bf16_fp32_k1, bfloat16, float, consistent_tile_k_fp32,
              1, square);
SUM_KN_KERNEL(sum_squared_bf16_fp32_kn, bfloat16, float, 1, square);

SUM_K1_KERNEL(sum_squared_fp16_fp32_k1, half, float, consistent_tile_k_fp32, 1,
              square);
SUM_KN_KERNEL(sum_squared_fp16_fp32_kn, half, float, 1, square);

SUM_K1_KERNEL(sum_squared_int32_k1, int32_t, int32_t, 1, 1, square);
SUM_KN_KERNEL(sum_squared_int32_kn, int32_t, int32_t, 1, square);

SUM_K1_KERNEL(sum_squared_int8_int32_k1, int8_t, int32_t, 1, 1, square);
SUM_KN_KERNEL(sum_squared_int8_int32_kn, int8_t, int32_t, 1, square);

SUM_K1_KERNEL(sum_squared_uint8_int32_k1, uint8_t, int32_t, 1, 1, square);
SUM_KN_KERNEL(sum_squared_uint8_int32_kn, uint8_t, int32_t, 1, square);

// min/max kernels
#define MIN_K1_KERNEL(name, T)                                    \
  void name(size_t n, size_t k, size_t a_stride_n, const void* a, \
            size_t c_stride_m, void* c) {                         \
    reduce<T, T, T, min_op<T>>(n, 1, 1, k, a_stride_n, 0, 0,      \
                               reinterpret_cast<const T*>(a),     \
                               reinterpret_cast<T*>(c));          \
  }
#define MIN_KN_KERNEL(name, T)                                       \
  void name(size_t n, size_t k, size_t a_stride_k, const void* a,    \
            size_t c_stride_m, void* c) {                            \
    reduce<T, T, T, min_op<T>>(n, 1, k, 1, sizeof(T), 0, a_stride_k, \
                               reinterpret_cast<const T*>(a),        \
                               reinterpret_cast<T*>(c));             \
  }
#define MAX_K1_KERNEL(name, T)                                    \
  void name(size_t n, size_t k, size_t a_stride_n, const void* a, \
            size_t c_stride_m, void* c) {                         \
    reduce<T, T, T, max_op<T>>(n, 1, 1, k, a_stride_n, 0, 0,      \
                               reinterpret_cast<const T*>(a),     \
                               reinterpret_cast<T*>(c));          \
  }
#define MAX_KN_KERNEL(name, T)                                       \
  void name(size_t n, size_t k, size_t a_stride_k, const void* a,    \
            size_t c_stride_m, void* c) {                            \
    reduce<T, T, T, max_op<T>>(n, 1, k, 1, sizeof(T), 0, a_stride_k, \
                               reinterpret_cast<const T*>(a),        \
                               reinterpret_cast<T*>(c));             \
  }
#define MIN_MAX_K1_KERNEL(name, T)                                      \
  void name(size_t n, size_t k, size_t a_stride_n, const void* a,       \
            size_t c_stride_m, void* c) {                               \
    min_max<T, T, min_op<T>, max_op<T>>(                                \
        n, 1, 1, k, a_stride_n, 0, 0, reinterpret_cast<const T*>(a),    \
        c_stride_m, reinterpret_cast<T*>(c), min_op<T>{}, max_op<T>{}); \
  }
#define MIN_MAX_KN_KERNEL(name, T)                                           \
  void name(size_t n, size_t k, size_t a_stride_k, const void* a,            \
            size_t c_stride_m, void* c) {                                    \
    min_max<T, T, min_op<T>, max_op<T>>(                                     \
        n, 1, k, 1, sizeof(T), 0, a_stride_k, reinterpret_cast<const T*>(a), \
        c_stride_m, reinterpret_cast<T*>(c), min_op<T>{}, max_op<T>{});      \
  }

MIN_K1_KERNEL(min_fp32_k1, float);
MIN_KN_KERNEL(min_fp32_kn, float);
MIN_K1_KERNEL(min_fp64_k1, double);
MIN_KN_KERNEL(min_fp64_kn, double);
MIN_K1_KERNEL(min_fp16_k1, half);
MIN_KN_KERNEL(min_fp16_kn, half);
MIN_K1_KERNEL(min_bf16_k1, bfloat16);
MIN_KN_KERNEL(min_bf16_kn, bfloat16);
MIN_K1_KERNEL(min_int8_k1, int8_t);
MIN_KN_KERNEL(min_int8_kn, int8_t);
MIN_K1_KERNEL(min_uint8_k1, uint8_t);
MIN_KN_KERNEL(min_uint8_kn, uint8_t);

MAX_K1_KERNEL(max_fp32_k1, float);
MAX_KN_KERNEL(max_fp32_kn, float);
MAX_K1_KERNEL(max_fp64_k1, double);
MAX_KN_KERNEL(max_fp64_kn, double);
MAX_K1_KERNEL(max_fp16_k1, half);
MAX_KN_KERNEL(max_fp16_kn, half);
MAX_K1_KERNEL(max_bf16_k1, bfloat16);
MAX_KN_KERNEL(max_bf16_kn, bfloat16);
MAX_K1_KERNEL(max_int8_k1, int8_t);
MAX_KN_KERNEL(max_int8_kn, int8_t);
MAX_K1_KERNEL(max_uint8_k1, uint8_t);
MAX_KN_KERNEL(max_uint8_kn, uint8_t);

MIN_MAX_K1_KERNEL(min_max_fp32_k1, float);
MIN_MAX_KN_KERNEL(min_max_fp32_kn, float);
MIN_MAX_K1_KERNEL(min_max_fp64_k1, double);
MIN_MAX_KN_KERNEL(min_max_fp64_kn, double);
MIN_MAX_K1_KERNEL(min_max_fp16_k1, half);
MIN_MAX_KN_KERNEL(min_max_fp16_kn, half);
MIN_MAX_K1_KERNEL(min_max_bf16_k1, bfloat16);
MIN_MAX_KN_KERNEL(min_max_bf16_kn, bfloat16);
MIN_MAX_K1_KERNEL(min_max_int8_k1, int8_t);
MIN_MAX_KN_KERNEL(min_max_int8_kn, int8_t);
MIN_MAX_K1_KERNEL(min_max_uint8_k1, uint8_t);
MIN_MAX_KN_KERNEL(min_max_uint8_kn, uint8_t);

reduce_kernel get_sum_kernel(ynn_type a_type, ynn_type c_type) {
  reduce_kernel res = {nullptr, nullptr};
#define YNN_REDUCE_K1_KERNEL(arch, name, A, C)               \
  if (is_arch_supported(arch)) {                             \
    if (type_of<A>() == a_type && type_of<C>() == c_type) {  \
      YNN_LOG_DEBUG() << "Using reduce k1 kernel " << #name; \
      res.k1 = name;                                         \
    }                                                        \
  }
#define YNN_REDUCE_KN_KERNEL(arch, name, A, C)               \
  if (is_arch_supported(arch)) {                             \
    if (type_of<A>() == a_type && type_of<C>() == c_type) {  \
      YNN_LOG_DEBUG() << "Using reduce kn kernel " << #name; \
      res.kn = name;                                         \
    }                                                        \
  }
#include "ynnpack/kernels/reduce/sum.inc"
#undef YNN_REDUCE_K1_KERNEL
#undef YNN_REDUCE_KN_KERNEL
  if (!res.k1 || !res.kn) {
    YNN_LOG_ERROR() << "Unsupported sum type " << a_type << "_" << c_type;
  }
  return res;
}

reduce_kernel get_sum_squared_kernel(ynn_type a_type, ynn_type c_type) {
  reduce_kernel res = {nullptr, nullptr};
#define YNN_REDUCE_K1_KERNEL(arch, name, A, C)               \
  if (is_arch_supported(arch)) {                             \
    if (type_of<A>() == a_type && type_of<C>() == c_type) {  \
      YNN_LOG_DEBUG() << "Using reduce k1 kernel " << #name; \
      res.k1 = name;                                         \
    }                                                        \
  }
#define YNN_REDUCE_KN_KERNEL(arch, name, A, C)               \
  if (is_arch_supported(arch)) {                             \
    if (type_of<A>() == a_type && type_of<C>() == c_type) {  \
      YNN_LOG_DEBUG() << "Using reduce kn kernel " << #name; \
      res.kn = name;                                         \
    }                                                        \
  }
#include "ynnpack/kernels/reduce/sum_squared.inc"
#undef YNN_REDUCE_K1_KERNEL
#undef YNN_REDUCE_KN_KERNEL
  if (!res.k1 || !res.kn) {
    YNN_LOG_ERROR() << "Unsupported sum_squared type " << a_type << "_"
                    << c_type;
  }
  return res;
}

reduce_kernel get_min_kernel(ynn_type type) {
  reduce_kernel res = {nullptr, nullptr};
#define YNN_REDUCE_K1_KERNEL(arch, name, A, C)               \
  if (is_arch_supported(arch)) {                             \
    if (type_of<A>() == type && type_of<C>() == type) {      \
      YNN_LOG_DEBUG() << "Using reduce k1 kernel " << #name; \
      res.k1 = name;                                         \
    }                                                        \
  }
#define YNN_REDUCE_KN_KERNEL(arch, name, A, C)               \
  if (is_arch_supported(arch)) {                             \
    if (type_of<A>() == type && type_of<C>() == type) {      \
      YNN_LOG_DEBUG() << "Using reduce kn kernel " << #name; \
      res.kn = name;                                         \
    }                                                        \
  }
#include "ynnpack/kernels/reduce/min.inc"
#undef YNN_REDUCE_K1_KERNEL
#undef YNN_REDUCE_KN_KERNEL
  if (!res.k1 || !res.kn) {
    YNN_LOG_ERROR() << "Unsupported min type " << type;
  }
  return res;
}

reduce_kernel get_max_kernel(ynn_type type) {
  reduce_kernel res = {nullptr, nullptr};
#define YNN_REDUCE_K1_KERNEL(arch, name, A, C)               \
  if (is_arch_supported(arch)) {                             \
    if (type_of<A>() == type && type_of<C>() == type) {      \
      YNN_LOG_DEBUG() << "Using reduce k1 kernel " << #name; \
      res.k1 = name;                                         \
    }                                                        \
  }
#define YNN_REDUCE_KN_KERNEL(arch, name, A, C)               \
  if (is_arch_supported(arch)) {                             \
    if (type_of<A>() == type && type_of<C>() == type) {      \
      YNN_LOG_DEBUG() << "Using reduce kn kernel " << #name; \
      res.kn = name;                                         \
    }                                                        \
  }
#include "ynnpack/kernels/reduce/max.inc"
#undef YNN_REDUCE_K1_KERNEL
#undef YNN_REDUCE_KN_KERNEL
  if (!res.k1 || !res.kn) {
    YNN_LOG_ERROR() << "Unsupported max type " << type;
  }
  return res;
}

reduce_kernel get_min_max_kernel(ynn_type type) {
  reduce_kernel res = {nullptr, nullptr};
#define YNN_REDUCE_K1_KERNEL(arch, name, A, C)               \
  if (is_arch_supported(arch)) {                             \
    if (type_of<A>() == type && type_of<C>() == type) {      \
      YNN_LOG_DEBUG() << "Using reduce k1 kernel " << #name; \
      res.k1 = name;                                         \
    }                                                        \
  }
#define YNN_REDUCE_KN_KERNEL(arch, name, A, C)               \
  if (is_arch_supported(arch)) {                             \
    if (type_of<A>() == type && type_of<C>() == type) {      \
      YNN_LOG_DEBUG() << "Using reduce kn kernel " << #name; \
      res.kn = name;                                         \
    }                                                        \
  }
#include "ynnpack/kernels/reduce/min_max.inc"
#undef YNN_REDUCE_K1_KERNEL
#undef YNN_REDUCE_KN_KERNEL
  if (!res.k1 || !res.kn) {
    YNN_LOG_ERROR() << "Unsupported min_max type " << type;
  }
  return res;
}

}  // namespace ynn
