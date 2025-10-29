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
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/reduce/generic.h"

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
      // For numerical consistency, always do the final k reduction as a binary
      // tree.
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
    for (size_t i = 0; i < n; ++i) {
      C[i] = op(static_cast<AccT>(C[i]), acc[i][0]);
    }
  }
};

template <typename AccT, typename T, size_t N_, typename ReduceOp, typename F>
struct accumulator_k1_1 {
  static constexpr std::integral_constant<size_t, N_> N = {};
  static constexpr std::integral_constant<size_t, 1> K2 = {};

  AccT acc[N];

  accumulator_k1_1() = default;

  YNN_ALWAYS_INLINE explicit accumulator_k1_1(size_t) {
    std::fill_n(acc, N, static_cast<AccT>(ReduceOp::identity));
  }

  template <typename AT, typename NT, typename K2T>
  YNN_ALWAYS_INLINE void reduce(const AT* __restrict A, NT n,
                                size_t /*A_stride_k2*/, K2T) {
    ReduceOp op;
    F f;
    for (size_t i = 0; i < n; ++i) {
      acc[i] = op(acc[i], f(A[i]));
    }
  }

  template <typename NT>
  YNN_ALWAYS_INLINE void accumulate(size_t /*C_stride_m*/, T* __restrict C,
                                    NT n) {
    ReduceOp op;
    for (size_t i = 0; i < n; ++i) {
      C[i] = op(static_cast<AccT>(C[i]), acc[i]);
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

  AccT acc_min[N];
  AccT acc_max[N];

  min_max_accumulator_k1_1() = default;

  YNN_ALWAYS_INLINE explicit min_max_accumulator_k1_1(size_t) {
    std::fill_n(acc_min, N, static_cast<AccT>(Min::identity));
    std::fill_n(acc_max, N, static_cast<AccT>(Max::identity));
  }

  template <typename AT, typename N, typename K2T>
  YNN_ALWAYS_INLINE void reduce(const AT* __restrict A, N n,
                                size_t /*A_stride_k2*/, K2T) {
    Min min;
    Max max;
    for (size_t i = 0; i < n; ++i) {
      acc_min[i] = min(acc_min[i], static_cast<AccT>(A[i]));
      acc_max[i] = max(acc_max[i], static_cast<AccT>(A[i]));
    }
  }

  template <typename NT>
  YNN_ALWAYS_INLINE void accumulate(size_t C_stride_m, T* __restrict C, NT n) {
    Min min;
    Max max;
    T* __restrict C_min = offset_bytes(C, 0 * C_stride_m);
    T* __restrict C_max = offset_bytes(C, 1 * C_stride_m);
    for (size_t i = 0; i < n; ++i) {
      C_min[i] = min(static_cast<AccT>(C_min[i]), acc_min[i]);
      C_max[i] = max(static_cast<AccT>(C_max[i]), acc_max[i]);
    }
  }
};

template <typename T>
struct min_op {
  T operator()(T a, T b) { return min(a, b); }

  static constexpr T identity = type_info<T>::min_identity();
  static constexpr bool is_associative = true;
};

template <typename T>
struct max_op {
  T operator()(T a, T b) { return max(a, b); }

  static constexpr T identity = type_info<T>::max_identity();
  static constexpr bool is_associative = true;
};

template <typename T>
struct sum_op {
  T operator()(T a, T b) { return a + b; }

  static constexpr T identity = type_info<T>::sum_identity();
  static constexpr bool is_associative = std::is_integral<T>::value;
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
    tiled_reduce<accumulator_k1_1<AccT, CT, 128 / sizeof(AccT), ReduceOp, F>,
                 AT, CT>(N, K3, K2, A_stride_k3, A_stride_k2, A,
                         /*C_stride_m=*/0, C);
  } else {
    tiled_reduce<accumulator<AccT, CT, 1, 128 / sizeof(AccT), ReduceOp, F>, AT,
                 CT>(N, K3, K2, K1, A_stride_n, A_stride_k3, A_stride_k2, A,
                     /*C_stride_m=*/0, C);
  }
}

template <typename AccT, typename T, typename Min, typename Max>
void min_max(size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_n,
             size_t A_stride_k3, size_t A_stride_k2, const T* A,
             size_t C_stride_m, T* C, Min, Max) {
  if (K1 == 1 && A_stride_n == sizeof(T)) {
    tiled_reduce<min_max_accumulator_k1_1<AccT, T, 64 / sizeof(AccT), Min, Max>,
                 T, T>(N, K3, K2, A_stride_k3, A_stride_k2, A, C_stride_m, C);
  } else {
    tiled_reduce<min_max_accumulator<AccT, T, 1, 64 / sizeof(AccT), Min, Max>,
                 T, T>(N, K3, K2, K1, A_stride_n, A_stride_k3, A_stride_k2, A,
                       C_stride_m, C);
  }
}

}  // namespace

void sum_fp32(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
              size_t a_stride_k3, size_t a_stride_k2, const void* a, size_t,
              void* c) {
  reduce<float>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                static_cast<const float*>(a), static_cast<float*>(c),
                sum_op<float>());
}

void sum_bf16_fp32(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
                   size_t a_stride_k3, size_t a_stride_k2, const void* a,
                   size_t, void* c) {
  reduce<float>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                static_cast<const bfloat16*>(a), static_cast<float*>(c),
                sum_op<float>());
}

void sum_fp16_fp32(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
                   size_t a_stride_k3, size_t a_stride_k2, const void* a,
                   size_t, void* c) {
  reduce<float>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                static_cast<const half*>(a), static_cast<float*>(c),
                sum_op<float>());
}

void sum_int8_int32(size_t n, size_t k3, size_t k2, size_t k1,
                    size_t a_stride_n, size_t a_stride_k3, size_t a_stride_k2,
                    const void* a, size_t, void* c) {
  reduce<int32_t>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                  static_cast<const int8_t*>(a), static_cast<int32_t*>(c),
                  sum_op<int32_t>());
}

void sum_uint8_int32(size_t n, size_t k3, size_t k2, size_t k1,
                     size_t a_stride_n, size_t a_stride_k3, size_t a_stride_k2,
                     const void* a, size_t, void* c) {
  reduce<int32_t>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                  static_cast<const uint8_t*>(a), static_cast<int32_t*>(c),
                  sum_op<int32_t>());
}

void sum_squared_fp32(size_t n, size_t k3, size_t k2, size_t k1,
                      size_t a_stride_n, size_t a_stride_k3, size_t a_stride_k2,
                      const void* a, size_t, void* c) {
  reduce<float>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                static_cast<const float*>(a), static_cast<float*>(c),
                sum_op<float>(), square_op<float, float>());
}

void sum_squared_bf16_fp32(size_t n, size_t k3, size_t k2, size_t k1,
                           size_t a_stride_n, size_t a_stride_k3,
                           size_t a_stride_k2, const void* a, size_t, void* c) {
  reduce<float>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                static_cast<const bfloat16*>(a), static_cast<float*>(c),
                sum_op<float>(), square_op<float, bfloat16>());
}

void sum_squared_fp16_fp32(size_t n, size_t k3, size_t k2, size_t k1,
                           size_t a_stride_n, size_t a_stride_k3,
                           size_t a_stride_k2, const void* a, size_t, void* c) {
  reduce<float>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                static_cast<const half*>(a), static_cast<float*>(c),
                sum_op<float>(), square_op<float, half>());
}

void sum_squared_int8_int32(size_t n, size_t k3, size_t k2, size_t k1,
                            size_t a_stride_n, size_t a_stride_k3,
                            size_t a_stride_k2, const void* a, size_t,
                            void* c) {
  reduce<int32_t>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                  static_cast<const int8_t*>(a), static_cast<int32_t*>(c),
                  sum_op<int32_t>(), square_op<int32_t, int8_t>());
}

void sum_squared_uint8_int32(size_t n, size_t k3, size_t k2, size_t k1,
                             size_t a_stride_n, size_t a_stride_k3,
                             size_t a_stride_k2, const void* a, size_t,
                             void* c) {
  reduce<int32_t>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                  static_cast<const uint8_t*>(a), static_cast<int32_t*>(c),
                  sum_op<int32_t>(), square_op<int32_t, uint8_t>());
}

void min_fp32(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
              size_t a_stride_k3, size_t a_stride_k2, const void* a, size_t,
              void* c) {
  reduce<float>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                static_cast<const float*>(a), static_cast<float*>(c),
                min_op<float>());
}

void min_fp16(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
              size_t a_stride_k3, size_t a_stride_k2, const void* a, size_t,
              void* c) {
  reduce<half_rvar>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                    static_cast<const half*>(a), static_cast<half*>(c),
                    min_op<half_rvar>());
}

void min_bf16(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
              size_t a_stride_k3, size_t a_stride_k2, const void* a, size_t,
              void* c) {
  reduce<bfloat16_rvar>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                        static_cast<const bfloat16*>(a),
                        static_cast<bfloat16*>(c), min_op<bfloat16_rvar>());
}

void min_int8(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
              size_t a_stride_k3, size_t a_stride_k2, const void* a, size_t,
              void* c) {
  reduce<int8_t>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                 static_cast<const int8_t*>(a), static_cast<int8_t*>(c),
                 min_op<int8_t>());
}

void min_uint8(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
               size_t a_stride_k3, size_t a_stride_k2, const void* a, size_t,
               void* c) {
  reduce<uint8_t>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                  static_cast<const uint8_t*>(a), static_cast<uint8_t*>(c),
                  min_op<uint8_t>());
}

void max_fp32(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
              size_t a_stride_k3, size_t a_stride_k2, const void* a, size_t,
              void* c) {
  reduce<float>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                static_cast<const float*>(a), static_cast<float*>(c),
                max_op<float>());
}

void max_fp16(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
              size_t a_stride_k3, size_t a_stride_k2, const void* a, size_t,
              void* c) {
  reduce<half_rvar>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                    static_cast<const half*>(a), static_cast<half*>(c),
                    max_op<half_rvar>());
}

void max_bf16(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
              size_t a_stride_k3, size_t a_stride_k2, const void* a, size_t,
              void* c) {
  reduce<bfloat16_rvar>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                        static_cast<const bfloat16*>(a),
                        static_cast<bfloat16*>(c), max_op<bfloat16_rvar>());
}

void max_int8(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
              size_t a_stride_k3, size_t a_stride_k2, const void* a, size_t,
              void* c) {
  reduce<int8_t>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                 static_cast<const int8_t*>(a), static_cast<int8_t*>(c),
                 max_op<int8_t>());
}

void max_uint8(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
               size_t a_stride_k3, size_t a_stride_k2, const void* a, size_t,
               void* c) {
  reduce<uint8_t>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                  static_cast<const uint8_t*>(a), static_cast<uint8_t*>(c),
                  max_op<uint8_t>());
}

void min_max_fp32(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
                  size_t a_stride_k3, size_t a_stride_k2, const void* a,
                  size_t c_stride_m, void* c) {
  min_max<float>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                 static_cast<const float*>(a), c_stride_m,
                 static_cast<float*>(c), min_op<float>(), max_op<float>());
}

void min_max_fp16(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
                  size_t a_stride_k3, size_t a_stride_k2, const void* a,
                  size_t c_stride_m, void* c) {
  min_max<half_rvar>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                     static_cast<const half*>(a), c_stride_m,
                     static_cast<half*>(c), min_op<half_rvar>(),
                     max_op<half_rvar>());
}

void min_max_bf16(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
                  size_t a_stride_k3, size_t a_stride_k2, const void* a,
                  size_t c_stride_m, void* c) {
  min_max<bfloat16_rvar>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                         static_cast<const bfloat16*>(a), c_stride_m,
                         static_cast<bfloat16*>(c), min_op<bfloat16_rvar>(),
                         max_op<bfloat16_rvar>());
}

void min_max_int8(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
                  size_t a_stride_k3, size_t a_stride_k2, const void* a,
                  size_t c_stride_m, void* c) {
  min_max<int8_t>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                  static_cast<const int8_t*>(a), c_stride_m,
                  static_cast<int8_t*>(c), min_op<int8_t>(), max_op<int8_t>());
}

void min_max_uint8(size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_n,
                   size_t a_stride_k3, size_t a_stride_k2, const void* a,
                   size_t c_stride_m, void* c) {
  min_max<uint8_t>(n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2,
                   static_cast<const uint8_t*>(a), c_stride_m,
                   static_cast<uint8_t*>(c), min_op<uint8_t>(),
                   max_op<uint8_t>());
}

unary_reduce_kernel_fn get_sum_kernel(ynn_type a_type, ynn_type c_type,
                                      size_t n, size_t k3, size_t k2,
                                      size_t k1) {
#define YNN_UNARY_REDUCE_KERNEL(arch, name, A, C)           \
  if (is_arch_supported(arch)) {                            \
    if (type_of<A>() == a_type && type_of<C>() == c_type) { \
      return name;                                          \
    }                                                       \
  }
#include "ynnpack/kernels/reduce/sum.inc"
#undef YNN_UNARY_REDUCE_KERNEL
  YNN_LOG_ERROR() << "Unsupported sum type " << a_type << "_" << c_type;
  return nullptr;
}

unary_reduce_kernel_fn get_sum_squared_kernel(ynn_type a_type, ynn_type c_type,
                                              size_t n, size_t k3, size_t k2,
                                              size_t k1) {
#define YNN_UNARY_REDUCE_KERNEL(arch, name, A, C)           \
  if (is_arch_supported(arch)) {                            \
    if (type_of<A>() == a_type && type_of<C>() == c_type) { \
      return name;                                          \
    }                                                       \
  }
#include "ynnpack/kernels/reduce/sum_squared.inc"
#undef YNN_UNARY_REDUCE_KERNEL
  YNN_LOG_ERROR() << "Unsupported sum_squared type " << a_type << "_" << c_type;
  return nullptr;
}

unary_reduce_kernel_fn get_min_kernel(ynn_type type, size_t n, size_t k3,
                                      size_t k2, size_t k1) {
#define YNN_UNARY_REDUCE_KERNEL(arch, name, A, C)       \
  if (is_arch_supported(arch)) {                        \
    if (type_of<A>() == type && type_of<C>() == type) { \
      return name;                                      \
    }                                                   \
  }
#include "ynnpack/kernels/reduce/min.inc"
#undef YNN_UNARY_REDUCE_KERNEL
  YNN_LOG_ERROR() << "Unsupported min type " << type;
  return nullptr;
}

unary_reduce_kernel_fn get_max_kernel(ynn_type type, size_t n, size_t k3,
                                      size_t k2, size_t k1) {
#define YNN_UNARY_REDUCE_KERNEL(arch, name, A, C)       \
  if (is_arch_supported(arch)) {                        \
    if (type_of<A>() == type && type_of<C>() == type) { \
      return name;                                      \
    }                                                   \
  }
#include "ynnpack/kernels/reduce/max.inc"
#undef YNN_UNARY_REDUCE_KERNEL
  YNN_LOG_ERROR() << "Unsupported max type " << type;
  return nullptr;
}

unary_reduce_kernel_fn get_min_max_kernel(ynn_type type, size_t n, size_t k3,
                                          size_t k2, size_t k1) {
#define YNN_UNARY_REDUCE_KERNEL(arch, name, A, C)       \
  if (is_arch_supported(arch)) {                        \
    if (type_of<A>() == type && type_of<C>() == type) { \
      return name;                                      \
    }                                                   \
  }
#include "ynnpack/kernels/reduce/min_max.inc"
#undef YNN_UNARY_REDUCE_KERNEL
  YNN_LOG_ERROR() << "Unsupported min_max type " << type;
  return nullptr;
}

}  // namespace ynn
