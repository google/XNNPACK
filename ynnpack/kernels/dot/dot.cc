// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/dot/dot.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/fp8.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/dot/cost_model/cost_model.h"
#ifdef YNN_ENABLE_CPUINFO
#include <cpuinfo.h>
#endif
#ifdef YNN_ARCH_X86
#include "ynnpack/kernels/dot/cost_model/broadwell.h"
#include "ynnpack/kernels/dot/cost_model/cascade_lake.h"
#include "ynnpack/kernels/dot/cost_model/golden_cove.h"
#include "ynnpack/kernels/dot/cost_model/haswell.h"
#include "ynnpack/kernels/dot/cost_model/redwood_cove.h"
#include "ynnpack/kernels/dot/cost_model/zen2.h"
#include "ynnpack/kernels/dot/cost_model/zen3.h"
#include "ynnpack/kernels/dot/cost_model/zen4.h"
#endif  // YNN_ARCH_X86
#ifdef YNN_ARCH_ARM
#include "ynnpack/kernels/dot/cost_model/cortex_a510.h"
#include "ynnpack/kernels/dot/cost_model/cortex_a520.h"
#include "ynnpack/kernels/dot/cost_model/cortex_a710.h"
#include "ynnpack/kernels/dot/cost_model/cortex_a715.h"
#include "ynnpack/kernels/dot/cost_model/cortex_a720.h"
#include "ynnpack/kernels/dot/cost_model/cortex_a725.h"
#include "ynnpack/kernels/dot/cost_model/cortex_x1.h"
#include "ynnpack/kernels/dot/cost_model/cortex_x4.h"
#include "ynnpack/kernels/dot/cost_model/donan_everest.h"
#include "ynnpack/kernels/dot/cost_model/donan_sawtooth.h"
#include "ynnpack/kernels/dot/cost_model/lumex_c1_pro.h"
#include "ynnpack/kernels/dot/cost_model/lumex_c1_ultra.h"
#include "ynnpack/kernels/dot/cost_model/neoverse_n1.h"
#include "ynnpack/kernels/dot/cost_model/neoverse_v2.h"
#include "ynnpack/kernels/dot/cost_model/oryon.h"
#endif  // YNN_ARCH_ARM

namespace ynn {

namespace {

template <typename AT, typename BT, typename CT>
void dot_1x1x1(size_t M, size_t N, size_t K3, size_t K2, size_t K1,
               size_t A_stride_m, size_t A_stride_k3, size_t A_stride_k2,
               const AT* A, size_t B_stride_k3, size_t B_stride_k2,
               size_t B_stride_k1, const BT* B, size_t C_in_stride_m,
               const CT* C_in, size_t C_out_stride_m, CT* C_out) {
  using B_info = type_info<BT>;
  assert(M == 1);
  CT* acc = YNN_ALLOCA(CT, N);
  std::fill_n(acc, N, static_cast<CT>(0));
  for (size_t k3 = 0; k3 < K3; ++k3) {
    const BT* B_k3 = offset_bytes(B, k3 * B_stride_k3);
    const AT* A_k3 = offset_bytes(A, k3 * A_stride_k3);
    for (size_t k2 = 0; k2 < K2; ++k2) {
      const BT* B_k2 = offset_bytes(B_k3, k2 * B_stride_k2);
      const AT* A_k2 = offset_bytes(A_k3, k2 * A_stride_k2);
      for (size_t k1 = 0; k1 < K1; ++k1) {
        const BT* B_k1 = offset_bytes(B_k2, k1 * B_stride_k1);
        const AT A_k1 = A_k2[k1];
        for (size_t j = 0; j < N; ++j) {
          acc[j] +=
              static_cast<CT>(A_k1) * static_cast<CT>(B_info::get(B_k1, j));
        }
      }
    }
  }
  if (C_in) {
    for (size_t j = 0; j < N; ++j) {
      C_out[j] = acc[j] + C_in[j];
    }
  } else {
    std::copy_n(acc, N, C_out);
  }
}

template <typename AT, typename BT, typename CT>
void dot_1x1x2(size_t M, size_t N, size_t K3, size_t K2, size_t K1,
               size_t A_stride_m, size_t A_stride_k3, size_t A_stride_k2,
               const AT* A, size_t B_stride_k3, size_t B_stride_k2,
               size_t B_stride_k1, const BT* B, size_t C_in_stride_m,
               const CT* C_in, size_t C_out_stride_m, CT* C_out) {
  using B_info = type_info<BT>;
  assert(M == 1);
  assert(K1 % 2 == 0);
  CT* acc = YNN_ALLOCA(CT, N);
  std::fill_n(acc, N, 0);
  for (size_t k3 = 0; k3 < K3; ++k3) {
    const BT* B_k3 = offset_bytes(B, k3 * B_stride_k3);
    const AT* A_k3 = offset_bytes(A, k3 * A_stride_k3);
    for (size_t k2 = 0; k2 < K2; ++k2) {
      const BT* B_k2 = offset_bytes(B_k3, k2 * B_stride_k2);
      const AT* A_k2 = offset_bytes(A_k3, k2 * A_stride_k2);
      for (size_t k1 = 0; k1 < K1; k1 += 2) {
        const BT* B_k1 = offset_bytes(B_k2, k1 * B_stride_k1);
        const AT A_k1_0 = A_k2[k1 + 0];
        const AT A_k1_1 = A_k2[k1 + 1];
        for (size_t j = 0; j < N; ++j) {
          const auto B_k1_0 = B_info::get(B_k1, 2 * j + 0);
          const auto B_k1_1 = B_info::get(B_k1, 2 * j + 1);
          acc[j] += static_cast<CT>(A_k1_0) * static_cast<CT>(B_k1_0);
          acc[j] += static_cast<CT>(A_k1_1) * static_cast<CT>(B_k1_1);
        }
      }
    }
  }
  if (C_in) {
    for (size_t j = 0; j < N; ++j) {
      C_out[j] = acc[j] + C_in[j];
    }
  } else {
    std::copy_n(acc, N, C_out);
  }
}

template <typename AT, typename BT, typename CT>
void dot_1x1x4(size_t M, size_t N, size_t K3, size_t K2, size_t K1,
               size_t A_stride_m, size_t A_stride_k3, size_t A_stride_k2,
               const AT* A, size_t B_stride_k3, size_t B_stride_k2,
               size_t B_stride_k1, const BT* B, size_t C_in_stride_m,
               const CT* C_in, size_t C_out_stride_m, CT* C_out) {
  using B_info = type_info<BT>;
  assert(M == 1);
  assert(K1 % 4 == 0);
  CT* acc = YNN_ALLOCA(CT, N);
  std::fill_n(acc, N, 0);
  for (size_t k3 = 0; k3 < K3; ++k3) {
    const BT* B_k3 = offset_bytes(B, k3 * B_stride_k3);
    const AT* A_k3 = offset_bytes(A, k3 * A_stride_k3);
    for (size_t k2 = 0; k2 < K2; ++k2) {
      const BT* B_k2 = offset_bytes(B_k3, k2 * B_stride_k2);
      const AT* A_k2 = offset_bytes(A_k3, k2 * A_stride_k2);
      for (size_t k1 = 0; k1 < K1; k1 += 4) {
        const BT* B_k1 = offset_bytes(B_k2, k1 * B_stride_k1);
        const AT A_k1_0 = A_k2[k1 + 0];
        const AT A_k1_1 = A_k2[k1 + 1];
        const AT A_k1_2 = A_k2[k1 + 2];
        const AT A_k1_3 = A_k2[k1 + 3];
        for (size_t j = 0; j < N; ++j) {
          const auto B_k1_0 = B_info::get(B_k1, 4 * j + 0);
          const auto B_k1_1 = B_info::get(B_k1, 4 * j + 1);
          const auto B_k1_2 = B_info::get(B_k1, 4 * j + 2);
          const auto B_k1_3 = B_info::get(B_k1, 4 * j + 3);
          acc[j] += static_cast<CT>(A_k1_0) * static_cast<CT>(B_k1_0);
          acc[j] += static_cast<CT>(A_k1_1) * static_cast<CT>(B_k1_1);
          acc[j] += static_cast<CT>(A_k1_2) * static_cast<CT>(B_k1_2);
          acc[j] += static_cast<CT>(A_k1_3) * static_cast<CT>(B_k1_3);
        }
      }
    }
  }
  if (C_in) {
    for (size_t j = 0; j < N; ++j) {
      C_out[j] = acc[j] + C_in[j];
    }
  } else {
    std::copy_n(acc, N, C_out);
  }
}

}  // namespace

void dot_fp32_1xNx1_1x1x1(size_t m, size_t n, size_t k3, size_t k2, size_t k1,
                          size_t a_stride_m, size_t a_stride_k3,
                          size_t a_stride_k2, const void* a, size_t b_stride_k3,
                          size_t b_stride_k2, size_t b_stride_k1, const void* b,
                          size_t c_in_stride_m, const void* c_in,
                          size_t c_out_stride_m, void* c_out,
                          dot_kernel_state* /*state*/) {
  dot_1x1x1(m, n, k3, k2, k1, a_stride_m, a_stride_k3, a_stride_k2,
            static_cast<const float*>(a), b_stride_k3, b_stride_k2, b_stride_k1,
            static_cast<const float*>(b), c_in_stride_m,
            static_cast<const float*>(c_in), c_out_stride_m,
            static_cast<float*>(c_out));
}

void dot_fp64_1xNx1_1x1x1(size_t m, size_t n, size_t k3, size_t k2, size_t k1,
                          size_t a_stride_m, size_t a_stride_k3,
                          size_t a_stride_k2, const void* a, size_t b_stride_k3,
                          size_t b_stride_k2, size_t b_stride_k1, const void* b,
                          size_t c_in_stride_m, const void* c_in,
                          size_t c_out_stride_m, void* c_out,
                          dot_kernel_state* /*state*/) {
  dot_1x1x1(m, n, k3, k2, k1, a_stride_m, a_stride_k3, a_stride_k2,
            static_cast<const double*>(a), b_stride_k3, b_stride_k2,
            b_stride_k1, static_cast<const double*>(b), c_in_stride_m,
            static_cast<const double*>(c_in), c_out_stride_m,
            static_cast<double*>(c_out));
}

void dot_fp16_fp16_fp32_1xNx1_1x1x1(size_t m, size_t n, size_t k3, size_t k2,
                                    size_t k1, size_t a_stride_m,
                                    size_t a_stride_k3, size_t a_stride_k2,
                                    const void* a, size_t b_stride_k3,
                                    size_t b_stride_k2, size_t b_stride_k1,
                                    const void* b, size_t c_in_stride_m,
                                    const void* c_in, size_t c_out_stride_m,
                                    void* c_out, dot_kernel_state* /*state*/) {
  dot_1x1x1(m, n, k3, k2, k1, a_stride_m, a_stride_k3, a_stride_k2,
            static_cast<const half*>(a), b_stride_k3, b_stride_k2, b_stride_k1,
            static_cast<const half*>(b), c_in_stride_m,
            static_cast<const float*>(c_in), c_out_stride_m,
            static_cast<float*>(c_out));
}

void dot_bf16_bf16_fp32_1xNx1_1x1x1(size_t m, size_t n, size_t k3, size_t k2,
                                    size_t k1, size_t a_stride_m,
                                    size_t a_stride_k3, size_t a_stride_k2,
                                    const void* a, size_t b_stride_k3,
                                    size_t b_stride_k2, size_t b_stride_k1,
                                    const void* b, size_t c_in_stride_m,
                                    const void* c_in, size_t c_out_stride_m,
                                    void* c_out, dot_kernel_state* /*state*/) {
  dot_1x1x1(m, n, k3, k2, k1, a_stride_m, a_stride_k3, a_stride_k2,
            static_cast<const bfloat16*>(a), b_stride_k3, b_stride_k2,
            b_stride_k1, static_cast<const bfloat16*>(b), c_in_stride_m,
            static_cast<const float*>(c_in), c_out_stride_m,
            static_cast<float*>(c_out));
}

void dot_int8_int8_int32_1xNx1_1x1x1(size_t m, size_t n, size_t k3, size_t k2,
                                     size_t k1, size_t a_stride_m,
                                     size_t a_stride_k3, size_t a_stride_k2,
                                     const void* a, size_t b_stride_k3,
                                     size_t b_stride_k2, size_t b_stride_k1,
                                     const void* b, size_t c_in_stride_m,
                                     const void* c_in, size_t c_out_stride_m,
                                     void* c_out, dot_kernel_state* /*state*/) {
  dot_1x1x1(m, n, k3, k2, k1, a_stride_m, a_stride_k3, a_stride_k2,
            static_cast<const int8_t*>(a), b_stride_k3, b_stride_k2,
            b_stride_k1, static_cast<const int8_t*>(b), c_in_stride_m,
            static_cast<const int32_t*>(c_in), c_out_stride_m,
            static_cast<int32_t*>(c_out));
}

void dot_uint8_int8_int32_1xNx1_1x1x1(
    size_t m, size_t n, size_t k3, size_t k2, size_t k1, size_t a_stride_m,
    size_t a_stride_k3, size_t a_stride_k2, const void* a, size_t b_stride_k3,
    size_t b_stride_k2, size_t b_stride_k1, const void* b, size_t c_in_stride_m,
    const void* c_in, size_t c_out_stride_m, void* c_out,
    dot_kernel_state* /*state*/) {
  dot_1x1x1(m, n, k3, k2, k1, a_stride_m, a_stride_k3, a_stride_k2,
            static_cast<const uint8_t*>(a), b_stride_k3, b_stride_k2,
            b_stride_k1, static_cast<const int8_t*>(b), c_in_stride_m,
            static_cast<const int32_t*>(c_in), c_out_stride_m,
            static_cast<int32_t*>(c_out));
}

void dot_int8_int4_int32_1xNx2_1x1x2(size_t m, size_t n, size_t k3, size_t k2,
                                     size_t k1, size_t a_stride_m,
                                     size_t a_stride_k3, size_t a_stride_k2,
                                     const void* a, size_t b_stride_k3,
                                     size_t b_stride_k2, size_t b_stride_k1,
                                     const void* b, size_t c_in_stride_m,
                                     const void* c_in, size_t c_out_stride_m,
                                     void* c_out, dot_kernel_state* /*state*/) {
  dot_1x1x2(m, n, k3, k2, k1, a_stride_m, a_stride_k3, a_stride_k2,
            static_cast<const int8_t*>(a), b_stride_k3, b_stride_k2,
            b_stride_k1, static_cast<const int4x2*>(b), c_in_stride_m,
            static_cast<const int32_t*>(c_in), c_out_stride_m,
            static_cast<int32_t*>(c_out));
}

void dot_int8_int2_int32_1xNx4_1x1x4(size_t m, size_t n, size_t k3, size_t k2,
                                     size_t k1, size_t a_stride_m,
                                     size_t a_stride_k3, size_t a_stride_k2,
                                     const void* a, size_t b_stride_k3,
                                     size_t b_stride_k2, size_t b_stride_k1,
                                     const void* b, size_t c_in_stride_m,
                                     const void* c_in, size_t c_out_stride_m,
                                     void* c_out, dot_kernel_state* /*state*/) {
  dot_1x1x4(m, n, k3, k2, k1, a_stride_m, a_stride_k3, a_stride_k2,
            static_cast<const int8_t*>(a), b_stride_k3, b_stride_k2,
            b_stride_k1, static_cast<const int2x4*>(b), c_in_stride_m,
            static_cast<const int32_t*>(c_in), c_out_stride_m,
            static_cast<int32_t*>(c_out));
}

namespace {

// dot_cost_model estimates the cost of a single block of a dot. When processing
// multiple blocks, there is a super-linear cost due to blocks not fitting in
// the cache. This is a continuous, cache-oblivious model of this cost.
YNN_ALWAYS_INLINE float locality_penalty(float blocks_m, float blocks_n) {
  return 1.0f + (fast_log2(blocks_m) + fast_log2(blocks_n)) * (1.0f / 32.0f);
}

YNN_ALWAYS_INLINE float estimate_dot_cost(uint32_t m, uint32_t n, uint32_t k,
                                          uint32_t block_m, uint32_t block_n,
                                          uint32_t block_k,
                                          const dot_cost_model& cost_model) {
  const float block_cost =
      cost_model.estimate_block_cost(k, block_m, block_n, block_k);

  const float blocks_m = ceil_div(m, block_m);
  const float blocks_n = ceil_div(n, block_n);

  return block_cost *
         (blocks_m * blocks_n * locality_penalty(blocks_m, blocks_n));
}

YNN_ALWAYS_INLINE float get_thread_factor(uint64_t arch, int thread_count) {
#ifdef YNN_ARCH_ARM64
  if (arch & (arch_flag::sme | arch_flag::sme2)) {
    // SME units are typically shared among 4 cores. Our cost modeling assumes
    // linear scaling with cores, which is false in the case of SME. This factor
    // accounts for this, by adjusting the per-core performance down.
    return std::min(4, std::max(1, thread_count));
  }
#endif
  return 1.0f;
}

template <typename A, typename B, typename C>
struct optimizer {
  // Inputs
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t required_tile_k;
  uint32_t required_block_n;
  uint32_t required_flags;
  uint32_t disallowed_flags;
  uint64_t supported_arch_flags;
  int thread_count;

  // Outputs
  dot_kernel result;
#if YNN_LOG_LEVEL >= YNN_LOG_LEVEL_DEBUG
  const char* kernel_used = nullptr;
#endif

  YNN_ALWAYS_INLINE void operator()(uint64_t arch, uint32_t block_m,
                                    uint32_t block_n, uint32_t block_k,
                                    uint32_t tile_m, uint32_t tile_n,
                                    uint32_t tile_k, uint32_t flags,
                                    dot_kernel_fn kernel,
                                    const dot_cost_model& cost_model,
                                    const char* name) {
    // These checks are ordered to minimize the cost of these checks:
    // - Checks that are invariant across many kernels should be first.
    // - Checks that are more likely to fail should be first.
    // - Checks that are cheap should be first.
    if (!is_arch_supported(arch, supported_arch_flags)) return;
    if ((required_flags & flags) != required_flags) return;
    if (disallowed_flags & flags) return;
    assert(block_m > 0);
    assert(block_n > 0);
    assert(block_k > 0);
    assert(tile_n > 0);
    assert(tile_k > 0);
    if (required_tile_k && tile_k != required_tile_k) return;
    if ((flags & dot_flag::unaligned_b) == 0 &&
        (required_block_n % tile_n != 0)) {
      return;
    }

#ifdef YNN_ARCH_X86
    if (arch & arch_flag::amxbf16) {
      // The AMX 32x48 kernel (2x3 configuration) currently only works better
      // than 32x32 kernels for small shapes. This may be due to memory
      // bandwidth limitations, as there are only 2 tiles for A/B and must be
      // frequently updated.
      if (block_m == 32 && block_n == 48 && tile_m == 16 && tile_n == 16) {
        if (n > 48) {
          return;
        }
      }
    }
#endif

    // We might use this kernel.
    result.max_block_n = std::max<int>(result.max_block_n, block_n);

    // This is equivalent to estimate_dot_cost, but we evaluate it in two steps,
    // so we can skip some work if part of the cost exceeds the optimal kernel
    // found so far.
    const float block_cost =
        cost_model.estimate_block_cost(k, block_m, block_n, block_k);

    const float blocks_m = ceil_div(m, block_m);
    const float blocks_n = ceil_div(n, block_n);
    const float unpenalized_cost = block_cost * (blocks_m * blocks_n) *
                                   get_thread_factor(arch, thread_count);
    if (unpenalized_cost >= result.cost) return;

    const float dot_cost_k =
        unpenalized_cost * locality_penalty(blocks_m, blocks_n);
    if (dot_cost_k >= result.cost) return;

    result = {
        kernel,
        static_cast<int>(block_m),
        static_cast<int>(block_n),
        static_cast<int>(block_k),
        static_cast<int>(tile_m),
        static_cast<int>(tile_n),
        static_cast<int>(tile_k),
        flags,
        &cost_model,
        dot_cost_k,
        static_cast<int>(result.max_block_n),
    };
#if YNN_LOG_LEVEL >= YNN_LOG_LEVEL_DEBUG
    kernel_used = name;
#endif
  }
};

YNN_UNUSED logger& operator<<(logger& os, std::optional<size_t> v) {
  return v ? os << *v : os << "?";
}
YNN_UNUSED null_logger& operator<<(null_logger& os, std::optional<size_t> v) {
  return v ? os << *v : os << "?";
}

template <typename A, typename B, typename C>
dot_kernel get_dot_kernel(const dot_cost_models& cost_models,
                          const dot_shape& shape, dot_packed_shape packed_shape,
                          uint32_t required_flags,
                          std::optional<bool> transpose_a, uint64_t arch_flags,
                          int thread_count) {
  if (packed_shape.tile_k == 0 && packed_shape.block_n == 0) {
    YNN_LOG_DEBUG() << "Selecting kernel for dot " << shape.m << "x" << shape.n
                    << "x" << shape.k1;
  }

  uint32_t strictly_required_flags = required_flags;
  uint32_t disallowed_flags = 0;
  if (required_flags & dot_flag::symmetric_b) {
    // We don't require the kernel to be symmetric_b, a non-symmetric_b kernel
    // might still be faster.
    strictly_required_flags &= ~dot_flag::symmetric_b;
  } else {
    // Don't use a symmetric_b kernel if the caller did not indicate that the
    // data is symmetric_b.
    disallowed_flags |= dot_flag::symmetric_b;
  }

  if (transpose_a.has_value()) {
    // We need the kernel to match the requested transpose_a.
    if (*transpose_a) {
      strictly_required_flags |= dot_flag::transpose_a;
    } else {
      disallowed_flags |= dot_flag::transpose_a;
    }
  }

  optimizer<A, B, C> optimizer{
      // These casts might saturate a size_t value, which should be OK, because
      // if m, n, k are that large, any tail cases will be negligible. Cast to
      // uint16 so we have some headroom for arithmetic.
      cast<uint16_t>(shape.m),
      cast<uint16_t>(shape.n),
      cast<uint16_t>(shape.k1),
      static_cast<uint32_t>(packed_shape.tile_k),
      static_cast<uint32_t>(packed_shape.block_n),
      strictly_required_flags,
      disallowed_flags,
      arch_flags,
      thread_count,
  };

// TODO: Limit this to only a subset of the "prod" kernels.
#define YNN_DOT_KERNEL(arch, name, block_m, block_n, block_k, tile_m, tile_n, \
                       tile_k, flags, a_type, b_type, c_type, cost_model)     \
  if constexpr (std::is_same_v<A, a_type> && std::is_same_v<B, b_type> &&     \
                std::is_same_v<C, c_type>) {                                  \
    optimizer(arch, block_m, block_n, block_k, tile_m, tile_n, tile_k, flags, \
              name, cost_model, #name);                                       \
  }
#include "ynnpack/kernels/dot/kernels.inc"
#undef YNN_DOT_KERNEL

#if YNN_LOG_LEVEL >= YNN_LOG_LEVEL_DEBUG
  if (packed_shape.tile_k == 0 && packed_shape.block_n == 0) {
    if (optimizer.result.kernel) {
      YNN_LOG_DEBUG() << "Using dot kernel " << optimizer.kernel_used
                      << " for dot " << shape.m << "x" << shape.n << "x"
                      << shape.k1;
    }
  }
#endif
  return optimizer.result;
}

// Moving this to a separate (non-inlined) function cleans up the stack a bit.
YNN_NO_INLINE dot_kernel get_unsupported_dot_kernel(const dot_type& type) {
  YNN_LOG_ERROR() << "Unsupported dot type " << type.a << "_" << type.b << "_"
                  << type.c;
  return {};
}

constexpr uint32_t dot_type_id(ynn_type a, ynn_type b, ynn_type c) {
  return (static_cast<uint32_t>(a) << 16) | (static_cast<uint32_t>(b) << 8) |
         static_cast<uint32_t>(c);
}

template <typename A, typename B, typename C>
constexpr uint32_t dot_type_id() {
  return dot_type_id(type_of<A>(), type_of<B>(), type_of<C>());
}

}  // namespace

dot_kernel get_dot_kernel(const dot_type& type,
                          const dot_cost_models& cost_models,
                          const dot_shape& shape, dot_packed_shape packed_shape,
                          uint32_t required_flags,
                          std::optional<bool> transpose_a, uint64_t arch_flags,
                          int thread_count) {
#define GET_DOT_KERNEL_CASE(a, b, c)                                        \
  case dot_type_id<a, b, c>():                                              \
    return get_dot_kernel<a, b, c>(cost_models, shape, packed_shape,        \
                                   required_flags, transpose_a, arch_flags, \
                                   thread_count);
  switch (dot_type_id(type.a, type.b, type.c)) {
    GET_DOT_KERNEL_CASE(double, double, double);
    GET_DOT_KERNEL_CASE(float, float, float);
    GET_DOT_KERNEL_CASE(half, half, float);
    GET_DOT_KERNEL_CASE(bfloat16, bfloat16, float);
    GET_DOT_KERNEL_CASE(int8_t, int8_t, int32_t);
    GET_DOT_KERNEL_CASE(uint8_t, int8_t, int32_t);
    GET_DOT_KERNEL_CASE(int8_t, int2x4, int32_t);
    GET_DOT_KERNEL_CASE(uint8_t, int2x4, int32_t);
    GET_DOT_KERNEL_CASE(int8_t, int4x2, int32_t);
    GET_DOT_KERNEL_CASE(uint8_t, int4x2, int32_t);
    GET_DOT_KERNEL_CASE(fp8_e5m2, fp8_e5m2, float);
    GET_DOT_KERNEL_CASE(fp8_e4m3, fp8_e4m3, float);
    default:
      return get_unsupported_dot_kernel(type);
  }
}

float dot_kernel::estimate_cost(size_t m, size_t n, size_t k) const {
  assert(cost_model);
  // Cast to uint16_t, because we need a bit of headroom to do arithmetic.
  return estimate_dot_cost(cast<uint16_t>(m), cast<uint16_t>(n),
                           cast<uint16_t>(k), block_m, block_n, block_k,
                           *cost_model);
}

const dot_cost_models& get_dot_cost_models() {
#ifdef YNN_ENABLE_CPUINFO
  if (cpuinfo_initialize()) {
    uint32_t uarch_index = cpuinfo_get_current_uarch_index_with_default(0);
    const cpuinfo_uarch_info* uarch = cpuinfo_get_uarch(uarch_index);
    assert(uarch);
    // TODO: b/549305639 - This is probably overkill, many of the uarchs we have
    // separate models for are probably very similar, and should be combined
    // into one model.
    switch (uarch->uarch) {
#ifdef YNN_ARCH_X86
      case cpuinfo_uarch_haswell:
      case cpuinfo_uarch_sandy_bridge:
      case cpuinfo_uarch_ivy_bridge:
        return haswell;
      case cpuinfo_uarch_broadwell:
        return broadwell;
      case cpuinfo_uarch_sky_lake:
      case cpuinfo_uarch_palm_cove:
      case cpuinfo_uarch_sunny_cove:
      case cpuinfo_uarch_willow_cove:
        return cascade_lake;
      case cpuinfo_uarch_golden_cove:
      case cpuinfo_uarch_raptor_cove:
        return golden_cove;
      case cpuinfo_uarch_redwood_cove:
      case cpuinfo_uarch_coyote_cove:
        return redwood_cove;
      case cpuinfo_uarch_zen:
      case cpuinfo_uarch_zen2:
        return zen2;
      case cpuinfo_uarch_zen3:
        return zen3;
      case cpuinfo_uarch_zen4:
      case cpuinfo_uarch_zen5:
      case cpuinfo_uarch_zen6:
        return zen4;
#endif  // YNN_ARCH_X86
#ifdef YNN_ARCH_ARM
      case cpuinfo_uarch_cortex_a53:
      case cpuinfo_uarch_cortex_a55:
      case cpuinfo_uarch_cortex_a55r0:
      case cpuinfo_uarch_cortex_a35:
      case cpuinfo_uarch_cortex_a32:
      case cpuinfo_uarch_cortex_a510:
        return cortex_a510;
      case cpuinfo_uarch_cortex_a520:
      case cpuinfo_uarch_cortex_a320:
        return cortex_a520;
      case cpuinfo_uarch_cortex_a78:
      case cpuinfo_uarch_cortex_a77:
      case cpuinfo_uarch_cortex_a76:
      case cpuinfo_uarch_cortex_a75:
      case cpuinfo_uarch_cortex_a73:
      case cpuinfo_uarch_cortex_a72:
      case cpuinfo_uarch_cortex_a57:
      case cpuinfo_uarch_cortex_a710:
        return cortex_a710;
      case cpuinfo_uarch_cortex_a715:
        return cortex_a715;
      case cpuinfo_uarch_cortex_a720:
        return cortex_a720;
      case cpuinfo_uarch_cortex_a725:
        return cortex_a725;
      case cpuinfo_uarch_cortex_x1:
        return cortex_x1;
      case cpuinfo_uarch_cortex_x2:
      case cpuinfo_uarch_cortex_x3:
      case cpuinfo_uarch_cortex_x4:
      case cpuinfo_uarch_cortex_x925:
        return cortex_x4;
      case cpuinfo_uarch_lumex_c1_pro:
      case cpuinfo_uarch_lumex_c1_nano:
        return lumex_c1_pro;
      case cpuinfo_uarch_lumex_c1_ultra:
      case cpuinfo_uarch_lumex_c1_premium:
        return lumex_c1_ultra;
      case cpuinfo_uarch_neoverse_n1:
      case cpuinfo_uarch_neoverse_e1:
        return neoverse_n1;
      case cpuinfo_uarch_neoverse_v1:
      case cpuinfo_uarch_neoverse_n2:
      case cpuinfo_uarch_neoverse_v2:
        return neoverse_v2;
      case cpuinfo_uarch_oryon:
      case cpuinfo_uarch_oryon_v3:
        return oryon;
      case cpuinfo_uarch_swift:
      case cpuinfo_uarch_cyclone:
      case cpuinfo_uarch_typhoon:
      case cpuinfo_uarch_twister:
      case cpuinfo_uarch_hurricane:
      case cpuinfo_uarch_monsoon:
      case cpuinfo_uarch_vortex:
      case cpuinfo_uarch_lightning:
      case cpuinfo_uarch_firestorm:
      case cpuinfo_uarch_avalanche:
      case cpuinfo_uarch_everest:
      case cpuinfo_uarch_coll_everest:
      case cpuinfo_uarch_tupai_everest:
      case cpuinfo_uarch_tahiti_everest:
      case cpuinfo_uarch_tilos_everest:
      case cpuinfo_uarch_donan_everest:
      case cpuinfo_uarch_sotra_super:
      case cpuinfo_uarch_sotra_performance:
        return donan_everest;
      case cpuinfo_uarch_mistral:
      case cpuinfo_uarch_tempest:
      case cpuinfo_uarch_thunder:
      case cpuinfo_uarch_icestorm:
      case cpuinfo_uarch_blizzard:
      case cpuinfo_uarch_sawtooth:
      case cpuinfo_uarch_coll_sawtooth:
      case cpuinfo_uarch_tupai_sawtooth:
      case cpuinfo_uarch_tahiti_sawtooth:
      case cpuinfo_uarch_tilos_sawtooth:
      case cpuinfo_uarch_donan_sawtooth:
        return donan_sawtooth;
#endif  // YNN_ARCH_ARM
      default:
        break;
    }
  }
#endif  // YNN_ENABLE_CPUINFO

  // We don't know what the CPU is. Use a very high end CPU, so we have coverage
  // of the advanced instruction sets.
#if defined(YNN_ARCH_ARM64)
  return donan_everest;
#elif defined(YNN_ARCH_ARM)
  return cortex_a510;
#elif defined(YNN_ARCH_X86)
  return redwood_cove;
#else
  static constexpr dot_cost_models default_models = {};
  return default_models;
#endif
}

}  // namespace ynn
