// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/dot/dot.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <type_traits>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

namespace {

template <typename AT, typename BT, typename CT>
void dot(size_t M, size_t N, size_t K3, size_t K2, size_t K1, size_t A_stride_m,
         size_t A_stride_k3, size_t A_stride_k2, const AT* A,
         size_t B_stride_k3, size_t B_stride_k2, size_t B_stride_k1,
         const BT* B, size_t C_in_stride_m, const CT* C_in,
         size_t C_out_stride_m, CT* C_out) {
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

// Unfortunately the compiler doesn't see that it should unroll the j loop by 2
// for `type_info::element_count() == 2`, if we do it manually we get a
// 5x speedup from this code.
template <typename AT, typename BT, typename CT>
void dot_unroll2(size_t M, size_t N, size_t K3, size_t K2, size_t K1,
                 size_t A_stride_m, size_t A_stride_k3, size_t A_stride_k2,
                 const AT* A, size_t B_stride_k3, size_t B_stride_k2,
                 size_t B_stride_k1, const BT* B, size_t C_in_stride_m,
                 const CT* C_in, size_t C_out_stride_m, CT* C_out) {
  using B_info = type_info<BT>;
  assert(M == 1);
  assert(N % 2 == 0);
  CT* acc = YNN_ALLOCA(CT, N);
  std::fill_n(acc, N, 0);
  for (size_t k3 = 0; k3 < K3; ++k3) {
    const BT* B_k3 = offset_bytes(B, k3 * B_stride_k3);
    const AT* A_k3 = offset_bytes(A, k3 * A_stride_k3);
    for (size_t k2 = 0; k2 < K2; ++k2) {
      const BT* B_k2 = offset_bytes(B_k3, k2 * B_stride_k2);
      const AT* A_k2 = offset_bytes(A_k3, k2 * A_stride_k2);
      for (size_t k1 = 0; k1 < K1; ++k1) {
        const BT* B_k1 = offset_bytes(B_k2, k1 * B_stride_k1);
        const AT A_k1 = A_k2[k1];
        for (size_t j = 0; j < N; j += 2) {
          acc[j + 0] +=
              static_cast<CT>(A_k1) * static_cast<CT>(B_info::get(B_k1, j + 0));
          acc[j + 1] +=
              static_cast<CT>(A_k1) * static_cast<CT>(B_info::get(B_k1, j + 1));
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

void dot_fp32(size_t m, size_t n, size_t k3, size_t k2, size_t k1,
              size_t a_stride_m, size_t a_stride_k3, size_t a_stride_k2,
              const void* a, size_t b_stride_k3, size_t b_stride_k2,
              size_t b_stride_k1, const void* b, size_t c_in_stride_m,
              const void* c_in, size_t c_out_stride_m, void* c_out) {
  dot(m, n, k3, k2, k1, a_stride_m, a_stride_k3, a_stride_k2,
      static_cast<const float*>(a), b_stride_k3, b_stride_k2, b_stride_k1,
      static_cast<const float*>(b), c_in_stride_m,
      static_cast<const float*>(c_in), c_out_stride_m,
      static_cast<float*>(c_out));
}

void dot_fp16_fp16_fp32(size_t m, size_t n, size_t k3, size_t k2, size_t k1,
                        size_t a_stride_m, size_t a_stride_k3,
                        size_t a_stride_k2, const void* a, size_t b_stride_k3,
                        size_t b_stride_k2, size_t b_stride_k1, const void* b,
                        size_t c_in_stride_m, const void* c_in,
                        size_t c_out_stride_m, void* c_out) {
  dot(m, n, k3, k2, k1, a_stride_m, a_stride_k3, a_stride_k2,
      static_cast<const half*>(a), b_stride_k3, b_stride_k2, b_stride_k1,
      static_cast<const half*>(b), c_in_stride_m,
      static_cast<const float*>(c_in), c_out_stride_m,
      static_cast<float*>(c_out));
}

void dot_bf16_bf16_fp32(size_t m, size_t n, size_t k3, size_t k2, size_t k1,
                        size_t a_stride_m, size_t a_stride_k3,
                        size_t a_stride_k2, const void* a, size_t b_stride_k3,
                        size_t b_stride_k2, size_t b_stride_k1, const void* b,
                        size_t c_in_stride_m, const void* c_in,
                        size_t c_out_stride_m, void* c_out) {
  dot(m, n, k3, k2, k1, a_stride_m, a_stride_k3, a_stride_k2,
      static_cast<const bfloat16*>(a), b_stride_k3, b_stride_k2, b_stride_k1,
      static_cast<const bfloat16*>(b), c_in_stride_m,
      static_cast<const float*>(c_in), c_out_stride_m,
      static_cast<float*>(c_out));
}

void dot_int8_int8_int32(size_t m, size_t n, size_t k3, size_t k2, size_t k1,
                         size_t a_stride_m, size_t a_stride_k3,
                         size_t a_stride_k2, const void* a, size_t b_stride_k3,
                         size_t b_stride_k2, size_t b_stride_k1, const void* b,
                         size_t c_in_stride_m, const void* c_in,
                         size_t c_out_stride_m, void* c_out) {
  dot(m, n, k3, k2, k1, a_stride_m, a_stride_k3, a_stride_k2,
      static_cast<const int8_t*>(a), b_stride_k3, b_stride_k2, b_stride_k1,
      static_cast<const int8_t*>(b), c_in_stride_m,
      static_cast<const int32_t*>(c_in), c_out_stride_m,
      static_cast<int32_t*>(c_out));
}

void dot_uint8_int8_int32(size_t m, size_t n, size_t k3, size_t k2, size_t k1,
                          size_t a_stride_m, size_t a_stride_k3,
                          size_t a_stride_k2, const void* a, size_t b_stride_k3,
                          size_t b_stride_k2, size_t b_stride_k1, const void* b,
                          size_t c_in_stride_m, const void* c_in,
                          size_t c_out_stride_m, void* c_out) {
  dot(m, n, k3, k2, k1, a_stride_m, a_stride_k3, a_stride_k2,
      static_cast<const uint8_t*>(a), b_stride_k3, b_stride_k2, b_stride_k1,
      static_cast<const int8_t*>(b), c_in_stride_m,
      static_cast<const int32_t*>(c_in), c_out_stride_m,
      static_cast<int32_t*>(c_out));
}

void dot_int8_int4_int32(size_t m, size_t n, size_t k3, size_t k2, size_t k1,
                         size_t a_stride_m, size_t a_stride_k3,
                         size_t a_stride_k2, const void* a, size_t b_stride_k3,
                         size_t b_stride_k2, size_t b_stride_k1, const void* b,
                         size_t c_in_stride_m, const void* c_in,
                         size_t c_out_stride_m, void* c_out) {
  dot_unroll2(m, n, k3, k2, k1, a_stride_m, a_stride_k3, a_stride_k2,
              static_cast<const int8_t*>(a), b_stride_k3, b_stride_k2,
              b_stride_k1, static_cast<const int4x2*>(b), c_in_stride_m,
              static_cast<const int32_t*>(c_in), c_out_stride_m,
              static_cast<int32_t*>(c_out));
}

float estimate_dot_cost(size_t m, size_t n, size_t k, size_t block_m,
                        size_t block_n, size_t block_k, size_t tile_m,
                        size_t tile_n, size_t tile_k) {
  const float blocks_m = ceil_div(m, block_m);
  const float blocks_n = ceil_div(n, block_n);
  const float blocks_k = ceil_div(k, block_k);

  // This model was derived by benchmarking each kernel on a [block_m, block_k *
  // 64] . [block_k * 64, block_n] dot operation (64 blocks should fit in L1
  // cache), and then fitting a linear model to a set of features (block/tile
  // dimensions, memory loaded, etc.). It turned out that the only features the
  // model really depends on is the number of loads it does, and that loads of b
  // are ~2x as expensive as loads from a.
  // TODO(dsharlet): This has been tested on Intel Skylake and AMD Rome, but not
  // ARM.
  const size_t loads_a = block_m * block_k / (tile_m * tile_k);
  const size_t loads_b = block_n * block_k / (tile_n * tile_k);
  const float block_cost = loads_a * 5 + loads_b * 11 + 9;

  return blocks_m * blocks_n * blocks_k * block_cost;
}

namespace {

// If we don't know the shape of a dot, just assume it's big.
constexpr size_t unknown_dot_extent = 2048;

// An additional penalty scale term on the cost of a dot kernel based on the
// architecture.
float dot_arch_cost_factor(uint64_t arch) {
  if (arch == arch_flag::none) {
    // We should only use the default dot kernel if there is no other choice.
    return 100.0f;
  } else {
    return 1.0f;
  }
}

template <typename A, typename B, typename C>
struct optimizer {
  // Inputs
  size_t m;
  size_t n;
  size_t k;
  int required_tile_k;
  int required_block_n;
  std::optional<bool> transpose_a;
  uint64_t supported_arch_flags;

  // Outputs
  dot_kernel result;
  float dot_cost = std::numeric_limits<float>::infinity();
  const char* kernel_used = nullptr;

  void operator()(uint64_t arch, int block_m, int block_n, int block_k,
                  int tile_m, int tile_n, int tile_k, uint32_t flags,
                  dot_kernel_fn kernel, const char* name) {
    if (transpose_a && *transpose_a != ((flags & dot_flag::transpose_a) != 0)) {
      // The caller wants a transposed (or not), and this kernel is not
      // transposed (or is).
      return;
    }
    if (!is_arch_supported(arch, supported_arch_flags)) {
      return;
    }
    if ((required_tile_k && tile_k != required_tile_k) ||
        (required_block_n % tile_n != 0)) {
      // We wanted a kernel compatible with `packed_shape`, but this kernel is
      // not.
      return;
    }
    const float dot_cost_k =
        estimate_dot_cost(m, n, k, block_m, block_n, block_k, tile_m, tile_n,
                          tile_k) *
        dot_arch_cost_factor(arch);
    if (!required_tile_k && !required_block_n) {
      char selected = dot_cost_k < dot_cost ? '*' : ' ';
      YNN_LOG_DEBUG() << " " << selected << name << " cost=" << dot_cost_k;
    }
    if (dot_cost_k >= dot_cost) {
      return;
    }
    result = {kernel, block_m, block_n, block_k, tile_n, tile_k, flags};
    kernel_used = name;
    dot_cost = dot_cost_k;
  }
};

YNN_UNUSED logger& operator<<(logger& os, std::optional<size_t> v) {
  return v ? os << *v : os << "?";
}
YNN_UNUSED null_logger& operator<<(null_logger& os, std::optional<size_t> v) {
  return v ? os << *v : os << "?";
}

template <typename A, typename B, typename C>
dot_kernel get_dot_kernel(const dot_shape& shape,
                          const dot_packed_shape* packed_shape,
                          std::optional<bool> transpose_a,
                          uint64_t arch_flags) {
  if (!packed_shape) {
    YNN_LOG_DEBUG() << "Selecting kernel for dot " << shape.m << "x" << shape.n
                    << "x" << shape.k1;
  }
  optimizer<A, B, C> optimizer{
      shape.m.value_or(unknown_dot_extent),
      shape.n.value_or(unknown_dot_extent),
      shape.k1.value_or(unknown_dot_extent) * shape.k2.value_or(1) *
          shape.k3.value_or(1),
      packed_shape ? packed_shape->tile_k : 0,
      packed_shape ? packed_shape->block_n : 0,
      transpose_a,
      arch_flags,
  };
  // TODO(dsharlet): If we ever have any tile_m != 1 kernels, we need to plumb
  // it out and handle it.
  constexpr int tile_m = 1;
// TODO: Limit this to only a subset of the "prod" kernels.
#define YNN_DOT_KERNEL(arch, name, block_m, block_n, block_k, tile_n, tile_k, \
                       flags, a_type, b_type, c_type)                         \
  if (std::is_same<A, a_type>::value && std::is_same<B, b_type>::value &&     \
      std::is_same<C, c_type>::value) {                                       \
    optimizer(arch, block_m, block_n, block_k, tile_m, tile_n, tile_k, flags, \
              name, #name);                                                   \
  }
#include "ynnpack/kernels/dot/kernels.inc"
#undef YNN_DOT_KERNEL
  if (!packed_shape) {
    YNN_LOG_INFO() << "Using dot kernel " << optimizer.kernel_used
                   << " for dot " << shape.m << "x" << shape.n << "x"
                   << shape.k1;
  }
  return optimizer.result;
}

}  // namespace

dot_kernel get_dot_kernel(const dot_type& type, const dot_shape& shape,
                          const dot_packed_shape* packed_shape,
                          std::optional<bool> transpose_a,
                          uint64_t arch_flags) {
  if (type.a == ynn_type_fp32 && type.b == ynn_type_fp32 &&
      type.c == ynn_type_fp32) {
    return get_dot_kernel<float, float, float>(shape, packed_shape, transpose_a,
                                               arch_flags);
  } else if (type.a == ynn_type_fp16 && type.b == ynn_type_fp16 &&
             type.c == ynn_type_fp32) {
    return get_dot_kernel<half, half, float>(shape, packed_shape, transpose_a,
                                             arch_flags);
  } else if (type.a == ynn_type_bf16 && type.b == ynn_type_bf16 &&
             type.c == ynn_type_fp32) {
    return get_dot_kernel<bfloat16, bfloat16, float>(shape, packed_shape,
                                                     transpose_a, arch_flags);
  } else if (type.a == ynn_type_int8 && type.b == ynn_type_int8 &&
             type.c == ynn_type_int32) {
    return get_dot_kernel<int8_t, int8_t, int32_t>(shape, packed_shape,
                                                   transpose_a, arch_flags);
  } else if (type.a == ynn_type_int8 && type.b == ynn_type_int4 &&
             type.c == ynn_type_int32) {
    return get_dot_kernel<uint8_t, int4x2, int32_t>(shape, packed_shape,
                                                    transpose_a, arch_flags);
  } else if (type.a == ynn_type_uint8 && type.b == ynn_type_int8 &&
             type.c == ynn_type_int32) {
    return get_dot_kernel<uint8_t, int8_t, int32_t>(shape, packed_shape,
                                                    transpose_a, arch_flags);
  } else {
    YNN_LOG_ERROR() << "Unsupported dot type " << type.a << "_" << type.b << "_"
                    << type.c;
    return {};
  }
}

}  // namespace ynn
