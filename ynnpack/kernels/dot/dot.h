// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_DOT_DOT_H_
#define XNNPACK_YNNPACK_KERNELS_DOT_DOT_H_

#include <cstddef>
#include <cstdint>  // IWYU pragma: keep
#include <limits>
#include <optional>

#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/dot/arm64_sme.h"  // IWYU pragma: keep

namespace ynn {

// `enum class` doesn't work well for bitfield values.
namespace dot_flag {

enum {
  // The `a` parameter of the dot must be transposed from [i, k3, k2, k1]
  // to [k1 / tile_k, k3, k2, {i, tile_k}], where:
  // - The {i, tile_k} dimension is dense (stride of 1 element).
  // - `a_stride_m` indicates the stride of the k1 / tile_k dimension.
  transpose_a = 1 << 0,

  // This kernel produces results that are numerically consistent with all other
  // kernels of the same type with this flag. For the most part,
  // fp32 `tile_k = 1` kernels to be numerically consistent, and bf16/fp16
  // `tile_k = 2` kernels to be numerically consistent for bf16 and fp16.
  consistent_arithmetic = 1 << 1,

  // This kernel supports an unaligned B
  unaligned_b = 1 << 2,
};

}  // namespace dot_flag

// Dot kernels compute the following:
//
//    C_out(i, j) = 0
//    C_out(i, j) += A(i, k3, k2, k1) * B(k3, k2, k1, j)
//    C_out(i, j) += C_in(i, j)
//
// for all i, j, k3, k2, k1
typedef void (*dot_kernel_fn)(size_t m, size_t n, size_t k3, size_t k2,
                              size_t k1, size_t a_stride_m, size_t a_stride_k3,
                              size_t a_stride_k2, const void* a,
                              size_t b_stride_k3, size_t b_stride_k2,
                              size_t b_stride_k1, const void* b,
                              size_t c_in_stride_m, const void* c_in,
                              size_t c_out_stride_m, void* c_out);

#define YNN_DOT_KERNEL(arch, name, block_m, block_n, block_k, tile_m, tile_n, \
                       tile_k, transpose_a, type_a, type_b, type_c)           \
  void name(size_t m, size_t n, size_t k3, size_t k2, size_t k1,              \
            size_t a_stride_m, size_t a_stride_k3, size_t a_stride_k2,        \
            const void* a, size_t b_stride_k3, size_t b_stride_k2,            \
            size_t b_stride_k1, const void* b, size_t c_in_stride_m,          \
            const void* c_in, size_t c_out_stride_m, void* c_out);
#include "ynnpack/kernels/dot/kernels.inc"
#undef YNN_DOT_KERNEL

struct dot_type {
  ynn_type a;
  ynn_type b;
  ynn_type c;
};

// A dot kernel is a function pointer, along with information about the block
// shape.
struct dot_kernel {
  dot_kernel_fn kernel = nullptr;
  // Dot kernels have two shapes that callers must be aware of:
  // - The "tile shape", which is the minimal element of work that the kernel
  // can compute. Work that is not aligned to this shape will be padded up to
  // this shape.
  // - The "block shape", which is an unrolling in all 3 dimensions of the tile
  // shape.
  //
  // Key impacts of this on calling code of these kernels:
  // - `m` must not be larger than `block_m`. Call the kernel in a loop to
  // handle this case.
  // - `n` can be anything, but performance may be sub-optimal if not aligned to
  // a multiple of `block_n` and/or `tile_n`.
  // - `tile_k` values of B must be contiguous in memory, i.e. `tile_k = K`
  // indicates that `K` rows of B should be interleaved at a time, such that
  // values from `K` rows of B are adjacent in memory.
  // - Kernels often assume that the memory of B is aligned such that the each
  // tile beings on a memory address aligned to the size of the tile.
  int block_m = 0;
  int block_n = 0;
  int block_k = 0;
  int tile_n = 0;
  int tile_k = 0;
  uint32_t flags = 0;
  float cost = std::numeric_limits<float>::infinity();
};

struct dot_shape {
  std::optional<size_t> m;
  std::optional<size_t> n;
  std::optional<size_t> k1;
  std::optional<size_t> k2;
  std::optional<size_t> k3;
};

// Compute an estimate of the cost of a dot operation. This number has no
// absolute meaning, it is only comparable to other return values of this
// function.
float estimate_dot_cost(size_t m, size_t n, size_t k, size_t block_m,
                        size_t block_n, size_t block_k, size_t tile_m,
                        size_t tile_n, size_t tile_k);

struct dot_packed_shape {
  int block_n = 0;
  int tile_k = 0;
};

// Find a dot kernel to use for the given `shape`. If not null, the chosen
// kernel will have the same `tile_n` and `tile_k` as `compatible_with` (i.e.
// both kernels can use the same packed data.). Similarly, if `transpose_a` is
// not `nullopt`, the chosen kernel will have the flag `dot_flag::transpose_a`
// if *transpose_a is true.
dot_kernel get_dot_kernel(const dot_type& type, const dot_shape& shape = {},
                          const dot_packed_shape* dot_packed_shape = nullptr,
                          bool consistent_arithmetic = false,
                          std::optional<bool> transpose_a = std::nullopt,
                          uint64_t arch_flags = get_supported_arch_flags());

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_DOT_DOT_H_
