// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_DEQUANTIZE_DOT_DEQUANTIZE_DOT_H_
#define XNNPACK_YNNPACK_KERNELS_DEQUANTIZE_DOT_DEQUANTIZE_DOT_H_

#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

struct dequantize_dot_params {};

inline bool operator==(const dequantize_dot_params& a,
                       const dequantize_dot_params& b) {
  return true;
}
inline bool operator<(const dequantize_dot_params& a,
                      const dequantize_dot_params& b) {
  return false;
}

// This kernel is an elementwise op that computes:
//
//   dot' = dot - a_offset * b_offset
//   output = cast<Output>(cast<float>(dot') * a_scale * b_scale + offset)
//
// where:
//   a_offset, a_scale are column vectors (or scalars)
//   b_offset, b_scale, offset are row vectors (or scalars)
typedef void (*dequantize_dot_kernel_fn)(
    size_t m, size_t n, size_t stride_dot_m, const void* dot,
    size_t stride_a_offset_m, const void* a_offset, size_t stride_b_offset_n,
    const void* b_offset, size_t stride_offset_n, const void* offset,
    size_t stride_a_scale_m, const void* a_scale, size_t stride_b_scale_n,
    const void* b_scale, size_t stride_output_m, void* output,
    const dequantize_dot_params* params);

#define YNN_DEQUANTIZE_DOT_KERNEL(arch, name, type)                           \
  void name(                                                                  \
      size_t m, size_t n, size_t stride_dot_m, const void* dot,               \
      size_t stride_a_offset_m, const void* a_offset,                         \
      size_t stride_b_offset_n, const void* b_offset, size_t stride_offset_n, \
      const void* offset, size_t stride_a_scale_m, const void* a_scale,       \
      size_t stride_b_scale_n, const void* b_scale, size_t stride_output_m,   \
      void* output, const dequantize_dot_params* params);
#include "ynnpack/kernels/dequantize_dot/kernels.inc"
#undef YNN_DEQUANTIZE_DOT_KERNEL

dequantize_dot_kernel_fn get_dequantize_dot_kernel(
    ynn_type type, uint64_t arch_flags = get_supported_arch_flags());

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_DEQUANTIZE_DOT_DEQUANTIZE_DOT_H_
