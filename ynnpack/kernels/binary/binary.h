// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_BINARY_H_
#define XNNPACK_YNNPACK_KERNELS_BINARY_H_

#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

// Elementwise ops are defined as x, y, ... = f(a, b, ...)
struct binary_reference_params {
  float a_scale;
  int32_t a_zero_point;
  float b_scale;
  int32_t b_zero_point;
  float inv_x_scale;
  int32_t x_zero_point;
};

union binary_params {
  binary_reference_params reference;
};

// The stride of dimension `n` for any operand must be 0 or the size of one
// element.
typedef void (*binary_kernel_fn)(size_t m, size_t n, size_t stride_a_m,
                                 size_t stride_a_n, const void* a,
                                 size_t stride_b_m, size_t stride_b_n,
                                 const void* b, size_t stride_x_m, void* x,
                                 const binary_params* params);
typedef void (*init_binary_params_fn)(float a_scale, int32_t a_zero_point,
                                      float b_scale, int32_t b_zero_point,
                                      float x_scale, int32_t x_zero_point,
                                      binary_params& params);

#define YNN_ELEMENTWISE_KERNEL(arch, name, op, init_params_fn, type_a, type_b, \
                               type_c)                                         \
  void name(size_t m, size_t n, size_t stride_a_m, size_t stride_a_n,          \
            const void* a, size_t stride_b_m, size_t stride_b_n,               \
            const void* b, size_t stride_x_m, void* x,                         \
            const binary_params* params);
#include "ynnpack/kernels/binary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

struct binary_kernel {
  binary_kernel_fn op;
  init_binary_params_fn init_params;
};

const binary_kernel* get_binary_reference_kernel(ynn_binary_operator op,
                                                 ynn_type input_type,
                                                 bool quantized);

template <typename T>
const binary_kernel* get_binary_reference_kernel(ynn_binary_operator op, T) {
  return get_binary_reference_kernel(op, type_of<T>(), is_quantized<T>{});
}

const binary_kernel* get_binary_kernel(
    ynn_binary_operator op, ynn_type input_type, bool quantized,
    uint64_t supported_arch_flags = get_supported_arch_flags());

binary_kernel_fn get_binary_multiply_kernel(ynn_type type_a, ynn_type type_b,
                                            ynn_type type_x);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_BINARY_H_
