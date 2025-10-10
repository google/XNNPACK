// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_UNARY_H_
#define XNNPACK_YNNPACK_KERNELS_UNARY_H_

#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

// Elementwise ops are defined as x, y, ... = f(a, b, ...)
struct unary_reference_params {
  float a_scale;
  int32_t a_zero_point;
  float inv_x_scale;
  int32_t x_zero_point;
};

union unary_params {
  unary_reference_params reference;
};

typedef void (*unary_kernel_fn)(size_t width, size_t height, size_t stride_a,
                                const void* a, size_t stride_x, void* x,
                                const unary_params* params);

typedef void (*init_unary_params_fn)(float a_scale, int32_t a_zero_point,
                                     float x_scale, int32_t x_zero_point,
                                     unary_params& params);

#define YNN_ELEMENTWISE_KERNEL(arch, name, op, init_params_fn, type_a, type_c) \
  void name(size_t m, size_t n, size_t stride_a_m, const void* a,              \
            size_t stride_x_m, void* x, const unary_params* params);
#include "ynnpack/kernels/unary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

struct unary_kernel {
  unary_kernel_fn op;
  init_unary_params_fn init_params;
};

const unary_kernel* get_unary_reference_kernel(ynn_unary_operator op,
                                               ynn_type a_type,
                                               bool a_quantized,
                                               ynn_type x_type,
                                               bool x_quantized);

template <typename A, typename X>
const unary_kernel* get_unary_reference_kernel(ynn_unary_operator op, A, X) {
  return get_unary_reference_kernel(op, type_of<A>(), is_quantized<A>{},
                                    type_of<X>(), is_quantized<X>{});
}

const unary_kernel* get_unary_kernel(
    ynn_unary_operator op, ynn_type a_type, bool a_quantized, ynn_type x_type,
    bool x_quantized,
    uint64_t supported_arch_flags = get_supported_arch_flags());

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_UNARY_H_
