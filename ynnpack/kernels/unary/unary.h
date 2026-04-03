// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_UNARY_H_
#define XNNPACK_YNNPACK_KERNELS_UNARY_H_

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

struct exp_params {
  float input_multiplier;

  friend bool operator==(const exp_params& a, const exp_params& b) {
    return a.input_multiplier == b.input_multiplier;
  }
  friend bool operator<(const exp_params& a, const exp_params& b) {
    return a.input_multiplier < b.input_multiplier;
  }
};

struct erf_params {
  float input_multiplier;
  float output_multiplier;
  float output_offset;

  friend bool operator==(const erf_params& a, const erf_params& b) {
    return std::tie(a.input_multiplier, a.output_multiplier, a.output_offset) ==
           std::tie(b.input_multiplier, b.output_multiplier, b.output_offset);
  }
  friend bool operator<(const erf_params& a, const erf_params& b) {
    return std::tie(a.input_multiplier, a.output_multiplier, a.output_offset) <
           std::tie(b.input_multiplier, b.output_multiplier, b.output_offset);
  }
};

union unary_params {
  exp_params exp;
  erf_params erf;
};

typedef void (*unary_kernel_fn)(size_t width, size_t height, size_t stride_a,
                                const void* a, size_t stride_x, void* x,
                                const unary_params* params);

#define YNN_ELEMENTWISE_KERNEL(arch, name, op, type_a, type_c)    \
  void name(size_t m, size_t n, size_t stride_a_m, const void* a, \
            size_t stride_x_m, void* x, const unary_params* params);
#include "ynnpack/kernels/unary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

unary_kernel_fn get_unary_reference_kernel(ynn_unary_operator op,
                                           ynn_type type);
unary_kernel_fn get_convert_reference_kernel(ynn_type a_type, ynn_type x_type);
template <typename A, typename X>
unary_kernel_fn get_convert_reference_kernel() {
  return get_convert_reference_kernel(type_of<A>(), type_of<X>());
}

unary_kernel_fn get_unary_kernel(
    ynn_unary_operator op, ynn_type a_type, ynn_type x_type,
    uint64_t supported_arch_flags = get_supported_arch_flags());

unary_params get_unary_params(ynn_unary_operator op);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_UNARY_H_
