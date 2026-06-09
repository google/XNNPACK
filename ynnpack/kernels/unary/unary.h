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

// `enum class` doesn't work well for bitfield values.
namespace unary_flag {

enum {
  // This kernel produces results that are numerically consistent with all other
  // kernels of the same type with this flag.
#if defined(YNN_ARCH_X86) || defined(YNN_ARCH_ARM)
  consistent_arithmetic = 1 << 0,
#else
  // We don't support consistent arithmetic for unary operators on other
  // targets.
  consistent_arithmetic = 0,
#endif
};

}  // namespace unary_flag

struct log_exp_params {
  real _;  // output_offset not supported by exp
  real output_multiplier;
  real input_multiplier;

  friend bool operator==(const log_exp_params& a, const log_exp_params& b) {
    return std::tie(a.input_multiplier, a.output_multiplier) ==
           std::tie(b.input_multiplier, b.output_multiplier);
  }
  friend bool operator<(const log_exp_params& a, const log_exp_params& b) {
    return std::tie(a.input_multiplier, a.output_multiplier) <
           std::tie(b.input_multiplier, b.output_multiplier);
  }
};

using exp_params = log_exp_params;
using expm1_params = log_exp_params;
using log_params = log_exp_params;
using log1p_params = log_exp_params;

struct erf_params {
  real output_offset;
  real output_multiplier;
  real input_multiplier;

  friend bool operator==(const erf_params& a, const erf_params& b) {
    return std::tie(a.input_multiplier, a.output_multiplier, a.output_offset) ==
           std::tie(b.input_multiplier, b.output_multiplier, b.output_offset);
  }
  friend bool operator<(const erf_params& a, const erf_params& b) {
    return std::tie(a.input_multiplier, a.output_multiplier, a.output_offset) <
           std::tie(b.input_multiplier, b.output_multiplier, b.output_offset);
  }
};

using approx_erf_params = erf_params;

struct affine_output_params {
  real output_offset;
  real output_multiplier;

  friend bool operator==(const affine_output_params& a,
                         const affine_output_params& b) {
    return std::tie(a.output_multiplier, a.output_offset) ==
           std::tie(b.output_multiplier, b.output_offset);
  }
  friend bool operator<(const affine_output_params& a,
                        const affine_output_params& b) {
    return std::tie(a.output_multiplier, a.output_offset) <
           std::tie(b.output_multiplier, b.output_offset);
  }
};

using tanh_params = affine_output_params;
using approx_tanh_params = tanh_params;
using sine_params = affine_output_params;
using cosine_params = affine_output_params;

struct poly3_params {
  real c0, c1, c2, c3;

  friend bool operator==(const poly3_params& x, const poly3_params& y) {
    return std::tie(x.c0, x.c1, x.c2, x.c3) == std::tie(y.c0, y.c1, y.c2, y.c3);
  }
  friend bool operator<(const poly3_params& x, const poly3_params& y) {
    return std::tie(x.c0, x.c1, x.c2, x.c3) < std::tie(y.c0, y.c1, y.c2, y.c3);
  }
};

union unary_params {
  exp_params exp;
  expm1_params expm1;
  log_params log;
  log1p_params log1p;

  // All of these params have the first two params a, b, such that they form
  // output offset and output scale parameters, respectively. We use this fact
  // to support all of them via poly3's parameters.
  erf_params erf;
  approx_erf_params approx_erf;
  approx_tanh_params approx_tanh;
  tanh_params tanh;
  sine_params sine;
  cosine_params cosine;
  poly3_params poly3;
};

typedef void (*unary_kernel_fn)(size_t width, size_t height, size_t stride_a,
                                const void* a, size_t stride_x, void* x,
                                const unary_params* params);

#define YNN_ELEMENTWISE_KERNEL(arch, name, op, flags, type_a, type_c) \
  void name(size_t m, size_t n, size_t stride_a_m, const void* a,     \
            size_t stride_x_m, void* x, const unary_params* params);
#include "ynnpack/kernels/unary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

unary_kernel_fn get_unary_reference_kernel(ynn_unary_operator op, ynn_type type,
                                           uint32_t required_flags = 0);
unary_kernel_fn get_convert_reference_kernel(ynn_type a_type, ynn_type x_type,
                                             uint32_t required_flags = 0);
template <typename A, typename X>
unary_kernel_fn get_convert_reference_kernel(uint32_t required_flags = 0) {
  return get_convert_reference_kernel(type_of<A>(), type_of<X>(),
                                      required_flags);
}

unary_kernel_fn get_unary_kernel(
    ynn_unary_operator op, ynn_type a_type, ynn_type x_type,
    uint32_t required_flags = 0,
    uint64_t supported_arch_flags = get_supported_arch_flags());

unary_params get_unary_params(ynn_unary_operator op);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_UNARY_H_
