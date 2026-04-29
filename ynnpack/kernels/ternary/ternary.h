// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_TERNARY_H_
#define XNNPACK_YNNPACK_KERNELS_TERNARY_H_

#include <cstddef>
#include <ostream>

#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

union ternary_params {};

// The stride of dimension `n` for any operand must be 0 or the size of one
// element.
typedef void (*ternary_kernel_fn)(size_t m, size_t n, size_t stride_a_m,
                                  size_t stride_a_n, const void* a,
                                  size_t stride_b_m, size_t stride_b_n,
                                  const void* b, size_t stride_c_m,
                                  size_t stride_c_n, const void* c,
                                  size_t stride_x_m, void* x,
                                  const ternary_params* params);

#define YNN_ELEMENTWISE_KERNEL(arch, name, op, type_a, type_b, type_c, type_x) \
  void name(size_t m, size_t n, size_t stride_a_m, size_t stride_a_n,          \
            const void* a, size_t stride_b_m, size_t stride_b_n,               \
            const void* b, size_t stride_c_m, size_t stride_c_n,               \
            const void* c, size_t stride_x_m, void* x,                         \
            const ternary_params* params);
#include "ynnpack/kernels/ternary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

enum class ternary_op {
  invalid,
  multiply,           // a*b*c
  multiply_add,       // a*b + c
  subtract_multiply,  // a - b*c
  clamp,              // min(max(a, b), c)
  quantize_int8,      // i8(a/b + c)
  quantize_uint8,     // u8(a/b + c)
  dequantize,         // (a - b)*c
};

const char* to_string(ternary_op op);

inline std::ostream& operator<<(std::ostream& os, ternary_op op) {
  return os << to_string(op);
}

ternary_kernel_fn get_ternary_kernel(ternary_op op, ynn_type type_a,
                                     ynn_type type_b, ynn_type type_c,
                                     ynn_type type_x);

ternary_params get_ternary_params(ternary_op op);

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_TERNARY_H_
