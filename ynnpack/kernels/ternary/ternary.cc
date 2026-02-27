// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/ternary/ternary.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/type.h"  // IWYU pragma: keep
#include "ynnpack/include/ynnpack.h"

namespace ynn {

namespace {

template <typename A, typename X>
void quantize(size_t m, size_t n, size_t stride_a_m, size_t stride_a_n,
              const A* a, size_t stride_b_m, size_t stride_b_n, const float* b,
              size_t stride_c_m, size_t stride_c_n, const int32_t* c,
              size_t stride_x_m, X* x, const ternary_params* params) {
  for (size_t i = 0; i < m; ++i) {
    // There are 8 cases of broadcasting. Here, we only specialize for
    // broadcasting b, because it permits lifting a division out of the loop.
    if (stride_b_n == 0) {
      const float b_0 = 1.0f / b[0];
      for (size_t j = 0; j < n; ++j) {
        const A a_j = *offset_bytes(a, j * stride_a_n);
        const int32_t c_j = *offset_bytes(c, j * stride_c_n);
        x[j] = ynn::quantize<X>(a_j, b_0, c_j);
      }
    } else {
      for (size_t j = 0; j < n; ++j) {
        const A a_j = *offset_bytes(a, j * stride_a_n);
        const int32_t c_j = *offset_bytes(c, j * stride_c_n);
        x[j] = ynn::quantize<X>(a_j, 1.0f / b[j], c_j);
      }
    }
    a = offset_bytes(a, stride_a_m);
    b = offset_bytes(b, stride_b_m);
    c = offset_bytes(c, stride_c_m);
    x = offset_bytes(x, stride_x_m);
  }
}

template <typename A, typename X>
void dequantize(size_t m, size_t n, size_t stride_a_m, size_t stride_a_n,
                const A* a, size_t stride_b_m, size_t stride_b_n,
                const int32_t* b, size_t stride_c_m, size_t stride_c_n,
                const float* c, size_t stride_x_m, X* x,
                const ternary_params* params) {
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      const A a_j = *offset_bytes(a, j * stride_a_n);
      const int32_t b_j = *offset_bytes(b, j * stride_b_n);
      const float c_j = *offset_bytes(c, j * stride_c_n);
      x[j] = (a_j - b_j) * c_j;
    }
    a = offset_bytes(a, stride_a_m);
    b = offset_bytes(b, stride_b_m);
    c = offset_bytes(c, stride_c_m);
    x = offset_bytes(x, stride_x_m);
  }
}

}  // namespace

void quantize_fp32_to_int8(size_t m, size_t n, size_t stride_a_m,
                           size_t stride_a_n, const void* a, size_t stride_b_m,
                           size_t stride_b_n, const void* b, size_t stride_c_m,
                           size_t stride_c_n, const void* c, size_t stride_x_m,
                           void* x, const ternary_params* params) {
  quantize(m, n, stride_a_m, stride_a_n, reinterpret_cast<const float*>(a),
           stride_b_m, stride_b_n, reinterpret_cast<const float*>(b),
           stride_c_m, stride_c_n, reinterpret_cast<const int32_t*>(c),
           stride_x_m, reinterpret_cast<int8_t*>(x), params);
}

void quantize_fp32_to_uint8(size_t m, size_t n, size_t stride_a_m,
                            size_t stride_a_n, const void* a, size_t stride_b_m,
                            size_t stride_b_n, const void* b, size_t stride_c_m,
                            size_t stride_c_n, const void* c, size_t stride_x_m,
                            void* x, const ternary_params* params) {
  quantize(m, n, stride_a_m, stride_a_n, reinterpret_cast<const float*>(a),
           stride_b_m, stride_b_n, reinterpret_cast<const float*>(b),
           stride_c_m, stride_c_n, reinterpret_cast<const int32_t*>(c),
           stride_x_m, reinterpret_cast<uint8_t*>(x), params);
}

void dequantize_int8_to_fp32(size_t m, size_t n, size_t stride_a_m,
                             size_t stride_a_n, const void* a,
                             size_t stride_b_m, size_t stride_b_n,
                             const void* b, size_t stride_c_m,
                             size_t stride_c_n, const void* c,
                             size_t stride_x_m, void* x,
                             const ternary_params* params) {
  dequantize(m, n, stride_a_m, stride_a_n, reinterpret_cast<const int8_t*>(a),
             stride_b_m, stride_b_n, reinterpret_cast<const int32_t*>(b),
             stride_c_m, stride_c_n, reinterpret_cast<const float*>(c),
             stride_x_m, reinterpret_cast<float*>(x), params);
}

void dequantize_uint8_to_fp32(size_t m, size_t n, size_t stride_a_m,
                              size_t stride_a_n, const void* a,
                              size_t stride_b_m, size_t stride_b_n,
                              const void* b, size_t stride_c_m,
                              size_t stride_c_n, const void* c,
                              size_t stride_x_m, void* x,
                              const ternary_params* params) {
  dequantize(m, n, stride_a_m, stride_a_n, reinterpret_cast<const uint8_t*>(a),
             stride_b_m, stride_b_n, reinterpret_cast<const int32_t*>(b),
             stride_c_m, stride_c_n, reinterpret_cast<const float*>(c),
             stride_x_m, reinterpret_cast<float*>(x), params);
}

void dequantize_int32_to_fp32(size_t m, size_t n, size_t stride_a_m,
                              size_t stride_a_n, const void* a,
                              size_t stride_b_m, size_t stride_b_n,
                              const void* b, size_t stride_c_m,
                              size_t stride_c_n, const void* c,
                              size_t stride_x_m, void* x,
                              const ternary_params* params) {
  dequantize(m, n, stride_a_m, stride_a_n, reinterpret_cast<const int32_t*>(a),
             stride_b_m, stride_b_n, reinterpret_cast<const int32_t*>(b),
             stride_c_m, stride_c_n, reinterpret_cast<const float*>(c),
             stride_x_m, reinterpret_cast<float*>(x), params);
}

ternary_kernel_fn get_ternary_kernel(ternary_op op, ynn_type type_a,
                                     ynn_type type_b, ynn_type type_c,
                                     ynn_type type_x) {
#define YNN_ELEMENTWISE_KERNEL(arch, name, kernel_op, init_params_fn, A, B, C, \
                               X)                                              \
  if (ternary_op::kernel_op == op && is_arch_supported(arch)) {                \
    if (type_of<A>() == type_a && type_of<B>() == type_b &&                    \
        type_of<C>() == type_c && type_of<X>() == type_x) {                    \
      return name;                                                             \
    }                                                                          \
  }
#include "ynnpack/kernels/ternary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL
  return nullptr;
}

const char* to_string(ternary_op op) {
  switch (op) {
    case ternary_op::multiply:
      return "multiply";
    case ternary_op::multiply_add:
      return "multiply_add";
    case ternary_op::subtract_multiply:
      return "subtract_multiply";
    case ternary_op::clamp:
      return "clamp";
    case ternary_op::quantize_int8:
      return "quantize_int8";
    case ternary_op::quantize_uint8:
      return "quantize_uint8";
    case ternary_op::dequantize:
      return "dequantize";
  }
  YNN_UNREACHABLE;
  return "unknown";
}

}  // namespace ynn
