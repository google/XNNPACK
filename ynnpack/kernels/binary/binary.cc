// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/binary/binary.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "ynnpack/base/arch.h"
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

namespace {

// These "reference" microkernels are not designed to be fast, only to support
// all possible operators and types with reasonable performance. We just
// intend to give the compiler a reasonable chance at optimizing them.
template <typename T, typename Operator>
void unquantized_impl(size_t m, size_t n, size_t stride_a_m, size_t stride_a_n,
                      const void* va, size_t stride_b_m, size_t stride_b_n,
                      const void* vb, size_t stride_x_m, void* vx,
                      const binary_params*) {
  auto a = reinterpret_cast<const T*>(va);
  auto b = reinterpret_cast<const T*>(vb);
  auto x = reinterpret_cast<T*>(vx);
  Operator op;
  for (size_t i = 0; i < m; ++i) {
    if (stride_a_n == 0) {
      const T a_0 = a[0];
      for (size_t j = 0; j < n; ++j) {
        x[j] = static_cast<T>(op(a_0, b[j]));
      }
    } else if (stride_b_n == 0) {
      // TODO: If Operator::commutative is true, we could swap the arguments and
      // avoid the code size cost of the third variant.
      const T b_0 = b[0];
      for (size_t j = 0; j < n; ++j) {
        x[j] = static_cast<T>(op(a[j], b_0));
      }
    } else {
      assert(stride_a_n == sizeof(T));
      assert(stride_b_n == sizeof(T));
      for (size_t j = 0; j < n; ++j) {
        x[j] = static_cast<T>(op(a[j], b[j]));
      }
    }
    a = offset_bytes(a, stride_a_m);
    b = offset_bytes(b, stride_b_m);
    x = offset_bytes(x, stride_x_m);
  }
}

template <typename Operator, typename T>
const binary_kernel& get_kernel(T) {
  static binary_kernel kernel = {
      &unquantized_impl<T, Operator>,
      nullptr,
  };
  return kernel;
}

template <typename T, typename Operator>
void quantized_impl(size_t m, size_t n, size_t stride_a_m, size_t stride_a_n,
                    const void* va, size_t stride_b_m, size_t stride_b_n,
                    const void* vb, size_t stride_x_m, void* vx,
                    const binary_params* params) {
  assert(params);
  auto a = reinterpret_cast<const T*>(va);
  auto b = reinterpret_cast<const T*>(vb);
  auto x = reinterpret_cast<T*>(vx);

  Operator op;
  for (size_t i = 0; i < m; ++i) {
    if (stride_a_n == 0) {
      const float a_0 = dequantize(a[0], params->reference.a_scale,
                                   params->reference.a_zero_point);
      for (size_t j = 0; j < n; ++j) {
        const float b_j = dequantize(b[j], params->reference.b_scale,
                                     params->reference.b_zero_point);
        const float result = op(a_0, b_j);
        x[j] = quantize<T>(result, params->reference.inv_x_scale,
                           params->reference.x_zero_point);
      }
    } else if (stride_b_n == 0) {
      // TODO: If Operator::commutative is true, we could swap the arguments and
      // avoid the code size cost of the third variant.
      const float b_0 = dequantize(b[0], params->reference.b_scale,
                                   params->reference.b_zero_point);
      for (size_t j = 0; j < n; ++j) {
        const float a_j = dequantize(a[j], params->reference.a_scale,
                                     params->reference.a_zero_point);
        const float result = op(a_j, b_0);
        x[j] = quantize<T>(result, params->reference.inv_x_scale,
                           params->reference.x_zero_point);
      }
    } else {
      assert(stride_a_n == sizeof(T));
      assert(stride_b_n == sizeof(T));
      for (size_t j = 0; j < n; ++j) {
        const float a_j = dequantize(a[j], params->reference.a_scale,
                                     params->reference.a_zero_point);
        const float b_j = dequantize(b[j], params->reference.b_scale,
                                     params->reference.b_zero_point);
        const float result = op(a_j, b_j);
        x[j] = quantize<T>(result, params->reference.inv_x_scale,
                           params->reference.x_zero_point);
      }
    }
    a = offset_bytes(a, stride_a_m);
    b = offset_bytes(b, stride_b_m);
    x = offset_bytes(x, stride_x_m);
  }
}

void init_reference_binary_params(float a_scale, int32_t a_zero_point,
                                  float b_scale, int32_t b_zero_point,
                                  float x_scale, int32_t x_zero_point,
                                  binary_params& params) {
  params.reference.a_scale = a_scale;
  params.reference.a_zero_point = a_zero_point;
  params.reference.b_scale = b_scale;
  params.reference.b_zero_point = b_zero_point;
  params.reference.inv_x_scale = 1.0f / x_scale;
  params.reference.x_zero_point = x_zero_point;
}

template <typename Operator, typename T>
const binary_kernel& get_kernel(quantized<T>) {
  static binary_kernel kernel = {
      &quantized_impl<quantized<T>, Operator>,
      init_reference_binary_params,
  };
  return kernel;
}

// We can't use STL's functional ops for integers because they will crash if
// they overflow.
template <typename T>
struct AddOp {
  T operator()(T a, T b) const { return a + b; }
  using commutative = std::true_type;
};

template <>
struct AddOp<int32_t> {
  int32_t operator()(int32_t a, int32_t b) const { return widen(a) + widen(b); }
  using commutative = std::true_type;
};

template <typename T>
struct SubtractOp {
  T operator()(T a, T b) const { return a - b; }
  using commutative = std::false_type;
};

template <>
struct SubtractOp<int32_t> {
  int32_t operator()(int32_t a, int32_t b) const { return widen(a) - widen(b); }
  using commutative = std::false_type;
};

template <typename T>
struct MultiplyOp {
  T operator()(T a, T b) const { return a * b; }
  using commutative = std::true_type;
};

template <>
struct MultiplyOp<int32_t> {
  int32_t operator()(int32_t a, int32_t b) const { return widen(a) * widen(b); }
  using commutative = std::true_type;
};

template <typename T>
struct DivideOp {
  T operator()(T a, T b) const { return a / b; }
  using commutative = std::false_type;
};

template <>
struct DivideOp<int32_t> {
  int32_t operator()(int32_t a, int32_t b) const { return euclidean_div(a, b); }
  using commutative = std::false_type;
};

template <typename T>
struct MaxOp {
  T operator()(T a, T b) const { return a > b ? a : b; }
  using commutative = std::true_type;
};

template <typename T>
struct MinOp {
  T operator()(T a, T b) const { return a < b ? a : b; }
  using commutative = std::true_type;
};

template <typename T>
struct SquaredDifferenceOp {
  T operator()(T a, T b) const { return (a - b) * (a - b); }
  using commutative = std::true_type;
};

template <typename T>
struct PowOp {
  float operator()(float a, float b) const { return std::pow(a, b); }
  using commutative = std::false_type;
};

template <>
struct PowOp<int32_t> {
  int32_t operator()(int32_t a, int32_t b) const { return integer_pow(a, b); }
  using commutative = std::false_type;
};

template <typename T>
struct LeakyReluOp {
  T operator()(T a, T b) const { return (a < 0) ? static_cast<T>(a * b) : a; }
  using commutative = std::false_type;
};

using std::copysign;

half copysign(half a, half b) {
  return half::from_bits((a.to_bits() & 0x7FFF) | (b.to_bits() & 0x8000));
}

bfloat16 copysign(bfloat16 a, bfloat16 b) {
  return bfloat16::from_bits((a.to_bits() & 0x7FFF) | (b.to_bits() & 0x8000));
}

template <typename T>
struct CopysignOp {
  T operator()(T a, T b) const { return copysign(a, b); }
  using commutative = std::false_type;
};

#define DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, op)               \
  switch (type) {                                                        \
    case ynn_type_fp32:                                                  \
      return &get_kernel<op<float>>(float());                            \
    case ynn_type_fp16:                                                  \
      return &get_kernel<op<half>>(half());                              \
    case ynn_type_bf16:                                                  \
      return &get_kernel<op<bfloat16>>(bfloat16());                      \
    case ynn_type_int8:                                                  \
      return &get_kernel<op<float>>(quantized<int8_t>());                \
    case ynn_type_uint8:                                                 \
      return &get_kernel<op<float>>(quantized<uint8_t>());               \
    case ynn_type_int32:                                                 \
      return is_quantized ? &get_kernel<op<float>>(quantized<int32_t>()) \
                          : &get_kernel<op<int32_t>>(int32_t());         \
    default:                                                             \
      return nullptr;                                                    \
  }

}  // namespace

const binary_kernel* get_binary_reference_kernel(ynn_binary_operator op,
                                                 ynn_type type,
                                                 bool is_quantized) {
  switch (op) {
    case ynn_binary_add:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, AddOp);
    case ynn_binary_subtract:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, SubtractOp);
    case ynn_binary_multiply:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, MultiplyOp);
    case ynn_binary_divide:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, DivideOp);
    case ynn_binary_max:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, MaxOp);
    case ynn_binary_min:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, MinOp);
    case ynn_binary_copysign:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, CopysignOp);
    case ynn_binary_squared_difference:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, SquaredDifferenceOp);
    case ynn_binary_pow:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, PowOp);
    case ynn_binary_leaky_relu:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, LeakyReluOp);
    case ynn_binary_invalid:
      return nullptr;
  }
  return nullptr;
}

const binary_kernel* get_binary_kernel(ynn_binary_operator op, ynn_type type,
                                       bool is_quantized,
                                       uint64_t supported_arch_flags) {
  // TODO(vksnk): select a better kernel based on the passed size.
#define YNN_ELEMENTWISE_KERNEL(arch, name, op_type, init_params_fn, A, B, X)  \
  if (type == type_of<A>() && type == type_of<B>() && type == type_of<X>() && \
      op == ynn_binary_##op_type &&                                           \
      is_arch_supported(arch, supported_arch_flags)) {                        \
    static binary_kernel kernel##name = {&name, nullptr};                     \
    YNN_LOG_INFO() << "Using binary kernel " << #name;                        \
    return &kernel##name;                                                     \
  }

#include "ynnpack/kernels/binary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

  return get_binary_reference_kernel(op, type, false);
}

binary_kernel_fn get_binary_multiply_kernel(ynn_type type_a, ynn_type type_b,
                                            ynn_type type_x) {
#define YNN_ELEMENTWISE_KERNEL(arch, name, op, init_params_fn, A, B, X)    \
  if (ynn_binary_##op == ynn_binary_multiply && is_arch_supported(arch)) { \
    if (type_of<A>() == type_a && type_of<B>() == type_b &&                \
        type_of<X>() == type_x) {                                          \
      return name;                                                         \
    }                                                                      \
  }
#include "ynnpack/kernels/binary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL
  return nullptr;
}

}  // namespace ynn
