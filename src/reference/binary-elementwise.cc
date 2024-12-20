// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "xnnpack.h"
#include "xnnpack/config-types.h"
#include "xnnpack/datatype.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "xnnpack/reference-utils.h"

using xnnpack::dequantize;
using xnnpack::euclidean_div;
using xnnpack::euclidean_mod;
using xnnpack::integer_pow;
using xnnpack::quantize;
using xnnpack::widen;

namespace {

// These "reference" microkernels are not designed to be fast, only to support
// all possible operators and datatypes with reasonable performance. We just
// intend to give the compiler a reasonable chance at optimizing them.
template <typename T, typename Operator>
void binary_ukernel_unquantized(size_t batch_size_bytes, const T* a, const T* b,
                                T* output, const xnn_binary_uparams*) {
  const size_t batch_size = batch_size_bytes / sizeof(T);
  Operator op;
  for (size_t i = 0; i < batch_size; ++i) {
    output[i] = op(a[i], b[i]);
  }
}

template <typename T, typename Operator>
void binaryc_ukernel_unquantized(size_t batch_size_bytes, const T* a,
                                 const T* b, T* output,
                                 const xnn_binary_uparams*) {
  const size_t batch_size = batch_size_bytes / sizeof(T);
  const T b_0 = *b;
  Operator op;
  for (size_t i = 0; i < batch_size; ++i) {
    output[i] = op(a[i], b_0);
  }
}

template <typename T, typename Operator>
void rbinaryc_ukernel_unquantized(size_t batch_size_bytes, const T* a,
                                  const T* b, T* output,
                                  const xnn_binary_uparams*) {
  const size_t batch_size = batch_size_bytes / sizeof(T);
  const T b_0 = *b;
  Operator op;
  for (size_t i = 0; i < batch_size; ++i) {
    output[i] = op(b_0, a[i]);
  }
}

template <typename Operator, typename T>
const xnn_binary_elementwise_config* get_config(T) {
  static_assert(!xnnpack::is_quantized<T>::value, "");
  static xnn_binary_elementwise_config config = {
      (xnn_vbinary_ukernel_fn)binary_ukernel_unquantized<T, Operator>,
      (xnn_vbinary_ukernel_fn)binaryc_ukernel_unquantized<T, Operator>,
      (xnn_vbinary_ukernel_fn)rbinaryc_ukernel_unquantized<T, Operator>,
      nullptr,
      /*element_tile=*/1,
  };
  return &config;
}

template <typename T, typename Operator>
void binary_ukernel_quantized(size_t batch_size_bytes, const T* a, const T* b,
                              T* output, const xnn_binary_uparams* params) {
  const size_t batch_size = batch_size_bytes / sizeof(T);
  Operator op;
  for (size_t i = 0; i < batch_size; ++i) {
    const float a_i = dequantize(a[i], params->reference.a_scale,
                                 params->reference.a_zero_point);
    const float b_i = dequantize(b[i], params->reference.b_scale,
                                 params->reference.b_zero_point);
    const float result = op(a_i, b_i);
    output[i] = quantize<T>(result, params->reference.inv_output_scale,
                            params->reference.output_zero_point);
  }
}

template <typename T, typename Operator>
void binaryc_ukernel_quantized(size_t batch_size_bytes, const T* a, const T* b,
                               T* output, const xnn_binary_uparams* params) {
  const size_t batch_size = batch_size_bytes / sizeof(T);
  Operator op;
  const float b_0 =
      dequantize(*b, params->reference.b_scale, params->reference.b_zero_point);
  for (size_t i = 0; i < batch_size; ++i) {
    const float a_i = dequantize(a[i], params->reference.a_scale,
                                 params->reference.a_zero_point);
    const float result = op(a_i, b_0);
    output[i] = quantize<T>(result, params->reference.inv_output_scale,
                            params->reference.output_zero_point);
  }
}

template <typename T, typename Operator>
void rbinaryc_ukernel_quantized(size_t batch_size_bytes, const T* a, const T* b,
                                T* output, const xnn_binary_uparams* params) {
  const size_t batch_size = batch_size_bytes / sizeof(T);
  Operator op;
  const float b_0 =
      dequantize(*b, params->reference.b_scale, params->reference.b_zero_point);
  for (size_t i = 0; i < batch_size; ++i) {
    const float a_i = dequantize(a[i], params->reference.a_scale,
                                 params->reference.a_zero_point);
    const float result = op(b_0, a_i);
    output[i] = quantize<T>(result, params->reference.inv_output_scale,
                            params->reference.output_zero_point);
  }
}

size_t init_quantized_binary_op(
    union xnn_binary_uparams* params,
    const struct xnn_quantization_params* a_params,
    const struct xnn_quantization_params* b_params,
    const struct xnn_quantization_params* output_params) {
  params->reference.a_scale = a_params->scale;
  params->reference.a_zero_point = a_params->zero_point;
  params->reference.b_scale = b_params->scale;
  params->reference.b_zero_point = b_params->zero_point;
  params->reference.inv_output_scale = 1.0f / output_params->scale;
  params->reference.output_zero_point = output_params->zero_point;
  return sizeof(params->reference);
}

template <typename Operator, typename T>
const xnn_binary_elementwise_config* get_config(xnnpack::quantized<T>) {
  static xnn_binary_elementwise_config config = {
      (xnn_vbinary_ukernel_fn)
          binary_ukernel_quantized<xnnpack::quantized<T>, Operator>,
      (xnn_vbinary_ukernel_fn)
          binaryc_ukernel_quantized<xnnpack::quantized<T>, Operator>,
      (xnn_vbinary_ukernel_fn)
          rbinaryc_ukernel_quantized<xnnpack::quantized<T>, Operator>,
      init_quantized_binary_op,
      /*element_tile=*/1,
  };
  return &config;
}

// We can't use STL's functional ops for integers because they will crash if
// they overflow.
template <typename T>
struct AddOp {
  T operator()(T a, T b) const { return a + b; }
};

template <>
struct AddOp<int32_t> {
  int32_t operator()(int32_t a, int32_t b) const { return widen(a) + widen(b); }
};

template <typename T>
struct SubOp {
  T operator()(T a, T b) const { return a - b; }
};

template <>
struct SubOp<int32_t> {
  int32_t operator()(int32_t a, int32_t b) const { return widen(a) - widen(b); }
};

template <typename T>
struct MultiplyOp {
  T operator()(T a, T b) const { return a * b; }
};

template <>
struct MultiplyOp<int32_t> {
  int32_t operator()(int32_t a, int32_t b) const { return widen(a) * widen(b); }
};

template <typename T>
struct DivideOp {
  T operator()(T a, T b) const { return a / b; }
};

template <>
struct DivideOp<int32_t> {
  int32_t operator()(int32_t a, int32_t b) const { return euclidean_div(a, b); }
};

template <typename T>
struct ModulusOp {
  float operator()(float a, float b) const {
    // Define division by zero to be 0?
    if (b == 0) {
      return 0;
    } else {
      return std::fmod(a, b);
    }
  }
};

template <>
struct ModulusOp<int32_t> {
  int32_t operator()(int32_t a, int32_t b) const { return euclidean_mod(a, b); }
};

template <typename T>
struct MaxOp {
  T operator()(T a, T b) const { return a > b ? a : b; }
};

template <typename T>
struct MinOp {
  T operator()(T a, T b) const { return a < b ? a : b; }
};

template <typename T>
struct SquaredDifferenceOp {
  T operator()(T a, T b) const { return (a - b) * (a - b); }
};

template <typename T>
struct PreluOp {
  T operator()(T a, T b) const { return (a < 0) ? static_cast<T>(a * b) : a; }
};

template <typename T>
struct Atan2Op {
  float operator()(float a, float b) const { return std::atan2(a, b); }
};

template <typename T>
struct PowOp {
  float operator()(float a, float b) const { return std::pow(a, b); }
};

template <>
struct PowOp<int32_t> {
  int32_t operator()(int32_t a, int32_t b) const {
    if (b < 0) {
      return 0;
    } else if (b == 0) {
      return 1;
    } else {
      return integer_pow(a, b);
    }
  }
};

template <typename T>
struct BitwiseAndOp {
  T operator()(T a, T b) const { return a & b; }
};

template <typename T>
struct BitwiseOrOp {
  T operator()(T a, T b) const { return a | b; }
};

template <typename T>
struct BitwiseXorOp {
  T operator()(T a, T b) const { return a ^ b; }
};

template <typename T>
struct ShiftLeftOp {
  static constexpr T type_mask = sizeof(T) * 8 - 1;
  T operator()(T a, T b) const { return a << (b & type_mask); }
};

template <typename T>
struct ShiftRightLogicalOp {
  T operator()(T a, T b) const {
    static constexpr T type_mask = sizeof(T) * 8 - 1;
    return static_cast<typename std::make_unsigned<T>::type>(a) >>
           (b & type_mask);
  }
};
template <typename T>
struct ShiftRightArithmeticOp {
  T operator()(T a, T b) const {
    static constexpr T type_mask = sizeof(T) * 8 - 1;
    return static_cast<typename std::make_signed<T>::type>(a) >>
           (b & type_mask);
  }
};

using std::copysign;

xnn_float16 copysign(xnn_float16 a, xnn_float16 b) {
  uint16_t a_bits = xnn_float16_to_bits(a);
  uint16_t b_bits = xnn_float16_to_bits(b);
  uint16_t sign_bit = b_bits & 0x8000;
  return xnn_float16_from_bits((a_bits & 0x7FFF) | (sign_bit & 0x8000));
}

xnn_bfloat16 copysign(xnn_bfloat16 a, xnn_bfloat16 b) {
  uint16_t a_bits = xnn_bfloat16_to_bits(a);
  uint16_t b_bits = xnn_bfloat16_to_bits(b);
  uint16_t sign_bit = b_bits & 0x8000;
  return xnn_bfloat16_from_bits((a_bits & 0x7FFF) | (sign_bit & 0x8000));
}

template <typename T>
struct CopysignOp {
  T operator()(T a, T b) const { return copysign(a, b); }
};

#define DISPATCH_OPERATOR_FOR_DATATYPE(datatype, op)               \
  switch (datatype) {                                              \
    case xnn_datatype_fp32:                                        \
      return get_config<op<float>>(float());                       \
    case xnn_datatype_fp16:                                        \
      return get_config<op<xnn_float16>>(xnn_float16());           \
    case xnn_datatype_bf16:                                        \
      return get_config<op<xnn_bfloat16>>(xnn_bfloat16());         \
    case xnn_datatype_qint8:                                       \
      return get_config<op<float>>(xnnpack::quantized<int8_t>());  \
    case xnn_datatype_quint8:                                      \
      return get_config<op<float>>(xnnpack::quantized<uint8_t>()); \
    case xnn_datatype_int32:                                       \
      return get_config<op<int32_t>>(int32_t());                   \
    default:                                                       \
      return nullptr;                                              \
  }

#define DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, op)          \
  switch (datatype) {                                              \
    case xnn_datatype_fp32:                                        \
      return get_config<op<float>>(float());                       \
    case xnn_datatype_fp16:                                        \
      return get_config<op<xnn_float16>>(xnn_float16());           \
    case xnn_datatype_bf16:                                        \
      return get_config<op<xnn_bfloat16>>(xnn_bfloat16());         \
    case xnn_datatype_qint8:                                       \
      return get_config<op<float>>(xnnpack::quantized<int8_t>());  \
    case xnn_datatype_quint8:                                      \
      return get_config<op<float>>(xnnpack::quantized<uint8_t>()); \
    default:                                                       \
      return nullptr;                                              \
  }

#define DISPATCH_OPERATOR_FOR_INTEGRAL_DATATYPE(datatype, op) \
  switch (datatype) {                                         \
    case xnn_datatype_int32:                                  \
      return get_config<op<int32_t>>(int32_t());              \
    default:                                                  \
      return nullptr;                                         \
  }

}  // namespace

extern "C" {

const struct xnn_binary_elementwise_config* xnn_init_binary_reference_config(
    enum xnn_binary_operator type, enum xnn_datatype datatype) {
  switch (type) {
    case xnn_binary_add:
      DISPATCH_OPERATOR_FOR_DATATYPE(datatype, AddOp);
    case xnn_binary_subtract:
      DISPATCH_OPERATOR_FOR_DATATYPE(datatype, SubOp);
    case xnn_binary_multiply:
      DISPATCH_OPERATOR_FOR_DATATYPE(datatype, MultiplyOp);
    case xnn_binary_divide:
      DISPATCH_OPERATOR_FOR_DATATYPE(datatype, DivideOp);
    case xnn_binary_maximum:
      DISPATCH_OPERATOR_FOR_DATATYPE(datatype, MaxOp);
    case xnn_binary_minimum:
      DISPATCH_OPERATOR_FOR_DATATYPE(datatype, MinOp);
    case xnn_binary_copysign:
      DISPATCH_OPERATOR_FOR_DATATYPE(datatype, CopysignOp);
    case xnn_binary_prelu:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, PreluOp);
    case xnn_binary_squared_difference:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, SquaredDifferenceOp);
    case xnn_binary_modulus:
      DISPATCH_OPERATOR_FOR_DATATYPE(datatype, ModulusOp);
    case xnn_binary_atan2:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, Atan2Op);
    case xnn_binary_pow:
      DISPATCH_OPERATOR_FOR_DATATYPE(datatype, PowOp);
    case xnn_binary_bitwise_and:
      DISPATCH_OPERATOR_FOR_INTEGRAL_DATATYPE(datatype, BitwiseAndOp);
    case xnn_binary_bitwise_or:
      DISPATCH_OPERATOR_FOR_INTEGRAL_DATATYPE(datatype, BitwiseOrOp);
    case xnn_binary_bitwise_xor:
      DISPATCH_OPERATOR_FOR_INTEGRAL_DATATYPE(datatype, BitwiseXorOp);
    case xnn_binary_shift_left:
      DISPATCH_OPERATOR_FOR_INTEGRAL_DATATYPE(datatype, ShiftLeftOp);
    case xnn_binary_shift_right_logical:
      DISPATCH_OPERATOR_FOR_INTEGRAL_DATATYPE(datatype, ShiftRightLogicalOp);
    case xnn_binary_shift_right_arithmetic:
      DISPATCH_OPERATOR_FOR_INTEGRAL_DATATYPE(datatype, ShiftRightArithmeticOp);
    case xnn_binary_invalid:
      return nullptr;
  }
  return nullptr;
}

}  // extern "C"
