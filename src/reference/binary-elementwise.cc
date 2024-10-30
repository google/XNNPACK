// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <type_traits>

#include "xnnpack.h"
#include "xnnpack/config-types.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"

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

template <typename T, typename Operator>
const xnn_binary_elementwise_config* get_config(
    std::false_type = std::false_type()) {
  static xnn_binary_elementwise_config config = {
      (xnn_vbinary_ukernel_fn)binary_ukernel_unquantized<T, Operator>,
      (xnn_vbinary_ukernel_fn)binaryc_ukernel_unquantized<T, Operator>,
      (xnn_vbinary_ukernel_fn)rbinaryc_ukernel_unquantized<T, Operator>,
      nullptr,
      /*element_tile=*/1,
  };
  return &config;
}

template <typename T>
float dequantize(T x, float scale, int32_t zero_point) {
  return (static_cast<float>(x) - static_cast<float>(zero_point)) * scale;
}

template <typename T>
T quantize(float x, float inv_scale, int32_t zero_point) {
  const float q = x * inv_scale + zero_point;
  return std::lround(
      std::min<float>(std::max<float>(q, std::numeric_limits<T>::min()),
                      std::numeric_limits<T>::max()));
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

template <typename T, typename Operator>
const xnn_binary_elementwise_config* get_config(std::true_type) {
  static xnn_binary_elementwise_config config = {
      (xnn_vbinary_ukernel_fn)binary_ukernel_quantized<T, Operator>,
      (xnn_vbinary_ukernel_fn)binaryc_ukernel_quantized<T, Operator>,
      (xnn_vbinary_ukernel_fn)rbinaryc_ukernel_quantized<T, Operator>,
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
  int32_t operator()(int32_t a, int32_t b) const {
    return static_cast<int64_t>(a) + static_cast<int64_t>(b);
  }
};

template <typename T>
struct SubOp {
  T operator()(T a, T b) const { return a - b; }
};

template <>
struct SubOp<int32_t> {
  int32_t operator()(int32_t a, int32_t b) const {
    return static_cast<int64_t>(a) - static_cast<int64_t>(b);
  }
};

template <typename T>
struct MultiplyOp {
  T operator()(T a, T b) const { return a * b; }
};

template <>
struct MultiplyOp<int32_t> {
  int32_t operator()(int32_t a, int32_t b) const {
    return static_cast<int64_t>(a) * static_cast<int64_t>(b);
  }
};

template <typename T>
struct DivideOp {
  T operator()(T a, T b) const {
    return a / b;
  }
};

template <>
struct DivideOp<int32_t> {
  int32_t operator()(int32_t a, int32_t b) const {
    // This implements "Euclidean division", which is the way integer division
    // should be: (a / b) * b + r = a, where r is always in [0, |b|). This is
    // unlike "computer division" where, annoyingly, a / b is rounded towards 0,
    // and the remainder may be positive or negative accordingly. This
    // implementation of Euclidean integer division is taken from
    // https://github.com/dsharlet/slinky/blob/5020dae47ecb176bcd917ecd07d37e19615b955b/base/arithmetic.h#L12-L26
    if (b == 0) {
      return 0;
    }
    int32_t q = a / b;
    int32_t r = a - q * b;
    int32_t bs = b >> (sizeof(int32_t) * 8 - 1);
    int32_t rs = r >> (sizeof(int32_t) * 8 - 1);
    return q - (rs & bs) + (rs & ~bs);
  }
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

template <>
struct SquaredDifferenceOp<int32_t> {
  int32_t operator()(int32_t a, int32_t b) const {
    int32_t diff = static_cast<int64_t>(a) - static_cast<int64_t>(b);
    return static_cast<int64_t>(diff) * static_cast<int64_t>(diff);
  }
};

template <typename T>
struct PreluOp {
  T operator()(T a, T b) const { return (a < 0) ? static_cast<T>(a * b) : a; }
};

using std::copysign;

xnn_float16 copysign(xnn_float16 a, xnn_float16 b) {
  uint16_t a_bits = xnn_float16_to_bits(a);
  uint16_t b_bits = xnn_float16_to_bits(b);
  uint16_t sign_bit = b_bits & 0x8000;
  return xnn_float16_from_bits((a_bits & 0x7FFF) | (sign_bit & 0x8000));
}

template <typename T>
struct CopysignOp {
  T operator()(T a, T b) const { return copysign(a, b); }
};

#define DISPATCH_OPERATOR_FOR_DATATYPE(datatype, op)                         \
  switch (datatype) {                                                        \
    case xnn_datatype_fp32:                                                  \
      return get_config<float, op<float>>();                                 \
    case xnn_datatype_fp16:                                                  \
      return get_config<xnn_float16, op<xnn_float16>>();                     \
    case xnn_datatype_qint8:                                                 \
      return get_config<int8_t, op<float>>(/*quantized=*/std::true_type());  \
    case xnn_datatype_quint8:                                                \
      return get_config<uint8_t, op<float>>(/*quantized=*/std::true_type()); \
    case xnn_datatype_int32:                                                 \
      return get_config<int32_t, op<int32_t>>();                             \
    default:                                                                 \
      return nullptr;                                                        \
  }

#define DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, op)                    \
  switch (datatype) {                                                        \
    case xnn_datatype_fp32:                                                  \
      return get_config<float, op<float>>();                                 \
    case xnn_datatype_fp16:                                                  \
      return get_config<xnn_float16, op<xnn_float16>>();                     \
    case xnn_datatype_qint8:                                                 \
      return get_config<int8_t, op<float>>(/*quantized=*/std::true_type());  \
    case xnn_datatype_quint8:                                                \
      return get_config<uint8_t, op<float>>(/*quantized=*/std::true_type()); \
    default:                                                                 \
      return nullptr;                                                        \
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
      DISPATCH_OPERATOR_FOR_DATATYPE(datatype, SquaredDifferenceOp);
    case xnn_binary_invalid:
      return nullptr;
  }
}

}  // extern "C"
