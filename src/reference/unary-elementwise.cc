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
template <typename TIn, typename TOut, typename Operator>
void unary_ukernel_unquantized(size_t input_batch_size_bytes, const TIn* x,
                               TOut* y, const xnn_unary_uparams* params) {
  const size_t batch_size = input_batch_size_bytes / sizeof(TIn);
  Operator op(params);
  for (size_t i = 0; i < batch_size; ++i) {
    y[i] = static_cast<TOut>(op(x[i]));
  }
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

template <typename TIn, typename TOut, typename Operator>
void unary_ukernel_quantized_input(size_t input_batch_size_bytes, const TIn* x,
                                   TOut* y, const xnn_unary_uparams* params) {
  const size_t batch_size = input_batch_size_bytes / sizeof(TIn);
  Operator op(params);
  for (size_t i = 0; i < batch_size; ++i) {
    const float x_i = dequantize(x[i], params->reference.x_scale,
                                 params->reference.x_zero_point);
    y[i] = static_cast<TOut>(op(x_i));
  }
}

template <typename TIn, typename TOut, typename Operator>
void unary_ukernel_quantized_output(size_t input_batch_size_bytes, const TIn* x,
                                    TOut* y, const xnn_unary_uparams* params) {
  const size_t batch_size = input_batch_size_bytes / sizeof(TIn);
  Operator op(params);
  for (size_t i = 0; i < batch_size; ++i) {
    const float result = op(x[i]);
    y[i] = quantize<TOut>(result, params->reference.inv_y_scale,
                          params->reference.y_zero_point);
  }
}

template <typename TIn, typename TOut, typename Operator>
void unary_ukernel_quantized(size_t input_batch_size_bytes, const TIn* x,
                             TOut* y, const xnn_unary_uparams* params) {
  const size_t batch_size = input_batch_size_bytes / sizeof(TIn);
  Operator op(params);
  for (size_t i = 0; i < batch_size; ++i) {
    const float x_i = dequantize(x[i], params->reference.x_scale,
                                 params->reference.x_zero_point);
    const float result = op(x_i);
    y[i] = quantize<TOut>(result, params->reference.inv_y_scale,
                          params->reference.y_zero_point);
  }
}

size_t init_reference_unary_params(
    xnn_unary_uparams* params, const xnn_unary_params* op_params,
    const struct xnn_quantization_params* input_quantization,
    const struct xnn_quantization_params* output_quantization) {
  if (input_quantization) {
    params->reference.x_scale = input_quantization->scale;
    params->reference.x_zero_point = input_quantization->zero_point;
  }
  if (output_quantization) {
    params->reference.inv_y_scale = 1.0f / output_quantization->scale;
    params->reference.y_zero_point = output_quantization->zero_point;
  }
  if (op_params) {
    params->reference.params = *op_params;
  }
  return sizeof(params->reference);
}

template <typename T, typename Operator>
const xnn_unary_elementwise_config* get_config(
    std::false_type /*quantized*/ = std::false_type()) {
  static xnn_unary_elementwise_config config = {
      (xnn_vunary_ukernel_fn)unary_ukernel_unquantized<T, T, Operator>,
      init_reference_unary_params,
  };
  return &config;
}

template <typename T, typename Operator>
const xnn_unary_elementwise_config* get_config(std::true_type /*quantized*/) {
  static xnn_unary_elementwise_config config = {
      (xnn_vunary_ukernel_fn)unary_ukernel_quantized<T, T, Operator>,
      init_reference_unary_params,
  };
  return &config;
}

template <typename TIn, typename TOut>
struct ConvertOp {
  explicit ConvertOp(const xnn_unary_uparams*) {}
  TOut operator()(TIn x) const {
    if (std::is_integral<TOut>::value && !std::is_integral<TIn>::value) {
      float x_f = static_cast<float>(x);
      x_f = std::min<float>(x_f, std::numeric_limits<TOut>::max());
      x_f = std::max<float>(x_f, std::numeric_limits<TOut>::min());
      return static_cast<TOut>(std::lround(x_f));
    } else {
      return static_cast<TOut>(x);
    }
  }
};

template <typename TIn, typename TOut>
const xnn_unary_elementwise_config* get_convert_config(
    std::true_type /*input_quantized*/, std::true_type /*output_quantized*/) {
  static xnn_unary_elementwise_config config = {
      (xnn_vunary_ukernel_fn)
          unary_ukernel_quantized<TIn, TOut, ConvertOp<float, float>>,
      init_reference_unary_params,
  };
  return &config;
}

template <typename TIn, typename TOut>
const xnn_unary_elementwise_config* get_convert_config(
    std::true_type /*input_quantized*/, std::false_type /*output_quantized*/) {
  static xnn_unary_elementwise_config config = {
      (xnn_vunary_ukernel_fn)
          unary_ukernel_quantized_input<TIn, TOut, ConvertOp<float, TOut>>,
      init_reference_unary_params,
  };
  return &config;
}

template <typename TIn, typename TOut>
const xnn_unary_elementwise_config* get_convert_config(
    std::false_type /*input_quantized*/, std::true_type /*output_quantized*/) {
  static xnn_unary_elementwise_config config = {
      (xnn_vunary_ukernel_fn)
          unary_ukernel_quantized_output<TIn, TOut, ConvertOp<TIn, float>>,
      init_reference_unary_params,
  };
  return &config;
}

template <typename TIn, typename TOut>
const xnn_unary_elementwise_config* get_convert_config(
    std::false_type /*input_quantized*/, std::false_type /*output_quantized*/) {
  static xnn_unary_elementwise_config config = {
      (xnn_vunary_ukernel_fn)
          unary_ukernel_unquantized<TIn, TOut, ConvertOp<TIn, TOut>>,
      init_reference_unary_params,
  };
  return &config;
}

template <typename TIn, typename InputQuantized>
const xnn_unary_elementwise_config* get_convert_config(xnn_datatype output,
                                                       InputQuantized) {
  switch (output) {
    case xnn_datatype_fp32:
      return get_convert_config<TIn, float>(
          InputQuantized(),
          /*output_quantized=*/std::false_type());
    case xnn_datatype_fp16:
      return get_convert_config<TIn, xnn_float16>(
          InputQuantized(),
          /*output_quantized=*/std::false_type());
    case xnn_datatype_qint8:
      return get_convert_config<TIn, int8_t>(
          InputQuantized(),
          /*output_quantized=*/std::true_type());
    case xnn_datatype_quint8:
      return get_convert_config<TIn, uint8_t>(
          InputQuantized(),
          /*output_quantized=*/std::true_type());
    case xnn_datatype_int32:
      return get_convert_config<TIn, int32_t>(
          InputQuantized(),
          /*output_quantized=*/std::false_type());
    default:
      return nullptr;
  }
}

const xnn_unary_elementwise_config* get_convert_config(xnn_datatype input,
                                                       xnn_datatype output) {
  switch (input) {
    case xnn_datatype_fp32:
      return get_convert_config<float>(output,
                                       /*input_quantized=*/std::false_type());
    case xnn_datatype_fp16:
      return get_convert_config<xnn_float16>(
          output, /*input_quantized=*/std::false_type());
    case xnn_datatype_qint8:
      return get_convert_config<int8_t>(output,
                                        /*input_quantized=*/std::true_type());
    case xnn_datatype_quint8:
      return get_convert_config<uint8_t>(output,
                                         /*input_quantized=*/std::true_type());
    case xnn_datatype_int32:
      return get_convert_config<int32_t>(output,
                                         /*input_quantized=*/std::false_type());
    default:
      return nullptr;
  }
}


template <typename T>
struct AbsOp {
  explicit AbsOp(const xnn_unary_uparams*) {}

  float operator()(float x) const { return std::abs(x); }
};

template <typename T>
struct ClampOp {
  T min;
  T max;

  explicit ClampOp(const xnn_unary_uparams* params)
      : min(static_cast<T>(params->reference.params.clamp.min)),
        max(static_cast<T>(params->reference.params.clamp.max)) {}

  T operator()(T x) const { return std::min(std::max(x, min), max); }
};

template <typename T>
struct ELUOp {
  float alpha;

  explicit ELUOp(const xnn_unary_uparams* params)
      : alpha(params->reference.params.elu.alpha) {}

  float operator()(float x) const { return x < 0 ? alpha * std::expm1(x) : x; }
};

template <typename T>
struct GELUOp {
  explicit GELUOp(const xnn_unary_uparams*) {}

  T operator()(T x) const {
    return static_cast<T>((x / 2) * (1 + std::erf(x * std::sqrt(2) / 2)));
  }
};

template <typename T>
struct HardSwishOp {
  explicit HardSwishOp(const xnn_unary_uparams*) {}

  T operator()(T x) const {
    return static_cast<T>((x / 6) * std::max<T>(std::min<T>(x + 3, 6), 0));
  }
};

template <typename T>
struct LeakyReLUOp {
  float negative_slope;

  explicit LeakyReLUOp(const xnn_unary_uparams* params)
      : negative_slope(params->reference.params.leaky_relu.negative_slope) {}

  T operator()(T x) const {
    return x < 0 ? static_cast<T>(x * negative_slope) : x;
  }
};

template <typename T>
struct NegateOp {
  explicit NegateOp(const xnn_unary_uparams*) {}

  T operator()(T x) const { return -x; }
};

template <typename T>
struct ReLUOp {
  explicit ReLUOp(const xnn_unary_uparams*) {}

  T operator()(T x) const { return std::max<T>(x, 0); }
};

template <typename T>
struct RoundToNearestOp {
  explicit RoundToNearestOp(const xnn_unary_uparams*) {}

  float operator()(float x) const { return std::nearbyint(x); }
};

template <typename T>
struct RoundTowardsZeroOp {
  explicit RoundTowardsZeroOp(const xnn_unary_uparams*) {}

  T operator()(T x) const { return std::trunc(x); }
};

template <typename T>
struct RoundUpOp {
  explicit RoundUpOp(const xnn_unary_uparams*) {}

  float operator()(float x) const { return std::ceil(x); }
};

template <typename T>
struct RoundDownOp {
  explicit RoundDownOp(const xnn_unary_uparams*) {}

  float operator()(float x) const { return std::floor(x); }
};

template <typename T>
struct SigmoidOp {
  explicit SigmoidOp(const xnn_unary_uparams*) {}

  T operator()(T x) const {
    if (x > 100) {
      return 1;
    } else if (x < -100) {
      return 0;
    } else {
      const double e = std::exp(static_cast<double>(x));
      return static_cast<T>(e / (1 + e));
    }
  }
};

template <typename T>
struct SquareOp {
  explicit SquareOp(const xnn_unary_uparams*) {}

  T operator()(T x) const { return x * x; }
};
template <>
struct SquareOp<int32_t> {
  explicit SquareOp(const xnn_unary_uparams*) {}

  int32_t operator()(int32_t x) const { return (int64_t)x * (int64_t)x; }
};

template <typename T>
struct SquareRootOp {
  explicit SquareRootOp(const xnn_unary_uparams*) {}

  float operator()(float x) const { return std::sqrt(x); }
};

template <typename T>
struct TanHOp {
  explicit TanHOp(const xnn_unary_uparams*) {}

  float operator()(float x) const { return std::tanh(x); }
};

template <typename T>
struct ReciprocalSquareRootOp {
  explicit ReciprocalSquareRootOp(const xnn_unary_uparams*) {}

  float operator()(float x) const { return 1 / std::sqrt(x); }
};

template <typename T>
struct LogOp {
  explicit LogOp(const xnn_unary_uparams*) {}

  float operator()(float x) const { return std::log(x); }
};

template <typename T>
struct ExpOp {
  explicit ExpOp(const xnn_unary_uparams*) {}

  float operator()(float x) const { return std::exp(x); }
};

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

const xnn_unary_elementwise_config* get_config(xnn_unary_operator op,
                                               xnn_datatype datatype) {
  switch (op) {
    case xnn_unary_clamp:
      DISPATCH_OPERATOR_FOR_DATATYPE(datatype, ClampOp);
    case xnn_unary_abs:
      DISPATCH_OPERATOR_FOR_DATATYPE(datatype, AbsOp);
    case xnn_unary_bankers_rounding:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, RoundToNearestOp);
    case xnn_unary_ceiling:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, RoundUpOp);
    case xnn_unary_elu:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, ELUOp);
    case xnn_unary_exp:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, ExpOp);
    case xnn_unary_floor:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, RoundDownOp);
    case xnn_unary_gelu:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, GELUOp);
    case xnn_unary_hardswish:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, HardSwishOp);
    case xnn_unary_leaky_relu:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, LeakyReLUOp);
    case xnn_unary_log:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, LogOp);
    case xnn_unary_negate:
      DISPATCH_OPERATOR_FOR_DATATYPE(datatype, NegateOp);
    case xnn_unary_sigmoid:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, SigmoidOp);
    case xnn_unary_square:
      DISPATCH_OPERATOR_FOR_DATATYPE(datatype, SquareOp);
    case xnn_unary_square_root:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, SquareRootOp);
    case xnn_unary_reciprocal_square_root:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, ReciprocalSquareRootOp);
    case xnn_unary_tanh:
      DISPATCH_OPERATOR_FOR_REAL_DATATYPE(datatype, TanHOp);
    default:
      return nullptr;
  }
}

}  // namespace

extern "C" {

// If we could guarantee that xnn_init_*_config did not return NULL, we could
// only support reference configs for the subset of ops/datatypes that we don't
// have a microkernel for. But, that is not the case, so we need the full set of
// reference ops implemented.
const xnn_unary_elementwise_config* xnn_init_unary_reference_config(
    xnn_unary_operator op, xnn_datatype input_datatype,
    xnn_datatype output_datatype) {
  if (op == xnn_unary_convert) {
    return get_convert_config(input_datatype, output_datatype);
  } else if (input_datatype == output_datatype) {
    return get_config(op, input_datatype);
  } else {
    return nullptr;
  }
}

}  // extern "C"
