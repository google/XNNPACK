// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/unary/unary.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

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
template <typename TIn, typename TOut, typename Operator>
void unquantized(size_t m, size_t n, size_t stride_x, const void* vx,
                 size_t stride_y, void* vy, const unary_params* params) {
  auto x = reinterpret_cast<const TIn*>(vx);
  auto y = reinterpret_cast<TOut*>(vy);

  Operator op(params);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      y[j] = static_cast<TOut>(op(x[j]));
    }
    x = offset_bytes(x, stride_x);
    y = offset_bytes(y, stride_y);
  }
}

template <typename TIn, typename TOut, typename Operator>
void quantized_input(size_t m, size_t n, size_t stride_x, const void* vx,
                     size_t stride_y, void* vy, const unary_params* params) {
  assert(params);
  auto x = reinterpret_cast<const TIn*>(vx);
  auto y = reinterpret_cast<TOut*>(vy);
  Operator op(params);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      const float x_j = dequantize(x[j], params->reference.a_scale,
                                   params->reference.a_zero_point);
      y[j] = static_cast<TOut>(op(x_j));
    }
    x = offset_bytes(x, stride_x);
    y = offset_bytes(y, stride_y);
  }
}

template <typename TIn, typename TOut, typename Operator>
void quantized_output(size_t m, size_t n, size_t stride_x, const void* vx,
                      size_t stride_y, void* vy, const unary_params* params) {
  assert(params);
  auto x = reinterpret_cast<const TIn*>(vx);
  auto y = reinterpret_cast<TOut*>(vy);
  Operator op(params);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      const float result = op(x[j]);
      y[j] = quantize<TOut>(result, params->reference.inv_x_scale,
                            params->reference.x_zero_point);
    }
    x = offset_bytes(x, stride_x);
    y = offset_bytes(y, stride_y);
  }
}

template <typename TIn, typename TOut, typename Operator>
void quantized_input_output(size_t m, size_t n, size_t stride_x, const void* vx,
                            size_t stride_y, void* vy,
                            const unary_params* params) {
  assert(params);
  auto x = reinterpret_cast<const TIn*>(vx);
  auto y = reinterpret_cast<TOut*>(vy);

  Operator op(params);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      const float x_j = dequantize(x[j], params->reference.a_scale,
                                   params->reference.a_zero_point);
      const float result = op(x_j);
      y[j] = quantize<TOut>(result, params->reference.inv_x_scale,
                            params->reference.x_zero_point);
    }
    x = offset_bytes(x, stride_x);
    y = offset_bytes(y, stride_y);
  }
}

void init_reference_unary_params(float a_scale, int32_t a_zero_point,
                                 float x_scale, int32_t x_zero_point,
                                 unary_params& params) {
  params.reference.a_scale = a_scale;
  params.reference.a_zero_point = a_zero_point;
  params.reference.inv_x_scale = 1.0f / x_scale;
  params.reference.x_zero_point = x_zero_point;
}

template <typename Operator, typename T>
const unary_kernel& get_kernel(T) {
  static unary_kernel kernel = {
      &unquantized<T, T, Operator>,
      nullptr,
  };
  return kernel;
}

template <typename Operator, typename T>
const unary_kernel& get_kernel(quantized<T>) {
  static unary_kernel kernel = {
      &quantized_input_output<quantized<T>, quantized<T>, Operator>,
      init_reference_unary_params,
  };
  return kernel;
}

template <typename TIn, typename TOut>
struct ConvertOp {
  explicit ConvertOp(const unary_params*) {}
  TOut operator()(TIn x) const {
    if (std::is_integral<TOut>::value && !std::is_integral<TIn>::value) {
      return round_float_to_int<TOut>(x);
    } else {
      return static_cast<TOut>(x);
    }
  }
};

#if XNN_HAVE_FLOAT16
template <>
struct ConvertOp<bfloat16, _Float16> {
  explicit ConvertOp(const unary_params*) {}
  _Float16 operator()(bfloat16 x) const {
    return static_cast<_Float16>(static_cast<float>(x));
  }
};
#endif

template <typename TIn, typename TOut>
const unary_kernel& get_convert_kernel(quantized<TIn>, quantized<TOut>) {
  static unary_kernel kernel = {
      &quantized_input_output<quantized<TIn>, quantized<TOut>,
                              ConvertOp<float, float>>,
      init_reference_unary_params,
  };
  return kernel;
}

template <typename TIn, typename TOut>
const unary_kernel& get_convert_kernel(quantized<TIn>, TOut) {
  static unary_kernel kernel = {
      &quantized_input<quantized<TIn>, TOut, ConvertOp<float, TOut>>,
      init_reference_unary_params,
  };
  return kernel;
}

template <typename TIn, typename TOut>
const unary_kernel& get_convert_kernel(TIn, quantized<TOut>) {
  static unary_kernel kernel = {
      &quantized_output<TIn, quantized<TOut>, ConvertOp<TIn, float>>,
      init_reference_unary_params,
  };
  return kernel;
}

template <typename TIn, typename TOut>
const unary_kernel& get_convert_kernel(TIn, TOut) {
  static unary_kernel kernel = {
      &unquantized<TIn, TOut, ConvertOp<TIn, TOut>>,
      nullptr,
  };
  return kernel;
}

template <typename TIn>
const unary_kernel* get_convert_kernel(ynn_type output, bool output_quantized) {
  switch (output) {
    case ynn_type_fp32:
      return &get_convert_kernel(TIn(), float());
    case ynn_type_fp16:
      return &get_convert_kernel(TIn(), half());
    case ynn_type_bf16:
      return &get_convert_kernel(TIn(), bfloat16());
    case ynn_type_int8:
      return &get_convert_kernel(TIn(), quantized<int8_t>());
    case ynn_type_uint8:
      return &get_convert_kernel(TIn(), quantized<uint8_t>());
    case ynn_type_int32:
      return output_quantized ? &get_convert_kernel(TIn(), quantized<int32_t>())
                              : &get_convert_kernel(TIn(), int32_t());
    default:
      return nullptr;
  }
}

const unary_kernel* get_convert_kernel(ynn_type input, bool input_quantized,
                                       ynn_type output, bool output_quantized) {
  switch (input) {
    case ynn_type_fp32:
      return get_convert_kernel<float>(output, output_quantized);
    case ynn_type_fp16:
      return get_convert_kernel<half>(output, output_quantized);
    case ynn_type_bf16:
      return get_convert_kernel<bfloat16>(output, output_quantized);
    case ynn_type_int8:
      return get_convert_kernel<quantized<int8_t>>(output, output_quantized);
    case ynn_type_uint8:
      return get_convert_kernel<quantized<uint8_t>>(output, output_quantized);
    case ynn_type_int32:
      return input_quantized
                 ? get_convert_kernel<quantized<int32_t>>(output,
                                                          output_quantized)
                 : get_convert_kernel<int32_t>(output, output_quantized);
    default:
      return nullptr;
  }
}

template <typename T>
struct abs_op {
  explicit abs_op(const unary_params*) {}

  int32_t operator()(int32_t x) const { return std::abs(x); }
  float operator()(float x) const { return std::abs(x); }
  half operator()(half x) const {
    return half::from_bits(x.to_bits() & 0x7fff);
  }
  bfloat16 operator()(bfloat16 x) const {
    return bfloat16::from_bits(x.to_bits() & 0x7fff);
  }
};

template <typename T>
struct negate_op {
  explicit negate_op(const unary_params*) {}

  static const uint16_t sign_mask = 0x8000;
  int32_t operator()(int32_t x) const { return -x; }
  float operator()(float x) const { return -x; }
  half operator()(half x) const {
    return half::from_bits(x.to_bits() ^ sign_mask);
  }
  bfloat16 operator()(bfloat16 x) const {
    return bfloat16::from_bits(x.to_bits() ^ sign_mask);
  }
};

template <typename T>
struct round_op {
  explicit round_op(const unary_params*) {}

  float operator()(float x) const { return std::nearbyint(x); }
};

template <typename T>
struct ceil_op {
  explicit ceil_op(const unary_params*) {}

  float operator()(float x) const { return std::ceil(x); }
};

template <typename T>
struct floor_op {
  explicit floor_op(const unary_params*) {}

  float operator()(float x) const { return std::floor(x); }
};

template <typename T>
struct square_op {
  explicit square_op(const unary_params*) {}

  T operator()(T x) const { return x * x; }
};
template <>
struct square_op<int32_t> {
  explicit square_op(const unary_params*) {}

  int32_t operator()(int32_t x) const { return (int64_t)x * (int64_t)x; }
};

template <typename T>
struct square_root_op {
  explicit square_root_op(const unary_params*) {}

  float operator()(float x) const { return std::sqrt(x); }
};

template <typename T>
struct cube_root_op {
  explicit cube_root_op(const unary_params*) {}

  float operator()(float x) const { return std::cbrt(x); }
};

template <typename T>
struct tanh_op {
  explicit tanh_op(const unary_params*) {}

  float operator()(float x) const { return std::tanh(x); }
};

template <typename T>
struct reciprocal_square_root_op {
  explicit reciprocal_square_root_op(const unary_params*) {}

  float operator()(float x) const { return 1 / std::sqrt(x); }
};

template <typename T>
struct log_op {
  explicit log_op(const unary_params*) {}

  float operator()(float x) const { return std::log(x); }
};

template <typename T>
struct log1p_op {
  explicit log1p_op(const unary_params*) {}

  float operator()(float x) const { return std::log1p(x); }
};

template <typename T>
struct exp_op {
  explicit exp_op(const unary_params*) {}

  float operator()(float x) const { return std::exp(x); }
};

template <typename T>
struct expm1_op {
  explicit expm1_op(const unary_params*) {}

  float operator()(float x) const { return std::expm1(x); }
};

template <typename T>
struct erf_op {
  explicit erf_op(const unary_params*) {}

  float operator()(float x) const { return std::erf(x); }
};

template <typename T>
struct sign_op {
  explicit sign_op(const unary_params*) {}

  int32_t operator()(int32_t x) const { return x < 0 ? -1 : x > 0 ? 1 : 0; }
  float operator()(float x) const { return x < 0 ? -1 : x > 0 ? 1 : 0; }

  static const uint16_t sign_mask = 0x8000;
  half operator()(half x) const {
    uint16_t sign = x.to_bits() & sign_mask;
    static constexpr uint16_t one = 0x3c00;
    return x.is_zero() ? x : half::from_bits(one | sign);
  }
  bfloat16 operator()(bfloat16 x) const {
    uint16_t sign = x.to_bits() & sign_mask;
    static constexpr uint16_t one = 0x3f80;
    return x.is_zero() ? x : bfloat16::from_bits(one | sign);
  }
};

template <typename T>
struct sine_op {
  explicit sine_op(const unary_params*) {}

  float operator()(float x) const { return std::sin(x); }
};

template <typename T>
struct cosine_op {
  explicit cosine_op(const unary_params*) {}

  float operator()(float x) const { return std::cos(x); }
};

template <typename T>
struct sigmoid_op {
  explicit sigmoid_op(const unary_params*) {}

  float operator()(float x) const {
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
struct hardswish_op {
  explicit hardswish_op(const unary_params*) {}

  T operator()(T x) const {
    return static_cast<T>((x / 6) * std::max<T>(std::min<T>(x + 3, 6), 0));
  }
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

#define DISPATCH_OPERATOR_FOR_INTEGRAL_TYPE(type, op) \
  switch (type) {                                     \
    case ynn_type_int32:                              \
      return &get_kernel<op<int32_t>>(int32_t());     \
    default:                                          \
      return nullptr;                                 \
  }

const unary_kernel* get_unary_reference_kernel(ynn_unary_operator op,
                                               ynn_type type,
                                               bool is_quantized) {
  switch (op) {
    case ynn_unary_abs:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, abs_op);
    case ynn_unary_round:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, round_op);
    case ynn_unary_ceil:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, ceil_op);
    case ynn_unary_exp:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, exp_op);
    case ynn_unary_expm1:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, expm1_op);
    case ynn_unary_erf:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, erf_op);
    case ynn_unary_floor:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, floor_op);
    case ynn_unary_log:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, log_op);
    case ynn_unary_log1p:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, log1p_op);
    case ynn_unary_negate:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, negate_op);
    case ynn_unary_square:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, square_op);
    case ynn_unary_square_root:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, square_root_op);
    case ynn_unary_reciprocal_square_root:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, reciprocal_square_root_op);
    case ynn_unary_tanh:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, tanh_op);
    case ynn_unary_cube_root:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, cube_root_op);
    case ynn_unary_sign:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, sign_op);
    case ynn_unary_sine:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, sine_op);
    case ynn_unary_cosine:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, cosine_op);
    case ynn_unary_sigmoid:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, sigmoid_op);
    case ynn_unary_hardswish:
      DISPATCH_OPERATOR_FOR_TYPE(type, is_quantized, hardswish_op);
    default:
      return nullptr;
  }
}

}  // namespace

// If we could guarantee that ynn_init_*_config did not return NULL, we could
// only support reference configs for the subset of ops/types that we don't
// have a microkernel for. But, that is not the case, so we need the full set of
// reference ops implemented.
const unary_kernel* get_unary_reference_kernel(ynn_unary_operator op,
                                               ynn_type a_type,
                                               bool a_quantized,
                                               ynn_type x_type,
                                               bool x_quantized) {
  if (op == ynn_unary_convert) {
    return get_convert_kernel(a_type, a_quantized, x_type, x_quantized);
  } else if (a_type == x_type) {
    return get_unary_reference_kernel(op, a_type, a_quantized);
  } else {
    return nullptr;
  }
}

const unary_kernel* get_unary_kernel(ynn_unary_operator op, ynn_type a_type,
                                     bool a_quantized, ynn_type x_type,
                                     bool x_quantized,
                                     uint64_t supported_arch_flags) {
  // TODO(vksnk): select a better kernel based on the passed size.
#define YNN_ELEMENTWISE_KERNEL(arch, name, op_type, init_params_fn, type_a, \
                               type_x)                                      \
  if (a_type == type_of<type_a>() && x_type == type_of<type_x>() &&         \
      op == ynn_unary_##op_type &&                                          \
      is_arch_supported(arch, supported_arch_flags)) {                      \
    static unary_kernel kernel##name = {&name, nullptr};                    \
    YNN_LOG_INFO() << "Using unary kernel " << #name;                       \
    return &kernel##name;                                                   \
  }

#include "ynnpack/kernels/unary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

  return get_unary_reference_kernel(op, a_type, a_quantized, x_type,
                                    x_quantized);
}

}  // namespace ynn
