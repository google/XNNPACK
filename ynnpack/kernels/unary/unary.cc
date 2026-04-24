// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/kernels/unary/unary.h"

#include <algorithm>
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
#include "ynnpack/include/ynnpack.h"

namespace ynn {

namespace {

// These "reference" microkernels are not designed to be fast, only to support
// all possible operators and types with reasonable performance. We just
// intend to give the compiler a reasonable chance at optimizing them.
template <typename TIn, typename TOut, typename Operator>
void unary_impl(size_t m, size_t n, size_t stride_x, const void* vx,
                size_t stride_y, void* vy, const unary_params* params) {
  auto x = reinterpret_cast<const TIn*>(vx);
  auto y = reinterpret_cast<TOut*>(vy);

  constexpr size_t unroll = std::max(type_info<TIn>::element_count(),
                                     type_info<TOut>::element_count());

  assert(n % unroll == 0);

  Operator op(*params);
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; j += unroll) {
      YNN_UNROLL
      for (size_t ji = 0; ji < unroll; ++ji) {
        auto y_j = static_cast<TOut>(op(type_info<TIn>::get(x, j + ji)));
        type_info<TOut>::set(y, j + ji, y_j);
      }
    }
    x = offset_bytes(x, stride_x);
    y = offset_bytes(y, stride_y);
  }
}

template <typename TOut>
struct convert_op {
  explicit convert_op(const unary_params& = {}) {}

  template <typename TIn>
  TOut operator()(TIn x) const {
    if constexpr (std::is_integral<TOut>::value) {
      if constexpr (std::is_integral<TIn>::value) {
        return saturate_cast<TOut>(x);
      } else {
        return round_float_to_int<TOut>(x);
      }
    } else {
      return static_cast<TOut>(x);
    }
  }

  // We need to give the compiler a little help for bf16 -> fp16
  TOut operator()(bfloat16 x) const {
    return operator()(static_cast<float>(x));
  }
};

template <typename TIn, typename TOut>
unary_kernel_fn get_convert_kernel(TIn, TOut) {
  return unary_impl<TIn, TOut, convert_op<TOut>>;
}

template <typename TIn>
unary_kernel_fn get_convert_kernel(ynn_type output) {
  switch (output) {
    case ynn_type_fp32:
      return get_convert_kernel(TIn(), float());
    case ynn_type_fp16:
      return get_convert_kernel(TIn(), half());
    case ynn_type_bf16:
      return get_convert_kernel(TIn(), bfloat16());
    case ynn_type_int8:
      return get_convert_kernel(TIn(), int8_t());
    case ynn_type_uint8:
      return get_convert_kernel(TIn(), uint8_t());
    case ynn_type_int32:
      return get_convert_kernel(TIn(), int32_t());
    default:
      return nullptr;
  }
}

struct abs_op {
  explicit abs_op(const unary_params& = {}) {}
  float operator()(float x) const { return std::abs(x); }
  int32_t operator()(int32_t x) const { return std::abs(x); }
};

struct negate_op {
  explicit negate_op(const unary_params& = {}) {}
  float operator()(float x) const { return -x; }
  int32_t operator()(int32_t x) const { return -x; }
};

struct round_op {
  explicit round_op(const unary_params& = {}) {}
  float operator()(float x) const { return std::nearbyint(x); }
};

struct ceil_op {
  explicit ceil_op(const unary_params& = {}) {}
  float operator()(float x) const { return std::ceil(x); }
};

struct floor_op {
  explicit floor_op(const unary_params& = {}) {}
  float operator()(float x) const { return std::floor(x); }
};

struct square_op {
  explicit square_op(const unary_params& = {}) {}
  float operator()(float x) const { return x * x; }
  int32_t operator()(int32_t x) const {
    return static_cast<int32_t>(static_cast<int64_t>(x) *
                                static_cast<int64_t>(x));
  }
};

struct square_root_op {
  explicit square_root_op(const unary_params& = {}) {}
  float operator()(float x) const { return std::sqrt(x); }
};

struct cube_root_op {
  explicit cube_root_op(const unary_params& = {}) {}
  float operator()(float x) const { return std::cbrt(x); }
};

struct tanh_op {
  tanh_params params;

  explicit tanh_op(const unary_params& params) : params(params.tanh) {}
  float operator()(float x) const {
    return std::tanh(x) * params.output_multiplier + params.output_offset;
  }
};

struct reciprocal_square_root_op {
  explicit reciprocal_square_root_op(const unary_params& = {}) {}
  float operator()(float x) const { return 1 / std::sqrt(x); }
};

struct log_op {
  log_params params;

  explicit log_op(const unary_params& params) : params(params.log) {}
  float operator()(float x) const {
    return std::log2(x * params.input_multiplier / std::sqrt(2.0f)) *
           params.output_multiplier;
  }
};

struct log1p_op {
  explicit log1p_op(const unary_params& = {}) {}
  float operator()(float x) const { return std::log1p(x); }
};

struct exp_op {
  exp_params params;

  explicit exp_op(const unary_params& params) : params(params.exp) {}
  float operator()(float x) const {
    return std::exp2(params.input_multiplier * x) * params.output_multiplier;
  }
};

struct expm1_op {
  explicit expm1_op(const unary_params& = {}) {}
  float operator()(float x) const { return std::expm1(x); }
};

struct erf_op {
  erf_params params;

  explicit erf_op(const unary_params& params) : params(params.erf) {}
  float operator()(float x) const {
    return std::erf(params.input_multiplier * x) * params.output_multiplier +
           params.output_offset;
  }
};

struct sign_op {
  explicit sign_op(const unary_params& = {}) {}
  float operator()(float x) const { return x < 0 ? -1 : x > 0 ? 1 : 0; }
  int32_t operator()(int32_t x) const { return x < 0 ? -1 : x > 0 ? 1 : 0; }
};

struct sine_op {
  sine_params params;

  explicit sine_op(const unary_params& params) : params(params.sine) {}
  float operator()(float x) const {
    return std::sin(x) * params.output_multiplier + params.output_offset;
  }
};

struct cosine_op {
  cosine_params params;

  explicit cosine_op(const unary_params& params) : params(params.cosine) {}
  float operator()(float x) const {
    return std::cos(x) * params.output_multiplier + params.output_offset;
  }
};

struct sigmoid_op {
  explicit sigmoid_op(const unary_params& = {}) {}
  float operator()(float x) const { return 1.0f / (1.0f + std::exp(-x)); }
};

struct hardswish_op {
  explicit hardswish_op(const unary_params& = {}) {}
  float operator()(float x) const {
    return (x * (1.0f / 6.0f)) * std::max(std::min(x + 3.0f, 6.0f), 0.0f);
  }
};

struct poly3_op {
  poly3_params params;

  explicit poly3_op(const unary_params& params) : params(params.poly3) {}
  float operator()(float x) const {
    return ((params.c3 * x + params.c2) * x + params.c1) * x + params.c0;
  }
};

}  // namespace

unary_kernel_fn get_unary_reference_kernel(ynn_unary_operator op,
                                           ynn_type type) {
  if (type == ynn_type_fp32) {
    switch (op) {
      case ynn_unary_abs:
        return unary_impl<float, float, abs_op>;
      case ynn_unary_round:
        return unary_impl<float, float, round_op>;
      case ynn_unary_ceil:
        return unary_impl<float, float, ceil_op>;
      case ynn_unary_exp:
        return unary_impl<float, float, exp_op>;
      case ynn_unary_expm1:
        return unary_impl<float, float, expm1_op>;
      case ynn_unary_erf:
        return unary_impl<float, float, erf_op>;
      case ynn_unary_floor:
        return unary_impl<float, float, floor_op>;
      case ynn_unary_log:
        return unary_impl<float, float, log_op>;
      case ynn_unary_log1p:
        return unary_impl<float, float, log1p_op>;
      case ynn_unary_negate:
        return unary_impl<float, float, negate_op>;
      case ynn_unary_square:
        return unary_impl<float, float, square_op>;
      case ynn_unary_square_root:
        return unary_impl<float, float, square_root_op>;
      case ynn_unary_reciprocal_square_root:
        return unary_impl<float, float, reciprocal_square_root_op>;
      case ynn_unary_tanh:
        return unary_impl<float, float, tanh_op>;
      case ynn_unary_cube_root:
        return unary_impl<float, float, cube_root_op>;
      case ynn_unary_sign:
        return unary_impl<float, float, sign_op>;
      case ynn_unary_sine:
        return unary_impl<float, float, sine_op>;
      case ynn_unary_cosine:
        return unary_impl<float, float, cosine_op>;
      case ynn_unary_sigmoid:
        return unary_impl<float, float, sigmoid_op>;
      case ynn_unary_hardswish:
        return unary_impl<float, float, hardswish_op>;
      case ynn_unary_poly3:
        return unary_impl<float, float, poly3_op>;
      case ynn_unary_convert:
      case ynn_unary_invalid:
        return nullptr;
    }
  } else if (type == ynn_type_int32) {
    switch (op) {
      case ynn_unary_abs:
        return unary_impl<int32_t, int32_t, abs_op>;
      case ynn_unary_negate:
        return unary_impl<int32_t, int32_t, negate_op>;
      case ynn_unary_square:
        return unary_impl<int32_t, int32_t, square_op>;
      case ynn_unary_sign:
        return unary_impl<int32_t, int32_t, sign_op>;
      default:
        return nullptr;
    }
  }
  return nullptr;
}

unary_kernel_fn get_convert_reference_kernel(ynn_type a_type, ynn_type x_type) {
  switch (a_type) {
    case ynn_type_fp32:
      return get_convert_kernel<float>(x_type);
    case ynn_type_fp16:
      return get_convert_kernel<half>(x_type);
    case ynn_type_bf16:
      return get_convert_kernel<bfloat16>(x_type);
    case ynn_type_int8:
      return get_convert_kernel<int8_t>(x_type);
    case ynn_type_uint8:
      return get_convert_kernel<uint8_t>(x_type);
    case ynn_type_int32:
      return get_convert_kernel<int32_t>(x_type);
    case ynn_type_int4:
      return get_convert_kernel<int4x2>(x_type);
    case ynn_type_int2:
      return get_convert_kernel<int2x4>(x_type);
    default:
      return nullptr;
  }
}

unary_kernel_fn get_unary_kernel(ynn_unary_operator op, ynn_type a_type,
                                 ynn_type x_type,
                                 uint64_t supported_arch_flags) {
  // TODO(vksnk): select a better kernel based on the passed size.
#define YNN_ELEMENTWISE_KERNEL(arch, name, op_type, type_a, type_x) \
  if (a_type == type_of<type_a>() && x_type == type_of<type_x>() && \
      op == ynn_unary_##op_type &&                                  \
      is_arch_supported(arch, supported_arch_flags)) {              \
    YNN_LOG_DEBUG() << "Using unary kernel " << #name;              \
    return &name;                                                   \
  }

#include "ynnpack/kernels/unary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

  if (op == ynn_unary_convert) {
    return get_convert_reference_kernel(a_type, x_type);
  } else if (a_type == ynn_type_fp32 && x_type == ynn_type_fp32) {
    return get_unary_reference_kernel(op, x_type);
  } else if (a_type == ynn_type_int32 && x_type == ynn_type_int32) {
    return get_unary_reference_kernel(op, x_type);
  } else {
    return nullptr;
  }
}

unary_params get_unary_params(ynn_unary_operator op) {
  switch (op) {
    case ynn_unary_exp:
      return unary_params{
          .exp = exp_params{
              ._ = 0.0f,
              .output_multiplier = 1.0f,
              .input_multiplier = static_cast<float>(std::log2(std::exp(1.0))),
          }};
    case ynn_unary_log:
      return unary_params{
          .log = log_params{
              ._ = 0.0f,
              .output_multiplier = static_cast<float>(std::log(2.0)),
              .input_multiplier = 1.4142134190e+00f,  // sqrt(2)
          }};
    case ynn_unary_erf:
      return unary_params{.erf = erf_params{.output_offset = 0.0f,
                                            .output_multiplier = 1.0f,
                                            .input_multiplier = 1.0f}};
    case ynn_unary_tanh:
      return unary_params{.tanh = tanh_params{.output_offset = 0.0f,
                                              .output_multiplier = 1.0f}};
    case ynn_unary_sine:
      return unary_params{.sine = sine_params{.output_offset = 0.0f,
                                              .output_multiplier = 1.0f}};
    case ynn_unary_cosine:
      return unary_params{.cosine = cosine_params{.output_offset = 0.0f,
                                                  .output_multiplier = 1.0f}};
    case ynn_unary_poly3:
      return unary_params{.poly3 = poly3_params{/*c0=*/0.0f, /*c1=*/0.0f,
                                                /*c2=*/0.0f, /*c3=*/0.0f}};
    default:
      return unary_params{};
  }

  return unary_params{};
}

}  // namespace ynn
