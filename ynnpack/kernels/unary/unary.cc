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

#include "ynnpack/base/arch.h"
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/bit_cast.h"
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
void unary_impl(size_t m, size_t n, size_t stride_x, const void* vx,
                size_t stride_y, void* vy, const unary_params* params) {
  auto x = reinterpret_cast<const TIn*>(vx);
  auto y = reinterpret_cast<TOut*>(vy);

  constexpr size_t unroll = std::max(type_info<TIn>::element_count(),
                                     type_info<TOut>::element_count());

  Operator op(*params);
  for (size_t i = 0; i < m; ++i) {
    size_t j = 0;
    for (; j + unroll <= n; j += unroll) {
      YNN_UNROLL
      for (size_t ji = 0; ji < unroll; ++ji) {
        auto y_j = static_cast<TOut>(op(type_info<TIn>::get(x, j + ji)));
        type_info<TOut>::set(y, j + ji, y_j);
      }
    }
    for (; j < n; ++j) {
      auto y_j = static_cast<TOut>(op(type_info<TIn>::get(x, j)));
      type_info<TOut>::set(y, j, y_j);
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
    return cast<TOut>(x);
  }
};

template <typename TIn, typename TOut>
unary_kernel_fn get_convert_kernel(TIn, TOut) {
  return unary_impl<TIn, TOut, convert_op<TOut>>;
}

template <typename TIn>
unary_kernel_fn get_convert_kernel(ynn_type output) {
  switch (output) {
    case ynn_type_fp64:
      return get_convert_kernel(TIn(), double());
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
  double operator()(double x) const { return std::abs(x); }
  int32_t operator()(int32_t x) const { return std::abs(x); }
};

struct negate_op {
  explicit negate_op(const unary_params& = {}) {}
  float operator()(float x) const { return -x; }
  double operator()(double x) const { return -x; }
  int32_t operator()(int32_t x) const { return -x; }
};

struct round_op {
  explicit round_op(const unary_params& = {}) {}
  float operator()(float x) const { return std::nearbyint(x); }
  double operator()(double x) const { return std::nearbyint(x); }
};

struct round_to_bf16_op {
  explicit round_to_bf16_op(const unary_params& = {}) {}

  float operator()(float x) const {
    return static_cast<float>(bfloat16(x));
  }
};

struct ceil_op {
  explicit ceil_op(const unary_params& = {}) {}
  float operator()(float x) const { return std::ceil(x); }
  double operator()(double x) const { return std::ceil(x); }
};

struct floor_op {
  explicit floor_op(const unary_params& = {}) {}
  float operator()(float x) const { return std::floor(x); }
  double operator()(double x) const { return std::floor(x); }
};

struct square_op {
  explicit square_op(const unary_params& = {}) {}
  float operator()(float x) const { return x * x; }
  double operator()(double x) const { return x * x; }
  int32_t operator()(int32_t x) const {
    return static_cast<int32_t>(static_cast<int64_t>(x) *
                                static_cast<int64_t>(x));
  }
};

struct square_root_op {
  explicit square_root_op(const unary_params& = {}) {}
  float operator()(float x) const { return std::sqrt(x); }
  double operator()(double x) const { return std::sqrt(x); }
};

struct cube_root_op {
  explicit cube_root_op(const unary_params& = {}) {}
  float operator()(float x) const { return std::cbrt(x); }
  double operator()(double x) const { return std::cbrt(x); }
};

struct tanh_op {
  tanh_params params;

  explicit tanh_op(const unary_params& params) : params(params.tanh) {}
  float operator()(float x) const {
    return std::tanh(x) * static_cast<float>(params.output_multiplier) +
           static_cast<float>(params.output_offset);
  }
  double operator()(double x) const {
    return std::tanh(x) * params.output_multiplier + params.output_offset;
  }
};

struct reciprocal_square_root_op {
  explicit reciprocal_square_root_op(const unary_params& = {}) {}
  float operator()(float x) const { return 1.0f / std::sqrt(x); }
  double operator()(double x) const { return 1.0 / std::sqrt(x); }
};

struct log_op {
  log_params params;

  explicit log_op(const unary_params& params) : params(params.log) {}
  float operator()(float x) const {
    return std::log(x * static_cast<float>(params.input_multiplier)) *
           static_cast<float>(params.output_multiplier);
  }
  double operator()(double x) const {
    return std::log(x * params.input_multiplier) * params.output_multiplier;
  }
};

struct log1p_op {
  log1p_params params;

  explicit log1p_op(const unary_params& params) : params(params.log1p) {}
  float operator()(float x) const {
    return std::log1p(x * static_cast<float>(params.input_multiplier)) *
           static_cast<float>(params.output_multiplier);
  }
  double operator()(double x) const {
    return std::log1p(x * params.input_multiplier) * params.output_multiplier;
  }
};

struct exp_op {
  exp_params params;

  explicit exp_op(const unary_params& params) : params(params.exp) {}
  float operator()(float x) const {
    return std::exp(static_cast<float>(params.input_multiplier) * x) *
           static_cast<float>(params.output_multiplier);
  }
  double operator()(double x) const {
    return std::exp(params.input_multiplier * x) * params.output_multiplier;
  }
};

struct expm1_op {
  exp_params params;

  explicit expm1_op(const unary_params& params) : params(params.expm1) {}
  float operator()(float x) const {
    return std::expm1(static_cast<float>(params.input_multiplier) * x) *
           static_cast<float>(params.output_multiplier);
  }
  double operator()(double x) const {
    return std::expm1(params.input_multiplier * x) * params.output_multiplier;
  }
};

struct erf_op {
  erf_params params;

  explicit erf_op(const unary_params& params) : params(params.erf) {}
  float operator()(float x) const {
    return std::erf(static_cast<float>(params.input_multiplier) * x) *
               static_cast<float>(params.output_multiplier) +
           static_cast<float>(params.output_offset);
  }
  double operator()(double x) const {
    return std::erf(params.input_multiplier * x) * params.output_multiplier +
           params.output_offset;
  }
};

struct sign_op {
  explicit sign_op(const unary_params& = {}) {}
  float operator()(float x) const {
    if (std::isnan(x)) return x;
    if (x < 0.0f) return -1.0f;
    if (x > 0.0f) return 1.0f;
    return 0.0f;
  }
  double operator()(double x) const {
    if (std::isnan(x)) return x;
    if (x < 0.0) return -1.0;
    if (x > 0.0) return 1.0;
    return 0.0;
  }
  int32_t operator()(int32_t x) const { return x < 0 ? -1 : x > 0 ? 1 : 0; }
};

struct sine_op {
  sine_params params;

  explicit sine_op(const unary_params& params) : params(params.sine) {}
  float operator()(float x) const {
    return std::sin(x) * static_cast<float>(params.output_multiplier) +
           static_cast<float>(params.output_offset);
  }
  double operator()(double x) const {
    return std::sin(x) * params.output_multiplier + params.output_offset;
  }
};

struct cosine_op {
  cosine_params params;

  explicit cosine_op(const unary_params& params) : params(params.cosine) {}
  float operator()(float x) const {
    return std::cos(x) * static_cast<float>(params.output_multiplier) +
           static_cast<float>(params.output_offset);
  }
  double operator()(double x) const {
    return std::cos(x) * params.output_multiplier + params.output_offset;
  }
};

struct sigmoid_op {
  explicit sigmoid_op(const unary_params& = {}) {}
  float operator()(float x) const { return 1.0f / (1.0f + std::exp(-x)); }
  double operator()(double x) const { return 1.0 / (1.0 + std::exp(-x)); }
};

struct hardswish_op {
  explicit hardswish_op(const unary_params& = {}) {}
  float operator()(float x) const {
    return (x * (1.0f / 6.0f)) * std::max(std::min(x + 3.0f, 6.0f), 0.0f);
  }
  double operator()(double x) const {
    return (x * (1.0 / 6.0)) * std::max(std::min(x + 3.0, 6.0), 0.0);
  }
};

struct poly3_op {
  poly3_params params;

  explicit poly3_op(const unary_params& params) : params(params.poly3) {}
  float operator()(float x) const {
    return ((static_cast<float>(params.c3) * x +
             static_cast<float>(params.c2)) *
                x +
            static_cast<float>(params.c1)) *
               x +
           static_cast<float>(params.c0);
  }
  double operator()(double x) const {
    return ((params.c3 * x + params.c2) * x + params.c1) * x + params.c0;
  }
};

template <typename T>
unary_kernel_fn get_float_unary_reference_kernel(ynn_unary_operator op,
                                                 uint32_t required_flags) {
  if (required_flags & unary_flag::consistent_arithmetic) {
    switch (op) {
      case ynn_unary_abs:
      case ynn_unary_round:
      case ynn_unary_ceil:
      case ynn_unary_floor:
      case ynn_unary_negate:
#ifndef YNN_ARCH_ARM32
      // 32-bit ARM handles denormals differently between scalar and NEON.
      case ynn_unary_square:
#endif  // !YNN_ARCH_ARM32
      case ynn_unary_sign:
      case ynn_unary_round_to_bf16:
      case ynn_unary_convert:
        // We assume these kernels are exact and thus numerically consistent.
        break;
      default:
        // We call into cmath, which might not be numerically consistent.
        return nullptr;
    }
  }
  switch (op) {
    case ynn_unary_abs:
      return unary_impl<T, T, abs_op>;
    case ynn_unary_round:
      return unary_impl<T, T, round_op>;
    case ynn_unary_ceil:
      return unary_impl<T, T, ceil_op>;
    case ynn_unary_exp:
      return unary_impl<T, T, exp_op>;
    case ynn_unary_expm1:
      return unary_impl<T, T, expm1_op>;
    case ynn_unary_erf:
      return unary_impl<T, T, erf_op>;
    case ynn_unary_approx_erf:
      return unary_impl<T, T, erf_op>;
    case ynn_unary_approx_tanh:
      return unary_impl<T, T, tanh_op>;
    case ynn_unary_floor:
      return unary_impl<T, T, floor_op>;
    case ynn_unary_log:
      return unary_impl<T, T, log_op>;
    case ynn_unary_log1p:
      return unary_impl<T, T, log1p_op>;
    case ynn_unary_negate:
      return unary_impl<T, T, negate_op>;
    case ynn_unary_square:
      return unary_impl<T, T, square_op>;
    case ynn_unary_square_root:
      return unary_impl<T, T, square_root_op>;
    case ynn_unary_reciprocal_square_root:
      return unary_impl<T, T, reciprocal_square_root_op>;
    case ynn_unary_tanh:
      return unary_impl<T, T, tanh_op>;
    case ynn_unary_cube_root:
      return unary_impl<T, T, cube_root_op>;
    case ynn_unary_sign:
      return unary_impl<T, T, sign_op>;
    case ynn_unary_sine:
      return unary_impl<T, T, sine_op>;
    case ynn_unary_cosine:
      return unary_impl<T, T, cosine_op>;
    case ynn_unary_sigmoid:
      return unary_impl<T, T, sigmoid_op>;
    case ynn_unary_hardswish:
      return unary_impl<T, T, hardswish_op>;
    case ynn_unary_poly3:
      return unary_impl<T, T, poly3_op>;
    case ynn_unary_round_to_bf16:
      return unary_impl<T, T, round_to_bf16_op>;
    case ynn_unary_convert:
    case ynn_unary_invalid:
      break;
  }
  return nullptr;
}

}  // namespace

unary_kernel_fn get_unary_reference_kernel(ynn_unary_operator op, ynn_type type,
                                           uint32_t required_flags) {
  if (type == ynn_type_fp32) {
    return get_float_unary_reference_kernel<float>(op, required_flags);
  } else if (type == ynn_type_fp64) {
    return get_float_unary_reference_kernel<double>(op, required_flags);
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

unary_kernel_fn get_convert_reference_kernel(ynn_type a_type, ynn_type x_type,
                                             uint32_t required_flags) {
  switch (a_type) {
    case ynn_type_fp64:
      return get_convert_kernel<double>(x_type);
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
                                 ynn_type x_type, uint32_t required_flags,
                                 uint64_t supported_arch_flags) {
  // TODO(vksnk): select a better kernel based on the passed size.
#define YNN_ELEMENTWISE_KERNEL(arch, name, op_type, flags, type_a, type_x) \
  if (a_type == type_of<type_a>() && x_type == type_of<type_x>() &&        \
      op == ynn_unary_##op_type &&                                         \
      (flags & required_flags) == required_flags &&                        \
      is_arch_supported(arch, supported_arch_flags)) {                     \
    YNN_LOG_DEBUG() << "Using unary kernel " << #name;                     \
    return &name;                                                          \
  }

#include "ynnpack/kernels/unary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

  switch (op) {
    case ynn_unary_approx_erf:
      return get_unary_kernel(ynn_unary_erf, a_type, x_type, required_flags,
                              supported_arch_flags);
    case ynn_unary_approx_tanh:
      return get_unary_kernel(ynn_unary_tanh, a_type, x_type, required_flags,
                              supported_arch_flags);
    case ynn_unary_convert:
      return get_convert_reference_kernel(a_type, x_type, required_flags);
    default:
      if (a_type == ynn_type_fp64 && x_type == ynn_type_fp64) {
        return get_unary_reference_kernel(op, x_type, required_flags);
      } else if (a_type == ynn_type_fp32 && x_type == ynn_type_fp32) {
        return get_unary_reference_kernel(op, x_type, required_flags);
      } else if (a_type == ynn_type_int32 && x_type == ynn_type_int32) {
        return get_unary_reference_kernel(op, x_type, required_flags);
      }
      return nullptr;
  }
}

unary_params get_unary_params(ynn_unary_operator op) {
  switch (op) {
    case ynn_unary_exp:
    case ynn_unary_expm1:
      return unary_params{
          .exp = exp_params{
              ._ = 0.0,
              .output_multiplier = 1.0,
              .input_multiplier = 1.0,
          }};
    case ynn_unary_log:
    case ynn_unary_log1p:
      return unary_params{.log = log_params{
                              ._ = 0.0,
                              .output_multiplier = 1.0,
                              .input_multiplier = 1.0,
                          }};
    case ynn_unary_erf:
      return unary_params{.erf = erf_params{.output_offset = 0.0,
                                            .output_multiplier = 1.0,
                                            .input_multiplier = 1.0}};
    case ynn_unary_approx_erf:
      return unary_params{.approx_erf =
                              approx_erf_params{.output_offset = 0.0,
                                                .output_multiplier = 1.0,
                                                .input_multiplier = 1.0}};
    case ynn_unary_approx_tanh:
      return unary_params{.approx_tanh = approx_tanh_params{
                              .output_offset = 0.0, .output_multiplier = 1.0}};
    case ynn_unary_tanh:
      return unary_params{
          .tanh = tanh_params{.output_offset = 0.0, .output_multiplier = 1.0}};
    case ynn_unary_sine:
      return unary_params{
          .sine = sine_params{.output_offset = 0.0, .output_multiplier = 1.0}};
    case ynn_unary_cosine:
      return unary_params{.cosine = cosine_params{.output_offset = 0.0,
                                                  .output_multiplier = 1.0}};
    case ynn_unary_poly3:
      return unary_params{.poly3 = poly3_params{/*c0=*/0.0, /*c1=*/0.0,
                                                /*c2=*/0.0, /*c3=*/0.0}};
    default:
      return unary_params{};
  }

  return unary_params{};
}

}  // namespace ynn
