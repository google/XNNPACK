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
                size_t stride_y, void* vy) {
  auto x = reinterpret_cast<const TIn*>(vx);
  auto y = reinterpret_cast<TOut*>(vy);

  Operator op;
  for (size_t i = 0; i < m; ++i) {
    for (size_t j = 0; j < n; ++j) {
      y[j] = static_cast<TOut>(op(x[j]));
    }
    x = offset_bytes(x, stride_x);
    y = offset_bytes(y, stride_y);
  }
}

template <typename TIn, typename TOut>
struct convert_op {
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
};

#if XNN_HAVE_FLOAT16
template <>
struct convert_op<bfloat16, _Float16> {
  _Float16 operator()(bfloat16 x) const {
    return static_cast<_Float16>(static_cast<float>(x));
  }
};
#endif

template <typename TIn, typename TOut>
unary_kernel_fn get_convert_kernel(TIn, TOut) {
  return unary_impl<TIn, TOut, convert_op<TIn, TOut>>;
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
  float operator()(float x) const { return std::abs(x); }
  int32_t operator()(int32_t x) const { return std::abs(x); }
};

struct negate_op {
  float operator()(float x) const { return -x; }
  int32_t operator()(int32_t x) const { return -x; }
};

struct round_op {
  float operator()(float x) const { return std::nearbyint(x); }
};

struct ceil_op {
  float operator()(float x) const { return std::ceil(x); }
};

struct floor_op {
  float operator()(float x) const { return std::floor(x); }
};

struct square_op {
  float operator()(float x) const { return x * x; }
  int32_t operator()(int32_t x) const {
    return static_cast<int32_t>(static_cast<int64_t>(x) *
                                static_cast<int64_t>(x));
  }
};

struct square_root_op {
  float operator()(float x) const { return std::sqrt(x); }
};

struct cube_root_op {
  float operator()(float x) const { return std::cbrt(x); }
};

struct tanh_op {
  float operator()(float x) const { return std::tanh(x); }
};

struct reciprocal_square_root_op {
  float operator()(float x) const { return 1 / std::sqrt(x); }
};

struct log_op {
  float operator()(float x) const { return std::log(x); }
};

struct log1p_op {
  float operator()(float x) const { return std::log1p(x); }
};

struct exp_op {
  float operator()(float x) const { return std::exp(x); }
};

struct expm1_op {
  float operator()(float x) const { return std::expm1(x); }
};

struct erf_op {
  float operator()(float x) const { return std::erf(x); }
};

struct sign_op {
  float operator()(float x) const { return x < 0 ? -1 : x > 0 ? 1 : 0; }
  int32_t operator()(int32_t x) const { return x < 0 ? -1 : x > 0 ? 1 : 0; }
};

struct sine_op {
  float operator()(float x) const { return std::sin(x); }
};

struct cosine_op {
  float operator()(float x) const { return std::cos(x); }
};

struct sigmoid_op {
  float operator()(float x) const { return 1.0f / (1.0f + std::exp(-x)); }
};

struct hardswish_op {
  float operator()(float x) const {
    return (x * (1.0f / 6.0f)) * std::max(std::min(x + 3.0f, 6.0f), 0.0f);
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
    YNN_LOG_INFO() << "Using unary kernel " << #name;               \
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

}  // namespace ynn
