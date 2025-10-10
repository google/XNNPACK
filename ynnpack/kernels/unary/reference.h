// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_TEST_UNARY_OPS_H_
#define XNNPACK_YNNPACK_KERNELS_TEST_UNARY_OPS_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include <gtest/gtest.h>
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

static float tol_exact(float) { return 0.0f; }
static float tol_exact16(float y_ref) {
  // The maximum of the relative tolerance and half the smallest positive
  // normal.
  return std::max(std::abs(y_ref) * type_info<half>::epsilon(),
                  0.5f * type_info<half>::epsilon());
}

static float tol_relative(float y_ref, float rel_tol) {
  // Note that `y_ref * rel_tol`, i.e. the expected absolute difference,
  // may round differently than `y_ref * (1 + rel_tol) - y_ref`, i.e. the
  // effective absolute difference computed in `float`s. We therefore use
  // the latter form since it is the true difference between two `float`s
  // within the given relative tolerance.
  return std::abs(y_ref * (1.0f + rel_tol)) - std::abs(y_ref);
}

static float tol_mixed(float y_ref, float abs_tol, float rel_tol) {
  return std::max(abs_tol,
                  std::abs(y_ref) * (1.0f + rel_tol) - std::abs(y_ref));
}

struct interval {
  float min;
  float max;

  static interval all() {
    return {-type_info<float>::infinity(), type_info<float>::infinity()};
  }

  static interval positive(ynn_type type) {
    switch (type) {
      case ynn_type_fp16:
        return {type_info<half>::epsilon(), type_info<half>::infinity()};
      case ynn_type_fp32:
        return {type_info<float>::epsilon(), type_info<float>::infinity()};
      default:
        return {1.0f, type_info<float>::infinity()};
    }
  }
};

// This struct describes a unary operator enough such that we can test them
// without knowing anything about the specific operator.
struct unary_op_info {
  virtual ~unary_op_info() = default;

  virtual float operator()(float x) const { YNN_UNREACHABLE; }
  virtual int32_t operator()(int32_t x) const { YNN_UNREACHABLE; }

  // Compute the tolerance for error given the reference result and the
  // type.
  virtual float tolerance(float y_ref, ynn_type type) const {
    switch (type) {
      case ynn_type_int8:
      case ynn_type_uint8:
        return 1;
      case ynn_type_fp16:
        return tol_exact16(y_ref);
      default:
        return tol_exact(y_ref);
    }
  }

  virtual interval domain(ynn_type) const { return interval::all(); }

  // Quantization parameters to use by default.
  virtual quantization_params input_quantization_params(ynn_type type) const {
    switch (type) {
      case ynn_type_uint8:
        return {150, 1.0f};
      default:
        return {0, 1.0f};
    }
  }
  virtual quantization_params output_quantization_params(ynn_type type) const {
    switch (type) {
      case ynn_type_uint8:
        return {100, 1.0f};
      default:
        return {0, 1.0f};
    }
  }

  // If this returns false, we do not promise to match the reference
  // implementation within `tolerance`.
  virtual bool is_in_supported_range(float y) const { return true; }
};

struct convert : public unary_op_info {
  float operator()(float x) const override { return x; }
  int32_t operator()(int32_t x) const override { return x; }

  float tolerance(float y_ref, ynn_type type) const override {
    // The epsilon of a 23-bit integer.
    constexpr float epsilon_int23 = 1.0f / (1 << 23);
    return type_is_integral(type)
               ? tol_relative(y_ref, epsilon_int23)
               : tol_mixed(y_ref, epsilon(type), epsilon(type));
  }
};

struct abs : public unary_op_info {
  float operator()(float x) const override { return std::abs(x); }
  int32_t operator()(int32_t x) const override { return std::abs(x); }
};

struct negate : public unary_op_info {
  float operator()(float x) const override { return -x; }
  int32_t operator()(int32_t x) const override { return -x; }
};

struct round : public unary_op_info {
  float operator()(float x) const override { return std::nearbyint(x); }
};

struct ceil : public unary_op_info {
  float operator()(float x) const override { return std::ceil(x); }
};

struct floor : public unary_op_info {
  float operator()(float x) const override { return std::floor(x); }
};

struct sigmoid : public unary_op_info {
  float operator()(float x) const override {
    if (x > 100) {
      return 1.0f;
    } else if (x < -100) {
      return 0.0f;
    } else {
      const double e = std::exp(static_cast<double>(x));
      return e / (1.0 + e);
    }
  }

  float tolerance(float y_ref, ynn_type type) const override {
    switch (type) {
      case ynn_type_fp32:
        return tol_mixed(y_ref, 5.0e-6f, 1.0e-5f);
      case ynn_type_fp16:
        return tol_mixed(y_ref, 1.0e-4f, 5.0e-3f);
      case ynn_type_bf16:
        return tol_mixed(y_ref, 1.0e-3f, 1.0e-2f);
      case ynn_type_int8:
      case ynn_type_uint8:
        return 1;
      default:
        return tol_exact(y_ref);
    }
  }

  interval domain(ynn_type type) const override {
    switch (type) {
      case ynn_type_fp16:
        return {-25.0f, 25.0f};
      default:
        return {-125.0f, 125.0f};
    }
  }
};

struct square : public unary_op_info {
  float operator()(float x) const override { return x * x; }
  int32_t operator()(int32_t x) const override {
    return static_cast<int32_t>(static_cast<int64_t>(x) *
                                static_cast<int64_t>(x));
  }

  float tolerance(float y_ref, ynn_type type) const override {
    switch (type) {
      case ynn_type_fp32:
      case ynn_type_fp16:
      case ynn_type_bf16:
        return tol_mixed(y_ref, epsilon(type), epsilon(type));
      case ynn_type_int8:
      case ynn_type_uint8:
        return 1;
      case ynn_type_int32:
        // Overflow makes this hard to test.
        return type_info<float>::infinity();
      default:
        YNN_UNREACHABLE;
    }
  }
};

struct square_root : public unary_op_info {
  float operator()(float x) const override { return std::sqrt(x); }

  float tolerance(float y_ref, ynn_type type) const override {
    switch (type) {
      case ynn_type_fp32:
        return tol_relative(y_ref, 2.0f * type_info<float>::epsilon());
      case ynn_type_fp16:
      case ynn_type_bf16:
        return tol_relative(y_ref, 3.0f * epsilon(type));
      case ynn_type_int8:
      case ynn_type_uint8:
        return 1;
      default:
        YNN_UNREACHABLE;
    }
  }

  interval domain(ynn_type type) const override {
    switch (type) {
      case ynn_type_fp16:
        return {1.0e-4f, 10.0f};
      case ynn_type_fp32:
        // The reciprocal square root estimate instructions (e.g. `vrsqrteq_f32`
        // for Arm or `_m*_rsqrt_ps` for Intel) seem to fail for values larger
        // than the inverse of the minimum normalized number when denormals are
        // switched off, so limit the range to that of normally inversible
        // numbers.
        return {type_info<float>::epsilon(), type_info<float>::max() / 4};
      default:
        return interval::positive(type);
    }
  }

  bool is_in_supported_range(float y) const override {
    // TODO(b/404943039): We have some cases where inf input produces NaN
    // output, that the reference implementation disagrees with.
    return !std::isnan(y) && !std::isinf(y);
  }
};

struct tanh : public unary_op_info {
  float operator()(float x) const override { return std::tanh(x); }

  float tolerance(float y_ref, ynn_type type) const override {
    switch (type) {
      case ynn_type_fp32:
      case ynn_type_fp16:
      case ynn_type_bf16:
        return tol_mixed(y_ref, epsilon(type),
                         4.0f * epsilon(type));  // 4 ULP
      default:
        return 1;
    }
  }

  interval domain(ynn_type type) const override {
    switch (type) {
      case ynn_type_fp16:
        return {-5.0f, 5.0f};
      default:
        return {-10.0f, 10.0f};
    }
  }
};

struct reciprocal_square_root : public unary_op_info {
  float operator()(float x) const override { return 1.0 / std::sqrt(x); }

  float tolerance(float y_ref, ynn_type type) const override {
    switch (type) {
      case ynn_type_fp32:
        return tol_relative(y_ref, 2 * type_info<float>::epsilon());
      case ynn_type_fp16:
      case ynn_type_bf16:
        return tol_mixed(y_ref, 1.0e-4f, 5.0e-3f);
      case ynn_type_int8:
      case ynn_type_uint8:
        return 1;
      default:
        return tol_exact(y_ref);
    }
  }

  interval domain(ynn_type type) const override {
    switch (type) {
      case ynn_type_fp16:
        return {1.0e-4f, 10.0f};
      case ynn_type_fp32:
        // The reciprocal square root estimate instructions (e.g. `vrsqrteq_f32`
        // for Arm or `_m*_rsqrt_ps` for Intel) seem to fail for values larger
        // than the inverse of the minimum normalized number when denormals are
        // switched off, so limit the range to that of normally inversible
        // numbers.
        return {type_info<float>::epsilon(), type_info<float>::max() / 4};
      default:
        return interval::positive(type);
    }
  }

  bool is_in_supported_range(float y) const override {
    // TODO(b/404943039): If the input is inf, the result is 0, but we produce
    // NaN.
    return y != 0.0f;
  }
};

struct log : public unary_op_info {
  float operator()(float x) const override { return std::log(x); }

  float tolerance(float y_ref, ynn_type type) const override {
    return tol_mixed(y_ref, 2 * epsilon(type), 6 * epsilon(type));
  }

  interval domain(ynn_type type) const override {
    return {type_info<float>::epsilon(), 1000.0f};
  }
};

struct exp : public unary_op_info {
  float operator()(float x) const override { return std::exp(x); }

  float tolerance(float y_ref, ynn_type type) const override {
    return tol_mixed(y_ref, 2 * epsilon(type), 6 * epsilon(type));
  }

  interval domain(ynn_type) const override { return {-10.0f, 10.0f}; }
};

struct log1p : public unary_op_info {
  float operator()(float x) const override { return std::log1p(x); }

  float tolerance(float y_ref, ynn_type type) const override {
    if (type == ynn_type_fp16) {
      return tol_mixed(y_ref, 2 * epsilon(type), epsilon(type));
    } else {
      return tol_relative(y_ref, 2 * epsilon(type));
    }
  }

  interval domain(ynn_type type) const override {
    return {type_info<float>::epsilon(), 1000.0f};
  }
};

struct expm1 : public unary_op_info {
  float operator()(float x) const override { return std::expm1(x); }

  float tolerance(float y_ref, ynn_type type) const override {
    if (type == ynn_type_fp16) {
      return tol_mixed(y_ref, 2 * epsilon(type), epsilon(type));
    } else {
      return tol_relative(y_ref, 2 * epsilon(type));
    }
  }
  interval domain(ynn_type) const override { return {-10.0f, 10.0f}; }
};

struct erf : public unary_op_info {
  float operator()(float x) const override { return std::erf(x); }

  float tolerance(float y_ref, ynn_type type) const override {
    if (type == ynn_type_fp16) {
      return tol_mixed(y_ref, 3 * epsilon(type), epsilon(type));
    } else {
      return tol_relative(y_ref, 3 * epsilon(type));
    }
  }
};

struct cube_root : public unary_op_info {
  float operator()(float x) const override { return std::cbrt(x); }

  float tolerance(float y_ref, ynn_type type) const override {
    return tol_relative(y_ref, 2.5f * epsilon(type));
  }
};

struct sign : public unary_op_info {
  float operator()(float x) const override {
    return x < 0 ? -1.0f : (x > 0 ? 1.0f : 0.0f);
  }
  int32_t operator()(int32_t x) const override {
    return x < 0 ? -1 : (x > 0 ? 1 : 0);
  }
};

struct trig : public unary_op_info {
  float tolerance(float y_ref, ynn_type type) const override {
    switch (type) {
      case ynn_type_fp32:
      case ynn_type_fp16:
      case ynn_type_bf16:
        return tol_mixed(y_ref, 3 * epsilon(type), 5 * epsilon(type));
      case ynn_type_int8:
      case ynn_type_uint8:
        return 1;
      default:
        return tol_exact(y_ref);
    }
  }

  interval domain(ynn_type type) const override { return {-100.0f, 100.0f}; }
};

struct sine : public trig {
  float operator()(float x) const override { return std::sin(x); }
};

struct cosine : public trig {
  float operator()(float x) const override { return std::cos(x); }
};

struct hardswish : public unary_op_info {
  float operator()(float x) const override {
    return (x / 6.0) * std::max(std::min(x + 3.0, 6.0), 0.0);
  }

  float tolerance(float y_ref, ynn_type type) const override {
    switch (type) {
      case ynn_type_fp32:
        return tol_mixed(y_ref, 5.0e-6f, 1.0e-5f);
      case ynn_type_fp16:
        return tol_mixed(y_ref, 1.0e-3f, 1.0e-2f);
      case ynn_type_bf16:
        return tol_mixed(y_ref, 1.0e-2f, 5.0e-2f);
      case ynn_type_int8:
      case ynn_type_uint8:
        return 1;
      default:
        YNN_UNREACHABLE;
    }
  }

  interval domain(ynn_type) const override { return {-4.0f, 4.0f}; }
};

const unary_op_info* get_unary_op_info(ynn_unary_operator op);

// Check that op(a) == x, within tolerances described by `op`.
template <typename A, typename X>
void check_results(const unary_op_info& op, Tensor<A> a, Tensor<X> x,
                   const quantization_params& a_quantization,
                   const quantization_params& x_quantization) {
  for (const auto& i : EnumerateIndices(x.extents())) {
    if (std::is_integral<X>::value) {
      if (std::is_integral<A>::value) {
        const int32_t expected = op(static_cast<int32_t>(a(i)));
        ASSERT_EQ(expected, x(i)) << "i = " << index_to_string(i)
                                  << ", a(i) = " << static_cast<int32_t>(a(i));
      } else {
        // Integral output, non-integral a. We need to potentially
        // dequantize the a, and avoid UB when converting to int.
        const float input_i = dequantize(a(i), a_quantization);
        const int32_t expected = round_float_to_int<X>(op(input_i));
        ASSERT_EQ(expected, x(i))
            << "i = " << index_to_string(i) << ", a(i) = " << input_i << " ("
            << static_cast<float>(a(i)) << ")";
      }
    } else if (is_quantized<X>()) {
      const float input_i = dequantize(a(i), a_quantization);
      float expected = op(input_i);
      expected = fake_quantize(expected, x_quantization);
      expected = std::max<float>(expected, type_info<X>::min());
      expected = std::min<float>(expected, type_info<X>::max());
      if (std::isnan(expected)) {
        // This is expected to overflow.
      } else {
        ASSERT_NEAR(expected, x(i), op.tolerance(expected, type_of<X>()))
            << "i = " << index_to_string(i) << ", a(i) = " << input_i << " ("
            << static_cast<float>(a(i)) << ")"
            << ", x(i) = " << static_cast<int32_t>(x(i));
      }
    } else {
      const float input_i = dequantize(a(i), a_quantization);
      float expected = op(input_i);
      // Force overflow to infinity if that is what should happen.
      expected = static_cast<float>(static_cast<X>(expected));
      if (std::abs(expected) < type_info<X>::smallest_normal()) {
        // Flush denormals to 0
        expected = 0.0f;
      }
      if (op.is_in_supported_range(expected)) {
        if (std::isnan(static_cast<float>(expected))) {
          ASSERT_TRUE(std::isnan(static_cast<float>(x(i))));
        } else {
          ASSERT_NEAR(expected, x(i), op.tolerance(expected, type_of<X>()))
              << "i = " << index_to_string(i) << ", a(i) = " << input_i << " ("
              << static_cast<float>(a(i)) << ")";
        }
      }
    }
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_TEST_UNARY_OPS_H_
