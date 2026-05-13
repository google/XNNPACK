#include "ynnpack/base/test/tolerance.h"
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
#include <memory>
#include <type_traits>

#include <gtest/gtest.h>
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/unary/unary.h"

namespace ynn {

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
      case ynn_type_bf16:
        return {type_info<bfloat16>::epsilon(),
                type_info<bfloat16>::infinity()};
      case ynn_type_fp32:
      case ynn_type_fp64:
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
  virtual double operator()(double x) const { YNN_UNREACHABLE; }
  virtual int32_t operator()(int32_t x) const { YNN_UNREACHABLE; }

  // Compute the tolerance for error given the reference result and the
  // type.
  virtual tolerance_spec tolerance(ynn_type type) const {
    return tolerance_spec{};
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
  explicit convert(const unary_params& = {}) {}
  float operator()(float x) const override { return x; }
  double operator()(double x) const override { return x; }
  int32_t operator()(int32_t x) const override { return x; }

  tolerance_spec tolerance(ynn_type type) const override {
    if (type_is_integral(type)) {
      // The epsilon of a 23-bit integer.
      constexpr float epsilon_int23 = 1.0f / (1 << 23);
      return tolerance_spec{epsilon_int23, /*absolute=*/1.0f};
    } else {
      return tolerance_spec{/*relative=*/1.0f, /*absolute=*/1.0f};
    }
  }
};

struct abs : public unary_op_info {
  explicit abs(const unary_params& = {}) {}
  float operator()(float x) const override { return std::abs(x); }
  double operator()(double x) const override { return std::abs(x); }
  int32_t operator()(int32_t x) const override { return std::abs(x); }

  tolerance_spec tolerance(ynn_type type) const override {
    return tolerance_spec{/*relative=*/0.0f,
                          /*absolute=*/type_is_integral(type) ? 1.0f : 0.0f};
  }
};

struct negate : public unary_op_info {
  explicit negate(const unary_params& = {}) {}
  float operator()(float x) const override { return -x; }
  double operator()(double x) const override { return -x; }
  int32_t operator()(int32_t x) const override { return -x; }

  tolerance_spec tolerance(ynn_type type) const override {
    return tolerance_spec{/*relative=*/0.0f,
                          /*absolute=*/type_is_integral(type) ? 1.0f : 0.0f};
  }
};

struct round : public unary_op_info {
  explicit round(const unary_params& = {}) {}
  float operator()(float x) const override { return std::nearbyint(x); }
  double operator()(double x) const override { return std::nearbyint(x); }

  tolerance_spec tolerance(ynn_type type) const override {
    return tolerance_spec{/*relative=*/0.0f,
                          /*absolute=*/type_is_integral(type) ? 1.0f : 0.0f};
  }
};

struct ceil : public unary_op_info {
  explicit ceil(const unary_params& = {}) {}
  float operator()(float x) const override { return std::ceil(x); }
  double operator()(double x) const override { return std::ceil(x); }

  tolerance_spec tolerance(ynn_type type) const override {
    return tolerance_spec{/*relative=*/0.0f,
                          /*absolute=*/type_is_integral(type) ? 1.0f : 0.0f};
  }
};

struct floor : public unary_op_info {
  explicit floor(const unary_params& = {}) {}
  float operator()(float x) const override { return std::floor(x); }
  double operator()(double x) const override { return std::floor(x); }

  tolerance_spec tolerance(ynn_type type) const override {
    return tolerance_spec{/*relative=*/0.0f,
                          /*absolute=*/type_is_integral(type) ? 1.0f : 0.0f};
  }
};

struct sigmoid : public unary_op_info {
  explicit sigmoid(const unary_params& = {}) {}
  float operator()(float x) const override {
    return static_cast<float>(1.0 / (1.0 + std::exp(static_cast<double>(-x))));
  }
  double operator()(double x) const override {
    return 1.0 / (1.0 + std::exp(-x));
  }

  tolerance_spec tolerance(ynn_type /*type*/) const override {
    return tolerance_spec{/*relative=*/1.0f, /*absolute=*/1.0f};
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
  explicit square(const unary_params& = {}) {}
  float operator()(float x) const override { return x * x; }
  double operator()(double x) const override { return x * x; }
  int32_t operator()(int32_t x) const override {
    return static_cast<int32_t>(static_cast<int64_t>(x) *
                                static_cast<int64_t>(x));
  }

  tolerance_spec tolerance(ynn_type type) const override {
    if (type != ynn_type_int32) {
      return tolerance_spec{/*relative=*/1.0f, /*absolute=*/1.0f};
    } else {
      // Overflow makes this hard to test.
      return tolerance_spec{type_info<float>::infinity()};
    }
  }
};

struct square_root : public unary_op_info {
  explicit square_root(const unary_params& = {}) {}
  float operator()(float x) const override { return std::sqrt(x); }
  double operator()(double x) const override { return std::sqrt(x); }

  tolerance_spec tolerance(ynn_type type) const override {
    switch (type) {
      case ynn_type_fp32:
      case ynn_type_fp64:
        return tolerance_spec{/*relative=*/2.0f};
      case ynn_type_fp16:
      case ynn_type_bf16:
        return tolerance_spec{/*relative=*/3.0f};
      case ynn_type_int8:
      case ynn_type_uint8:
        return tolerance_spec{/*relative=*/0.0f, /*absolute=*/1.0f};
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
  tanh_params params;

  explicit tanh(const unary_params& params = {}) : params(params.tanh) {}
  float operator()(float x) const override {
    return std::tanh(x) * params.output_multiplier + params.output_offset;
  }
  double operator()(double x) const override {
    return std::tanh(x) * params.output_multiplier + params.output_offset;
  }

  tolerance_spec tolerance(ynn_type /*type*/) const override {
    return tolerance_spec{/*relative=*/5.0f, /*absolute=*/1.0f};
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
  explicit reciprocal_square_root(const unary_params& = {}) {}
  float operator()(float x) const override {
    return static_cast<float>(1.0 / std::sqrt(static_cast<double>(x)));
  }
  double operator()(double x) const override { return 1.0 / std::sqrt(x); }

  tolerance_spec tolerance(ynn_type /*type*/) const override {
    return tolerance_spec{/*relative=*/2.0f};
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
  log_params params;

  explicit log(const unary_params& params) : params(params.log) {}
  float operator()(float x) const override {
    return std::log2(x * params.input_multiplier / std::sqrt(2.0f)) *
           params.output_multiplier;
  }
  double operator()(double x) const override {
    return std::log2(x * params.input_multiplier / std::sqrt(2.0)) *
           params.output_multiplier;
  }

  tolerance_spec tolerance(ynn_type /*type*/) const override {
    return tolerance_spec{/*relative=*/6.0f, /*absolute=*/2.0f};
  }

  interval domain(ynn_type type) const override {
    return {type_info<float>::epsilon(), 1000.0f};
  }
};

struct exp : public unary_op_info {
  exp_params params;

  explicit exp(const unary_params& params) : params(params.exp) {}
  float operator()(float x) const override {
    return std::exp2(params.input_multiplier * x) * params.output_multiplier;
  }
  double operator()(double x) const override {
    return std::exp2(params.input_multiplier * x) * params.output_multiplier;
  }

  tolerance_spec tolerance(ynn_type /*type*/) const override {
    return tolerance_spec{/*relative=*/6.0f, /*absolute=*/2.0f};
  }

  interval domain(ynn_type) const override { return {-10.0f, 10.0f}; }
};

struct log1p : public unary_op_info {
  explicit log1p(const unary_params& = {}) {}
  float operator()(float x) const override { return std::log1p(x); }
  double operator()(double x) const override { return std::log1p(x); }


  tolerance_spec tolerance(ynn_type /*type*/) const override {
    return tolerance_spec{/*relative=*/2.0f};
  }

  interval domain(ynn_type type) const override {
    return {type_info<float>::epsilon(), 1000.0f};
  }
};

struct expm1 : public unary_op_info {
  explicit expm1(const unary_params& = {}) {}
  float operator()(float x) const override { return std::expm1(x); }
  double operator()(double x) const override { return std::expm1(x); }

  tolerance_spec tolerance(ynn_type type) const override {
    return tolerance_spec{/*relative=*/1.0f, /*absolute=*/2.0f};
  }
  interval domain(ynn_type) const override { return {-10.0f, 10.0f}; }
};

struct erf : public unary_op_info {
  erf_params params;

  explicit erf(const unary_params& params) : params(params.erf) {}
  float operator()(float x) const override {
    return std::erf(params.input_multiplier * x) * params.output_multiplier +
           params.output_offset;
  }
  double operator()(double x) const override {
    return std::erf(params.input_multiplier * x) * params.output_multiplier +
           params.output_offset;
  }

  tolerance_spec tolerance(ynn_type /*type*/) const override {
    return tolerance_spec{/*relative=*/1.0f, /*absolute=*/3.0f};
  }
};

struct cube_root : public unary_op_info {
  explicit cube_root(const unary_params& = {}) {}
  float operator()(float x) const override { return std::cbrt(x); }
  double operator()(double x) const override { return std::cbrt(x); }

  tolerance_spec tolerance(ynn_type /*type*/) const override {
    return tolerance_spec{/*relative=*/2.5f};
  }
};

struct sign : public unary_op_info {
  explicit sign(const unary_params& = {}) {}
  float operator()(float x) const override {
    return x < 0 ? -1.0f : (x > 0 ? 1.0f : 0.0f);
  }
  double operator()(double x) const override {
    return x < 0 ? -1.0 : (x > 0 ? 1.0 : 0.0);
  }
  int32_t operator()(int32_t x) const override {
    return x < 0 ? -1 : (x > 0 ? 1 : 0);
  }
};

struct trig : public unary_op_info {
  explicit trig(const unary_params& = {}) {}
  tolerance_spec tolerance(ynn_type /*type*/) const override {
    return tolerance_spec{/*relative=*/5.0f, /*absolute=*/3.0f};
  }

  interval domain(ynn_type type) const override { return {-100.0f, 100.0f}; }
};

struct sine : public trig {
  sine_params params;

  explicit sine(const unary_params& params = {}) : params(params.sine) {}
  float operator()(float x) const override {
    return std::sin(x) * params.output_multiplier + params.output_offset;
  }
  double operator()(double x) const override {
    return std::sin(x) * params.output_multiplier + params.output_offset;
  }
};

struct cosine : public trig {
  cosine_params params;

  explicit cosine(const unary_params& params = {}) : params(params.cosine) {}
  float operator()(float x) const override {
    return std::cos(x) * params.output_multiplier + params.output_offset;
  }
  double operator()(double x) const override {
    return std::cos(x) * params.output_multiplier + params.output_offset;
  }
};

struct hardswish : public unary_op_info {
  explicit hardswish(const unary_params& = {}) {}
  float operator()(float x) const override {
    return (x / 6.0f) * std::max(std::min(x + 3.0f, 6.0f), 0.0f);
  }
  double operator()(double x) const override {
    return (x / 6.0) * std::max(std::min(x + 3.0, 6.0), 0.0);
  }

  tolerance_spec tolerance(ynn_type /*type*/) const override {
    return tolerance_spec{/*relative=*/2.0f};
  }

  interval domain(ynn_type) const override { return {-4.0f, 4.0f}; }
};

struct poly3 : public unary_op_info {
  poly3_params params;

  explicit poly3(const unary_params& params) : params(params.poly3) {}
  float operator()(float x) const override {
    return ((params.c3 * x + params.c2) * x + params.c1) * x + params.c0;
  }
  double operator()(double x) const override {
    return ((static_cast<double>(params.c3) * x +
             static_cast<double>(params.c2)) *
                x +
            static_cast<double>(params.c1)) *
               x +
           static_cast<double>(params.c0);
  }

  // Polynomials are tricky to test, because the tolerance should be based on
  // maximum value of the argument. We don't know what that is, but we can
  // define the domain to be [-1, 1].
  interval domain(ynn_type) const override { return {-1.0f, 1.0f}; }

  tolerance_spec tolerance(ynn_type /*type*/) const override {
    return tolerance_spec{/*relative=*/1.0f, /*absolute=*/5.0f};
  }
};

std::unique_ptr<unary_op_info> get_unary_op_info(
    ynn_unary_operator op, const unary_params& params = {});

// Check that op(a) == x, within tolerances described by `op`.
template <typename A, typename X>
void check_results(const unary_op_info& op, Tensor<A> a, Tensor<X> x,
                   const quantization_params& a_quantization = {},
                   const quantization_params& x_quantization = {}) {
  using Float =
      std::conditional_t<std::is_same<X, double>::value, double, float>;
  tolerance_spec tol = op.tolerance(type_of<X>());
  (void)tol;
  for (const auto& i : EnumerateIndices(x.extents())) {
    if constexpr (is_integral<X>::value) {
      if constexpr (is_integral<A>::value) {
        int32_t expected = op(static_cast<int32_t>(a(i)));
        if (expected > type_info<X>::max()) expected = type_info<X>::max();
        if (expected < type_info<X>::min()) expected = type_info<X>::min();
        ASSERT_EQ(expected, x(i)) << "i = " << index_to_string(i)
                                  << ", a(i) = " << static_cast<int32_t>(a(i));
      } else {
        // Integral output, non-integral a. We need to potentially
        // dequantize the a, and avoid UB when converting to int.
        const Float input_i = dequantize(a(i), a_quantization);
        const int32_t expected = round_float_to_int<X>(op(input_i));
        ASSERT_EQ(expected, x(i))
            << "i = " << index_to_string(i) << ", a(i) = " << input_i << " ("
            << static_cast<Float>(a(i)) << ")";
      }
    } else if constexpr (is_quantized<X>()) {
      const Float input_i = dequantize(a(i), a_quantization);
      Float expected = op(input_i);
      expected = fake_quantize(expected, x_quantization);
      expected = clamp_float_to_int<X>(expected);
      if (std::isnan(expected)) {
        // This is expected to overflow.
      } else {
        ASSERT_NEAR(expected, x(i), tol.absolute_error<X>(expected))
            << "i = " << index_to_string(i) << ", a(i) = " << input_i << " ("
            << static_cast<Float>(a(i)) << ")"
            << ", x(i) = " << static_cast<int32_t>(x(i)) << " ("
            << dequantize(x(i), x_quantization) << ")" << std::endl;
      }
    } else {
      const Float input_i = dequantize(a(i), a_quantization);
      Float expected = op(input_i);
      // Force overflow to infinity if that is what should happen.
      expected = static_cast<Float>(static_cast<X>(expected));
      if (std::abs(expected) < type_info<X>::smallest_normal()) {
        // Flush denormals to 0
        expected = 0.0f;
      }
      if (op.is_in_supported_range(expected)) {
        if (std::isnan(static_cast<Float>(expected))) {
          ASSERT_TRUE(std::isnan(static_cast<Float>(x(i))));
        } else {
          ASSERT_NEAR(expected, x(i), tol.absolute_error<X>(expected))
              << "i = " << index_to_string(i) << ", a(i) = " << input_i << " ("
              << static_cast<Float>(a(i)) << ")";
        }
      }
    }
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_TEST_UNARY_OPS_H_
