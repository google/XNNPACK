// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_KERNELS_BINARY_REFERENCE_H_
#define XNNPACK_YNNPACK_KERNELS_BINARY_REFERENCE_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include <gtest/gtest.h>
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

// This struct describes a unary operator enough such that we can test them
// without knowing anything about the specific operator.
struct binary_op_info {
  virtual ~binary_op_info() = default;

  virtual float operator()(float a, float b) const { YNN_UNREACHABLE; }
  virtual int32_t operator()(int32_t a, int32_t b) const { YNN_UNREACHABLE; }

  // Compute the tolerance for error given the reference result and the type.
  virtual float Tolerance(float x_ref, ynn_type type) const {
    return (std::abs(x_ref) + 1.0f) * 3.0f * epsilon(type);
  }
};

struct add : public binary_op_info {
  float operator()(float a, float b) const override { return a + b; }
  int32_t operator()(int32_t a, int32_t b) const override {
    return static_cast<int32_t>(widen(a) + widen(b));
  }
};

struct subtract : public binary_op_info {
  float operator()(float a, float b) const override { return a - b; }
  int32_t operator()(int32_t a, int32_t b) const override {
    return static_cast<int32_t>(widen(a) - widen(b));
  }
};

struct multiply : public binary_op_info {
  float operator()(float a, float b) const override { return a * b; }
  float operator()(int32_t a, float b) const {
    return static_cast<float>(a) * b;
  }
  int32_t operator()(int32_t a, int32_t b) const override {
    return static_cast<int32_t>(widen(a) * widen(b));
  }
};

struct divide : public binary_op_info {
  float operator()(float a, float b) const override { return a / b; }
  int32_t operator()(int32_t a, int32_t b) const override {
    return euclidean_div(a, b);
  }
};

struct min : public binary_op_info {
  float operator()(float a, float b) const override { return std::min(a, b); }
  int32_t operator()(int32_t a, int32_t b) const override {
    return std::min(a, b);
  }
};

struct max : public binary_op_info {
  float operator()(float a, float b) const override { return std::max(a, b); }
  int32_t operator()(int32_t a, int32_t b) const override {
    return std::max(a, b);
  }
};

struct copysign : public binary_op_info {
  float operator()(float a, float b) const override {
    return std::copysign(a, b);
  }
  int32_t operator()(int32_t a, int32_t b) const override {
    return std::copysign(a, b);
  }
};

struct pow : public binary_op_info {
  float operator()(float a, float b) const override { return std::pow(a, b); }
  int32_t operator()(int32_t a, int32_t b) const override {
    return integer_pow(a, b);
  }
};

struct squared_difference : public binary_op_info {
  float operator()(float a, float b) const override {
    return (a - b) * (a - b);
  }
};

struct leaky_relu : public binary_op_info {
  float operator()(float a, float b) const override {
    return a < 0 ? a * b : a;
  }
};

const binary_op_info* get_binary_op_info(ynn_binary_operator op);

// Check that `op(a, b)` == x, within tolerances described by `op`.
template <typename A, typename B, typename X, typename OpInfo>
void check_results(const OpInfo& op, const Tensor<A>& a, const Tensor<B>& b,
                   const Tensor<X>& x, const quantization_params&,
                   const quantization_params&, const quantization_params&) {
  for (const auto& i : EnumerateIndices(x.extents())) {
    if (std::is_integral<X>::value) {
      const int32_t expected = op(a(i), b(i));
      ASSERT_EQ(expected, x(i)) << "i = " << index_to_string(i)
                                << ", a(i) = " << a(i) << ", b(i) = " << b(i);
    } else {
      float expected = op(a(i), b(i));
      if (expected < type_info<X>::min()) {
        expected = -type_info<float>::infinity();
      }
      if (expected > type_info<X>::max()) {
        expected = type_info<float>::infinity();
      }
      if (std::isnan(expected)) {
        // Checking the x is NaN could make sense, but it fails in
        // a variety of cases.
      } else {
        ASSERT_NEAR(expected, x(i), op.Tolerance(expected, type_of<X>()))
            << "i = " << index_to_string(i) << ", a(i) = " << a(i)
            << ", b(i) = " << b(i);
      }
    }
  }
}

template <typename A, typename B, typename X, typename OpInfo>
void check_results(const OpInfo& op, const Tensor<quantized<A>>& a,
                   const Tensor<quantized<B>>& b, const Tensor<quantized<X>>& x,
                   const quantization_params& a_quantization,
                   const quantization_params& b_quantization,
                   const quantization_params& x_quantization) {
  for (const auto& i : EnumerateIndices(x.extents())) {
    const float a_i = dequantize(a(i), a_quantization);
    const float b_i = dequantize(b(i), b_quantization);
    float expected = op(a_i, b_i);
    expected = fake_quantize(expected, x_quantization);
    expected = std::max<float>(expected, type_info<X>::min());
    expected = std::min<float>(expected, type_info<X>::max());
    if (std::isnan(expected)) {
      // We don't know how to represent NaN for quantized types.
    } else {
      ASSERT_NEAR(expected, x(i), 1)
          << "i = " << index_to_string(i) << ", a(i) = " << a_i << " ("
          << static_cast<int32_t>(a(i)) << ")"
          << ", b(i) = " << b_i << " (" << static_cast<int32_t>(b(i)) << ")"
          << ", x(i) = " << static_cast<int32_t>(x(i));
    }
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_BINARY_REFERENCE_H_
