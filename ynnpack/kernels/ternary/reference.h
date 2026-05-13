// Copyright 2025 Google LLC
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

#include <gtest/gtest.h>
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

inline float epsilon(ynn_type type) {
  switch (type) {
    case ynn_type_fp32:
      return type_info<float>::epsilon();
    case ynn_type_fp64:
      return type_info<double>::epsilon();
    case ynn_type_fp16:
      return type_info<half>::epsilon();
    case ynn_type_bf16:
      return type_info<bfloat16>::epsilon();
    case ynn_type_int8:
    case ynn_type_uint8:
      return 1.0f;
    default:
      return 0.0f;
  }
}

struct multiply {
  float operator()(float a, float b, float c) const { return a * b * c; }
  double operator()(double a, double b, double c) const { return a * b * c; }
  float operator()(int32_t a, float b, float c) const {
    return static_cast<float>(a) * b * c;
  }

  double tolerance(double a, double b, double c, ynn_type type) const {
    return 0.0;
  }
};

struct multiply_add {
  float operator()(float a, float b, float c) const { return a * b + c; }
  double operator()(double a, double b, double c) const { return a * b + c; }
  int32_t operator()(int32_t a, int32_t b, int32_t c) const {
    return narrow(widen(a) * widen(b) + widen(c));
  }

  double tolerance(double a, double b, double c, ynn_type type) const {
    // This might differ due to using fused multiply-add instructions.
    return std::max(std::abs(a * b), std::abs(c)) * 2.0 * epsilon(type);
  }
};

struct subtract_multiply {
  float operator()(int32_t a, int32_t b, int32_t c) const {
    return narrow(widen(a) - widen(b) * widen(c));
  }

  double tolerance(double a, double b, double c, ynn_type type) const {
    // This might differ due to using fused multiply-add instructions.
    return std::max(std::abs(a), std::abs(b * c)) * 2.0 * epsilon(type);
  }
};

struct clamp {
  int32_t operator()(int32_t a, int32_t mn, int32_t mx) const {
    return std::min(std::max(a, mn), mx);
  }
  float operator()(float a, float mn, float mx) const {
    return std::min(std::max(a, mn), mx);
  }
  double operator()(double a, double mn, double mx) const {
    return std::min(std::max(a, mn), mx);
  }

  double tolerance(double a, double b, double c, ynn_type type) const {
    return 0.0;
  }
};

struct quantize_int8 {
  int8_t operator()(float a, float scale, int32_t zp) const {
    return quantize<int8_t>(a, 1.0f / scale, zp);
  }

  double tolerance(double, double, double, ynn_type type) const { return 1.0; }
};

struct quantize_uint8 {
  uint8_t operator()(float a, float scale, int32_t zp) const {
    return quantize<uint8_t>(a, 1.0f / scale, zp);
  }

  double tolerance(double, double, double, ynn_type type) const { return 1.0; }
};

struct dequantize {
  float operator()(float a, float zero_point, float scale) const {
    return (a - zero_point) * scale;
  }

  double tolerance(double a, double b, double c, ynn_type type) const {
    return epsilon(type);
  }
};


// Check that `op(a, b, c)` == x, within tolerances described by `op`.
template <typename A, typename B, typename C, typename X, typename OpInfo>
void check_results(const OpInfo& op, const Tensor<A>& a, const Tensor<B>& b,
                   const Tensor<C>& c, const Tensor<X>& x,
                   const quantization_params&, const quantization_params&,
                   const quantization_params&, const quantization_params&) {
  for (const auto& i : EnumerateIndices(x.extents())) {
    auto a_i = a(i);
    auto b_i = b(i);
    auto c_i = c(i);
    if (is_integral<X>::value) {
      const int32_t expected = op(a_i, b_i, c_i);
      const int32_t tolerance =
          std::nearbyint(op.tolerance(a_i, b_i, c_i, type_of<X>()));
      ASSERT_NEAR(expected, x(i), tolerance)
          << "i = " << index_to_string(i) << ", a(i) = " << a_i
          << ", b(i) = " << b_i << ", c(i) = " << c_i;
    } else {
      double expected = op(a_i, b_i, c_i);
      if (expected < type_info<X>::min()) {
        expected = -type_info<double>::infinity();
      }
      if (expected > type_info<X>::max()) {
        expected = type_info<double>::infinity();
      }
      if (std::isnan(expected)) {
        // Checking the x is NaN could make sense, but it fails in
        // a variety of cases.
      } else {
        const double tolerance = op.tolerance(a_i, b_i, c_i, type_of<X>());
        ASSERT_NEAR(expected, x(i), tolerance)
            << "i = " << index_to_string(i) << ", a(i) = " << a_i
            << ", b(i) = " << b_i << ", c(i) = " << c_i;
      }
    }
  }
}

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_KERNELS_BINARY_REFERENCE_H_
