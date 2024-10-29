// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_TEST_UNARY_OPS_H_
#define THIRD_PARTY_XNNPACK_TEST_UNARY_OPS_H_

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <type_traits>

#include "xnnpack.h"
#include "xnnpack/buffer.h"
#include "xnnpack/math.h"

static float TolExact(float) { return 0.0f; }
static float TolExact16(float y_ref) { return std::abs(y_ref) * 1.0e-3f; }

static float TolRelative(float y_ref, float rel_tol) {
  // Note that `y_ref * rel_tol`, i.e. the expected absolute difference,
  // may round differently than `y_ref * (1 + rel_tol) - y_ref`, i.e. the
  // effective absolute difference computed in `float`s. We therefore use
  // the latter form since it is the true difference between two `float`s
  // within the given relative tolerance.
  return std::abs(y_ref * (1.0f + rel_tol)) - std::abs(y_ref);
}

static float TolMixed(float y_ref, float abs_tol, float rel_tol) {
  return std::max(abs_tol,
                  std::abs(y_ref) * (1.0f + rel_tol) - std::abs(y_ref));
}

struct Interval {
  float min;
  float max;

  static Interval All() {
    return {-std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity()};
  }

  static Interval Positive(xnn_datatype datatype) {
    switch (datatype) {
      case xnn_datatype_fp16:
        return {0.001f, std::numeric_limits<float>::infinity()};
      case xnn_datatype_fp32:
        return {std::numeric_limits<float>::epsilon(),
                std::numeric_limits<float>::infinity()};
      default:
        return {1.0f, std::numeric_limits<float>::infinity()};
    }
  }
};

// This struct describes a unary operator enough such that we can test them
// without knowing anything about the specific operator.
struct UnaryOpInfo {
  virtual ~UnaryOpInfo() = default;

  virtual float ReferenceImpl(float x, const xnn_unary_params& params) const {
    XNN_UNREACHABLE;
  }
  virtual int ReferenceImpl(int x, const xnn_unary_params& params) const {
    XNN_UNREACHABLE;
  }

  // Get the parameters to use by default for this operator.
  virtual xnn_unary_params DefaultParams() const { return xnn_unary_params(); }

  // Compute the tolerance for error given the reference result and the
  // datatype.
  virtual float Tolerance(float y_ref, xnn_datatype datatype) const {
    switch (datatype) {
      case xnn_datatype_qint8:
      case xnn_datatype_quint8:
        return 1;
      case xnn_datatype_fp16:
        return TolExact16(y_ref);
      default:
        return TolExact(y_ref);
    }
  }

  virtual Interval Domain(xnn_datatype) const { return Interval::All(); }

  // Quantization parameters to use by default.
  virtual xnn_quantization_params InputQuantizationParams(
      xnn_datatype datatype) const {
    switch (datatype) {
      case xnn_datatype_quint8:
        return {150, 1.0f};
      default:
        return {0, 1.0f};
    }
  }
  virtual xnn_quantization_params OutputQuantizationParams(
      xnn_datatype datatype) const {
    switch (datatype) {
      case xnn_datatype_quint8:
        return {100, 1.0f};
      default:
        return {0, 1.0f};
    }
  }
};

struct Convert : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return x;
  }
  int ReferenceImpl(int x, const xnn_unary_params&) const override {
    return x;
  }
};

struct ReLU : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::max(x, 0.0f);
  }
  int ReferenceImpl(int x, const xnn_unary_params&) const override {
    return std::max(x, 0);
  }
};

struct Abs : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::abs(x);
  }
  int ReferenceImpl(int x, const xnn_unary_params&) const override {
    return std::abs(x);
  }
};

struct Negate : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return -x;
  }
  int ReferenceImpl(int x, const xnn_unary_params&) const override {
    return -x;
  }
};

struct Clamp : public UnaryOpInfo {
  xnn_unary_params DefaultParams() const override {
    xnn_unary_params params;
    params.clamp.min = -40.0f;
    params.clamp.max = 50.0f;
    return params;
  }

  float ReferenceImpl(float x, const xnn_unary_params& params) const override {
    return std::min<float>(std::max<float>(x, params.clamp.min),
                           params.clamp.max);
  }
  int ReferenceImpl(int x, const xnn_unary_params& params) const override {
    return std::min<int>(std::max<int>(x, params.clamp.min),
                         params.clamp.max);
  }

  xnn_quantization_params InputQuantizationParams(
      xnn_datatype datatype) const override {
    return {0, 1.0f};
  }
  xnn_quantization_params OutputQuantizationParams(
      xnn_datatype datatype) const override {
    return {0, 1.0f};
  }
};

struct ELU : public UnaryOpInfo {
  xnn_unary_params DefaultParams() const override {
    xnn_unary_params params;
    params.elu.alpha = 1.0f;
    return params;
  }

  float ReferenceImpl(float x, const xnn_unary_params& params) const override {
    return std::signbit(x) ? params.elu.alpha * std::expm1(x) : x;
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp32:
        return TolMixed(y_ref, 5.0e-6f, 1.0e-5f);
      case xnn_datatype_fp16:
        return TolMixed(y_ref, 1.0e-4f, 5.0e-3f);
      default:
        return 1;
    }
  }

  Interval Domain(xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp16:
        return {-9.0f, 9.0f};
      default:
        return {-20.0f, 20.0f};
    }
  }
};

struct GELU : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return x * 0.5f * (1.0f + std::erf(x * std::sqrt(2) / 2));
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    return TolMixed(y_ref, 10 * std::numeric_limits<float>::epsilon(),
                    5 * std::numeric_limits<float>::epsilon());
  }

  Interval Domain(xnn_datatype) const override { return {-10.0f, 10.0f}; }
};

struct HardSwish : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return (x / 6.0) * std::max(std::min(x + 3.0, 6.0), 0.0);
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp32:
        return TolMixed(y_ref, 5.0e-6f, 1.0e-5f);
      case xnn_datatype_fp16:
        return TolMixed(y_ref, 1.0e-3f, 1.0e-2f);
      case xnn_datatype_qint8:
      case xnn_datatype_quint8:
        return 1;
      default:
        XNN_UNREACHABLE;
    }
  }

  Interval Domain(xnn_datatype) const override { return {-4.0f, 4.0f}; }
};

struct LeakyReLU : public UnaryOpInfo {
  xnn_unary_params DefaultParams() const override {
    xnn_unary_params params;
    params.leaky_relu.negative_slope = 0.5f;
    return params;
  }

  float ReferenceImpl(float x, const xnn_unary_params& params) const override {
    return std::signbit(x) ? x * params.leaky_relu.negative_slope : x;
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp32:
        return TolExact(y_ref);
      case xnn_datatype_fp16:
        return TolMixed(y_ref, 1.0e-4f, 1.0e-3f);
      case xnn_datatype_qint8:
      case xnn_datatype_quint8:
        return 1;
      default:
        XNN_UNREACHABLE;
    }
  }
};

struct RoundToNearestEven : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::nearbyint(x);
  }
};

struct RoundTowardsZero : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::trunc(x);
  }
};

struct RoundUp : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::ceil(x);
  }
};

struct RoundDown : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::floor(x);
  }
};

struct Sigmoid : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    if (x > 100) {
      return 1.0f;
    } else if (x < -100) {
      return 0.0f;
    } else {
      const double e = std::exp(static_cast<double>(x));
      return e / (1.0 + e);
    }
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp32:
        return TolMixed(y_ref, 5.0e-6f, 1.0e-5f);
      case xnn_datatype_fp16:
        return TolMixed(y_ref, 1.0e-4f, 5.0e-3f);
      default:
        return TolExact(y_ref);
    }
  }

  Interval Domain(xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp16:
        return {-25.0f, 25.0f};
      default:
        return {-125.0f, 125.0f};
    }
  }
};

struct Square : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return x * x;
  }
  int ReferenceImpl(int x, const xnn_unary_params&) const override {
    return static_cast<int64_t>(x) * static_cast<int64_t>(x);
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp32:
        return TolExact(y_ref);
      case xnn_datatype_fp16:
        return TolMixed(y_ref, 1.0e-4f, 5.0e-3f);
      default:
        return TolExact(y_ref);
    }
  }
};

struct SquareRoot : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::sqrt(x);
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp32:
        return TolRelative(y_ref, 2.5f * std::numeric_limits<float>::epsilon());
      case xnn_datatype_fp16:
        return TolMixed(y_ref, 1.0e-4f, 5.0e-3f);
      default:
        return TolExact(y_ref);
    }
  }

  Interval Domain(xnn_datatype datatype) const override {
    if (datatype == xnn_datatype_fp16) {
      return {0.001f, 10.0f};
    } else {
      return Interval::Positive(datatype);
    }
  }
};

struct TanH : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::tanh(x);
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp32:
        return TolRelative(
            y_ref,
            4.0f * std::numeric_limits<float>::epsilon());  // 4 ULP
      case xnn_datatype_fp16:
        return TolMixed(y_ref, /*abs_tol=*/1.0e-4f, /*rel_tol=*/5.0e-3f);
      default:
        return TolExact(y_ref);
    }
  }

  Interval Domain(xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp16:
        return {-5.0f, 5.0f};
      default:
        return {-10.0f, 10.0f};
    }
  }
};

struct ReciprocalSquareRoot : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return 1.0 / std::sqrt(x);
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp32:
        return TolRelative(y_ref, 4 * std::numeric_limits<float>::epsilon());
      case xnn_datatype_fp16:
        return TolMixed(y_ref, 1.0e-4f, 5.0e-3f);
      default:
        return TolExact(y_ref);
    }
  }

  Interval Domain(xnn_datatype datatype) const override {
    if (datatype == xnn_datatype_fp16) {
      return {1.0e-4f, 10.0f};
    } else {
      return Interval::Positive(datatype);
    }
  }
};

struct Log : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::log(x);
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    return TolMixed(y_ref, 2 * std::numeric_limits<float>::epsilon(),
                    6 * std::numeric_limits<float>::epsilon());
  }

  Interval Domain(xnn_datatype datatype) const override {
    return {std::numeric_limits<float>::epsilon(), 1000.0f};
  }
};

struct Exp : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::exp(x);
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    return TolMixed(y_ref, 2 * std::numeric_limits<float>::epsilon(),
                    6 * std::numeric_limits<float>::epsilon());
  }
  Interval Domain(xnn_datatype) const override { return {-10.0f, 10.0f}; }
};

const UnaryOpInfo* GetUnaryOpInfo(xnn_unary_operator op);

// Generate random data in the given domain, where the domain is given as
// unquantized values.
template <typename T, typename Rng>
void FillRandom(Rng& rng, T* x, size_t n, const Interval& domain,
                const xnn_quantization_params& quantization = {0, 1.0f}) {
  float min = domain.min;
  float max = domain.max;
  min = min * quantization.scale + quantization.zero_point;
  max = max * quantization.scale + quantization.zero_point;
  min = std::max<float>(domain.min, xnnpack::NumericLimits<T>::min());
  max = std::min<float>(domain.max, xnnpack::NumericLimits<T>::max());
  min = std::max<float>(min, -1e6f);
  max = std::min<float>(max, 1e6f);

  std::uniform_real_distribution<float> dist(min, max);
  for (size_t i = 0; i < n; ++i) {
    x[i] = static_cast<T>(dist(rng));
  }
}

// Compute the result of a unary operator using the reference implementation.
template <typename In, typename Out, typename UnaryOp>
void UnaryReferenceImpl(
    const In* x, size_t n, Out* y, const UnaryOp& op_info,
    const xnn_quantization_params& input_quantization = {0, 1.0f},
    const xnn_quantization_params& output_quantization = {0, 1.0f},
    const xnn_unary_params& params = xnn_unary_params()) {
  for (size_t i = 0; i < n; i++) {
    float x_i = static_cast<float>(x[i]);
    if (std::is_integral<In>::value) {
      x_i = (x_i - input_quantization.zero_point) * input_quantization.scale;
    }
    float y_i = op_info.ReferenceImpl(x_i, params);
    if (std::is_integral<Out>::value) {
      y_i = y_i / output_quantization.scale + output_quantization.zero_point;
      y_i = std::max<float>(y_i, xnnpack::NumericLimits<Out>::min());
      y_i = std::min<float>(y_i, xnnpack::NumericLimits<Out>::max());
      y[i] = static_cast<Out>(std::lrint(y_i));
    } else {
      y[i] = y_i;
    }
  }
}

#endif  // THIRD_PARTY_XNNPACK_TEST_UNARY_OPS_H_
