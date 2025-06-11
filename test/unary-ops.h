// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_TEST_UNARY_OPS_H_
#define THIRD_PARTY_XNNPACK_TEST_UNARY_OPS_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <type_traits>

#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/reference-utils.h"

static float TolExact(float) { return 0.0f; }
static float TolExact16(float y_ref) {
  // The maximum of the relative tolerance and half the smallest positive
  // normal.
  return std::max(
      std::abs(y_ref) * xnnpack::NumericLimits<xnn_float16>::epsilon(),
      0.5f * xnnpack::NumericLimits<xnn_float16>::epsilon());
}

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
        return {xnnpack::NumericLimits<xnn_float16>::epsilon(),
                xnnpack::NumericLimits<xnn_float16>::infinity()};
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
  virtual int32_t ReferenceImpl(int32_t x,
                                const xnn_unary_params& params) const {
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
#if XNN_ARCH_HEXAGON
        return TolExact(y_ref) * 4.f;
#else
        return TolExact(y_ref);
#endif
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

  // If this returns false, we do not promise to match the reference
  // implementation within `Tolerance`.
  virtual bool IsInSupportedRange(float y) const { return true; }
};

struct Convert : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return x;
  }
  int32_t ReferenceImpl(int32_t x, const xnn_unary_params&) const override {
    return x;
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    return xnn_datatype_is_quantized(datatype)
               ? 1.0f
               : TolMixed(y_ref, xnnpack::epsilon(datatype),
                          xnnpack::epsilon(datatype));
  }
};

struct Abs : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::abs(x);
  }
  int32_t ReferenceImpl(int32_t x, const xnn_unary_params&) const override {
    return std::abs(x);
  }
};

struct Negate : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return -x;
  }
  int32_t ReferenceImpl(int32_t x, const xnn_unary_params&) const override {
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
  int32_t ReferenceImpl(int32_t x,
                        const xnn_unary_params& params) const override {
    return std::min<int32_t>(std::max<int32_t>(x, params.clamp.min),
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
    switch (datatype) {
      case xnn_datatype_fp32:
        return TolMixed(y_ref, 10 * std::numeric_limits<float>::epsilon(),
                        5 * std::numeric_limits<float>::epsilon());
      case xnn_datatype_fp16:
        return TolMixed(y_ref, 10 * 9.77e-04, 5 * 9.77e-04);
      case xnn_datatype_bf16:
        return TolMixed(y_ref, 10 * 7.8125e-3, 5 * 7.8125e-3);
      case xnn_datatype_qint8:
      case xnn_datatype_quint8:
        return 1;
      default:
        XNN_UNREACHABLE;
    }
  }

  Interval Domain(xnn_datatype) const override { return {-10.0f, 10.0f}; }
};

struct ApproxGELU : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return x * 0.5f *
           (1.0f +
            std::tanh(std::sqrt(2.0f / M_PI) * x * (1 + 0.044715f * x * x)));
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp32:
        return TolMixed(y_ref, 10 * std::numeric_limits<float>::epsilon(),
                        5 * std::numeric_limits<float>::epsilon());
      case xnn_datatype_fp16:
        return TolMixed(y_ref, 10 * 9.77e-04, 5 * 9.77e-04);
      case xnn_datatype_bf16:
        return TolMixed(y_ref, 10 * 7.8125e-3, 5 * 7.8125e-3);
      case xnn_datatype_qint8:
      case xnn_datatype_quint8:
        return 1;
      default:
        XNN_UNREACHABLE;
    }
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
      case xnn_datatype_bf16:
        return TolMixed(y_ref, 1.0e-2f, 5.0e-2f);
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
      case xnn_datatype_fp16:
      case xnn_datatype_bf16:
        return TolMixed(y_ref, xnnpack::epsilon(datatype),
                        5 * xnnpack::epsilon(datatype));
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

#if XNN_ARCH_RISCV
  bool IsInSupportedRange(float y) const override {
    // TODO(#8087): These ops are broken for large inputs on RISCV.
    return std::abs(y) < 1e6f;
  }
#endif
};

struct RoundTowardsZero : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::trunc(x);
  }

#if XNN_ARCH_RISCV
  bool IsInSupportedRange(float y) const override {
    // TODO(#8087): These ops are broken for large inputs on RISCV.
    return std::abs(y) < 1e6f;
  }
#endif
};

struct RoundUp : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::ceil(x);
  }

#if XNN_ARCH_RISCV
  bool IsInSupportedRange(float y) const override {
    // TODO(#8087): These ops are broken for large inputs on RISCV.
    return std::abs(y) < 1e6f;
  }
#endif
};

struct RoundDown : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::floor(x);
  }

#if XNN_ARCH_RISCV
  bool IsInSupportedRange(float y) const override {
    // TODO(#8087): These ops are broken for large inputs on RISCV.
    return std::abs(y) < 1e6f;
  }
#endif
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
      case xnn_datatype_bf16:
        return TolMixed(y_ref, 1.0e-3f, 1.0e-2f);
      case xnn_datatype_qint8:
      case xnn_datatype_quint8:
        return 1;
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

struct Sine : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::sin(x);
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp32:
      case xnn_datatype_fp16:
      case xnn_datatype_bf16:
        return TolMixed(y_ref, 3 * xnnpack::epsilon(datatype),
                        5 * xnnpack::epsilon(datatype));
      case xnn_datatype_qint8:
      case xnn_datatype_quint8:
        return 1;
      default:
        return TolExact(y_ref);
    }
  }

  Interval Domain(xnn_datatype datatype) const override {
    return {-100.0f, 100.0f};
  }
};

struct Square : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return x * x;
  }
  int32_t ReferenceImpl(int32_t x, const xnn_unary_params&) const override {
    return static_cast<int32_t>(static_cast<int64_t>(x) *
                                static_cast<int64_t>(x));
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp32:
      case xnn_datatype_fp16:
      case xnn_datatype_bf16:
        return TolMixed(y_ref, xnnpack::epsilon(datatype),
                        xnnpack::epsilon(datatype));
      case xnn_datatype_qint8:
      case xnn_datatype_quint8:
        return 1;
      case xnn_datatype_int32:
        // Overflow makes this hard to test.
        return std::numeric_limits<float>::infinity();
      default:
        XNN_UNREACHABLE;
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
        return TolRelative(y_ref, 2.0f * std::numeric_limits<float>::epsilon());
      case xnn_datatype_fp16:
      case xnn_datatype_bf16:
        return TolRelative(y_ref, 3.0f * xnnpack::epsilon(datatype));
      case xnn_datatype_qint8:
      case xnn_datatype_quint8:
        return 1;
      default:
        XNN_UNREACHABLE;
    }
  }

  Interval Domain(xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp16:
        return {1.0e-4f, 10.0f};
      case xnn_datatype_fp32:
        // The reciprocal square root estimate instructions (e.g. `vrsqrteq_f32`
        // for Arm or `_m*_rsqrt_ps` for Intel) seem to fail for values larger
        // than the inverse of the minimum normalized number when denormals are
        // switched off, so limit the range to that of normally inversible
        // numbers.
        return {std::numeric_limits<float>::epsilon(),
                std::numeric_limits<float>::max() / 4};
      default:
        return Interval::Positive(datatype);
    }
  }

  bool IsInSupportedRange(float y) const override {
    // TODO(b/404943039): We have some cases where inf input produces NaN
    // output, that the reference implementation disagrees with.
    return !std::isnan(y) && !std::isinf(y);
  }
};

struct TanH : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::tanh(x);
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp32:
      case xnn_datatype_fp16:
      case xnn_datatype_bf16:
        return TolMixed(y_ref, xnnpack::epsilon(datatype),
                        4.0f * xnnpack::epsilon(datatype));  // 4 ULP
      default:
        return 1;
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
        return TolRelative(y_ref, 2 * std::numeric_limits<float>::epsilon());
      case xnn_datatype_fp16:
      case xnn_datatype_bf16:
        return TolMixed(y_ref, 1.0e-4f, 5.0e-3f);
      case xnn_datatype_qint8:
      case xnn_datatype_quint8:
        return 1;
      default:
        return TolExact(y_ref);
    }
  }

  Interval Domain(xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp16:
        return {1.0e-4f, 10.0f};
      case xnn_datatype_fp32:
        // The reciprocal square root estimate instructions (e.g. `vrsqrteq_f32`
        // for Arm or `_m*_rsqrt_ps` for Intel) seem to fail for values larger
        // than the inverse of the minimum normalized number when denormals are
        // switched off, so limit the range to that of normally inversible
        // numbers.
        return {std::numeric_limits<float>::epsilon(),
                std::numeric_limits<float>::max() / 4};
      default:
        return Interval::Positive(datatype);
    }
  }

  bool IsInSupportedRange(float y) const override {
    // TODO(b/404943039): If the input is inf, the result is 0, but we produce
    // NaN.
    return y != 0.0f;
  }
};

struct Log : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::log(x);
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    return TolMixed(y_ref, 2 * xnnpack::epsilon(datatype),
                    6 * xnnpack::epsilon(datatype));
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
    return TolMixed(y_ref, 2 * xnnpack::epsilon(datatype),
                    6 * xnnpack::epsilon(datatype));
  }
  Interval Domain(xnn_datatype) const override { return {-10.0f, 10.0f}; }
};

struct CubeRoot : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::cbrt(x);
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    return TolRelative(y_ref, 2.5f * xnnpack::epsilon(datatype));
  }

  Interval Domain(xnn_datatype datatype) const override {
    if (datatype == xnn_datatype_fp16 || datatype == xnn_datatype_bf16) {
      return {0.001f, 10.0f};
    } else {
      return Interval::Positive(datatype);
    }
  }
};

struct Cosine : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return std::cos(x);
  }

  float Tolerance(float y_ref, xnn_datatype datatype) const override {
    switch (datatype) {
      case xnn_datatype_fp32:
      case xnn_datatype_fp16:
      case xnn_datatype_bf16:
        return TolMixed(y_ref, 3 * xnnpack::epsilon(datatype),
                        5 * xnnpack::epsilon(datatype));
      case xnn_datatype_qint8:
      case xnn_datatype_quint8:
        return 1;
      default:
        return TolExact(y_ref);
    }
  }

  Interval Domain(xnn_datatype datatype) const override {
    return {-100.0f, 100.0f};
  }
};

struct CountLeadingZeros : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return (float)math_clz_u32((int)x);
  }
  int32_t ReferenceImpl(int32_t x, const xnn_unary_params&) const override {
    return math_clz_u32(x);
  }
};

struct BitwiseNot : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return ~(int)x;
  }
  int32_t ReferenceImpl(int32_t x, const xnn_unary_params&) const override {
    return ~x;
  }
};

struct Popcount : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return (float)math_popcount_u32((int)x);
  }
  int32_t ReferenceImpl(int32_t x, const xnn_unary_params&) const override {
    return math_popcount_u32(x);
  }
};

struct Sign : public UnaryOpInfo {
  float ReferenceImpl(float x, const xnn_unary_params&) const override {
    return x < 0.0f ? -1.0f : x > 0.0f ? 1.0f : 0.0f;
  }
  int32_t ReferenceImpl(int32_t x, const xnn_unary_params&) const override {
    return x < 0 ? -1 : x > 0 ? 1 : 0;
  }
};

const UnaryOpInfo* GetUnaryOpInfo(xnn_unary_operator op);

// Compute the result of a unary operator using the reference implementation.
template <typename In, typename Out, typename UnaryOp>
void UnaryReferenceImpl(
    const xnnpack::quantized<In>* x, size_t n, xnnpack::quantized<Out>* y,
    const UnaryOp& op_info,
    const xnn_quantization_params& input_quantization = {0, 1.0f},
    const xnn_quantization_params& output_quantization = {0, 1.0f},
    const xnn_unary_params& params = xnn_unary_params()) {
  for (size_t i = 0; i < n; i++) {
    float x_i =
        (x[i] - input_quantization.zero_point) * input_quantization.scale;
    float y_i = op_info.ReferenceImpl(x_i, params);
    y_i = y_i / output_quantization.scale + output_quantization.zero_point;
    y[i] = xnnpack::round_float_to_int<Out>(y_i);
  }
}

// Compute the result of a unary operator using the reference implementation.
template <typename In, typename Out, typename UnaryOp>
void UnaryReferenceImpl(
    const In* x, size_t n, xnnpack::quantized<Out>* y, const UnaryOp& op_info,
    const xnn_quantization_params& input_quantization = {0, 1.0f},
    const xnn_quantization_params& output_quantization = {0, 1.0f},
    const xnn_unary_params& params = xnn_unary_params()) {
  static_assert(!xnnpack::is_quantized<In>::value, "");
  for (size_t i = 0; i < n; i++) {
    float y_i = op_info.ReferenceImpl(static_cast<float>(x[i]), params);
    y_i = y_i / output_quantization.scale + output_quantization.zero_point;
    y[i] = xnnpack::round_float_to_int<Out>(y_i);
  }
}

// Compute the result of a unary operator using the reference implementation.
template <typename In, typename Out, typename UnaryOp>
void UnaryReferenceImpl(
    const xnnpack::quantized<In>* x, size_t n, Out* y, const UnaryOp& op_info,
    const xnn_quantization_params& input_quantization = {0, 1.0f},
    const xnn_quantization_params& output_quantization = {0, 1.0f},
    const xnn_unary_params& params = xnn_unary_params()) {
  static_assert(!xnnpack::is_quantized<Out>::value, "");
  for (size_t i = 0; i < n; i++) {
    float x_i =
        (x[i] - input_quantization.zero_point) * input_quantization.scale;
    float y_i = op_info.ReferenceImpl(x_i, params);
    if (std::is_integral<Out>::value) {
      y[i] = xnnpack::round_float_to_int<Out>(y_i);
    } else {
      y[i] = y_i;
    }
  }
}

// Compute the result of a unary operator using the reference implementation.
template <typename In, typename Out, typename UnaryOp>
void UnaryReferenceImpl(
    const In* x, size_t n, Out* y, const UnaryOp& op_info,
    const xnn_quantization_params& input_quantization = {0, 1.0f},
    const xnn_quantization_params& output_quantization = {0, 1.0f},
    const xnn_unary_params& params = xnn_unary_params()) {
  static_assert(!xnnpack::is_quantized<In>::value, "");
  static_assert(!xnnpack::is_quantized<Out>::value, "");
  for (size_t i = 0; i < n; i++) {
    float y_i;
    if (std::is_integral<In>::value && std::is_integral<Out>::value) {
      y[i] = op_info.ReferenceImpl((int32_t)x[i], params);
    } else {
      if (std::is_integral<In>::value) {
        y_i = op_info.ReferenceImpl((int32_t)x[i], params);
      } else {
        y_i = op_info.ReferenceImpl(static_cast<float>(x[i]), params);
      }
      if (std::is_integral<Out>::value) {
        y[i] = xnnpack::round_float_to_int<Out>(y_i);
      } else {
        y[i] = y_i;
      }
    }
  }
}

#endif  // THIRD_PARTY_XNNPACK_TEST_UNARY_OPS_H_
