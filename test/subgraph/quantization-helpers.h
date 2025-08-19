// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_TEST_SUBGRAPH_CALCULATE_QUANTIZATION_PARAMS_H_
#define XNNPACK_TEST_SUBGRAPH_CALCULATE_QUANTIZATION_PARAMS_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/quantization.h"

namespace xnnpack {

// Dynamic quantization looks a lot like a float input/output, but the error is
// hard to quantify and test well. Rather than do that, we can just generate
// input data that has (close to) zero error when dynamically quantized, which
// makes it easier to test.
template <typename Data>
void FakeDynamicQuantize(
    Tensor<Data> input, float qmin, float qmax,
    const struct xnn_qd8_quantization_params* quantization_params) {
  for (auto& i : input) {
    const int32_t q =
        std::min(std::max(std::round(i * quantization_params->inv_scale) +
                              quantization_params->zero_point,
                          qmin),
                 qmax);
    i = (q - quantization_params->zero_point) / quantization_params->inv_scale;
  }
}

template <typename Data>
void FakeDynamicQuantize(Tensor<Data> input, xnn_datatype datatype) {
  // Since the re-quantized values may have a different min/max than the
  // original values, and thus a different scale, we have to requantize
  // iteratively until the quantization scale converges. This usually happens in
  // a single iteration, modulo rounding errors.
  for (size_t iter = 0; iter < 10; iter++) {
    auto minmax = std::minmax_element(input.begin(), input.end());
    const float rmin = *minmax.first;
    const float rmax = *minmax.second;
    struct xnn_qd8_quantization_params quantization_params;
    if (datatype == xnn_datatype_qdint8) {
      quantization_params = xnn_qd8_asymmetric_quantization_params(rmin, rmax);
      FakeDynamicQuantize(input, INT8_MIN, INT8_MAX, &quantization_params);
    } else if (datatype == xnn_datatype_qduint8) {
      quantization_params = xnn_qdu8_asymmetric_quantization_params(rmin, rmax);
      FakeDynamicQuantize(input, 0, UINT8_MAX, &quantization_params);
    } else {
      XNN_UNREACHABLE;
    }
    if (std::max(std::abs(rmin - *minmax.first),
                 std::abs(rmax - *minmax.second)) <
        1e-6 * quantization_params.inv_scale) {
      return;
    }
  }
  GTEST_FAIL() << "FakeDynamicQuantize failed to converge.";
}

template <typename Data>
void FakeDynamicQuantize(const Tensor<quantized<Data>>& input, xnn_datatype) {}

// The next chunk of things enables us to work with int4 data in the
// datatype/Tensor template system. It's not a perfect abstraction, I think with
// a little bit of improvement, this could be a clean mechanism.

// Two int4 values stored in an int8.
struct int4x2 {
  uint8_t value;

  int4x2() = default;
  int4x2(uint8_t value) : value(value) {}  // NOLINT

  int8_t operator[](size_t i) const {
    int8_t result = (value >> (i * 4)) & 0xf;
    // Sign extend
    result = static_cast<int8_t>(result << 4) >> 4;
    return result;
  }
};

struct uint4x2 {
  uint8_t value;

  uint4x2() = default;
  uint4x2(uint8_t value) : value(value) {}  // NOLINT

  uint8_t operator[](size_t i) const { return (value >> (i * 4)) & 0xf; }
};

using quint8 = quantized<uint8_t>;
using qint8 = quantized<int8_t>;
using qcint8 = quantized<int8_t, channelwise>;
using qint32 = quantized<int32_t>;
using qcint32 = quantized<int32_t, channelwise>;
using qcint4 = quantized<int4x2, channelwise>;

// Bogus datatype used to indicate an invalid type.
using invalid_type = quantized<float>;

// This is not a "real" XNNPACK datatype, but it is required to match the
// behavior of F32QC4W (b/407771627).
using qcuint4 = quantized<uint4x2, channelwise>;

template <>
class NumericLimits<qcint4> {
 public:
  static int32_t min() { return -8; }
  static int32_t max() { return 7; }
  static int32_t smallest_normal() { return 0; }
  static int32_t min_identity() { return max(); }
  static int32_t max_identity() { return min(); }
};

template <>
class NumericLimits<qcuint4> {
 public:
  static int32_t min() { return 0; }
  static int32_t max() { return 15; }
  static int32_t smallest_normal() { return 0; }
  static int32_t min_identity() { return max(); }
  static int32_t max_identity() { return min(); }
};

template <typename T>
xnn_datatype datatype_of() {
  if (std::is_same<T, qcint4>::value) {
    return xnn_datatype_qcint4;
  } else if (std::is_same<T, qcuint4>::value) {
    return xnn_datatype_qcint4;
  } else {
    return xnn_datatype_of<T>();
  }
}

template <typename T>
static float max_abs_bias() {
  if (std::is_same<T, qint32>::value) {
    return 10000;
  }
  return 1.0f;
};

template <typename T>
xnn_quantization_params quantization_for_range(float min, float max) {
  xnn_quantization_params result;
  result.scale = (max - min) / (static_cast<float>(NumericLimits<T>::max()) -
                                static_cast<float>(NumericLimits<T>::min()));
  result.zero_point = NumericLimits<T>::min() - min / result.scale;
  return result;
}

template <typename Input, typename Filter, typename Output,
          typename Bias = invalid_type>
xnn_quantization_params CalculateGEMMQuantizationParams(
    size_t reduction_size, xnn_quantization_params input_quantization,
    xnn_quantization_params filter_quantization,
    xnn_quantization_params bias_quantization) {
  if (!xnn_datatype_is_quantized(datatype_of<Output>())) {
    return {0, 1.0f};
  }

  // Get the dequantized input and filter ranges.
  const float input_min =
      dequantize(NumericLimits<Input>::min(), input_quantization);
  const float input_max =
      dequantize(NumericLimits<Input>::max(), input_quantization);
  const float filter_min =
      dequantize(NumericLimits<Filter>::min(), filter_quantization);
  const float filter_max =
      dequantize(NumericLimits<Filter>::max(), filter_quantization);
  const float bias_min = dequantize(-max_abs_bias<Bias>(), bias_quantization);
  const float bias_max = dequantize(max_abs_bias<Bias>(), bias_quantization);

  // Find the range of the product of an input and a filter value.
  std::array<float, 4> corners = {
      input_min * filter_min,
      input_max * filter_min,
      input_min * filter_max,
      input_max * filter_max,
  };
  auto input_filter_minmax =
      std::minmax_element(corners.begin(), corners.end());

  const bool use_bias = std::is_same<Bias, invalid_type>::value;
  const float output_min =
      *input_filter_minmax.first * reduction_size + bias_min * use_bias;
  const float output_max =
      *input_filter_minmax.second * reduction_size + bias_max * use_bias;

  // Now we want the output quantization to hold the range of the output.
  return quantization_for_range<Output>(output_min, output_max);
}

// For float types, generate data in [-1, 1]
template <typename T>
DatatypeGenerator<T> MakeDatatypeGenerator(T,
                                           bool /*symmetric_range*/ = false) {
  return DatatypeGenerator<T>(-1.0f, 1.0f);
}
template <typename T>
T MaxDatatype(T) {
  return 1.0f;
}

// For quantized types, generate the full range of the type.
template <typename T, typename Kind>
DatatypeGenerator<quantized<T, Kind>> MakeDatatypeGenerator(
    quantized<T, Kind>, bool symmetric_range = false) {
  if (symmetric_range) {
    const int type_min = NumericLimits<quantized<T, Kind>>::min();
    const int type_max = NumericLimits<quantized<T, Kind>>::max();
    // We don't allow the RHS of a GEMM to use the most negative value to avoid
    // overflow issues for some kernels.
    return DatatypeGenerator<quantized<T, Kind>>(std::max(type_min, -type_max),
                                                 type_max);
  } else {
    return DatatypeGenerator<quantized<T, Kind>>();
  }
}

template <typename T, typename Kind>
int32_t MaxDatatype(quantized<T, Kind>) {
  return NumericLimits<quantized<T, Kind>>::max();
}

template <>
inline DatatypeGenerator<quantized<int32_t>> MakeDatatypeGenerator(
    quantized<int32_t>, bool /*symmetric_range*/) {
  return DatatypeGenerator<quantized<int32_t>>(-10000, 10000, {0, 1.0f});
}

}  // namespace xnnpack

#endif  // XNNPACK_TEST_SUBGRAPH_CALCULATE_QUANTIZATION_PARAMS_H_
