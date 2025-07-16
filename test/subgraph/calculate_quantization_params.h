// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_TEST_SUBGRAPH_CALCULATE_QUANTIZATION_PARAMS_H_
#define XNNPACK_TEST_SUBGRAPH_CALCULATE_QUANTIZATION_PARAMS_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"

namespace xnnpack {

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
using qcuint4 = quantized<uint4x2, channelwise>;
using qbint4 = quantized<int4x2, blockwise>;
using qbuint4 = quantized<uint4x2, blockwise>;

// Bogus datatype used to indicate an invalid type.
using invalid_type = quantized<float>;

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

template <>
class NumericLimits<qbint4> {
 public:
  static int32_t min() { return -8; }
  static int32_t max() { return 7; }
  static int32_t smallest_normal() { return 0; }
  static int32_t min_identity() { return max(); }
  static int32_t max_identity() { return min(); }
};

template <>
class NumericLimits<qbuint4> {
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
    return xnn_datatype_qcuint4;
  } else if (std::is_same<T, qbint4>::value) {
    return xnn_datatype_qbint4;
  } else if (std::is_same<T, qbuint4>::value) {
    return xnn_datatype_qbuint4;
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

}  // namespace xnnpack

#endif  // XNNPACK_TEST_SUBGRAPH_CALCULATE_QUANTIZATION_PARAMS_H_
