// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: src/enums/operator-type.yaml
//   Generator: tools/generate-enum.py


#include <assert.h>
#include <stdint.h>

#include <xnnpack/operator-type.h>


static const uint16_t offset[130] = {
  0, 8, 22, 36, 50, 64, 78, 92, 119, 147, 175, 203, 230, 257, 283, 310, 337, 355, 373, 398, 424, 440, 456, 471, 486,
  508, 531, 554, 577, 600, 623, 646, 664, 687, 705, 728, 752, 776, 800, 824, 848, 872, 896, 910, 925, 940, 966, 992,
  1018, 1044, 1076, 1108, 1134, 1161, 1188, 1205, 1222, 1236, 1250, 1264, 1280, 1296, 1322, 1348, 1374, 1400, 1434,
  1468, 1502, 1536, 1570, 1604, 1624, 1644, 1665, 1686, 1707, 1728, 1752, 1776, 1799, 1822, 1840, 1858, 1876, 1894,
  1913, 1932, 1951, 1970, 1987, 2004, 2020, 2036, 2064, 2092, 2120, 2148, 2175, 2202, 2220, 2238, 2256, 2274, 2289,
  2305, 2321, 2339, 2357, 2375, 2401, 2428, 2455, 2472, 2489, 2511, 2533, 2562, 2591, 2610, 2629, 2648, 2667, 2682,
  2697, 2716, 2736, 2756, 2777, 2798
};

static const char data[] = 
  "Invalid\0"
  "Abs (NC, F16)\0"
  "Abs (NC, F32)\0"
  "Add (ND, F16)\0"
  "Add (ND, F32)\0"
  "Add (ND, QS8)\0"
  "Add (ND, QU8)\0"
  "ArgMax Pooling (NHWC, F32)\0"
  "Average Pooling (NHWC, F16)\0"
  "Average Pooling (NHWC, F32)\0"
  "Average Pooling (NHWC, QU8)\0"
  "Bankers Rounding (NC, F16)\0"
  "Bankers Rounding (NC, F32)\0"
  "Batch to Space (NHWC, X8)\0"
  "Batch to Space (NHWC, X16)\0"
  "Batch to Space (NHWC, X32)\0"
  "Ceiling (NC, F16)\0"
  "Ceiling (NC, F32)\0"
  "Channel Shuffle (NC, X8)\0"
  "Channel Shuffle (NC, X32)\0"
  "Clamp (NC, F16)\0"
  "Clamp (NC, F32)\0"
  "Clamp (NC, S8)\0"
  "Clamp (NC, U8)\0"
  "Constant Pad (ND, X8)\0"
  "Constant Pad (ND, X16)\0"
  "Constant Pad (ND, X32)\0"
  "Convert (NC, F16, F32)\0"
  "Convert (NC, F32, F16)\0"
  "Convert (NC, F32, QS8)\0"
  "Convert (NC, F32, QU8)\0"
  "Convert (NC, QS8)\0"
  "Convert (NC, QS8, F32)\0"
  "Convert (NC, QU8)\0"
  "Convert (NC, QU8, F32)\0"
  "Convolution (NCHW, F16)\0"
  "Convolution (NCHW, F32)\0"
  "Convolution (NHWC, F16)\0"
  "Convolution (NHWC, F32)\0"
  "Convolution (NHWC, QC8)\0"
  "Convolution (NHWC, QS8)\0"
  "Convolution (NHWC, QU8)\0"
  "Copy (NC, X8)\0"
  "Copy (NC, X16)\0"
  "Copy (NC, X32)\0"
  "Deconvolution (NHWC, F16)\0"
  "Deconvolution (NHWC, F32)\0"
  "Deconvolution (NHWC, QS8)\0"
  "Deconvolution (NHWC, QU8)\0"
  "Depth To Space (NCHW2NHWC, X16)\0"
  "Depth To Space (NCHW2NHWC, X32)\0"
  "Depth To Space (NHWC, X8)\0"
  "Depth To Space (NHWC, X16)\0"
  "Depth To Space (NHWC, X32)\0"
  "Divide (ND, F16)\0"
  "Divide (ND, F32)\0"
  "ELU (NC, F16)\0"
  "ELU (NC, F32)\0"
  "ELU (NC, QS8)\0"
  "Floor (NC, F16)\0"
  "Floor (NC, F32)\0"
  "Fully Connected (NC, F16)\0"
  "Fully Connected (NC, F32)\0"
  "Fully Connected (NC, QS8)\0"
  "Fully Connected (NC, QU8)\0"
  "Global Average Pooling (NCW, F16)\0"
  "Global Average Pooling (NCW, F32)\0"
  "Global Average Pooling (NWC, F16)\0"
  "Global Average Pooling (NWC, F32)\0"
  "Global Average Pooling (NWC, QS8)\0"
  "Global Average Pooling (NWC, QU8)\0"
  "HardSwish (NC, F16)\0"
  "HardSwish (NC, F32)\0"
  "Leaky ReLU (NC, F16)\0"
  "Leaky ReLU (NC, F32)\0"
  "Leaky ReLU (NC, QS8)\0"
  "Leaky ReLU (NC, QU8)\0"
  "Max Pooling (NHWC, F16)\0"
  "Max Pooling (NHWC, F32)\0"
  "Max Pooling (NHWC, S8)\0"
  "Max Pooling (NHWC, U8)\0"
  "Maximum (ND, F16)\0"
  "Maximum (ND, F32)\0"
  "Minimum (ND, F16)\0"
  "Minimum (ND, F32)\0"
  "Multiply (ND, F16)\0"
  "Multiply (ND, F32)\0"
  "Multiply (ND, QS8)\0"
  "Multiply (ND, QU8)\0"
  "Negate (NC, F16)\0"
  "Negate (NC, F32)\0"
  "PReLU (NC, F16)\0"
  "PReLU (NC, F32)\0"
  "Resize Bilinear (NCHW, F16)\0"
  "Resize Bilinear (NCHW, F32)\0"
  "Resize Bilinear (NHWC, F16)\0"
  "Resize Bilinear (NHWC, F32)\0"
  "Resize Bilinear (NHWC, S8)\0"
  "Resize Bilinear (NHWC, U8)\0"
  "Sigmoid (NC, F16)\0"
  "Sigmoid (NC, F32)\0"
  "Sigmoid (NC, QS8)\0"
  "Sigmoid (NC, QU8)\0"
  "Slice (ND, X8)\0"
  "Slice (ND, X16)\0"
  "Slice (ND, X32)\0"
  "Softmax (NC, F16)\0"
  "Softmax (NC, F32)\0"
  "Softmax (NC, QU8)\0"
  "Space To Depth (NHWC, X8)\0"
  "Space To Depth (NHWC, X16)\0"
  "Space To Depth (NHWC, X32)\0"
  "Square (NC, F16)\0"
  "Square (NC, F32)\0"
  "Square Root (NC, F16)\0"
  "Square Root (NC, F32)\0"
  "Squared Difference (NC, F16)\0"
  "Squared Difference (NC, F32)\0"
  "Subtract (ND, F16)\0"
  "Subtract (ND, F32)\0"
  "Subtract (ND, QS8)\0"
  "Subtract (ND, QU8)\0"
  "Tanh (NC, QS8)\0"
  "Tanh (NC, QU8)\0"
  "Transpose (ND, X8)\0"
  "Transpose (ND, X16)\0"
  "Transpose (ND, X32)\0"
  "Truncation (NC, F16)\0"
  "Truncation (NC, F32)\0"
  "Unpooling (NHWC, X32)";

const char* xnn_operator_type_to_string(enum xnn_operator_type operator_type) {
  assert(operator_type >= xnn_operator_type_invalid);
  assert(operator_type <= xnn_operator_type_unpooling_nhwc_x32);
  return &data[offset[operator_type]];
}
