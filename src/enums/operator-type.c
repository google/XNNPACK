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


static const uint16_t offset[138] = {
  0, 8, 22, 36, 50, 64, 78, 92, 119, 147, 175, 203, 230, 257, 289, 307, 325, 350, 376, 392, 408, 423, 438, 460, 483,
  506, 529, 552, 575, 598, 616, 639, 663, 681, 704, 728, 752, 776, 800, 824, 848, 872, 886, 901, 916, 942, 968, 994,
  1020, 1052, 1084, 1110, 1137, 1164, 1181, 1198, 1232, 1266, 1280, 1294, 1308, 1324, 1340, 1366, 1392, 1418, 1444,
  1478, 1512, 1546, 1580, 1614, 1648, 1678, 1708, 1728, 1748, 1769, 1790, 1811, 1832, 1856, 1880, 1903, 1926, 1944,
  1962, 1977, 1992, 2010, 2028, 2047, 2066, 2085, 2104, 2121, 2138, 2154, 2170, 2198, 2226, 2254, 2282, 2309, 2336,
  2353, 2371, 2389, 2407, 2425, 2440, 2456, 2472, 2490, 2508, 2526, 2552, 2579, 2606, 2623, 2640, 2662, 2684, 2713,
  2742, 2761, 2780, 2799, 2818, 2833, 2848, 2863, 2878, 2897, 2917, 2937, 2958, 2979
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
  "Batch Matrix Multiply (NC, F32)\0"
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
  "Convert (NC, QS16, QS8)\0"
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
  "Dynamic Fully Connected (NC, F16)\0"
  "Dynamic Fully Connected (NC, F32)\0"
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
  "Global Sum Pooling (NWC, F16)\0"
  "Global Sum Pooling (NWC, F32)\0"
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
  "Mean (ND, F16)\0"
  "Mean (ND, F32)\0"
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
  "RoPE (NTHC, F32)\0"
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
  "Tanh (NC, F16)\0"
  "Tanh (NC, F32)\0"
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
