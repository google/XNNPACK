// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: src/operator-strings.yaml
//   Generator: tools/generate-enum-strings.py


#include <assert.h>
#include <stdint.h>

#include <xnnpack/operator-type.h>

static const uint16_t offset[] = {0,8,22,36,50,64,78,105,133,161,188,206,231,257,273,289,304,319,341,364,387,410,433,456,479,502,525,549,573,597,621,645,669,683,698,713,739,765,791,817,849,875,902,929,946,960,974,990,1016,1042,1068,1094,1128,1162,1196,1230,1264,1284,1304,1325,1346,1367,1391,1415,1438,1461,1479,1497,1516,1535,1554,1573,1590,1606,1622,1650,1678,1705,1732,1760,1778,1796,1814,1832,1850,1868,1886,1903,1925,1954,1973,1992,2011,2026,2041,2062,2082,2102};

static const char *data =
    "Invalid\0"
    "Abs (NC, F32)\0"
    "Add (ND, F16)\0"
    "Add (ND, F32)\0"
    "Add (ND, QS8)\0"
    "Add (ND, QU8)\0"
    "ArgMax Pooling (NHWC, F32)\0"
    "Average Pooling (NHWC, F32)\0"
    "Average Pooling (NHWC, QU8)\0"
    "Bankers Rounding (NC, F32)\0"
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
    "Convert (NC, QS8, F32)\0"
    "Convert (NC, QU8, F32)\0"
    "Convolution (NHWC, F16)\0"
    "Convolution (NHWC, F32)\0"
    "Convolution (NHWC, QC8)\0"
    "Convolution (NHWC, QS8)\0"
    "Convolution (NHWC, QU8)\0"
    "Convolution (NCHW, F32)\0"
    "Copy (NC, X8)\0"
    "Copy (NC, X16)\0"
    "Copy (NC, X32)\0"
    "Deconvolution (NHWC, F16)\0"
    "Deconvolution (NHWC, F32)\0"
    "Deconvolution (NHWC, QS8)\0"
    "Deconvolution (NHWC, QU8)\0"
    "Depth To Space (NCHW2NHWC, X32)\0"
    "Depth To Space (NHWC, X8)\0"
    "Depth To Space (NHWC, X16)\0"
    "Depth To Space (NHWC, X32)\0"
    "Divide (ND, F32)\0"
    "ELU (NC, F32)\0"
    "ELU (NC, QS8)\0"
    "Floor (NC, F32)\0"
    "Fully Connected (NC, F16)\0"
    "Fully Connected (NC, F32)\0"
    "Fully Connected (NC, QS8)\0"
    "Fully Connected (NC, QU8)\0"
    "Global Average Pooling (NWC, F16)\0"
    "Global Average Pooling (NWC, F32)\0"
    "Global Average Pooling (NWC, QS8)\0"
    "Global Average Pooling (NWC, QU8)\0"
    "Global Average Pooling (NCW, F32)\0"
    "HardSwish (NC, F16)\0"
    "HardSwish (NC, F32)\0"
    "Leaky ReLU (NC, F16)\0"
    "Leaky ReLU (NC, F32)\0"
    "Leaky ReLU (NC, QU8)\0"
    "Max Pooling (NHWC, F16)\0"
    "Max Pooling (NHWC, F32)\0"
    "Max Pooling (NHWC, S8)\0"
    "Max Pooling (NHWC, U8)\0"
    "Maximum (ND, F32)\0"
    "Minimum (ND, F32)\0"
    "Multiply (ND, F16)\0"
    "Multiply (ND, F32)\0"
    "Multiply (ND, QS8)\0"
    "Multiply (ND, QU8)\0"
    "Negate (NC, F32)\0"
    "PReLU (NC, F16)\0"
    "PReLU (NC, F32)\0"
    "Resize Bilinear (NHWC, F16)\0"
    "Resize Bilinear (NHWC, F32)\0"
    "Resize Bilinear (NHWC, S8)\0"
    "Resize Bilinear (NHWC, U8)\0"
    "Resize Bilinear (NCHW, F32)\0"
    "Sigmoid (NC, F16)\0"
    "Sigmoid (NC, F32)\0"
    "Sigmoid (NC, QS8)\0"
    "Sigmoid (NC, QU8)\0"
    "Softmax (NC, F16)\0"
    "Softmax (NC, F32)\0"
    "Softmax (NC, QU8)\0"
    "Square (NC, F32)\0"
    "Square Root (NC, F32)\0"
    "Squared Difference (NC, F32)\0"
    "Subtract (ND, F32)\0"
    "Subtract (ND, QS8)\0"
    "Subtract (ND, QU8)\0"
    "Tanh (NC, QS8)\0"
    "Tanh (NC, QU8)\0"
    "Truncation (NC, F32)\0"
    "Transpose (ND, X16)\0"
    "Transpose (ND, X32)\0"
    "Unpooling (NHWC, X32)\0"
;

const char* xnn_operator_type_to_string(enum xnn_operator_type type) {
  assert(type <= xnn_operator_type_unpooling_nhwc_x32);
  return &data[offset[type]];
}