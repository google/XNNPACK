// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: src/enums/node-type.yaml
//   Generator: tools/generate-enum.py


#include <assert.h>
#include <stdint.h>

#include <xnnpack/node-type.h>


#if XNN_LOG_LEVEL > 0
static const uint16_t offset[55] = {
  0, 8, 12, 17, 35, 54, 71, 93, 101, 107, 120, 133, 146, 154, 169, 174, 191, 209, 234, 241, 245, 257, 269, 281, 287,
  303, 326, 352, 378, 400, 422, 432, 443, 458, 467, 476, 486, 493, 499, 504, 533, 541, 549, 567, 574, 586, 605, 625,
  637, 652, 678, 691, 708, 717, 722
};

static const char data[] =
  "Invalid\0"
  "Abs\0"
  "Add2\0"
  "ArgMax Pooling 2D\0"
  "Average Pooling 2D\0"
  "Bankers Rounding\0"
  "Batch Matrix Multiply\0"
  "Ceiling\0"
  "Clamp\0"
  "Concatenate2\0"
  "Concatenate3\0"
  "Concatenate4\0"
  "Convert\0"
  "Convolution 2D\0"
  "Copy\0"
  "Deconvolution 2D\0"
  "Depth To Space 2D\0"
  "Depthwise Convolution 2D\0"
  "Divide\0"
  "ELU\0"
  "Even Split2\0"
  "Even Split3\0"
  "Even Split4\0"
  "Floor\0"
  "Fully Connected\0"
  "Fully Connected Sparse\0"
  "Global Average Pooling 1D\0"
  "Global Average Pooling 2D\0"
  "Global Sum Pooling 1D\0"
  "Global Sum Pooling 2D\0"
  "HardSwish\0"
  "Leaky ReLU\0"
  "Max Pooling 2D\0"
  "Maximum2\0"
  "Minimum2\0"
  "Multiply2\0"
  "Negate\0"
  "PReLU\0"
  "RoPE\0"
  "Scaled Dot Product Attention\0"
  "Sigmoid\0"
  "Softmax\0"
  "Space To Depth 2D\0"
  "Square\0"
  "Square Root\0"
  "Squared Difference\0"
  "Static Constant Pad\0"
  "Static Mean\0"
  "Static Reshape\0"
  "Static Resize Bilinear 2D\0"
  "Static Slice\0"
  "Static Transpose\0"
  "Subtract\0"
  "Tanh\0"
  "Unpooling 2D";

const char* xnn_node_type_to_string(enum xnn_node_type node_type) {
  assert(node_type >= xnn_node_type_invalid);
  assert(node_type <= xnn_node_type_unpooling_2d);
  return &data[offset[node_type]];
}
#endif  // XNN_LOG_LEVEL > 0
