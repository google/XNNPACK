// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>
#include <xnnpack/log.h>
#include <xnnpack/node-type.h>


// This function is defined inline when logging is disabled
#if XNN_LOG_LEVEL > 0
const char* xnn_node_type_to_string(enum xnn_node_type type) {
  switch (type) {
    case xnn_node_type_invalid:
      return "Invalid";
    case xnn_node_type_abs:
      return "Abs";
    case xnn_node_type_add2:
      return "Add2";
    case xnn_node_type_argmax_pooling_2d:
      return "ArgMax Pooling 2D";
    case xnn_node_type_average_pooling_2d:
      return "Average Pooling 2D";
    case xnn_node_type_bankers_rounding:
      return "Bankers Rounding";
    case xnn_node_type_ceiling:
      return "Ceiling";
    case xnn_node_type_clamp:
      return "Clamp";
    case xnn_node_type_concatenate2:
      return "Concatenate2";
    case xnn_node_type_concatenate3:
      return "Concatenate3";
    case xnn_node_type_concatenate4:
      return "Concatenate4";
    case xnn_node_type_convert:
      return "Convert";
    case xnn_node_type_convolution_2d:
      return "Convolution 2D";
    case xnn_node_type_copy:
      return "Copy";
    case xnn_node_type_deconvolution_2d:
      return "Deconvolution 2D";
    case xnn_node_type_depth_to_space:
      return "Depth To Space";
    case xnn_node_type_depthwise_convolution_2d:
      return "Depthwise Convolution 2D";
    case xnn_node_type_divide:
      return "Divide";
    case xnn_node_type_elu:
      return "ELU";
    case xnn_node_type_even_split2:
      return "Even Split2";
    case xnn_node_type_even_split3:
      return "Even Split3";
    case xnn_node_type_even_split4:
      return "Even Split4";
    case xnn_node_type_floor:
      return "Floor";
    case xnn_node_type_fully_connected:
      return "Fully Connected";
    case xnn_node_type_global_average_pooling_1d:
      return "Global Average Pooling 1D";
    case xnn_node_type_global_average_pooling_2d:
      return "Global Average Pooling 2D";
    case xnn_node_type_hardswish:
      return "HardSwish";
    case xnn_node_type_leaky_relu:
      return "Leaky ReLU";
    case xnn_node_type_max_pooling_2d:
      return "Max Pooling 2D";
    case xnn_node_type_maximum2:
      return "Maximum2";
    case xnn_node_type_minimum2:
      return "Minimum2";
    case xnn_node_type_multiply2:
      return "Multiply2";
    case xnn_node_type_negate:
      return "Negate";
    case xnn_node_type_prelu:
      return "PReLU";
    case xnn_node_type_sigmoid:
      return "Sigmoid";
    case xnn_node_type_softmax:
      return "Softmax";
    case xnn_node_type_space_to_depth_2d:
      return "Space To Depth 2D";
    case xnn_node_type_square:
      return "Square";
    case xnn_node_type_square_root:
      return "Square Root";
    case xnn_node_type_squared_difference:
      return "Squared Difference";
    case xnn_node_type_static_constant_pad:
      return "Static Constant Pad";
    case xnn_node_type_static_reshape:
      return "Static Reshape";
    case xnn_node_type_static_resize_bilinear_2d:
      return "Static Resize Bilinear 2D";
    case xnn_node_type_static_slice:
      return "Static Slice";
    case xnn_node_type_static_transpose:
      return "Static Transpose";
    case xnn_node_type_subtract:
      return "Subtract";
    case xnn_node_type_unpooling_2d:
      return "Unpooling 2D";
  }
  XNN_UNREACHABLE;
  return NULL;
}
#endif  // XNN_LOG_LEVEL > 0
