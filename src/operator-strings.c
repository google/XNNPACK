// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>
#include <xnnpack/log.h>
#include <xnnpack/subgraph.h>


// This function is defined inline when logging is disabled
#if XNN_LOG_LEVEL > 0
const char* xnn_operator_type_to_string(enum xnn_operator_type type) {
  switch (type) {
    case xnn_operator_type_invalid:
      return "Invalid";
    case xnn_operator_type_abs_nc_f32:
      return "Abs (NC, F32)";
    case xnn_operator_type_add_nd_f16:
      return "Add (ND, F16)";
    case xnn_operator_type_add_nd_f32:
      return "Add (ND, F32)";
    case xnn_operator_type_add_nd_qs8:
      return "Add (ND, QS8)";
    case xnn_operator_type_add_nd_qu8:
      return "Add (ND, QU8)";
    case xnn_operator_type_argmax_pooling_nhwc_f32:
      return "ArgMax Pooling (NHWC, F32)";
    case xnn_operator_type_average_pooling_nhwc_f32:
      return "Average Pooling (NHWC, F32)";
    case xnn_operator_type_average_pooling_nhwc_qu8:
      return "Average Pooling (NHWC, QU8)";
    case xnn_operator_type_bankers_rounding_nc_f32:
      return "Bankers Rounding (NC, F32)";
    case xnn_operator_type_ceiling_nc_f32:
      return "Ceiling (NC, F32)";
    case xnn_operator_type_channel_shuffle_nc_x32:
      return "Channel Shuffle (NC, X32)";
    case xnn_operator_type_channel_shuffle_nc_x8:
      return "Channel Shuffle (NC, X8)";
    case xnn_operator_type_clamp_nc_f32:
      return "Clamp (NC, F32)";
    case xnn_operator_type_clamp_nc_u8:
      return "Clamp (NC, U8)";
    case xnn_operator_type_constant_pad_nd_x32:
      return "Constant Pad (ND, X32)";
    case xnn_operator_type_convolution_nhwc_f16:
      return "Convolution (NHWC, F16)";
    case xnn_operator_type_convolution_nhwc_f32:
      return "Convolution (NHWC, F32)";
    case xnn_operator_type_convolution_nhwc_qc8:
      return "Convolution (NHWC, QC8)";
    case xnn_operator_type_convolution_nhwc_qs8:
      return "Convolution (NHWC, QS8)";
    case xnn_operator_type_convolution_nhwc_qu8:
      return "Convolution (NHWC, QU8)";
    case xnn_operator_type_convolution_nchw_f32:
      return "Convolution (NCHW, F32)";
    case xnn_operator_type_copy_nc_x32:
      return "Copy (NC, X32)";
    case xnn_operator_type_deconvolution_nhwc_f32:
      return "Deconvolution (NHWC, F32)";
    case xnn_operator_type_deconvolution_nhwc_qs8:
      return "Deconvolution (NHWC, QS8)";
    case xnn_operator_type_deconvolution_nhwc_qu8:
      return "Deconvolution (NHWC, QU8)";
    case xnn_operator_type_depth_to_space_nchw2nhwc_x32:
      return "Depth To Space (NCHW2NHWC, X32)";
    case xnn_operator_type_depth_to_space_nhwc_x32:
      return "Depth To Space (NHWC, X32)";
    case xnn_operator_type_divide_nd_f32:
      return "Divide (ND, F32)";
    case xnn_operator_type_elu_nc_f32:
      return "ELU (NC, F32)";
    case xnn_operator_type_floor_nc_f32:
      return "Floor (NC, F32)";
    case xnn_operator_type_fully_connected_nc_f32:
      return "Fully Connected (NC, F32)";
    case xnn_operator_type_fully_connected_nc_qs8:
      return "Fully Connected (NC, QS8)";
    case xnn_operator_type_fully_connected_nc_qu8:
      return "Fully Connected (NC, QU8)";
    case xnn_operator_type_global_average_pooling_nwc_f16:
      return "Global Average Pooling (NWC, F16)";
    case xnn_operator_type_global_average_pooling_nwc_f32:
      return "Global Average Pooling (NWC, F32)";
    case xnn_operator_type_global_average_pooling_nwc_qs8:
      return "Global Average Pooling (NWC, QS8)";
    case xnn_operator_type_global_average_pooling_nwc_qu8:
      return "Global Average Pooling (NWC, QU8)";
    case xnn_operator_type_global_average_pooling_ncw_f32:
      return "Global Average Pooling (NCW, F32)";
    case xnn_operator_type_hardswish_nc_f16:
      return "HardSwish (NC, F16)";
    case xnn_operator_type_hardswish_nc_f32:
      return "HardSwish (NC, F32)";
    case xnn_operator_type_leaky_relu_nc_f32:
      return "Leaky ReLU (NC, F32)";
    case xnn_operator_type_leaky_relu_nc_qu8:
      return "Leaky ReLU (NC, QU8)";
    case xnn_operator_type_max_pooling_nhwc_f32:
      return "Max Pooling (NHWC, F32)";
    case xnn_operator_type_max_pooling_nhwc_u8:
      return "Max Pooling (NHWC, U8)";
    case xnn_operator_type_maximum_nd_f32:
      return "Maximum (ND, F32)";
    case xnn_operator_type_minimum_nd_f32:
      return "Minimum (ND, F32)";
    case xnn_operator_type_multiply_nd_f16:
      return "Multiply (ND, F16)";
    case xnn_operator_type_multiply_nd_f32:
      return "Multiply (ND, F32)";
    case xnn_operator_type_multiply_nd_qs8:
      return "Multiply (ND, QS8)";
    case xnn_operator_type_multiply_nd_qu8:
      return "Multiply (ND, QU8)";
    case xnn_operator_type_negate_nc_f32:
      return "Negate (NC, F32)";
    case xnn_operator_type_prelu_nc_f32:
      return "PReLU (NC, F32)";
    case xnn_operator_type_resize_bilinear_nhwc_f32:
      return "Resize Bilinear (NHWC, F32)";
    case xnn_operator_type_resize_bilinear_nchw_f32:
      return "Resize Bilinear (NCHW, F32)";
    case xnn_operator_type_sigmoid_nc_f32:
      return "Sigmoid (NC, F32)";
    case xnn_operator_type_sigmoid_nc_qu8:
      return "Sigmoid (NC, QU8)";
    case xnn_operator_type_softmax_nc_f32:
      return "Softmax (NC, F32)";
    case xnn_operator_type_softmax_nc_qu8:
      return "Softmax (NC, QU8)";
    case xnn_operator_type_square_nc_f32:
      return "Square (NC, F32)";
    case xnn_operator_type_square_root_nc_f32:
      return "Square Root (NC, F32)";
    case xnn_operator_type_squared_difference_nd_f32:
      return "Squared Difference (NC, F32)";
    case xnn_operator_type_subtract_nd_f32:
      return "Subtract (ND, F32)";
    case xnn_operator_type_truncation_nc_f32:
      return "Truncation (NC, F32)";
    case xnn_operator_type_unpooling_nhwc_x32:
      return "Unpooling (NHWC, X32)";
  }
  XNN_UNREACHABLE;
  return NULL;
}
#endif  // XNN_LOG_LEVEL > 0
