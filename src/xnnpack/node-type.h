// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: src/enums/node-type.yaml
//   Generator: tools/generate-enum.py

#pragma once

#include "xnnpack/common.h"


#ifdef __cplusplus
extern "C" {
#endif

enum xnn_node_type {
  xnn_node_type_invalid = 0,
  xnn_node_type_abs,
  xnn_node_type_add2,
  xnn_node_type_argmax_pooling_2d,
  xnn_node_type_average_pooling_2d,
  xnn_node_type_bankers_rounding,
  xnn_node_type_batch_matrix_multiply,
  xnn_node_type_ceiling,
  xnn_node_type_clamp,
  xnn_node_type_concatenate2,
  xnn_node_type_concatenate3,
  xnn_node_type_concatenate4,
  xnn_node_type_concatenate5,
  xnn_node_type_convert,
  xnn_node_type_convolution_2d,
  xnn_node_type_copy,
  xnn_node_type_copysign,
  xnn_node_type_deconvolution_2d,
  xnn_node_type_depth_to_space_2d,
  xnn_node_type_depthwise_convolution_2d,
  xnn_node_type_divide,
  xnn_node_type_elu,
  xnn_node_type_exp,
  xnn_node_type_even_split2,
  xnn_node_type_even_split3,
  xnn_node_type_even_split4,
  xnn_node_type_floor,
  xnn_node_type_fully_connected,
  xnn_node_type_fully_connected_sparse,
  xnn_node_type_gelu,
  xnn_node_type_global_average_pooling_1d,
  xnn_node_type_global_average_pooling_2d,
  xnn_node_type_global_sum_pooling_1d,
  xnn_node_type_global_sum_pooling_2d,
  xnn_node_type_hardswish,
  xnn_node_type_log,
  xnn_node_type_leaky_relu,
  xnn_node_type_max_pooling_2d,
  xnn_node_type_maximum2,
  xnn_node_type_minimum2,
  xnn_node_type_multiply2,
  xnn_node_type_negate,
  xnn_node_type_prelu,
  xnn_node_type_reciprocal_square_root,
  xnn_node_type_reshape_2d,
  xnn_node_type_rope,
  xnn_node_type_scaled_dot_product_attention,
  xnn_node_type_sigmoid,
  xnn_node_type_softmax,
  xnn_node_type_space_to_depth_2d,
  xnn_node_type_square,
  xnn_node_type_square_root,
  xnn_node_type_squared_difference,
  xnn_node_type_static_constant_pad,
  xnn_node_type_static_mean,
  xnn_node_type_static_reshape,
  xnn_node_type_static_resize_bilinear_2d,
  xnn_node_type_static_slice,
  xnn_node_type_static_transpose,
  xnn_node_type_subtract,
  xnn_node_type_tanh,
  xnn_node_type_unpooling_2d,
};

#if XNN_LOG_LEVEL <= 0
  XNN_INLINE static const char* xnn_node_type_to_string(enum xnn_node_type type) {
    return "<unknown>";
  }
#else
  XNN_INTERNAL const char* xnn_node_type_to_string(enum xnn_node_type type);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif
