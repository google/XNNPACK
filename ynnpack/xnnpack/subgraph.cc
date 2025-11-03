// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/subgraph.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "include/xnnpack.h"
#include "ynnpack/base/log.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/xnnpack/dynamic_quantization.h"
#include "ynnpack/xnnpack/utils.h"
#include "ynnpack/xnnpack/xnnpack.h"

extern "C" {

xnn_status xnn_initialize(const xnn_allocator* allocator) {
  return xnn_status_success;
}

xnn_status xnn_deinitialize(void) { return xnn_status_success; }

const void* xnn_experimental_get_build_identifier_data() {
  static uint64_t data = 0;
  return &data;
}

size_t xnn_experimental_get_build_identifier_size() { return sizeof(uint64_t); }

bool xnn_experimental_check_build_identifier(const void* data, size_t size) {
  return true;
}

xnn_status xnn_create_subgraph(uint32_t external_value_ids, uint32_t flags,
                               xnn_subgraph_t* subgraph_out) {
  *subgraph_out = new xnn_subgraph();
  return ynn::xnn_status_from_ynn(
      ynn_create_subgraph(external_value_ids, flags, &(*subgraph_out)->ynn));
}

xnn_status xnn_delete_subgraph(xnn_subgraph_t subgraph) {
  ynn_delete_subgraph(subgraph->ynn);
  delete subgraph;
  return xnn_status_success;
}

uint32_t xnn_subgraph_get_value_flags(xnn_subgraph_t subgraph,
                                      uint32_t value_id) {
  uint32_t ynn_flags = subgraph->ynn->value(value_id).flags;
  uint32_t xnn_flags = 0;
  if (ynn_flags & YNN_VALUE_FLAG_EXTERNAL_INPUT) {
    xnn_flags |= XNN_VALUE_FLAG_EXTERNAL_INPUT;
  }
  if (ynn_flags & YNN_VALUE_FLAG_EXTERNAL_OUTPUT) {
    xnn_flags |= XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  }
  return xnn_flags;
}

xnn_datatype xnn_subgraph_get_value_datatype(xnn_subgraph_t subgraph,
                                             uint32_t value_id) {
  const ynn_value& value = subgraph->ynn->value(value_id);
  return ynn::xnn_datatype_from_ynn(value.type);
}

uint32_t xnn_subgraph_get_num_external_values(xnn_subgraph_t subgraph) {
  // Not right, but XNNPACK only uses this to assert a value is an external
  // value ID.
  return subgraph->ynn->values.size();
}

uint32_t xnn_subgraph_get_num_nodes(xnn_subgraph_t subgraph) {
  // XNNPACK only uses this functions to test internal implementation details,
  // which we do not implement.
  return 0;
}

uint32_t xnn_subgraph_get_num_values(xnn_subgraph_t subgraph) {
  // XNNPACK only uses this functions to test internal implementation details,
  // which we do not implement.
  return 0;
}

int xnn_reduce_operator_to_node_type(int) {
  // XNNPACK only uses this functions to test internal implementation details,
  // which we do not implement.
  return 0;
}

xnn_status xnn_subgraph_optimize(xnn_subgraph_t subgraph) {
  YNN_LOG_ERROR() << "Test-only XNNPACK operation not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_subgraph_rewrite_for_nchw(xnn_subgraph_t subgraph) {
  YNN_LOG_ERROR() << "Test-only XNNPACK operation not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_subgraph_rewrite_for_fp16(xnn_subgraph_t subgraph) {
  YNN_LOG_ERROR() << "Test-only XNNPACK operation not supported";
  return xnn_status_deprecated;
}

xnn_status xnn_define_unary(xnn_subgraph_t subgraph, xnn_unary_operator type,
                            const union xnn_unary_params* params,
                            uint32_t input_id, uint32_t output_id,
                            uint32_t flags) {
  if (type == xnn_unary_convert) {
    // This might be a dynamic quantization conversion.
    const ynn_value& output = subgraph->ynn->value(output_id);
    const ynn_value* scale = output.scale_id != YNN_INVALID_VALUE_ID
                                 ? &subgraph->ynn->value(output.scale_id)
                                 : nullptr;
    const ynn_value* zero_point =
        output.zero_point_id != YNN_INVALID_VALUE_ID
            ? &subgraph->ynn->value(output.zero_point_id)
            : nullptr;
    if (scale && zero_point && !scale->is_static() &&
        !zero_point->is_static()) {
      // This is a qd8 dynamic quantization. We need to compute the quantization
      // params.
      assert(output.type == ynn_type_uint8 || output.type == ynn_type_int8);
      assert(subgraph->num_nonbatch_axes.count(output_id));
      ynn_status status = ynn::compute_qd8_params(
          subgraph->ynn, subgraph->num_nonbatch_axes[output_id], input_id,
          output_id);
      if (status != ynn_status_success) {
        return ynn::xnn_status_from_ynn(status);
      }

      // Now that we've computed the params,
      // We still need the convert node below.
    }
  }
  if (type == xnn_unary_leaky_relu) {
    return ynn::xnn_status_from_ynn(ynn::define_binary_scalar_b(
        subgraph->ynn, ynn_binary_leaky_relu, input_id,
        params->leaky_relu.negative_slope, &output_id));
  } else if (type == xnn_unary_clamp) {
    return ynn::xnn_status_from_ynn(
        ynn::define_clamp(subgraph->ynn, params->clamp.min, params->clamp.max,
                          input_id, &output_id));
  } else if (type == xnn_unary_elu) {
    // return x < 0 ? alpha * expm1(x) : x;
  } else if (type == xnn_unary_gelu) {
    return ynn::xnn_status_from_ynn(
        ynn::implement_gelu(subgraph->ynn, input_id, output_id));
  } else if (type == xnn_unary_approxgelu) {
    // return (x / 2) * (1 + tanh(sqrt(2.0 / pi) * x * (1 + 0.044715 * x *
    // x))));
  } else {
    ynn_unary_operator ynn_type = ynn::unary_operator_from_xnn(type);
    if (ynn_type != ynn_unary_invalid) {
      // This is a simple op without any relevant params.
      return ynn::xnn_status_from_ynn(ynn_define_unary(
          subgraph->ynn, ynn_type, input_id, &output_id, /*flags=*/0));
    }
  }
  YNN_LOG_ERROR() << "Unsupported unary operator " << type;
  return xnn_status_deprecated;
}

xnn_status xnn_define_convolution_2d(
    xnn_subgraph_t subgraph, uint32_t input_padding_top,
    uint32_t input_padding_right, uint32_t input_padding_bottom,
    uint32_t input_padding_left, uint32_t kernel_height, uint32_t kernel_width,
    uint32_t subsampling_height, uint32_t subsampling_width,
    uint32_t dilation_height, uint32_t dilation_width, uint32_t groups,
    size_t group_input_channels, size_t group_output_channels, float output_min,
    float output_max, uint32_t input_id, uint32_t filter_id, uint32_t bias_id,
    uint32_t output_id, uint32_t flags) {
  ynn_status status;

  if (groups != 1) {
    uint32_t split_id = YNN_INVALID_VALUE_ID;

    // [n, h, w, ci] -> [n, h, w, g, 1, ci/g].
    const size_t input_split[] = {groups, 1, group_input_channels};
    status = ynn_define_split_dim(subgraph->ynn, 3, 3, input_split, input_id,
                                  &split_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }
    input_id = split_id;

    // [co, kh, kw, ci] -> [g, co/g, kh, kw, ci].
    split_id = YNN_INVALID_VALUE_ID;
    const size_t filter_split[] = {groups, group_output_channels};
    status = ynn_define_split_dim(subgraph->ynn, 0, 2, filter_split, filter_id,
                                  &split_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }
    filter_id = split_id;

    uint32_t filter_scale_id = subgraph->ynn->value(filter_id).scale_id;
    if (filter_scale_id != YNN_INVALID_VALUE_ID &&
        ynn::rank_of_value(subgraph->ynn, filter_scale_id) >= 1) {
      split_id = YNN_INVALID_VALUE_ID;
      // There is a bit of gotcha moment here, because it would seem logical
      // to have splits the same as filter has. However, quantized dots are
      // factored into multiple parts and the scale is applied in the end, so
      // it has to match the dims of the output (that being said they are pretty
      // close: first two dims are the same between filter and output, except
      // that output has extra dimension of extent 1).
      const size_t filter_scale_split[] = {groups, 1, group_output_channels};
      status = ynn_define_split_dim(subgraph->ynn, 0, 3, filter_scale_split,
                                    filter_scale_id, &split_id, /*flags=*/0);
      if (status != ynn_status_success) {
        return ynn::xnn_status_from_ynn(status);
      }
      subgraph->ynn->value(filter_id).scale_id = split_id;
    }

    split_id = YNN_INVALID_VALUE_ID;
    if (bias_id != XNN_INVALID_VALUE_ID) {
      const size_t bias_split[] = {groups, 1, group_output_channels};

      status = ynn_define_split_dim(subgraph->ynn, 0, 3, bias_split, bias_id,
                                    &split_id, /*flags=*/0);
      if (status != ynn_status_success) {
        return ynn::xnn_status_from_ynn(status);
      }
      bias_id = split_id;
    }

    uint32_t input_zero_point_id = subgraph->ynn->value(input_id).zero_point_id;
    uint32_t input_scale_id = subgraph->ynn->value(input_id).scale_id;
    if (input_zero_point_id != YNN_INVALID_VALUE_ID &&
        input_scale_id != YNN_INVALID_VALUE_ID) {
      // We assume this is a dynamically quantized input.
      assert(ynn::rank_of_value(subgraph->ynn, input_zero_point_id) == 4);
      assert(ynn::rank_of_value(subgraph->ynn, input_scale_id) == 4);
      uint32_t zero_point_id = YNN_INVALID_VALUE_ID;
      // We assume that non-batch dims have extent 1, so just insert another two
      // dimensions to match the shape of the input after the group split.
      const int32_t dims[2] = {1, 2};
      status = ynn_define_static_expand_dims(subgraph->ynn, 2, dims,
                                             input_zero_point_id,
                                             &zero_point_id, /*flags=*/0);
      if (status != ynn_status_success) {
        return ynn::xnn_status_from_ynn(status);
      }

      uint32_t scale_id = YNN_INVALID_VALUE_ID;
      status = ynn_define_static_expand_dims(
          subgraph->ynn, 2, dims, input_scale_id, &scale_id, /*flags=*/0);
      if (status != ynn_status_success) {
        return ynn::xnn_status_from_ynn(status);
      }

      subgraph->ynn->value(input_id).zero_point_id = zero_point_id;
      subgraph->ynn->value(input_id).scale_id = scale_id;
    }
  }

  // Make a stenciled view of the input.
  // [n, h, w, ci] -> [n, h, w, kh, kw, ci] or [n, h, w, g, 1, ci] -> [n, h, w,
  // g, 1, kh, kw, ci] for group convolutions.
  uint32_t stencil_id = YNN_INVALID_VALUE_ID;
  status = ynn::define_xnn_stencil(
      subgraph->ynn, input_padding_top, input_padding_right,
      input_padding_bottom, input_padding_left, 0.0f, kernel_height,
      kernel_width, subsampling_height, subsampling_width, dilation_height,
      dilation_width, input_id, &stencil_id, flags);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  if (bias_id == XNN_INVALID_VALUE_ID) {
    bias_id = YNN_INVALID_VALUE_ID;
  }

  // XNNPACK specifies convolution filters as [co, h, w, ci], but we need them
  // to be [h, w, ci, co]. Because XNNPACK requires convolution filters be
  // constants, this transpose will always be constant folded.
  uint32_t transposed_filter_id = YNN_INVALID_VALUE_ID;
  if (groups == 1) {
    int32_t swap_co_ci[4] = {1, 2, 3, 0};

    status =
        ynn_define_static_transpose(subgraph->ynn, 4, swap_co_ci, filter_id,
                                    &transposed_filter_id, /*flags=*/0);
  } else {
    // Group convolutions have an additional split, so in this case the
    // transpose is: [g, co / g, kh, kw, ci] -> [g, kh, kw, ci, co / g]
    int32_t swap_co_ci[5] = {0, 2, 3, 4, 1};

    status =
        ynn_define_static_transpose(subgraph->ynn, 5, swap_co_ci, filter_id,
                                    &transposed_filter_id, /*flags=*/0);
  }

  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  uint32_t output_unfused_id = groups != 1 ? YNN_INVALID_VALUE_ID : output_id;

  if (output_unfused_id == YNN_INVALID_VALUE_ID) {
    status = ynn::define_tensor_value_like(subgraph->ynn, output_id,
                                           &output_unfused_id);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }
  }

  // Now we can compute [n, h, w, kh, kw, ci] * [kh, kw, ci, co] or
  // [n, h, w, g, 1, kh, kw, ci/g] * [g, kh, kw, ci, co/g] for group
  // convolutions.
  status =
      ynn::define_xnn_dot(subgraph->ynn, 3, stencil_id, transposed_filter_id,
                          bias_id, output_unfused_id);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  if (output_unfused_id != output_id) {
    // The output of the group convolution is [n, h, w, g, 1, co/g], so we need
    // to fuse three of the innermost dimensions.
    status =
        ynn_define_fuse_dim(subgraph->ynn, 3, 3, output_unfused_id, &output_id,
                            /*flags=*/0);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }
  }

  // Apply requested clamping.
  return ynn::xnn_status_from_ynn(
      ynn::implement_clamp(subgraph->ynn, output_min, output_max, output_id));
}

xnn_status xnn_define_deconvolution_2d(
    xnn_subgraph_t subgraph, uint32_t padding_top, uint32_t padding_right,
    uint32_t padding_bottom, uint32_t padding_left, uint32_t adjustment_height,
    uint32_t adjustment_width, uint32_t kernel_height, uint32_t kernel_width,
    uint32_t upsampling_height, uint32_t upsampling_width,
    uint32_t dilation_height, uint32_t dilation_width, uint32_t groups,
    size_t group_input_channels, size_t group_output_channels, float output_min,
    float output_max, uint32_t input_id, uint32_t filter_id, uint32_t bias_id,
    uint32_t output_id, uint32_t flags) {
  YNN_LOG_ERROR() << "Unsupported xnn_define_deconvolution_2d";
  return xnn_status_deprecated;
}

xnn_status xnn_define_depthwise_convolution_2d(
    xnn_subgraph_t subgraph, uint32_t input_padding_top,
    uint32_t input_padding_right, uint32_t input_padding_bottom,
    uint32_t input_padding_left, uint32_t kernel_height, uint32_t kernel_width,
    uint32_t subsampling_height, uint32_t subsampling_width,
    uint32_t dilation_height, uint32_t dilation_width,
    uint32_t depth_multiplier, size_t input_channels, float output_min,
    float output_max, uint32_t input_id, uint32_t filter_id, uint32_t bias_id,
    uint32_t output_id, uint32_t flags) {
  ynn_status status;

  uint32_t transposed_filter_id = YNN_INVALID_VALUE_ID;

  // We need to transpose and expand a filter buffer, so it matches format
  // expected by the grouped convolution.
  // [kh, kw, ci * dm] -> [ci * dm, kh, kw]
  int32_t swap_dims[3] = {2, 0, 1};

  status = ynn_define_static_transpose(subgraph->ynn, 3, swap_dims, filter_id,
                                       &transposed_filter_id, /*flags=*/0);

  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  uint32_t expanded_transposed_filter_id = YNN_INVALID_VALUE_ID;
  // [kh, kw, ci * dm] -> [ci * dm, kh, kw, 1]
  int32_t new_axes[] = {3};
  status = ynn_define_static_expand_dims(subgraph->ynn, 1, new_axes,
                                         transposed_filter_id,
                                         &expanded_transposed_filter_id,
                                         /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  return xnn_define_convolution_2d(
      subgraph, input_padding_top, input_padding_right, input_padding_bottom,
      input_padding_left, kernel_height, kernel_width, subsampling_height,
      subsampling_width, dilation_height, dilation_width, input_channels,
      /*group_input_channels=*/1, depth_multiplier, output_min, output_max,
      input_id, expanded_transposed_filter_id, bias_id, output_id, flags);
}

xnn_status xnn_define_space_to_depth_2d(xnn_subgraph_t subgraph,
                                        uint32_t block_size, uint32_t input_id,
                                        uint32_t output_id, uint32_t flags) {
  assert(ynn::rank_of_value(subgraph->ynn, input_id) == 4);

  // Stencil copy to split [n, y_dy, x_dx, c] -> [n, y, x, dy, dx, c]
  uint32_t tiled_id = YNN_INVALID_VALUE_ID;
  const int32_t stencil_axes[] = {1, 2};
  const int32_t new_axes[] = {3, 4};
  const size_t strides[] = {block_size, block_size};
  const size_t extents[] = {block_size, block_size};
  const size_t dilations[] = {1, 1};
  ynn_status status = ynn_define_stencil_copy(
      subgraph->ynn, 2, stencil_axes, new_axes, extents, strides, dilations,
      input_id, /*padding_id=*/YNN_INVALID_VALUE_ID, &tiled_id,
      /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  // Fuse [n, y, x, dy, dx, c] -> [n, y, x, dy_dx_c]
  return ynn::xnn_status_from_ynn(ynn_define_fuse_dim(
      subgraph->ynn, 3, 3, tiled_id, &output_id, /*flags=*/0));
}

xnn_status xnn_define_depth_to_space_2d(xnn_subgraph_t subgraph,
                                        uint32_t block_size, uint32_t input_id,
                                        uint32_t output_id, uint32_t flags) {
  assert(ynn::rank_of_value(subgraph->ynn, input_id) == 4);

  // Split [n, y, x, dy_dx_c] -> [n, y, x, dy, dx, c]
  uint32_t transposed_id = YNN_INVALID_VALUE_ID;
  const size_t splits[] = {block_size, block_size, 0};
  ynn_status status = ynn_define_split_dim(
      subgraph->ynn, 3, 3, splits, input_id, &transposed_id, /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  // Transpose [n, y, x, dy, dx, c] -> [n, y, dy, x, dx, c]
  uint32_t tiled_id = YNN_INVALID_VALUE_ID;
  const int32_t to_space_axes[] = {0, 1, 3, 2, 4, 5};
  status = ynn_define_static_transpose(subgraph->ynn, 6, to_space_axes,
                                       transposed_id, &tiled_id, /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  // Fuse [n, y, dy, x, dx, c] -> [n, y_dy, x_dx, c]
  const int32_t fuse_axes[] = {1, 3};
  return ynn::xnn_status_from_ynn(ynn_define_fuse_dims(
      subgraph->ynn, 2, fuse_axes, tiled_id, &output_id, /*flags=*/0));
}

xnn_status xnn_define_depth_to_space(xnn_subgraph_t subgraph, uint32_t input_id,
                                     uint32_t output_id, uint32_t block_size,
                                     uint32_t flags) {
  return xnn_define_depth_to_space_2d(subgraph, block_size, input_id, output_id,
                                      flags);
}

xnn_status xnn_define_average_pooling_2d(
    xnn_subgraph_t subgraph, uint32_t input_padding_top,
    uint32_t input_padding_right, uint32_t input_padding_bottom,
    uint32_t input_padding_left, uint32_t pooling_height,
    uint32_t pooling_width, uint32_t stride_height, uint32_t stride_width,
    float output_min, float output_max, uint32_t input_id, uint32_t output_id,
    uint32_t flags) {
  const int32_t reduce_axes[] = {3, 4};

  uint32_t norm_id = YNN_INVALID_VALUE_ID;
  if (input_padding_top || input_padding_right || input_padding_bottom ||
      input_padding_left || (flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0) {
    // To implement the per-pixel normalization, we're going to compute the
    // pooling on a buffer of ones of the same shape as the input. This computes
    // a buffer that is mostly pooling_width*pooling_height, which seems
    // wasteful, but it is what XNNPACK does too, so this shouldn't be a
    // significant regression (in memory usage at least). XNNPACK only
    // recomputes this on reshape, not invoke, so this is likely a regression in
    // overhead.
    //
    // It's tempting to think this should be something that can be constant
    // folded, but it is not (unless the YNN_FLAG_STATIC_BOUNDS flag is being
    // used).
    //
    // We could compute this in the width and height dimensions separately and
    // multiply the results later...
    uint32_t ones_squezed_id = YNN_INVALID_VALUE_ID;
    ynn_status status = ynn::define_scalar_value_like(subgraph->ynn, input_id,
                                                      1.0f, &ones_squezed_id);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }

    uint32_t ones_id = YNN_INVALID_VALUE_ID;
    const int32_t dims[4] = {0, 1, 2, 3};
    status = ynn_define_static_expand_dims(
        subgraph->ynn, 4, dims, ones_squezed_id, &ones_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }

    uint32_t ones_broadcasted_id = YNN_INVALID_VALUE_ID;
    const int32_t xy[2] = {1, 2};
    status =
        ynn_define_broadcast_like(subgraph->ynn, /*num_axes=*/2, xy, ones_id,
                                  input_id, &ones_broadcasted_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }

    uint32_t ones_stencil_id = YNN_INVALID_VALUE_ID;
    status = ynn::define_xnn_stencil(
        subgraph->ynn, input_padding_top, input_padding_right,
        input_padding_bottom, input_padding_left, /*padding_value=*/0.0f,
        pooling_height, pooling_width, stride_height, stride_width,
        /*dilation_height=*/1,
        /*dilation_width=*/1, ones_broadcasted_id, &ones_stencil_id, flags);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }

    status = ynn_define_reduce(subgraph->ynn, ynn_reduce_sum, /*num_axes=*/2,
                               reduce_axes, ones_stencil_id,
                               YNN_INVALID_VALUE_ID, &norm_id,
                               /*flags=*/0);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }
  }

  // Make a stenciled view of the input.
  uint32_t stencil_id = YNN_INVALID_VALUE_ID;
  ynn_status status = ynn::define_xnn_stencil(
      subgraph->ynn, input_padding_top, input_padding_right,
      input_padding_bottom, input_padding_left, /*padding_value=*/0.0f,
      pooling_height, pooling_width, stride_height, stride_width,
      /*dilation_height=*/1,
      /*dilation_width=*/1, input_id, &stencil_id, flags);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  if (norm_id != YNN_INVALID_VALUE_ID) {
    // There is padding, so the value we need to divide by is not simply the
    // size of the pooled area. We need to compute the sum, and divide by the
    // normalization we computed above.
    uint32_t sum_id = YNN_INVALID_VALUE_ID;
    status = ynn_define_reduce(subgraph->ynn, ynn_reduce_sum, /*num_axes=*/2,
                               reduce_axes, stencil_id, YNN_INVALID_VALUE_ID,
                               &sum_id,
                               /*flags=*/0);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }

    uint32_t output_widened_id = output_id;
    if (ynn::type_of_value(subgraph->ynn, output_id) !=
        ynn::type_of_value(subgraph->ynn, sum_id)) {
      // The sum is higher precision than the output, compute the division into
      // a temporary value.
      output_widened_id = YNN_INVALID_VALUE_ID;
    }
    status = ynn_define_binary(subgraph->ynn, ynn_binary_divide, sum_id,
                               norm_id, &output_widened_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }

    if (output_widened_id != output_id) {
      // The sum is higher precision than the output, convert the division
      // result to the output.
      status = ynn_define_unary(subgraph->ynn, ynn_unary_convert,
                                output_widened_id, &output_id, /*flags=*/0);
      if (status != ynn_status_success) {
        return ynn::xnn_status_from_ynn(status);
      }
    }
  } else {
    // Being lazy here, we just use the XNNPACK static reduce, which implements
    // the widening accumulator and mean normalization for us.
    const int64_t reduce_axes[] = {3, 4};
    xnn_status xnn_status =
        xnn_define_static_reduce_v2(subgraph, xnn_reduce_mean, 2, reduce_axes,
                                    stencil_id, output_id, /*flags=*/0);
    if (xnn_status != xnn_status_success) {
      return xnn_status;
    }
  }

  // Apply requested clamping.
  return ynn::xnn_status_from_ynn(
      ynn::implement_clamp(subgraph->ynn, output_min, output_max, output_id));
}

xnn_status xnn_define_fully_connected(xnn_subgraph_t subgraph, float output_min,
                                      float output_max, uint32_t input_id,
                                      uint32_t filter_id, uint32_t bias_id,
                                      uint32_t output_id, uint32_t flags) {
  if (!(flags & XNN_FLAG_TRANSPOSE_WEIGHTS)) {
    uint32_t filter_id_transposed = YNN_INVALID_VALUE_ID;
    assert(ynn::rank_of_value(subgraph->ynn, filter_id) == 2);
    const int32_t perm[] = {1, 0};
    ynn_status status = ynn_define_static_transpose(
        subgraph->ynn,
        /*num_dims=*/2, perm, filter_id, &filter_id_transposed,
        /*flags=*/0);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }
    filter_id = filter_id_transposed;
  }

  ynn_status status = ynn::define_xnn_dot(subgraph->ynn,
                                          /*num_k_dims=*/1, input_id, filter_id,
                                          bias_id, output_id);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  return ynn::xnn_status_from_ynn(
      ynn::implement_clamp(subgraph->ynn, output_min, output_max, output_id));
}

xnn_status xnn_define_fully_connected_sparse(
    xnn_subgraph_t subgraph, float output_min, float output_max,
    uint32_t input_id, uint32_t filter_id, uint32_t bias_id, uint32_t output_id,
    uint32_t flags) {
  return xnn_define_fully_connected(subgraph, output_min, output_max, input_id,
                                    filter_id, bias_id, output_id, flags);
}

xnn_status xnn_define_max_pooling_2d(
    xnn_subgraph_t subgraph, uint32_t input_padding_top,
    uint32_t input_padding_right, uint32_t input_padding_bottom,
    uint32_t input_padding_left, uint32_t pooling_height,
    uint32_t pooling_width, uint32_t stride_height, uint32_t stride_width,
    uint32_t dilation_height, uint32_t dilation_width, float output_min,
    float output_max, uint32_t input_id, uint32_t output_id, uint32_t flags) {
  // Make a stenciled view of the input.
  uint32_t stencil_id = YNN_INVALID_VALUE_ID;
  ynn_status status = ynn::define_xnn_stencil(
      subgraph->ynn, input_padding_top, input_padding_right,
      input_padding_bottom, input_padding_left, output_min, pooling_height,
      pooling_width, stride_height, stride_width, dilation_height,
      dilation_width, input_id, &stencil_id, flags);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  // Do a reduction on the stencil dimensions.
  uint32_t zero_id = YNN_INVALID_VALUE_ID;
  status = ynn::define_scalar_value_like(subgraph->ynn, output_id, output_min,
                                         &zero_id);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  const int32_t reduce_axes[] = {3, 4};
  status = ynn_define_reduce(subgraph->ynn, ynn_reduce_max, 2, reduce_axes,
                             stencil_id, zero_id, &output_id,
                             /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  // Apply requested clamping.
  return ynn::xnn_status_from_ynn(ynn::implement_clamp(
      subgraph->ynn, -std::numeric_limits<float>::infinity(), output_max,
      output_id));
}

xnn_status xnn_define_argmax_pooling_2d(
    xnn_subgraph_t subgraph, uint32_t input_padding_top,
    uint32_t input_padding_right, uint32_t input_padding_bottom,
    uint32_t input_padding_left, uint32_t pooling_height,
    uint32_t pooling_width, uint32_t input_id, uint32_t output_value_id,
    uint32_t output_index_id, uint32_t flags) {
  YNN_LOG_ERROR() << "Unsupported xnn_define_argmax_pooling_2d";
  return xnn_status_deprecated;
}

xnn_status xnn_define_unpooling_2d(
    xnn_subgraph_t subgraph, uint32_t padding_top, uint32_t padding_right,
    uint32_t padding_bottom, uint32_t padding_left, uint32_t pooling_height,
    uint32_t pooling_width, uint32_t input_value_id, uint32_t input_index_id,
    uint32_t output_id, uint32_t flags) {
  YNN_LOG_ERROR() << "Unsupported xnn_define_unpooling_2d";
  return xnn_status_deprecated;
}

xnn_status xnn_define_binary(xnn_subgraph_t subgraph, xnn_binary_operator type,
                             const xnn_binary_params* params,
                             uint32_t input1_id, uint32_t input2_id,
                             uint32_t output_id, uint32_t flags) {
  ynn_status status = ynn::implement_xnn_broadcasting(subgraph->ynn, &input1_id,
                                                      &input2_id, flags);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }
  status = ynn_define_binary(subgraph->ynn, ynn::binary_operator_from_xnn(type),
                             input1_id, input2_id, &output_id, flags);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }
  if (params) {
    return ynn::xnn_status_from_ynn(ynn::implement_clamp(
        subgraph->ynn, params->output_min, params->output_max, output_id));
  } else {
    return xnn_status_success;
  }
}

xnn_status xnn_define_static_constant_pad(xnn_subgraph_t subgraph,
                                          const size_t* pre_paddings,
                                          const size_t* post_paddings,
                                          float padding_value,
                                          uint32_t input_id, uint32_t output_id,
                                          uint32_t flags) {
  uint32_t padding_id = YNN_INVALID_VALUE_ID;
  ynn_status status = ynn::define_scalar_value_like(subgraph->ynn, input_id,
                                                    padding_value, &padding_id);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  size_t rank = ynn::rank_of_value(subgraph->ynn, input_id);

  int64_t ynn_pre_paddings[YNN_MAX_TENSOR_RANK];
  int64_t ynn_post_paddings[YNN_MAX_TENSOR_RANK];
  std::copy_n(pre_paddings, rank, ynn_pre_paddings);
  std::copy_n(post_paddings, rank, ynn_post_paddings);
  int32_t axes[YNN_MAX_TENSOR_RANK];
  std::iota(axes, axes + rank, 0);
  return ynn::xnn_status_from_ynn(ynn_define_static_pad(
      subgraph->ynn, rank, axes, ynn_pre_paddings, ynn_post_paddings, input_id,
      padding_id, &output_id, /*flags=*/0));
}

xnn_status xnn_define_static_expand_dims(xnn_subgraph_t subgraph,
                                         size_t num_new_axes,
                                         const size_t* new_axes,
                                         uint32_t input_id, uint32_t output_id,
                                         uint32_t flags) {
  int32_t ynn_axes[XNN_MAX_TENSOR_DIMS];
  for (size_t i = 0; i < num_new_axes; ++i) {
    ynn_axes[i] = new_axes[i];
  }
  return ynn::xnn_status_from_ynn(ynn_define_static_expand_dims(
      subgraph->ynn, num_new_axes, ynn_axes, input_id, &output_id,
      /*flags=*/0));
}

xnn_status xnn_define_fuse_dims(xnn_subgraph_t subgraph, size_t axis,
                                size_t axes_count, uint32_t input_id,
                                uint32_t output_id, uint32_t flags) {
  return ynn::xnn_status_from_ynn(ynn_define_fuse_dim(
      subgraph->ynn, axis, axes_count, input_id, &output_id, /*flags=*/0));
}

xnn_status xnn_define_split_dim(xnn_subgraph_t subgraph, size_t axis,
                                size_t num_splits, const size_t* splits,
                                uint32_t input_id, uint32_t output_id,
                                uint32_t flags) {
  return ynn::xnn_status_from_ynn(
      ynn_define_split_dim(subgraph->ynn, axis, num_splits, splits, input_id,
                           &output_id, /*flags=*/0));
}

xnn_status xnn_define_static_reduce(xnn_subgraph_t subgraph,
                                    xnn_reduce_operator reduce_operator,
                                    size_t num_reduction_axes,
                                    const size_t* reduction_axes,
                                    uint32_t input_id, uint32_t output_id,
                                    uint32_t flags) {
  int64_t signed_reduction_axes[XNN_MAX_TENSOR_DIMS];
  for (int i = 0; i < num_reduction_axes; i++) {
    signed_reduction_axes[i] = reduction_axes[i];
  }
  return xnn_define_static_reduce_v2(subgraph, reduce_operator,
                                     num_reduction_axes, signed_reduction_axes,
                                     input_id, output_id, flags);
}

xnn_status xnn_define_static_reduce_v2(xnn_subgraph_t subgraph,
                                       xnn_reduce_operator reduce_operator_type,
                                       size_t num_reduction_axes,
                                       const int64_t* reduction_axes,
                                       uint32_t input_id, uint32_t output_id,
                                       uint32_t flags) {
  const ynn_value& input = subgraph->ynn->value(input_id);
  ynn_value& output = subgraph->ynn->value(output_id);
  const bool is_mean = reduce_operator_type == xnn_reduce_mean ||
                       reduce_operator_type == xnn_reduce_mean_squared;

  ynn_reduce_operator reduce_op;
  ynn_type accumulator_type = input.type;
  switch (reduce_operator_type) {
    case xnn_reduce_mean:
    case xnn_reduce_sum:
      reduce_op = ynn_reduce_sum;
      accumulator_type = ynn::accumulator_for_type(input.type);
      break;
    case xnn_reduce_mean_squared:
    case xnn_reduce_sum_squared:
      reduce_op = ynn_reduce_sum_squared;
      accumulator_type = ynn::accumulator_for_type(input.type);
      break;
    case xnn_reduce_max:
      reduce_op = ynn_reduce_max;
      break;
    case xnn_reduce_min:
      reduce_op = ynn_reduce_min;
      break;
    default:
      return xnn_status_deprecated;
  }

  int32_t ynn_axes[XNN_MAX_TENSOR_DIMS];
  for (size_t i = 0; i < num_reduction_axes; ++i) {
    ynn_axes[i] = reduction_axes[i];
  }
  uint32_t ynn_flags = 0;
  if (flags & XNN_FLAG_KEEP_DIMS) {
    ynn_flags |= YNN_NODE_FLAG_KEEP_DIMS;
  }

  uint32_t accumulator_id = output_id;
  if (accumulator_type != output.type || is_mean) {
    // We need a separate accumulator if we need to widen, or we are computing
    // the mean.
    accumulator_id = YNN_INVALID_VALUE_ID;
  }

  uint32_t init_accumulator_id = YNN_INVALID_VALUE_ID;
  ynn_status status = ynn_define_reduce(
      subgraph->ynn, reduce_op, num_reduction_axes, ynn_axes, input_id,
      init_accumulator_id, &accumulator_id, ynn_flags);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  if (is_mean) {
    // Get the shape of the input axes we reduced.
    uint32_t reduce_size_id = YNN_INVALID_VALUE_ID;
    status = ynn_define_get_tensor_shape(subgraph->ynn, num_reduction_axes,
                                         ynn_axes, ynn_type_fp32,
                                         /*rank=*/0, input_id, &reduce_size_id,
                                         /*flags=*/YNN_NODE_FLAG_RESHAPE_1D);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }

    if (ynn::type_is_integral(accumulator_type)) {
      // We implement the division for quantized mean by dividing the scale of
      // the accumulator instead of the result itself.
      ynn_value& accumulator = subgraph->ynn->value(accumulator_id);
      if (input.scale_id == YNN_INVALID_VALUE_ID) {
        status = ynn::define_scalar_value_like(subgraph->ynn, reduce_size_id,
                                               1.0f, &accumulator.scale_id);
      }
      accumulator.scale_id = YNN_INVALID_VALUE_ID;
      status =
          ynn_define_binary(subgraph->ynn, ynn_binary_divide, input.scale_id,
                            reduce_size_id, &accumulator.scale_id,
                            /*flags=*/0);
      if (status != ynn_status_success) {
        return ynn::xnn_status_from_ynn(status);
      }
    } else {
      // For floating point outputs, just divide the result.
      uint32_t normalized_id =
          output.type == accumulator_type ? output_id : YNN_INVALID_VALUE_ID;
      status = ynn_define_binary(subgraph->ynn, ynn_binary_divide,
                                 accumulator_id, reduce_size_id, &normalized_id,
                                 /*flags=*/0);
      if (status != ynn_status_success) {
        return ynn::xnn_status_from_ynn(status);
      }
      accumulator_id = normalized_id;
    }
  }
  if (accumulator_id != output_id) {
    status = ynn_define_unary(subgraph->ynn, ynn_unary_convert, accumulator_id,
                              &output_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }
  }

  return xnn_status_success;
}

xnn_status xnn_define_concatenate(xnn_subgraph_t subgraph, int32_t axis,
                                  size_t num_inputs, const uint32_t* inputs,
                                  uint32_t output_id, uint32_t flags) {
  return ynn::xnn_status_from_ynn(ynn_define_concatenate(
      subgraph->ynn, axis, num_inputs, inputs, &output_id, /*flags=*/0));
}

xnn_status xnn_define_copy(xnn_subgraph_t subgraph, uint32_t input_id,
                           uint32_t output_id, uint32_t flags) {
  return ynn::xnn_status_from_ynn(
      ynn_define_copy(subgraph->ynn, input_id, &output_id, /*flags=*/0));
}

xnn_status xnn_define_even_split(xnn_subgraph_t subgraph, int32_t split_dim,
                                 uint32_t input_id, size_t num_outputs,
                                 const uint32_t* outputs, uint32_t flags) {
  std::vector<uint32_t> output_ids(outputs, outputs + num_outputs);
  return ynn::xnn_status_from_ynn(
      ynn_define_even_split(subgraph->ynn, split_dim, input_id, num_outputs,
                            output_ids.data(), /*flags=*/0));
}

xnn_status xnn_define_static_reshape(xnn_subgraph_t subgraph, size_t num_dims,
                                     const size_t* new_shape, uint32_t input_id,
                                     uint32_t output_id, uint32_t flags) {
  return ynn::xnn_status_from_ynn(ynn_define_static_reshape(
      subgraph->ynn, num_dims, new_shape, input_id, &output_id, /*flags=*/0));
}

xnn_status xnn_define_static_broadcast(xnn_subgraph_t subgraph, size_t num_dims,
                                       const size_t* new_shape,
                                       uint32_t input_id, uint32_t output_id,
                                       uint32_t flags) {
  return ynn::xnn_status_from_ynn(ynn_define_static_broadcast(
      subgraph->ynn, num_dims, new_shape, input_id, &output_id, /*flags=*/0));
}

xnn_status xnn_define_static_resize_bilinear_2d(
    xnn_subgraph_t subgraph, size_t new_height, size_t new_width,
    uint32_t input_id, uint32_t output_id, uint32_t flags) {
  YNN_LOG_ERROR() << "Unsupported xnn_define_static_resize_bilinear_2d";
  return xnn_status_deprecated;
}

xnn_status xnn_define_rope(xnn_subgraph_t subgraph, size_t max_sequence_size,
                           uint32_t input_id, uint32_t weights_id,
                           uint32_t output_id, uint32_t flags) {
  // This operation is basically a complex multiply, where the real and
  // imaginary parts of input and weights are a split of the channel dimension:
  //
  //   y(n, t, h, 0, c) =
  //       x(n, t, h, 0, c) * w(t, 0, c) - x(n, t, h, 1, c) * w(t, 1, c)
  //   y(n, t, h, 1, c) =
  //       x(n, t, h, 0, c) * w(t, 1, c) + x(n, t, h, 1, c) * w(t, 0, c)

  // [t, c] -> [0, t, 0, c]
  const int32_t weights_broadcast_axes[] = {0, 2};
  uint32_t weights_broadcasted_id = YNN_INVALID_VALUE_ID;
  ynn_status status =
      ynn_define_static_expand_dims(subgraph->ynn, 2, weights_broadcast_axes,
                                    weights_id, &weights_broadcasted_id,
                                    /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  const int32_t z_axis = 3;

  // [n, t, h, {re, im}_c] -> [n, t, h, {re, im}, c]
  const size_t split_dims[] = {2, 0};
  uint32_t weights_re_im_id = YNN_INVALID_VALUE_ID;
  status =
      ynn_define_split_dim(subgraph->ynn, z_axis, /*num_splits=*/2, split_dims,
                           weights_broadcasted_id, &weights_re_im_id,
                           /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  // [n, t, h, {re, im}_c] -> [n, t, h, re_c], [n, t, h, im_c]
  uint32_t input_re_im_ids[2] = {YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID};
  status =
      ynn_define_even_split(subgraph->ynn, z_axis, input_id,
                            /*num_outputs=*/2, input_re_im_ids, /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  // [n, t, h, c] -> [n, t, h, 0, c]
  uint32_t input_re_id = YNN_INVALID_VALUE_ID;
  uint32_t input_im_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_static_expand_dims(subgraph->ynn, /*num_new_axes=*/1,
                                         &z_axis, input_re_im_ids[0],
                                         &input_re_id, /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }
  status = ynn_define_static_expand_dims(subgraph->ynn, /*num_new_axes=*/1,
                                         &z_axis, input_re_im_ids[1],
                                         &input_im_id, /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  // broadcast(re(input)) * weights
  uint32_t input_re_times_weights_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_binary(subgraph->ynn, ynn_binary_multiply, input_re_id,
                             weights_re_im_id, &input_re_times_weights_id,
                             /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }
  // broadcast(im(input)) * weights
  uint32_t input_im_times_weights_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_binary(subgraph->ynn, ynn_binary_multiply, input_im_id,
                             weights_re_im_id, &input_im_times_weights_id,
                             /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  // Slice out the re, im parts of each multiplication.
  const int64_t zero = 0;
  const int64_t one = 1;
  uint32_t input_re_times_weights_re_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_static_slice(
      subgraph->ynn, /*num_dims=*/1, &z_axis, &zero, nullptr, nullptr,
      input_re_times_weights_id, &input_re_times_weights_re_id,
      /*flags=*/YNN_NODE_FLAG_SLICE_DIMS);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }
  uint32_t input_re_times_weights_im_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_static_slice(subgraph->ynn, /*num_dims=*/1, &z_axis, &one,
                                   nullptr, nullptr, input_re_times_weights_id,
                                   &input_re_times_weights_im_id,
                                   /*flags=*/YNN_NODE_FLAG_SLICE_DIMS);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  uint32_t input_im_times_weights_re_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_static_slice(
      subgraph->ynn, /*num_dims=*/1, &z_axis, &zero, nullptr, nullptr,
      input_im_times_weights_id, &input_im_times_weights_re_id,
      /*flags=*/YNN_NODE_FLAG_SLICE_DIMS);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }
  uint32_t input_im_times_weights_im_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_static_slice(subgraph->ynn, /*num_dims=*/1, &z_axis, &one,
                                   nullptr, nullptr, input_im_times_weights_id,
                                   &input_im_times_weights_im_id,
                                   /*flags=*/YNN_NODE_FLAG_SLICE_DIMS);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  // Subtract the real parts.
  uint32_t result_ids[2] = {YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID};
  status = ynn_define_binary(
      subgraph->ynn, ynn_binary_subtract, input_re_times_weights_re_id,
      input_im_times_weights_im_id, &result_ids[0], /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }
  // Add the imaginary parts.
  status = ynn_define_binary(
      subgraph->ynn, ynn_binary_add, input_re_times_weights_im_id,
      input_im_times_weights_re_id, &result_ids[1], /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  // Concatenate the real, imaginary parts back to the result.
  status = ynn_define_concatenate(subgraph->ynn, /*axis=*/3, /*num_inputs=*/2,
                                  result_ids, &output_id, /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  return xnn_status_success;
}

xnn_status xnn_define_batch_matrix_multiply(xnn_subgraph_t subgraph,
                                            uint32_t input1_id,
                                            uint32_t input2_id,
                                            uint32_t output_id,
                                            uint32_t flags) {
  ynn_status status = ynn::implement_xnn_broadcasting(
      subgraph->ynn, &input1_id, &input2_id, flags, /*exclude_a=*/2,
      /*exclude_b=*/2);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  if (flags & XNN_FLAG_TRANSPOSE_B) {
    uint32_t input2_id_transposed = YNN_INVALID_VALUE_ID;
    const size_t b_rank = ynn::rank_of_value(subgraph->ynn, input2_id);
    std::array<int32_t, YNN_MAX_TENSOR_RANK> perm;
    std::iota(perm.begin(), perm.end(), 0);
    std::swap(perm[b_rank - 1], perm[b_rank - 2]);
    status = ynn_define_static_transpose(subgraph->ynn,
                                         /*num_dims=*/b_rank, perm.data(),
                                         input2_id, &input2_id_transposed,
                                         /*flags=*/0);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }
    input2_id = input2_id_transposed;
  }

  status = ynn::define_xnn_dot(subgraph->ynn,
                               /*num_k_dims=*/1, input1_id, input2_id,
                               /*bias_id=*/YNN_INVALID_VALUE_ID, output_id);
  return ynn::xnn_status_from_ynn(status);
}

xnn_status xnn_define_softmax(xnn_subgraph_t subgraph, uint32_t input_id,
                              uint32_t output_id, uint32_t flags) {
  // TODO: This implementation needs helper functions...

  uint32_t max_input_id = YNN_INVALID_VALUE_ID;
  uint32_t max_identity_id = YNN_INVALID_VALUE_ID;
  const int32_t last_axis[] = {-1};
  ynn_status status =
      ynn_define_reduce(subgraph->ynn, ynn_reduce_max, 1, last_axis, input_id,
                        max_identity_id, &max_input_id,
                        /*flags=*/YNN_NODE_FLAG_KEEP_DIMS);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  uint32_t input_minus_max_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_binary(subgraph->ynn, ynn_binary_subtract, input_id,
                             max_input_id, &input_minus_max_id,
                             /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  uint32_t exp_input_minus_max_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_unary(subgraph->ynn, ynn_unary_exp, input_minus_max_id,
                            &exp_input_minus_max_id, /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  uint32_t sum_exp_input_minus_max_id = YNN_INVALID_VALUE_ID;
  uint32_t init_sum_exp_input_minus_max_id = YNN_INVALID_VALUE_ID;
  status = ynn_define_reduce(
      subgraph->ynn, ynn_reduce_sum, 1, last_axis, exp_input_minus_max_id,
      init_sum_exp_input_minus_max_id, &sum_exp_input_minus_max_id,
      /*flags=*/YNN_NODE_FLAG_KEEP_DIMS);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  uint32_t inv_sum_id = YNN_INVALID_VALUE_ID;
  status = ynn::define_binary_scalar_a(subgraph->ynn, ynn_binary_divide, 1.0f,
                                       sum_exp_input_minus_max_id, &inv_sum_id);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }

  if (ynn::type_of_value(subgraph->ynn, inv_sum_id) !=
      ynn::type_of_value(subgraph->ynn, input_id)) {
    uint32_t inv_sum_cast_id = YNN_INVALID_VALUE_ID;
    status = ynn::define_tensor_value_like(subgraph->ynn, input_id,
                                           &inv_sum_cast_id);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }

    status = ynn_define_unary(subgraph->ynn, ynn_unary_convert, inv_sum_id,
                              &inv_sum_cast_id, /*flags=*/0);
    if (status != ynn_status_success) {
      return ynn::xnn_status_from_ynn(status);
    }
    inv_sum_id = inv_sum_cast_id;
  }

  status = ynn_define_binary(subgraph->ynn, ynn_binary_multiply,
                             exp_input_minus_max_id, inv_sum_id, &output_id,
                             /*flags=*/0);
  if (status != ynn_status_success) {
    return ynn::xnn_status_from_ynn(status);
  }
  return xnn_status_success;
}

xnn_status xnn_define_static_slice(xnn_subgraph_t subgraph, size_t num_dims,
                                   const size_t* offsets, const size_t* sizes,
                                   uint32_t input_id, uint32_t output_id,
                                   uint32_t flags) {
  int64_t ynn_offsets[XNN_MAX_TENSOR_DIMS];
  std::copy_n(offsets, num_dims, ynn_offsets);
  return xnn_define_static_slice_v2(subgraph, num_dims, ynn_offsets, sizes,
                                    input_id, output_id, flags);
}

xnn_status xnn_define_static_slice_v2(xnn_subgraph_t subgraph, size_t num_dims,
                                      const int64_t* offsets,
                                      const size_t* sizes, uint32_t input_id,
                                      uint32_t output_id, uint32_t flags) {
  int64_t ends[XNN_MAX_TENSOR_DIMS];
  for (int i = 0; i < num_dims; i++) {
    ends[i] = offsets[i] + (int64_t)sizes[i];
  }
  return xnn_define_static_slice_v3(subgraph, num_dims, offsets, ends,
                                    /*strides=*/nullptr, input_id, output_id,
                                    flags);
}

xnn_status xnn_define_static_slice_v3(xnn_subgraph_t subgraph, size_t num_dims,
                                      const int64_t* begins,
                                      const int64_t* ends,
                                      const int64_t* strides, uint32_t input_id,
                                      uint32_t output_id, uint32_t flags) {
  int32_t axes[YNN_MAX_TENSOR_RANK];
  std::iota(axes, axes + YNN_MAX_TENSOR_RANK, 0);
  return ynn::xnn_status_from_ynn(
      ynn_define_static_slice(subgraph->ynn, num_dims, axes, begins, ends,
                              strides, input_id, &output_id, /*flags=*/0));
}

xnn_status xnn_define_static_transpose(xnn_subgraph_t subgraph, size_t num_dims,
                                       const size_t* perm, uint32_t input_id,
                                       uint32_t output_id, uint32_t flags) {
  int32_t ynn_perm[XNN_MAX_TENSOR_DIMS];
  std::copy_n(perm, num_dims, ynn_perm);
  return ynn::xnn_status_from_ynn(ynn_define_static_transpose(
      subgraph->ynn, num_dims, ynn_perm, input_id, &output_id, /*flags=*/0));
}

}  // extern "C"
