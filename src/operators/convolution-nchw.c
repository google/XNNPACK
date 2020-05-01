// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/compute.h>
#include <xnnpack/indirection.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


static inline size_t compute_output_dimension(
    size_t padded_input_dimension,
    size_t kernel_dimension,
    size_t dilation_dimension,
    size_t subsampling_dimension)
{
  const size_t effective_kernel_dimension = (kernel_dimension - 1) * dilation_dimension + 1;
  return doz(padded_input_dimension, effective_kernel_dimension) / subsampling_dimension + 1;
}

enum xnn_status xnn_create_convolution2d_nchw_f32(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t subsampling_height,
    uint32_t subsampling_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    const float* kernel,
    const float* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* convolution_op_out)
{
  xnn_operator_t convolution_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create Convolution operator: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      kernel_width, kernel_height);
    goto error;
  }

  if (subsampling_width == 0 || subsampling_height == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %" PRIu32 "x%" PRIu32 " subsampling: "
      "subsampling dimensions must be non-zero",
      subsampling_width, subsampling_height);
    goto error;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %" PRIu32 "x%" PRIu32 " dilation: "
      "dilation dimensions must be non-zero",
      dilation_width, dilation_height);
    goto error;
  }

  if (groups == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %" PRIu32 " groups: number of groups must be non-zero", groups);
    goto error;
  }

  if (group_input_channels == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %zu input channels per group: "
      "number of channels must be non-zero",
      group_input_channels);
    goto error;
  }

  if (group_output_channels == 0) {
    xnn_log_error(
      "failed to create Convolution operator with %zu output channels per group: "
      "number of channels must be non-zero",
      group_output_channels);
    goto error;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create Convolution operator with NaN output lower bound: lower bound must be non-NaN");
    goto error;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create Convolution operator with NaN output upper bound: upper bound must be non-NaN");
    goto error;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create Convolution operator with [%.7g, %.7g] output range: "
      "lower bound must be below upper bound",
      output_min, output_max);
    goto error;
  }

  if ((flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) != 0 && group_input_channels != 1) {
    xnn_log_error(
      "failed to create Depthwise Convolution operator with %zu input channels per group: "
      "Depthwise Convolution must have exactly 1 input channel per group",
      group_input_channels);
    goto error;
  }

  status = xnn_status_unsupported_parameter;

  enum xnn_ukernel_type ukernel_type;
  struct spchw_dwconv_parameters* dwconv_parameters = NULL;
  // Supported cases:
  // + 1x1 convolution (no groups)
  // + 3x3 stride-2 with 3 input channels and NHWC input layout
  // + 3x3 stride-2 depthwise convolution with horizontal padding 1 & no vertical padding
  // - 3x3 stride-1 depthwise convolution with horizontal padding 1 & no vertical padding
  // - 5x5 stride-2 depthwise convolution with horizontal padding 2 & no vertical padding
  // - 5x5 stride-1 depthwise convolution with horizontal padding 2 & no vertical padding
  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  const bool is_1x1 = kernel_width == 1 && kernel_height == 1 && subsampling_height == 1 && subsampling_width == 1;
  const bool is_3x3 = kernel_width == 3 && kernel_height == 3 && dilation_height == 1 && dilation_width == 1;
  const bool is_5x5 = kernel_width == 5 && kernel_height == 5 && dilation_height == 1 && dilation_width == 1;
  const bool nhwc_input = (flags & XNN_FLAG_INPUT_NHWC) != 0;
  if (is_1x1 && !any_padding && !nhwc_input && groups == 1 && xnn_params.f32.spmm.ukernel != NULL) {
    ukernel_type = xnn_ukernel_type_spmm;
  } else if (is_3x3 && subsampling_height == 2 && subsampling_width == 2 &&
    input_padding_top == 1 && input_padding_left == 1 && input_padding_bottom == 1 && input_padding_right == 1 &&
    nhwc_input && groups == 1 && xnn_params.f32.hwc2spchw_dconv3x3c3s2.ukernel_with_symm_padding != NULL)
  {
    ukernel_type = xnn_ukernel_type_dconv2d_hwc2spchw;
  } else if (is_3x3 && subsampling_height == 1 && subsampling_width == 1 &&
    input_padding_top == 0 && input_padding_left == 1 && input_padding_bottom == 0 && input_padding_right == 1 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1 && xnn_params.f32.spchw_dwconv3x3.ukernel != NULL)
  {
    ukernel_type = xnn_ukernel_type_dwconv;
    dwconv_parameters = &xnn_params.f32.spchw_dwconv3x3;
  } else if (is_3x3 && subsampling_height == 2 && subsampling_width == 2 &&
    input_padding_top == 0 && input_padding_left == 1 && input_padding_bottom == 0 && input_padding_right == 1 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1 && xnn_params.f32.spchw_dwconv3x3s2.ukernel != NULL)
  {
    ukernel_type = xnn_ukernel_type_dwconv;
    dwconv_parameters = &xnn_params.f32.spchw_dwconv3x3s2;
  } else if (is_5x5 && subsampling_height == 1 && subsampling_width == 1 &&
    input_padding_top == 0 && input_padding_left == 2 && input_padding_bottom == 0 && input_padding_right == 2 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1 && xnn_params.f32.spchw_dwconv5x5.ukernel != NULL)
  {
    ukernel_type = xnn_ukernel_type_dwconv;
    dwconv_parameters = &xnn_params.f32.spchw_dwconv5x5;
  } else if (is_5x5 && subsampling_height == 2 && subsampling_width == 2 &&
    input_padding_top == 0 && input_padding_left == 2 && input_padding_bottom == 0 && input_padding_right == 2 &&
    !nhwc_input && group_input_channels == 1 && group_output_channels == 1 && xnn_params.f32.spchw_dwconv5x5s2.ukernel != NULL)
  {
    ukernel_type = xnn_ukernel_type_dwconv;
    dwconv_parameters = &xnn_params.f32.spchw_dwconv5x5s2;
  } else {
    xnn_log_error(
      "failed to create Convolution operator: only selected Convolution parameters are supported");
    goto error;
  }

  status = xnn_status_out_of_memory;

  convolution_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (convolution_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for Convolution operator descriptor", sizeof(struct xnn_operator));
    goto error;
  }

  switch (ukernel_type) {
    case xnn_ukernel_type_spmm:
    {
      assert(kernel_height == 1);
      assert(kernel_width == 1);
      assert(groups == 1);

      size_t num_nonzeroes = 0;
      size_t num_nonzero_blocks2 = 0;
      size_t num_nonzero_blocks4 = 0;
      for (size_t oc = 0; oc < round_down_po2(group_output_channels, 4); oc += 4) {
        for (size_t ic = 0; ic < group_input_channels; ic++) {
          const size_t row0_nonzero = (size_t) (kernel[oc * group_input_channels + ic] != 0.0f);
          const size_t row1_nonzero = (size_t) (kernel[(oc + 1) * group_input_channels + ic] != 0.0f);
          const size_t row2_nonzero = (size_t) (kernel[(oc + 2) * group_input_channels + ic] != 0.0f);
          const size_t row3_nonzero = (size_t) (kernel[(oc + 3) * group_input_channels + ic] != 0.0f);
          num_nonzeroes += row0_nonzero + row1_nonzero + row2_nonzero + row3_nonzero;
          num_nonzero_blocks2 += (row0_nonzero | row1_nonzero) + (row2_nonzero | row3_nonzero);
          num_nonzero_blocks4 += (row0_nonzero | row1_nonzero | row2_nonzero | row3_nonzero);
        }
      }
      const size_t num_block4_nonzeroes = num_nonzeroes;
      for (size_t oc = round_down_po2(group_output_channels, 4); oc < round_down_po2(group_output_channels, 2); oc += 2) {
        for (size_t ic = 0; ic < group_input_channels; ic++) {
          const size_t row0_nonzero = (size_t) (kernel[oc * group_input_channels + ic] != 0.0f);
          const size_t row1_nonzero = (size_t) (kernel[(oc + 1) * group_input_channels + ic] != 0.0f);
          num_nonzeroes += row0_nonzero + row1_nonzero;
          num_nonzero_blocks2 += (row0_nonzero | row1_nonzero);
        }
      }
      const size_t num_block2_nonzeroes = num_nonzeroes;
      for (size_t oc = round_down_po2(group_output_channels, 2); oc < group_output_channels; oc++) {
        for (size_t ic = 0; ic < group_input_channels; ic++) {
          num_nonzeroes += (size_t) (kernel[oc * group_input_channels + ic] != 0.0f);
        }
      }
      size_t output_channels_block_size = 1;
      size_t num_output_channel_blocks = group_output_channels;
      size_t num_nonzero_values = num_nonzeroes;
      size_t num_nonzero_blocks = num_nonzeroes;
      const struct spmm_parameters* spmm_parameters = &xnn_params.f32.spmm;
      if (num_block4_nonzeroes * 5 >= num_nonzero_blocks4 * 18 && xnn_params.f32.spmm4.ukernel != NULL) {
        // 4-channel blocks have 90%+ non-zeroes

        output_channels_block_size = 4;
        num_output_channel_blocks = num_output_channel_blocks / 4 + num_output_channel_blocks % 4;
        spmm_parameters = &xnn_params.f32.spmm4;
        // Non-zeroes which don't fit into whole 4-channel blocks, processed one-by-one
        const size_t num_remaining_nonzeroes = num_nonzeroes - num_block4_nonzeroes;
        num_nonzero_values = num_nonzero_blocks4 * 4 + num_remaining_nonzeroes;
        num_nonzero_blocks = num_nonzero_blocks4 + num_remaining_nonzeroes;
      } else if (num_block2_nonzeroes * 5 >= num_nonzero_blocks2 * 9 && xnn_params.f32.spmm2.ukernel != NULL) {
        // 2-channel blocks have 90%+ non-zeroes

        output_channels_block_size = 2;
        num_output_channel_blocks = num_output_channel_blocks / 2 + num_output_channel_blocks % 2;
        spmm_parameters = &xnn_params.f32.spmm2;
        // Non-zeroes which don't fit into whole 2-channel blocks, processed one-by-one
        const size_t num_remaining_nonzeroes = num_nonzeroes - num_block2_nonzeroes;
        num_nonzero_values = num_nonzero_blocks2 * 2 + num_remaining_nonzeroes;
        num_nonzero_blocks = num_nonzero_blocks2 + num_remaining_nonzeroes;
      }

      // Sparse representation of weights consists of four components:
      // 1. An array of float values storing non-zero kernel elements, and all (group_output_channels) bias elements.
      //    All elements within non-zero block are assumed to be non-zero.
      // 2. An array of int32_t values storing increment for input pointer after each processed tile. This array is
      //    derived from scaled difference in array 2 using parameters to setup function.
      // 3. An array of uint32_t values storing the number of non-zero kernel elements per each output channel.
      // 4. An array of int32_t values storing scaled [by sizeof(input element)] difference between input channels
      //    corresponding to successive non-zero blocks.
      const size_t packed_weights_size = num_output_channel_blocks * sizeof(uint32_t) +
        (num_nonzero_blocks * 2) * sizeof(int32_t) + (num_nonzero_values + group_output_channels) * sizeof(float);

      convolution_op->packed_weights = xnn_allocate_simd_memory(packed_weights_size);
      if (convolution_op->packed_weights == NULL) {
        xnn_log_error("failed to allocate %zu bytes for packed weights", packed_weights_size);
        goto error;
      }
      convolution_op->num_nonzero_values = num_nonzero_values;
      convolution_op->num_nonzero_blocks = num_nonzero_blocks;
      convolution_op->num_output_channel_blocks = num_output_channel_blocks;

      float* nonzero_values = convolution_op->packed_weights;
      int32_t* input_increments = (int32_t*) (nonzero_values + num_nonzero_values + group_output_channels);
      uint32_t* output_channel_nonzeros = (uint32_t*) (input_increments + num_nonzero_blocks);
      int32_t* input_channel_diffs = (int32_t*) (output_channel_nonzeros + num_output_channel_blocks);
      memset(output_channel_nonzeros, 0, num_output_channel_blocks * sizeof(uint32_t));

      status = xnn_status_unsupported_parameter;

      size_t first_ic = 0, last_ic = 0;
      bool first_nonzero = true;
      for (size_t ocb = 0; ocb < round_down_po2(group_output_channels, output_channels_block_size); ocb += output_channels_block_size) {
        if XNN_LIKELY(bias != NULL) {
          for (size_t oco = 0; oco < output_channels_block_size; oco++) {
            *nonzero_values++ = bias[ocb + oco];
          }
        } else {
          for (size_t oco = 0; oco < output_channels_block_size; oco++) {
            *nonzero_values++ = 0.0f;
          }
        }
        for (size_t ic = 0; ic < group_input_channels; ic++) {
          bool is_nonzero_block = false;
          for (size_t oco = 0; oco < output_channels_block_size; oco++) {
            is_nonzero_block |= (kernel[(ocb + oco) * group_input_channels + ic] != 0.0f);
          }
          if (is_nonzero_block) {
            for (size_t oco = 0; oco < output_channels_block_size; oco++) {
              *nonzero_values++ = kernel[(ocb + oco) * group_input_channels + ic];
            }
            if (first_nonzero) {
              first_ic = ic;
            } else {
              const int64_t diff = (int64_t) ((uint64_t) ic - (uint64_t) last_ic) * (int64_t) sizeof(float);
              if (diff != (int64_t) (int32_t) diff) {
                xnn_log_error("failed to convert kernel to sparse representation: "
                  "scaled difference in input channels exceeds int32_t range");
                goto error;
              }
              *input_channel_diffs++ = (int32_t) diff;
            }
            first_nonzero = false;
            last_ic = ic;
            *output_channel_nonzeros += 1;
          }
        }
        output_channel_nonzeros += 1;
      }
      for (size_t oc = round_down_po2(group_output_channels, output_channels_block_size); oc < group_output_channels; oc++) {
        if XNN_LIKELY(bias != NULL) {
          *nonzero_values++ = bias[oc];
        } else {
          *nonzero_values++ = 0.0f;
        }
        for (size_t ic = 0; ic < group_input_channels; ic++) {
          const float weight = kernel[oc * group_input_channels + ic];
          if (weight != 0.0f) {
            *nonzero_values++ = weight;
            if (first_nonzero) {
              first_ic = ic;
            } else {
              const int64_t diff = (int64_t) ((uint64_t) ic - (uint64_t) last_ic) * (int64_t) sizeof(float);
              if (diff != (int64_t) (int32_t) diff) {
                xnn_log_error("failed to convert kernel to sparse representation: "
                  "scaled difference in input channels exceeds int32_t range");
                goto error;
              }
              *input_channel_diffs++ = (int32_t) diff;
            }
            first_nonzero = false;
            last_ic = ic;
            *output_channel_nonzeros += 1;
          }
        }
        output_channel_nonzeros += 1;
      }
      // If there are any non-zero elements, we have to return to the initial input channel.
      if (!first_nonzero) {
        const int64_t diff = (int64_t) ((uint64_t) first_ic - (uint64_t) last_ic) * (int64_t) sizeof(float);
        if (diff != (int64_t) (int32_t) diff) {
          xnn_log_error("failed to convert kernel to sparse representation: "
            "scaled difference in input channels exceeds int32_t range");
          goto error;
        }
        *input_channel_diffs++ = (int32_t) diff;
      }
      convolution_op->first_input_channel = first_ic;

      convolution_op->ukernel.spmm = (struct xnn_ukernel_spmm) {
        .function = spmm_parameters->ukernel,
        .mr = spmm_parameters->mr,
      };

      break;
    }
    case xnn_ukernel_type_dconv2d_hwc2spchw:
    {
      assert(groups == 1);

      const size_t packed_group_output_channels =
        round_up(group_output_channels, xnn_params.f32.hwc2spchw_dconv3x3c3s2.output_channel_tile);
      const size_t packed_weights_size = groups * packed_group_output_channels *
        (group_input_channels * kernel_height * kernel_width + 1 /* bias */) * sizeof(float);
      convolution_op->packed_weights = xnn_allocate_simd_memory(packed_weights_size);
      if (convolution_op->packed_weights == NULL) {
        xnn_log_error("failed to allocate %zu bytes for packed weights", packed_weights_size);
        goto error;
      }

      xnn_pack_f32_dconv_oki_w(
        group_output_channels,
        group_input_channels,
        xnn_params.f32.hwc2spchw_dconv3x3c3s2.output_channel_tile,
        kernel_height, kernel_width,
        kernel, bias, convolution_op->packed_weights);

      convolution_op->ukernel.dconv2d = (struct xnn_ukernel_dconv2d) {
        .hwc2spchw_function = xnn_params.f32.hwc2spchw_dconv3x3c3s2.ukernel_with_symm_padding,
        .output_height_tile = xnn_params.f32.hwc2spchw_dconv3x3c3s2.output_height_tile,
        .output_channel_tile = xnn_params.f32.hwc2spchw_dconv3x3c3s2.output_channel_tile,
      };

      break;
    }
    case xnn_ukernel_type_dwconv:
    {
      assert(dwconv_parameters != NULL);
      assert(group_input_channels == 1);
      assert(group_output_channels == 1);

      const size_t packed_weights_size = groups * (kernel_height * kernel_width + 1 /* bias */) * sizeof(float);
      convolution_op->packed_weights = xnn_allocate_simd_memory(packed_weights_size);
      if (convolution_op->packed_weights == NULL) {
        xnn_log_error("failed to allocate %zu bytes for packed weights", packed_weights_size);
        goto error;
      }

      xnn_pack_f32_spchw_dwconv_ghw_w(
        kernel_height * kernel_width, groups,
        kernel, bias, convolution_op->packed_weights);

      convolution_op->ukernel.dwconv2d = (struct xnn_ukernel_dwconv2d) {
        .spchw_function = dwconv_parameters->ukernel,
        .input_width_tile = dwconv_parameters->input_width_tile,
        .output_width_tile = dwconv_parameters->output_width_tile,
      };

      break;
    }
    default:
      XNN_UNREACHABLE;
  }

  convolution_op->padding_top = input_padding_top;
  convolution_op->padding_right = input_padding_right;
  convolution_op->padding_bottom = input_padding_bottom;
  convolution_op->padding_left = input_padding_left;

  convolution_op->kernel_height = kernel_height;
  convolution_op->kernel_width = kernel_width;
  convolution_op->stride_height = subsampling_height;
  convolution_op->stride_width = subsampling_width;
  convolution_op->dilation_height = dilation_height;
  convolution_op->dilation_width = dilation_width;
  convolution_op->groups = groups;
  convolution_op->group_input_channels = group_input_channels;
  convolution_op->group_output_channels = group_output_channels;

  if (ukernel_type == xnn_ukernel_type_dwconv) {
    convolution_op->f32_spchw_params = xnn_init_f32_spchw_params(0, output_min, output_max);
  } else {
    convolution_op->f32_minmax_params = xnn_init_f32_minmax_params(output_min, output_max);
  }

  convolution_op->type = xnn_operator_type_convolution_nchw_f32;
  convolution_op->ukernel.type = ukernel_type;

  convolution_op->state = xnn_run_state_invalid;

  *convolution_op_out = convolution_op;
  return xnn_status_success;

error:
  xnn_delete_operator(convolution_op);
  return status;
}

static enum xnn_status setup_convolution2d_nchw(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_batch_stride,
  size_t output_batch_stride,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
  uint32_t log2_input_element_size,
  uint32_t log2_filter_element_size,
  uint32_t bias_element_size,
  uint32_t log2_output_element_size,
  const void* params,
  const void* spchw_params,
  size_t num_threads)
{
  convolution_op->state = xnn_run_state_invalid;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to setup Convolution operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
      "failed to setup Convolution operator with %zux%zu input: input dimensions must be non-zero",
      input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  const uint32_t groups = convolution_op->groups;
  const size_t group_input_channels = convolution_op->group_input_channels;
  const size_t input_neurons = groups * group_input_channels * input_height * input_width;
  if (input_batch_stride < input_neurons) {
    xnn_log_error(
      "failed to setup Convolution operator with input batch stride of %zu: "
      "stride must be at least as large as the number of input neurons (%" PRIu32 "x%zux%zux%zu)",
      input_batch_stride, groups, group_input_channels, input_height, input_width);
    return xnn_status_invalid_parameter;
  }

  const size_t output_height = compute_output_dimension(
      convolution_op->padding_top + input_height + convolution_op->padding_bottom,
      convolution_op->kernel_height,
      convolution_op->dilation_height,
      convolution_op->stride_height);
  const size_t output_width = compute_output_dimension(
      convolution_op->padding_left + input_width + convolution_op->padding_right,
      convolution_op->kernel_width,
      convolution_op->dilation_width,
      convolution_op->stride_width);

  const size_t group_output_channels = convolution_op->group_output_channels;
  const size_t output_neurons = groups * group_output_channels * output_height * output_width;
  if (output_batch_stride < output_neurons) {
    xnn_log_error(
      "failed to setup Convolution operator with output batch stride of %zu: "
      "stride must be at least as large as the number of output neurons (%" PRIu32 "x%zux%zux%zu)",
      output_batch_stride, groups, group_output_channels, output_height, output_width);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    convolution_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  convolution_op->batch_size = batch_size;
  convolution_op->input_height = input_height;
  convolution_op->input_width = input_width;
  convolution_op->input = input;
  convolution_op->output = output;

  switch (convolution_op->ukernel.type) {
    case xnn_ukernel_type_spmm:
    {
      const size_t num_nonzero_values = convolution_op->num_nonzero_values;
      const size_t num_nonzero_blocks = convolution_op->num_nonzero_blocks;
      const size_t num_output_channel_blocks = convolution_op->num_output_channel_blocks;

      convolution_op->num_nonzero_values = num_nonzero_values;
      convolution_op->num_nonzero_blocks = num_nonzero_blocks;
      convolution_op->num_output_channel_blocks = num_output_channel_blocks;

      float* nonzero_values = convolution_op->packed_weights;
      int32_t* input_increments = (int32_t*) (nonzero_values + num_nonzero_values + convolution_op->group_output_channels);
      uint32_t* output_channel_nonzeros = (uint32_t*) (input_increments + num_nonzero_blocks);
      int32_t* input_channel_diffs = (int32_t*) (output_channel_nonzeros + num_output_channel_blocks);

      const size_t input_size = input_height * input_width;
      for (size_t i = 0; i < num_nonzero_blocks; i++) {
        const int32_t diff = input_channel_diffs[i];
        const int64_t increment = (int64_t) diff * input_size;
        if ((int64_t) (int32_t) increment != increment) {
          xnn_log_error("failed to setup Convolution operator with sparse kernel representation: "
            "input increment exceeds int32_t range");
          return xnn_status_unsupported_parameter;
        }
        input_increments[i] = (int32_t) increment;
      }

      convolution_op->context.spmm = (struct spmm_context) {
          .n = group_output_channels,
          .a = (const void*) ((uintptr_t) input + (convolution_op->first_input_channel * input_size * sizeof(float))),
          .packed_weights = nonzero_values,
          .input_increments = input_increments,
          .output_channel_nonzeros = output_channel_nonzeros,
          .c = output,
          .batched_a_stride = input_batch_stride << log2_input_element_size,
          .batched_c_stride = output_batch_stride << log2_output_element_size,
          .ukernel = convolution_op->ukernel.spmm.function,
      };
      memcpy(&convolution_op->context.spmm.params, params, sizeof(convolution_op->context.spmm.params));

      const size_t mr = convolution_op->ukernel.spmm.mr;
      size_t mc = input_size;
      if (num_threads > 1) {
        const size_t target_tiles_per_thread = 5;
        const size_t max_mc = divide_round_up(input_size, num_threads * target_tiles_per_thread);
        if (max_mc < mc) {
          mc = min(mc, divide_round_up(mc, max_mc * mr) * mr);
        }
      }
      convolution_op->compute.type = xnn_parallelization_type_2d_tile_1d;
      convolution_op->compute.task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_spmm;
      convolution_op->compute.range[0] = batch_size;
      convolution_op->compute.range[1] = input_size;
      convolution_op->compute.tile[0] = mc;
      convolution_op->state = xnn_run_state_ready;

      return xnn_status_success;
    }
    case xnn_ukernel_type_dconv2d_hwc2spchw:
    {
      const size_t zero_size = (input_width * convolution_op->group_input_channels << log2_input_element_size) + XNN_EXTRA_BYTES;
      void* zero_buffer = xnn_reallocate_memory(convolution_op->zero_buffer, zero_size);
      if (zero_buffer == NULL) {
        xnn_log_error("failed to allocate %zu bytes for zero padding", sizeof(struct xnn_operator));
        return xnn_status_out_of_memory;
      }
      memset(zero_buffer, 0, zero_size);
      convolution_op->zero_buffer = zero_buffer;

      convolution_op->context.dconv2d = (struct dconv2d_context) {
        .input_height = input_height,
        .input_width = input_width,
        .input = input,
        .input_batch_stride = input_batch_stride << log2_input_element_size,
        .zero = zero_buffer,
        .packed_weights = convolution_op->packed_weights,
        .output = output,
        .output_batch_stride = output_batch_stride << log2_input_element_size,
        .input_padding_top = convolution_op->padding_top,
        .output_channels = convolution_op->group_output_channels,
        .output_height_stride = output_width << log2_output_element_size,
        .output_channel_stride = output_height * output_width << log2_output_element_size,
        .hwc2spchw_ukernel = convolution_op->ukernel.dconv2d.hwc2spchw_function,
      };
      memcpy(&convolution_op->context.dconv2d.params, params, sizeof(convolution_op->context.dconv2d.params));

      size_t output_height_slice = output_height;
      const size_t output_height_tile = convolution_op->ukernel.dconv2d.output_height_tile;
      if (num_threads > 1) {
        const size_t target_tiles_per_thread = 5;
        const size_t max_output_height_slice = divide_round_up(output_height, num_threads * target_tiles_per_thread);
        if (max_output_height_slice < output_height_slice) {
          output_height_slice = min(output_height_slice,
            divide_round_up(output_height_slice, max_output_height_slice * output_height_tile) * output_height_tile);
        }
      }
      convolution_op->compute.type = xnn_parallelization_type_2d_tile_1d;
      convolution_op->compute.task_2d_tile_1d = (pthreadpool_task_2d_tile_1d_t) xnn_compute_dconv2d_hwc2spchw;
      convolution_op->compute.range[0] = batch_size;
      convolution_op->compute.range[1] = output_height;
      convolution_op->compute.tile[0] = output_height_slice;
      convolution_op->state = xnn_run_state_ready;

      return xnn_status_success;
    }
    case xnn_ukernel_type_dwconv:
    {
      xnn_update_f32_spchw_params((union xnn_f32_spchw_params*) spchw_params, (uint32_t) input_width);
      convolution_op->context.dwconv2d = (struct dwconv2d_context) {
        .output_height = output_height,
        .input_width = input_width,
        .input = input,
        .input_channel_stride = input_height * input_width << log2_input_element_size,
        .input_batch_stride = input_batch_stride << log2_input_element_size,
        .packed_weights = convolution_op->packed_weights,
        .weights_channel_stride = bias_element_size +
          (convolution_op->kernel_height * convolution_op->kernel_width << log2_filter_element_size),
        .output = output,
        .output_channel_stride = output_height * output_width << log2_output_element_size,
        .output_batch_stride = output_batch_stride << log2_output_element_size,
        .input_tuple_stride = convolution_op->ukernel.dwconv2d.input_width_tile << log2_input_element_size,
        .output_tuple_stride = convolution_op->ukernel.dwconv2d.output_width_tile << log2_output_element_size,
        .input_pixel_stride = input_width << log2_input_element_size,
        .output_pixel_stride = output_width << log2_output_element_size,
        .spchw_ukernel = convolution_op->ukernel.dwconv2d.spchw_function,
      };
      memcpy(&convolution_op->context.dwconv2d.params, spchw_params, sizeof(convolution_op->context.dwconv2d.params));

      convolution_op->compute.type = xnn_parallelization_type_2d;
      convolution_op->compute.task_2d = (pthreadpool_task_2d_t) xnn_compute_dwconv2d_spchw;
      convolution_op->compute.range[0] = batch_size;
      convolution_op->compute.range[1] = groups;
      convolution_op->state = xnn_run_state_ready;

      return xnn_status_success;
    }
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_setup_convolution2d_nchw_f32(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_batch_stride,
    size_t output_batch_stride,
    size_t input_height,
    size_t input_width,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (convolution_op->type != xnn_operator_type_convolution_nchw_f32) {
    xnn_log_error("failed to setup Convolution (NCHW, F32) operator: operator type mismatch");
    return xnn_status_invalid_parameter;
  }

  return setup_convolution2d_nchw(
    convolution_op,
    batch_size, input_batch_stride, output_batch_stride,
    input_height, input_width,
    input, output,
    2 /* log2(sizeof(input element)) = log2(sizeof(float)) */,
    2 /* log2(sizeof(filter element)) = log2(sizeof(float)) */,
    sizeof(float) /* sizeof(bias element) */,
    2 /* log2(sizeof(output element)) = log2(sizeof(float)) */,
    &convolution_op->f32_minmax_params,
    &convolution_op->f32_spchw_params,
    pthreadpool_get_threads_count(threadpool));
}
