// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
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

#include <fp16.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/compute.h>
#include <xnnpack/indirection.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>
#include <xnnpack/pack.h>
#include <xnnpack/params.h>
#include <xnnpack/params-init.h>


static inline size_t compute_output_dimension(
    size_t padded_input_dimension,
    size_t kernel_dimension,
    size_t dilation_dimension,
    size_t subsampling_dimension)
{
  const size_t effective_kernel_dimension = (kernel_dimension - 1) * dilation_dimension + 1;
  return doz(padded_input_dimension, effective_kernel_dimension) / subsampling_dimension + 1;
}

static inline size_t compute_output_dimension_with_tf_same_padding(
    size_t input_dimension,
    size_t subsampling_dimension)
{
  return divide_round_up(input_dimension, subsampling_dimension);
}

static inline const struct dwconv_parameters* find_dwconv_ukernel(
    size_t kernel_size,
    const struct dwconv_parameters* ukernel,
    size_t num_ukernels)
{
  while (num_ukernels-- != 0) {
    if (ukernel->primary_tile == kernel_size) {
      return ukernel;
    }
    ukernel++;
  }
  return NULL;
}

static enum xnn_status create_convolution2d_nhwc(
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
    size_t input_channel_stride,
    size_t output_channel_stride,
    const void* kernel,
    const void* bias,
    uint32_t flags,
    uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size,
    uint32_t bias_element_size,
    xnn_pack_vmulcaddc_w_function pack_vmulcaddc_w,
    xnn_pack_dwconv_hwg_w_function pack_dwconv_hwg_w,
    xnn_pack_dwconv_ghw_w_function pack_dwconv_ghw_w,
    xnn_pack_gemm_goi_w_function pack_gemm_goi_w,
    xnn_pack_conv_kgo_w_function pack_conv_kgo_w,
    xnn_pack_conv_goki_w_function pack_conv_goki_w,
    const void* packing_params,
    int input_padding_byte,
    int packed_weights_padding_byte,
    size_t extra_weights_bytes,
    xnn_init_qc8_scale_params_fn init_scale_params,
    const float* scale_params,
    const void* gemm_params,
    size_t gemm_params_size,
    const void* dwconv_params,
    size_t dwconv_params_size,
    const void* vmulcaddc_params,
    size_t vmulcaddc_params_size,
    const struct gemm_parameters* gemm_parameters,
    const struct dwconv_parameters* dwconv_ukernel,
    const struct vmulcaddc_parameters* vmulcaddc_parameters,
    bool linear_activation,
    bool relu_activation,
    uint32_t datatype_init_flags,
    enum xnn_operator_type operator_type,
    xnn_operator_t* convolution_op_out)
{
  xnn_operator_t convolution_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error(
      "failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_unsupported_hardware;

  if ((xnn_params.init_flags & datatype_init_flags) != datatype_init_flags) {
    xnn_log_error(
      "failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (kernel_width == 0 || kernel_height == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " kernel: kernel dimensions must be non-zero",
      xnn_operator_type_to_string(operator_type), kernel_width, kernel_height);
    goto error;
  }

  if (subsampling_width == 0 || subsampling_height == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " subsampling: subsampling dimensions must be non-zero",
      xnn_operator_type_to_string(operator_type), subsampling_width, subsampling_height);
    goto error;
  }

  if (dilation_width == 0 || dilation_height == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " dilation: dilation dimensions must be non-zero",
      xnn_operator_type_to_string(operator_type), dilation_width, dilation_height);
    goto error;
  }

  if (groups == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 " groups: number of groups must be non-zero",
      xnn_operator_type_to_string(operator_type), groups);
    goto error;
  }

  if (group_input_channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu input channels per group: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), group_input_channels);
    goto error;
  }

  if (group_output_channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu output channels per group: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), group_output_channels);
    goto error;
  }

  const size_t input_channels = groups * group_input_channels;
  if (input_channel_stride < input_channels) {
    xnn_log_error(
      "failed to create %s operator with input channel stride of %zu: "
      "stride must be at least as large as the number of input channels (%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(operator_type),
      input_channel_stride, groups, group_input_channels);
    goto error;
  }

  const size_t output_channels = groups * group_output_channels;
  if (output_channel_stride < output_channels) {
    xnn_log_error(
      "failed to create %s operator with output channel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(operator_type),
      output_channel_stride, groups, group_output_channels);
    goto error;
  }

  if ((flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) != 0 && group_input_channels != 1) {
    xnn_log_error(
      "failed to create depthwise %s operator with %zu input channels per group: "
      "depthwise convolution must have exactly 1 input channel per group",
      xnn_operator_type_to_string(operator_type), group_input_channels);
    goto error;
  }

  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  if ((flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0) {
    if (any_padding) {
      xnn_log_error(
        "failed to create %s operator with %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding: "
        "TensorFlow SAME padding can't be combined with explicit padding specification",
        xnn_operator_type_to_string(operator_type),
        input_padding_top, input_padding_left, input_padding_bottom, input_padding_right);
      goto error;
    }
  }

  status = xnn_status_out_of_memory;

  convolution_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (convolution_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  const size_t kernel_size = kernel_height * kernel_width;

  enum xnn_ukernel_type ukernel_type = xnn_ukernel_type_default;
  const bool unit_subsampling = (subsampling_width | subsampling_height) == 1;
  if (group_input_channels == 1 && group_output_channels == 1 && kernel_size == 1 && unit_subsampling && !any_padding && vmulcaddc_parameters != NULL) {
    ukernel_type = xnn_ukernel_type_vmulcaddc;
  } else if (group_input_channels == 1 && group_output_channels == 1 && dwconv_ukernel != NULL)
  {
    ukernel_type = xnn_ukernel_type_dwconv;
  } else if (kernel_size == 1 && unit_subsampling && !any_padding) {
    ukernel_type = xnn_ukernel_type_gemm;
  } else {
    ukernel_type = xnn_ukernel_type_igemm;
  }
  assert(ukernel_type != xnn_ukernel_type_default);

  size_t zero_size = 0;
  switch (ukernel_type) {
    case xnn_ukernel_type_vmulcaddc:
    {
      assert(vmulcaddc_parameters != NULL);
      assert(vmulcaddc_params != NULL);

      const size_t c_stride = round_up_po2(groups, vmulcaddc_parameters->channel_tile);
      const size_t packed_weights_size = ((UINT32_C(1) << log2_filter_element_size) + bias_element_size) * c_stride;
      convolution_op->packed_weights = xnn_allocate_simd_memory(packed_weights_size);
      if (convolution_op->packed_weights == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator packed weights",
          packed_weights_size, xnn_operator_type_to_string(operator_type));
        goto error;
      }

      pack_vmulcaddc_w(
        groups, vmulcaddc_parameters->channel_tile,
        kernel, bias, convolution_op->packed_weights, packing_params);

      memcpy(&convolution_op->params, vmulcaddc_params, vmulcaddc_params_size);

      convolution_op->ukernel.vmulcaddc = (struct xnn_ukernel_vmulcaddc) {
        .function = vmulcaddc_parameters->ukernel,
        .mr = vmulcaddc_parameters->row_tile,
      };
      break;
    }
    case xnn_ukernel_type_dwconv:
    {
      assert(dwconv_ukernel != NULL);
      assert(dwconv_ukernel->primary_tile == kernel_size);

      const size_t c_stride = round_up_po2(groups, dwconv_ukernel->channel_tile);
      const size_t packed_weights_size = ((kernel_size << log2_filter_element_size) + bias_element_size + extra_weights_bytes) * c_stride;
      convolution_op->packed_weights = xnn_allocate_simd_memory(packed_weights_size);
      if (convolution_op->packed_weights == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator packed weights",
          packed_weights_size, xnn_operator_type_to_string(operator_type));
        goto error;
      }
      memset(convolution_op->packed_weights, packed_weights_padding_byte, packed_weights_size);
      memcpy(&convolution_op->params, dwconv_params, dwconv_params_size);

      if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
        pack_dwconv_hwg_w(
          kernel_height, kernel_width,
          groups, dwconv_ukernel->channel_tile,
          kernel, bias, convolution_op->packed_weights,
          dwconv_ukernel->channel_tile * extra_weights_bytes,
          packing_params);
      } else {
        pack_dwconv_ghw_w(
          kernel_height, kernel_width,
          groups, dwconv_ukernel->channel_tile,
          kernel, bias, convolution_op->packed_weights,
          dwconv_ukernel->channel_tile * extra_weights_bytes,
          packing_params);
      }

      if (scale_params != NULL) {
        assert(init_scale_params != NULL);

        init_scale_params(
          groups, dwconv_ukernel->channel_tile,
          dwconv_ukernel->channel_tile * ((kernel_size << log2_filter_element_size) + bias_element_size + extra_weights_bytes),
          scale_params,
          (void*) ((uintptr_t) convolution_op->packed_weights + dwconv_ukernel->channel_tile * ((kernel_size << log2_filter_element_size) + bias_element_size)));
      }

      const union dwconv_fused_ukernels* ukernels = &dwconv_ukernel->minmax;
      if (linear_activation && dwconv_ukernel->linear.unipass != NULL) {
        ukernels = &dwconv_ukernel->linear;
      }
      convolution_op->ukernel.dwconv = (struct xnn_ukernel_dwconv) {
        .unipass_function = ukernels->unipass,
        .primary_tile = dwconv_ukernel->primary_tile,
        .incremental_tile = dwconv_ukernel->incremental_tile,
      };

      zero_size = XNN_EXTRA_BYTES + (c_stride << log2_input_element_size);
      break;
    }
    case xnn_ukernel_type_gemm:
    case xnn_ukernel_type_igemm:
    {
      const uint32_t nr = gemm_parameters->nr;
      const uint32_t kr = UINT32_C(1) << gemm_parameters->log2_kr;
      const uint32_t sr = UINT32_C(1) << gemm_parameters->log2_sr;
      const size_t n_stride = round_up(group_output_channels, nr);
      const size_t k_stride = round_up_po2(group_input_channels, kr);

      const size_t packed_group_weights_size = ((kernel_size * k_stride << log2_filter_element_size) + bias_element_size + extra_weights_bytes) * n_stride;
      convolution_op->packed_weights = xnn_allocate_simd_memory(packed_group_weights_size * groups);
      if (convolution_op->packed_weights == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator packed weights",
          packed_group_weights_size * groups, xnn_operator_type_to_string(operator_type));
        goto error;
      }
      memset(convolution_op->packed_weights, packed_weights_padding_byte, packed_group_weights_size * groups);
      memcpy(&convolution_op->params, gemm_params, gemm_params_size);

      const struct gemm_fused_ukernels* gemm_ukernels = &gemm_parameters->minmax;
      if (linear_activation && gemm_parameters->linear.gemm.function[XNN_UARCH_DEFAULT] != NULL) {
        gemm_ukernels = &gemm_parameters->linear;
      } else if (relu_activation && gemm_parameters->relu.gemm.function[XNN_UARCH_DEFAULT] != NULL) {
        gemm_ukernels = &gemm_parameters->relu;
      }
      switch (ukernel_type) {
        case xnn_ukernel_type_gemm:
          pack_gemm_goi_w(
              groups, group_output_channels, group_input_channels,
              nr, kr, sr,
              kernel, bias, convolution_op->packed_weights, gemm_parameters->nr * extra_weights_bytes, packing_params);
          convolution_op->ukernel.gemm = (struct xnn_ukernel_gemm) {
            .mr = gemm_parameters->mr,
            .nr = nr,
            .kr = kr,
            .general_case = gemm_ukernels->gemm,
            .mr1_case = gemm_ukernels->gemm1,
          };
          break;
        case xnn_ukernel_type_igemm:
          if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
            pack_conv_kgo_w(
              groups, group_output_channels, kernel_size,
              nr, kr,
              kernel, bias, convolution_op->packed_weights, gemm_parameters->nr * extra_weights_bytes, packing_params);
          } else {
            pack_conv_goki_w(
              groups, group_output_channels, kernel_size, group_input_channels,
              nr, kr, sr,
              kernel, bias, convolution_op->packed_weights, gemm_parameters->nr * extra_weights_bytes, packing_params);
          }
          convolution_op->ukernel.igemm = (struct xnn_ukernel_igemm) {
            .mr = gemm_parameters->mr,
            .nr = nr,
            .kr = kr,
            .general_case = gemm_ukernels->igemm,
            .mr1_case = gemm_ukernels->igemm1,
          };
          break;
        default:
          XNN_UNREACHABLE;
      }

      if (scale_params != NULL) {
        assert(init_scale_params != NULL);

        void* group_weights = (void*)
          ((uintptr_t) convolution_op->packed_weights + gemm_parameters->nr * ((kernel_size * k_stride << log2_filter_element_size) + bias_element_size));
        const size_t weights_stride = (kernel_size * k_stride << log2_filter_element_size) + bias_element_size + extra_weights_bytes;
        for (uint32_t group = 0; group < groups; group++) {
          init_scale_params(
            group_output_channels, gemm_parameters->nr,
            gemm_parameters->nr * weights_stride,
            scale_params, group_weights);
          scale_params += group_output_channels;
          group_weights = (void*) ((uintptr_t) group_weights + n_stride * weights_stride);
        }
      }

      zero_size = XNN_EXTRA_BYTES + (k_stride << log2_input_element_size);
      break;
    }
    default:
      XNN_UNREACHABLE;
  }

  const bool tf_same_padding = (flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0 && kernel_size != 1;
  if (any_padding || tf_same_padding) {
    convolution_op->zero_buffer = xnn_allocate_simd_memory(zero_size);
    if (convolution_op->zero_buffer == NULL) {
      xnn_log_error(
        "failed to allocate %zu bytes for %s operator zero padding",
        zero_size, xnn_operator_type_to_string(operator_type));
      goto error;
    }
    memset(convolution_op->zero_buffer, input_padding_byte, zero_size);
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
  convolution_op->input_pixel_stride = input_channel_stride;
  convolution_op->output_pixel_stride = output_channel_stride;

  convolution_op->type = operator_type;
  convolution_op->ukernel.type = ukernel_type;
  convolution_op->flags = flags & ~XNN_FLAG_TENSORFLOW_SAME_PADDING;
  if (tf_same_padding) {
    convolution_op->flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }

  convolution_op->state = xnn_run_state_invalid;

  *convolution_op_out = convolution_op;
  return xnn_status_success;

error:
  xnn_delete_operator(convolution_op);
  return status;
}

enum xnn_status xnn_create_convolution2d_nhwc_qu8(
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
    size_t input_channel_stride,
    size_t output_channel_stride,
    uint8_t input_zero_point,
    float input_scale,
    uint8_t kernel_zero_point,
    float kernel_scale,
    const uint8_t* kernel,
    const int32_t* bias,
    uint8_t output_zero_point,
    float output_scale,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    xnn_operator_t* convolution_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qu8), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g kernel scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qu8), kernel_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qu8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: range min must be below range max",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qu8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float requantization_scale = input_scale * kernel_scale / output_scale;
  if (requantization_scale >= 1.0f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "requantization scale %.7g is greater or equal to 1.0",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qu8),
      input_scale, kernel_scale, output_scale, requantization_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_qu8_packing_params packing_params = {
    .input_zero_point = input_zero_point,
    .kernel_zero_point = kernel_zero_point,
  };


  union xnn_qu8_conv_minmax_params gemm_params;
  if XNN_LIKELY(xnn_params.qu8.gemm.init.qu8 != NULL) {
    xnn_params.qu8.gemm.init.qu8(&gemm_params,
      kernel_zero_point, requantization_scale, output_zero_point, output_min, output_max);
  }

  union xnn_qu8_conv_minmax_params dwconv_params;
  const struct dwconv_parameters* dwconv_ukernel =
    find_dwconv_ukernel(kernel_height * kernel_width, xnn_params.qu8.dwconv, XNN_MAX_QU8_DWCONV_UKERNELS);
  if XNN_LIKELY(dwconv_ukernel != NULL) {
    dwconv_ukernel->init.qu8(&dwconv_params,
      kernel_zero_point, requantization_scale, output_zero_point, output_min, output_max);
  }

  return create_convolution2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    kernel_height, kernel_width,
    subsampling_height, subsampling_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_channel_stride, output_channel_stride,
    kernel, bias, flags,
    0 /* log2(sizeof(input element)) = log2(sizeof(uint8_t)) */,
    0 /* log2(sizeof(filter element)) = log2(sizeof(uint8_t)) */,
    sizeof(int32_t) /* sizeof(bias element) */,
    (xnn_pack_vmulcaddc_w_function) NULL,
    (xnn_pack_dwconv_hwg_w_function) xnn_pack_qu8_dwconv_hwg_w,
    (xnn_pack_dwconv_ghw_w_function) xnn_pack_qu8_dwconv_ghw_w,
    (xnn_pack_gemm_goi_w_function) xnn_pack_qu8_gemm_goi_w,
    (xnn_pack_conv_kgo_w_function) xnn_pack_qu8_conv_kgo_w,
    (xnn_pack_conv_goki_w_function) xnn_pack_qu8_conv_goki_w,
    &packing_params, input_zero_point /* input padding byte */, kernel_zero_point /* packed weights padding byte */,
    0 /* extra weights bytes */, NULL /* init scale params fn */, NULL /* scale params */,
    &gemm_params, sizeof(gemm_params),
    &dwconv_params, sizeof(dwconv_params),
    NULL /* vmulcaddc params */, 0,
    &xnn_params.qu8.gemm, dwconv_ukernel, NULL /* vmulcaddc parameters */,
    false /* linear activation */, false /* relu activation */, XNN_INIT_FLAG_QU8,
    xnn_operator_type_convolution_nhwc_qu8,
    convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_qs8(
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
    size_t input_channel_stride,
    size_t output_channel_stride,
    int8_t input_zero_point,
    float input_scale,
    float kernel_scale,
    const int8_t* kernel,
    const int32_t* bias,
    int8_t output_zero_point,
    float output_scale,
    int8_t output_min,
    int8_t output_max,
    uint32_t flags,
    xnn_operator_t* convolution_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qs8), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g kernel scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qs8), kernel_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qs8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: range min must be below range max",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qs8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float requantization_scale = input_scale * kernel_scale / output_scale;
  if (requantization_scale >= 1.0f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "requantization scale %.7g is greater or equal to 1.0",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qs8),
      input_scale, kernel_scale, output_scale, requantization_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_qs8_packing_params packing_params = { .input_zero_point = input_zero_point, };

  union xnn_qs8_conv_minmax_params gemm_params;
  if XNN_LIKELY(xnn_params.qs8.gemm.init.qs8 != NULL) {
    xnn_params.qs8.gemm.init.qs8(&gemm_params,
      requantization_scale, output_zero_point, output_min, output_max);
  }

  union xnn_qs8_conv_minmax_params dwconv_params;
  const struct dwconv_parameters* dwconv_ukernel =
    find_dwconv_ukernel(kernel_height * kernel_width, xnn_params.qs8.dwconv, XNN_MAX_QS8_DWCONV_UKERNELS);
  if XNN_LIKELY(dwconv_ukernel != NULL) {
    dwconv_ukernel->init.qs8(&dwconv_params,
      requantization_scale, output_zero_point, output_min, output_max);
  }

  return create_convolution2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    kernel_height, kernel_width,
    subsampling_height, subsampling_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_channel_stride, output_channel_stride,
    kernel, bias, flags,
    0 /* log2(sizeof(input element)) = log2(sizeof(int8_t)) */,
    0 /* log2(sizeof(filter element)) = log2(sizeof(int8_t)) */,
    sizeof(int32_t) /* sizeof(bias element) */,
    (xnn_pack_vmulcaddc_w_function) NULL,
    (xnn_pack_dwconv_hwg_w_function) xnn_pack_qs8_dwconv_hwg_w,
    (xnn_pack_dwconv_ghw_w_function) xnn_pack_qs8_dwconv_ghw_w,
    (xnn_pack_gemm_goi_w_function) xnn_pack_qs8_gemm_goi_w,
    (xnn_pack_conv_kgo_w_function) xnn_pack_qs8_conv_kgo_w,
    (xnn_pack_conv_goki_w_function) xnn_pack_qs8_conv_goki_w,
    &packing_params, input_zero_point /* input padding byte */, 0 /* packed weights padding byte */,
    0 /* extra weights bytes */, NULL /* init scale params fn */, NULL /* scale params */,
    &gemm_params, sizeof(gemm_params),
    &dwconv_params, sizeof(dwconv_params),
    NULL /* vmulcaddc params */, 0,
    &xnn_params.qs8.gemm, dwconv_ukernel, NULL /* vmulcaddc parameters */,
    false /* linear activation */, false /* relu activation */, XNN_INIT_FLAG_QS8,
    xnn_operator_type_convolution_nhwc_qs8,
    convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_qc8(
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
    size_t input_channel_stride,
    size_t output_channel_stride,
    int8_t input_zero_point,
    float input_scale,
    const float* kernel_scale,
    const int8_t* kernel,
    const int32_t* bias,
    int8_t output_zero_point,
    float output_scale,
    int8_t output_min,
    int8_t output_max,
    uint32_t flags,
    xnn_operator_t* convolution_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8), input_scale);
    return xnn_status_invalid_parameter;
  }

  for (size_t output_channel = 0; output_channel < groups * group_output_channels; output_channel++) {
    if (kernel_scale[output_channel] <= 0.0f || !isnormal(kernel_scale[output_channel])) {
      xnn_log_error(
        "failed to create %s operator with %.7g kernel scale in output channel #%zu: "
        "scale must be finite, normalized, and positive",
        xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8), kernel_scale[output_channel],
        output_channel);
      return xnn_status_invalid_parameter;
    }
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: range min must be below range max",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  float* requantization_scale = XNN_SIMD_ALLOCA(groups * group_output_channels * sizeof(float));
  for (size_t output_channel = 0; output_channel < groups * group_output_channels; output_channel++) {
    requantization_scale[output_channel] = input_scale * kernel_scale[output_channel] / output_scale;
    if (requantization_scale[output_channel] >= 1.0f) {
      xnn_log_error(
        "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale in output channel #%zu: "
        "requantization scale %.7g is greater or equal to 1.0",
        xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8),
        input_scale, kernel_scale[output_channel], output_scale,
        output_channel, requantization_scale[output_channel]);
      return xnn_status_unsupported_parameter;
    }
  }

  const struct xnn_qs8_packing_params packing_params = { .input_zero_point = input_zero_point, };

  union xnn_qs8_minmax_params gemm_params;
  if XNN_LIKELY(xnn_params.qc8.gemm.init.qc8 != NULL) {
    xnn_params.qc8.gemm.init.qc8(&gemm_params,
      output_zero_point, output_min, output_max);
  }

  union xnn_qs8_minmax_params dwconv_params;
  const struct dwconv_parameters* dwconv_ukernel =
    find_dwconv_ukernel(kernel_height * kernel_width, xnn_params.qc8.dwconv, XNN_MAX_QC8_DWCONV_UKERNELS);
  if XNN_LIKELY(dwconv_ukernel != NULL) {
    dwconv_ukernel->init.qc8(&dwconv_params,
      output_zero_point, output_min, output_max);
  }

  return create_convolution2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    kernel_height, kernel_width,
    subsampling_height, subsampling_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_channel_stride, output_channel_stride,
    kernel, bias, flags,
    0 /* log2(sizeof(input element)) = log2(sizeof(int8_t)) */,
    0 /* log2(sizeof(filter element)) = log2(sizeof(int8_t)) */,
    sizeof(int32_t) /* sizeof(bias element) */,
    (xnn_pack_vmulcaddc_w_function) NULL,
    (xnn_pack_dwconv_hwg_w_function) xnn_pack_qs8_dwconv_hwg_w,
    (xnn_pack_dwconv_ghw_w_function) xnn_pack_qs8_dwconv_ghw_w,
    (xnn_pack_gemm_goi_w_function) xnn_pack_qs8_gemm_goi_w,
    (xnn_pack_conv_kgo_w_function) xnn_pack_qs8_conv_kgo_w,
    (xnn_pack_conv_goki_w_function) xnn_pack_qs8_conv_goki_w,
    &packing_params, input_zero_point /* input padding byte */, 0 /* packed weights padding byte */,
    sizeof(float) /* extra weights bytes */, xnn_init_qc8_scale_fp32_params, requantization_scale,
    &gemm_params, sizeof(gemm_params),
    &dwconv_params, sizeof(dwconv_params),
    NULL /* vmulcaddc params */, 0,
    &xnn_params.qc8.gemm, dwconv_ukernel, NULL /* vmulcaddc parameters */,
    false /* linear activation */, false /* relu activation */, XNN_INIT_FLAG_QC8,
    xnn_operator_type_convolution_nhwc_qc8,
    convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_f16(
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
    size_t input_channel_stride,
    size_t output_channel_stride,
    const void* kernel,
    const void* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* convolution_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f16));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f16));
    return xnn_status_invalid_parameter;
  }

  const uint16_t fp16_output_min = fp16_ieee_from_fp32_value(output_min);
  const uint16_t fp16_output_max = fp16_ieee_from_fp32_value(output_max);
  const float rounded_output_min = fp16_ieee_to_fp32_value(fp16_output_min);
  const float rounded_output_max = fp16_ieee_to_fp32_value(fp16_output_max);
  if (rounded_output_min >= rounded_output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f16), rounded_output_min, rounded_output_max);
    return xnn_status_invalid_parameter;
  }

  struct xnn_f16_scaleminmax_params gemm_params;
  if XNN_LIKELY(xnn_params.f16.gemm.init.f16 != NULL) {
    xnn_params.f16.gemm.init.f16(&gemm_params,
      UINT16_C(0x3C00) /* 1.0 */, fp16_output_min, fp16_output_max);
  }

  struct xnn_f16_minmax_params dwconv_params;
  const struct dwconv_parameters* dwconv_ukernel =
    find_dwconv_ukernel(kernel_height * kernel_width, xnn_params.f16.dwconv, XNN_MAX_F16_DWCONV_UKERNELS);
  if XNN_LIKELY(dwconv_ukernel != NULL) {
    dwconv_ukernel->init.f16(&dwconv_params, fp16_output_min, fp16_output_max);
  }

  struct xnn_f16_minmax_params vmulcaddc_params;
  if XNN_LIKELY(xnn_params.f16.vmulcaddc.init.f16 != NULL) {
    xnn_params.f16.vmulcaddc.init.f16(&vmulcaddc_params, fp16_output_min, fp16_output_max);
  }

  return create_convolution2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    kernel_height, kernel_width,
    subsampling_height, subsampling_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_channel_stride, output_channel_stride,
    kernel, bias, flags,
    1 /* log2(sizeof(input element)) = log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(filter element)) = log2(sizeof(uint16_t)) */,
    sizeof(uint16_t) /* sizeof(bias element) */,
    (xnn_pack_vmulcaddc_w_function) xnn_pack_f16_vmulcaddc_w,
    (xnn_pack_dwconv_hwg_w_function) xnn_pack_f16_dwconv_hwg_w,
    (xnn_pack_dwconv_ghw_w_function) xnn_pack_f16_dwconv_ghw_w,
    (xnn_pack_gemm_goi_w_function) xnn_pack_f16_gemm_goi_w,
    (xnn_pack_conv_kgo_w_function) xnn_pack_f16_conv_kgo_w,
    (xnn_pack_conv_goki_w_function) xnn_pack_f16_conv_goki_w,
    NULL /* packing params */, 0 /* input padding byte */, 0 /* packed weights padding byte */,
    0 /* extra weights bytes */, NULL /* init scale params fn */, NULL /* scale params */,
    &gemm_params, sizeof(gemm_params),
    &dwconv_params, sizeof(dwconv_params),
    &vmulcaddc_params, sizeof(vmulcaddc_params),
    &xnn_params.f16.gemm, dwconv_ukernel, &xnn_params.f16.vmulcaddc,
    false /* linear activation */, false /* relu activation */, XNN_INIT_FLAG_F16,
    xnn_operator_type_convolution_nhwc_f16,
    convolution_op_out);
}

enum xnn_status xnn_create_convolution2d_nhwc_f32(
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
    size_t input_channel_stride,
    size_t output_channel_stride,
    const float* kernel,
    const float* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* convolution_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f32));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f32));
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f32), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const bool linear_activation = (output_max == INFINITY) && (output_min == -output_max);
  const bool relu_activation = (output_max == INFINITY) && (output_min == 0.0f);

  union xnn_f32_minmax_params gemm_params;
  if XNN_LIKELY(xnn_params.f32.gemm.init.f32 != NULL) {
    xnn_params.f32.gemm.init.f32(&gemm_params, output_min, output_max);
  }

  union xnn_f32_minmax_params dwconv_params;
  const struct dwconv_parameters* dwconv_ukernel =
    find_dwconv_ukernel(kernel_height * kernel_width, xnn_params.f32.dwconv, XNN_MAX_F32_DWCONV_UKERNELS);
  if XNN_LIKELY(dwconv_ukernel != NULL) {
    dwconv_ukernel->init.f32(&dwconv_params, output_min, output_max);
  }

  union xnn_f32_minmax_params vmulcaddc_params;
  if XNN_LIKELY(xnn_params.f32.vmulcaddc.init.f32 != NULL) {
    xnn_params.f32.vmulcaddc.init.f32(&vmulcaddc_params, output_min, output_max);
  }

  return create_convolution2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    kernel_height, kernel_width,
    subsampling_height, subsampling_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_channel_stride, output_channel_stride,
    kernel, bias, flags,
    2 /* log2(sizeof(input element)) = log2(sizeof(float)) */,
    2 /* log2(sizeof(filter element)) = log2(sizeof(float)) */,
    sizeof(float) /* sizeof(bias element) */,
    (xnn_pack_vmulcaddc_w_function) xnn_pack_f32_vmulcaddc_w,
    (xnn_pack_dwconv_hwg_w_function) xnn_pack_f32_dwconv_hwg_w,
    (xnn_pack_dwconv_ghw_w_function) xnn_pack_f32_dwconv_ghw_w,
    (xnn_pack_gemm_goi_w_function) xnn_pack_f32_gemm_goi_w,
    (xnn_pack_conv_kgo_w_function) xnn_pack_f32_conv_kgo_w,
    (xnn_pack_conv_goki_w_function) xnn_pack_f32_conv_goki_w,
    NULL /* packing params */, 0 /* input padding byte */, 0 /* packed weights padding byte */,
    0 /* extra weights bytes */, NULL /* init scale params fn */, NULL /* scale params */,
    &gemm_params, sizeof(gemm_params),
    &dwconv_params, sizeof(dwconv_params),
    &vmulcaddc_params, sizeof(vmulcaddc_params),
    &xnn_params.f32.gemm, dwconv_ukernel, &xnn_params.f32.vmulcaddc,
    linear_activation, relu_activation, XNN_INIT_FLAG_F32,
    xnn_operator_type_convolution_nhwc_f32,
    convolution_op_out);
}

static enum xnn_status setup_convolution2d_nhwc(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
  uint32_t datatype_init_flags,
  uint32_t log2_input_element_size,
  uint32_t log2_filter_element_size,
  uint32_t extra_weights_elements_size,
  uint32_t log2_output_element_size,
  size_t num_threads)
{
  convolution_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_uninitialized;
  }

  if ((xnn_params.init_flags & datatype_init_flags) != datatype_init_flags) {
    xnn_log_error(
      "failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_unsupported_hardware;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
      "failed to setup %s operator with %zux%zu input: input dimensions must be non-zero",
      xnn_operator_type_to_string(convolution_op->type), input_width, input_height);
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

  if (convolution_op->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) {
    convolution_op->output_height = compute_output_dimension_with_tf_same_padding(
        input_height, convolution_op->stride_height);
    convolution_op->output_width = compute_output_dimension_with_tf_same_padding(
        input_width, convolution_op->stride_width);

    const uint32_t effective_kernel_height = (convolution_op->kernel_height - 1) * convolution_op->dilation_height + 1;
    const uint32_t effective_kernel_width = (convolution_op->kernel_width - 1) * convolution_op->dilation_width + 1;
    const size_t total_padding_height =
      (convolution_op->output_height - 1) * convolution_op->stride_height + effective_kernel_height - input_height;
    const size_t total_padding_width =
      (convolution_op->output_width - 1) * convolution_op->stride_width + effective_kernel_width - input_width;
    convolution_op->padding_top = total_padding_height / 2;
    convolution_op->padding_left = total_padding_width / 2;
    convolution_op->padding_bottom = total_padding_height - convolution_op->padding_top;
    convolution_op->padding_right = total_padding_width - convolution_op->padding_left;
  } else {
    convolution_op->output_height = compute_output_dimension(
        convolution_op->padding_top + input_height + convolution_op->padding_bottom,
        convolution_op->kernel_height,
        convolution_op->dilation_height,
        convolution_op->stride_height);
    convolution_op->output_width = compute_output_dimension(
        convolution_op->padding_left + input_width + convolution_op->padding_right,
        convolution_op->kernel_width,
        convolution_op->dilation_width,
        convolution_op->stride_width);
  }
  convolution_op->output = output;

  switch (convolution_op->ukernel.type) {
    case xnn_ukernel_type_gemm:
    {
      // Convolution maps directly to GEMM and doesn't use indirection buffer.

      const size_t output_height = convolution_op->output_height;
      const size_t output_width = convolution_op->output_width;
      const size_t output_size = output_height * output_width;
      const size_t batch_output_size = batch_size * output_size;

      const size_t groups = convolution_op->groups;
      const size_t group_input_channels = convolution_op->group_input_channels;
      const size_t w_stride = extra_weights_elements_size +
        (round_up_po2(group_input_channels, convolution_op->ukernel.gemm.kr) << log2_filter_element_size);
      const size_t group_output_channels = convolution_op->group_output_channels;

      uint32_t mr = convolution_op->ukernel.gemm.mr;
      const uint32_t nr = convolution_op->ukernel.gemm.nr;
      struct xnn_hmp_gemm_ukernel gemm_ukernel = convolution_op->ukernel.gemm.general_case;
      if (batch_output_size == 1 && convolution_op->ukernel.gemm.mr1_case.function[XNN_UARCH_DEFAULT] != NULL) {
        mr = 1;
        gemm_ukernel = convolution_op->ukernel.gemm.mr1_case;
      }

      convolution_op->context.gemm = (struct gemm_context) {
          .k_scaled = group_input_channels << log2_input_element_size,
          .a = input,
          .a_stride = convolution_op->input_pixel_stride << log2_input_element_size,
          .packed_w = convolution_op->packed_weights,
          .w_stride = w_stride,
          .wg_stride = w_stride * round_up(group_output_channels, nr),
          .c = output,
          .cm_stride = convolution_op->output_pixel_stride << log2_output_element_size,
          .cn_stride = nr << log2_output_element_size,
          .cg_stride = group_output_channels << log2_output_element_size,
          .log2_csize = log2_output_element_size,
          .ukernel = gemm_ukernel,
      };
      memcpy(&convolution_op->context.gemm.params, &convolution_op->params, sizeof(convolution_op->context.gemm.params));

      size_t nc = group_output_channels;
      if (num_threads > 1) {
        const size_t num_other_tiles = groups * divide_round_up(batch_output_size, mr);
        const size_t target_tiles_per_thread = 5;
        const size_t max_nc = divide_round_up(group_output_channels * num_other_tiles, num_threads * target_tiles_per_thread);
        if (max_nc < nc) {
          nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
        }
      }
      if (groups == 1) {
        #if XNN_MAX_UARCH_TYPES > 1
          if (xnn_is_hmp_gemm_ukernel(gemm_ukernel)) {
            convolution_op->compute.type = xnn_parallelization_type_2d_tile_2d_with_uarch;
            convolution_op->compute.task_2d_tile_2d_with_id = (pthreadpool_task_2d_tile_2d_with_id_t) xnn_compute_hmp_gemm;
          } else {
            convolution_op->compute.type = xnn_parallelization_type_2d_tile_2d;
            convolution_op->compute.task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_gemm;
          }
        #else
          convolution_op->compute.type = xnn_parallelization_type_2d_tile_2d;
          convolution_op->compute.task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_gemm;
        #endif
        convolution_op->compute.range[0] = batch_output_size;
        convolution_op->compute.range[1] = group_output_channels;
        convolution_op->compute.tile[0] = mr;
        convolution_op->compute.tile[1] = nc;
      } else {
        #if XNN_MAX_UARCH_TYPES > 1
          if (xnn_is_hmp_gemm_ukernel(gemm_ukernel)) {
            convolution_op->compute.type = xnn_parallelization_type_3d_tile_2d_with_uarch;
            convolution_op->compute.task_3d_tile_2d_with_id = (pthreadpool_task_3d_tile_2d_with_id_t) xnn_compute_hmp_grouped_gemm;
          } else {
            convolution_op->compute.type = xnn_parallelization_type_3d_tile_2d;
            convolution_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_gemm;
          }
        #else
          convolution_op->compute.type = xnn_parallelization_type_3d_tile_2d;
          convolution_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_gemm;
        #endif
        convolution_op->compute.range[0] = groups;
        convolution_op->compute.range[1] = batch_output_size;
        convolution_op->compute.range[2] = group_output_channels;
        convolution_op->compute.tile[0] = mr;
        convolution_op->compute.tile[1] = nc;
      }
      convolution_op->state = xnn_run_state_ready;

      return xnn_status_success;
    }
    case xnn_ukernel_type_igemm:
    {
      const size_t groups = convolution_op->groups;
      const size_t kernel_height = convolution_op->kernel_height;
      const size_t kernel_width = convolution_op->kernel_width;
      const size_t kernel_size = kernel_height * kernel_width;
      const size_t output_height = convolution_op->output_height;
      const size_t output_width = convolution_op->output_width;
      const size_t output_size = output_height * output_width;

      uint32_t mr = convolution_op->ukernel.igemm.mr;
      const uint32_t nr = convolution_op->ukernel.igemm.nr;
      struct xnn_hmp_igemm_ukernel igemm_ukernel = convolution_op->ukernel.igemm.general_case;
      if (output_size == 1 && convolution_op->ukernel.igemm.mr1_case.function[XNN_UARCH_DEFAULT] != NULL) {
        mr = 1;
        igemm_ukernel = convolution_op->ukernel.igemm.mr1_case;
      }

      const size_t tiled_output_size = round_up(output_size, mr);
      const size_t indirection_buffer_size = sizeof(void*) * kernel_size * tiled_output_size;

      if (input_height != convolution_op->last_input_height ||
          input_width != convolution_op->last_input_width)
      {
        const void** indirection_buffer = (const void**) xnn_reallocate_memory((void*) convolution_op->indirection_buffer, indirection_buffer_size);
        if (indirection_buffer == NULL) {
          xnn_log_error(
            "failed to allocate %zu bytes for %s operator indirection buffer",
            indirection_buffer_size, xnn_operator_type_to_string(convolution_op->type));
          return xnn_status_out_of_memory;
        }
        convolution_op->indirection_buffer = indirection_buffer;
        convolution_op->last_input = input;
        convolution_op->last_input_height = input_height;
        convolution_op->last_input_width = input_width;

        xnn_indirection_init_conv2d(convolution_op, mr, log2_input_element_size);
      }

      const size_t group_input_channels = convolution_op->group_input_channels;
      const size_t w_stride = extra_weights_elements_size +
        (round_up_po2(group_input_channels, convolution_op->ukernel.igemm.kr) * kernel_size << log2_filter_element_size);
      const size_t group_output_channels = convolution_op->group_output_channels;
      convolution_op->context.igemm = (struct igemm_context) {
          .ks = kernel_size,
          .ks_scaled = kernel_size * mr * sizeof(void*),
          .kc = group_input_channels << log2_input_element_size,
          .w_stride = w_stride,
          .indirect_a = convolution_op->indirection_buffer,
          .a_offset = (size_t) ((uintptr_t) input - (uintptr_t) convolution_op->last_input),
          .zero = convolution_op->zero_buffer,
          .packed_w = convolution_op->packed_weights,
          .c = convolution_op->output,
          .cm_stride = convolution_op->output_pixel_stride << log2_output_element_size,
          .cn_stride = nr << log2_output_element_size,
          .ga_stride = group_input_channels << log2_input_element_size,
          .gw_stride = w_stride * round_up(group_output_channels, nr),
          .gc_stride = group_output_channels << log2_output_element_size,
          .ba_stride = input_height * input_width * convolution_op->input_pixel_stride << log2_input_element_size,
          .bc_stride = output_size * convolution_op->output_pixel_stride << log2_output_element_size,
          .log2_csize = log2_output_element_size,
          .ukernel = igemm_ukernel,
      };
      memcpy(&convolution_op->context.igemm.params, &convolution_op->params, sizeof(convolution_op->context.igemm.params));

      size_t nc = group_output_channels;
      if (num_threads > 1) {
        const size_t num_other_tiles = groups * batch_size * divide_round_up(output_size, mr);
        const size_t target_tiles_per_thread = 5;
        const size_t max_nc = divide_round_up(group_output_channels * num_other_tiles, num_threads * target_tiles_per_thread);
        if (max_nc < nc) {
          nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
        }
      }
      if (groups == 1) {
        #if XNN_MAX_UARCH_TYPES > 1
          if (xnn_is_hmp_igemm_ukernel(igemm_ukernel)) {
            if (batch_size > 1) {
              convolution_op->compute.type = xnn_parallelization_type_3d_tile_2d_with_uarch;
              convolution_op->compute.task_3d_tile_2d_with_id = (pthreadpool_task_3d_tile_2d_with_id_t) xnn_compute_batch_hmp_igemm;
            } else {
              convolution_op->compute.type = xnn_parallelization_type_2d_tile_2d_with_uarch;
              convolution_op->compute.task_2d_tile_2d_with_id = (pthreadpool_task_2d_tile_2d_with_id_t) xnn_compute_hmp_igemm;
            }
          } else {
            if (batch_size > 1) {
              convolution_op->compute.type = xnn_parallelization_type_3d_tile_2d;
              convolution_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_batch_igemm;
            } else {
              convolution_op->compute.type = xnn_parallelization_type_2d_tile_2d;
              convolution_op->compute.task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_igemm;
            }
          }
        #else
          if (batch_size > 1) {
            convolution_op->compute.type = xnn_parallelization_type_3d_tile_2d;
            convolution_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_batch_igemm;
          } else {
            convolution_op->compute.type = xnn_parallelization_type_2d_tile_2d;
            convolution_op->compute.task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_igemm;
          }
        #endif
        if (batch_size > 1) {
          convolution_op->compute.range[0] = batch_size;
          convolution_op->compute.range[1] = output_size;
          convolution_op->compute.range[2] = group_output_channels;
        } else {
          convolution_op->compute.range[0] = output_size;
          convolution_op->compute.range[1] = group_output_channels;
        }
        convolution_op->compute.tile[0] = mr;
        convolution_op->compute.tile[1] = nc;
      } else {
        #if XNN_MAX_UARCH_TYPES > 1
          if (xnn_is_hmp_igemm_ukernel(igemm_ukernel)) {
            if (batch_size > 1) {
              convolution_op->compute.type = xnn_parallelization_type_4d_tile_2d_with_uarch;
              convolution_op->compute.task_4d_tile_2d_with_id = (pthreadpool_task_4d_tile_2d_with_id_t) xnn_compute_hmp_grouped_batch_igemm;
            } else {
              convolution_op->compute.type = xnn_parallelization_type_3d_tile_2d_with_uarch;
              convolution_op->compute.task_3d_tile_2d_with_id = (pthreadpool_task_3d_tile_2d_with_id_t) xnn_compute_hmp_grouped_igemm;
            }
          } else {
            if (batch_size > 1) {
              convolution_op->compute.type = xnn_parallelization_type_4d_tile_2d;
              convolution_op->compute.task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_grouped_batch_igemm;
            } else {
              convolution_op->compute.type = xnn_parallelization_type_3d_tile_2d;
              convolution_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_igemm;
            }
          }
        #else
          if (batch_size > 1) {
            convolution_op->compute.type = xnn_parallelization_type_4d_tile_2d;
            convolution_op->compute.task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_grouped_batch_igemm;
          } else {
            convolution_op->compute.type = xnn_parallelization_type_3d_tile_2d;
            convolution_op->compute.task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_igemm;
          }
        #endif
        if (batch_size > 1) {
          convolution_op->compute.range[0] = batch_size;
          convolution_op->compute.range[1] = groups;
          convolution_op->compute.range[2] = output_size;
          convolution_op->compute.range[3] = group_output_channels;
        } else {
          convolution_op->compute.range[0] = groups;
          convolution_op->compute.range[1] = output_size;
          convolution_op->compute.range[2] = group_output_channels;
        }
        convolution_op->compute.tile[0] = mr;
        convolution_op->compute.tile[1] = nc;
      }
      convolution_op->state = xnn_run_state_ready;

      return xnn_status_success;
    }
    case xnn_ukernel_type_dwconv:
    {
      const size_t kernel_height = convolution_op->kernel_height;
      const size_t kernel_width = convolution_op->kernel_width;
      const size_t kernel_size = kernel_height * kernel_width;
      const size_t output_height = convolution_op->output_height;
      const size_t output_width = convolution_op->output_width;
      const size_t step_width = convolution_op->dilation_width == 1 ? convolution_op->stride_width : kernel_width;
      const size_t step_height = kernel_size + (output_width - 1) * step_width * kernel_height;
      if (input_height != convolution_op->last_input_height || input_width != convolution_op->last_input_width) {
        const size_t indirection_buffer_size = sizeof(void*) * output_height * step_height;

        const void** indirection_buffer =
          (const void**) xnn_reallocate_memory(convolution_op->indirection_buffer, indirection_buffer_size);
        if (indirection_buffer == NULL) {
          xnn_log_error("failed to allocate %zu bytes for %s operator indirection buffer",
            indirection_buffer_size, xnn_operator_type_to_string(convolution_op->type));
          return xnn_status_out_of_memory;
        }
        convolution_op->indirection_buffer = indirection_buffer;

        xnn_indirection_init_dwconv2d(convolution_op, step_height, step_width, log2_input_element_size);

        convolution_op->last_input = input;
        convolution_op->last_input_height = input_height;
        convolution_op->last_input_width = input_width;
      }

      const size_t groups = convolution_op->groups;
      convolution_op->context.dwconv = (struct dwconv_context) {
          .indirect_input = convolution_op->indirection_buffer,
          .indirect_input_width_stride = kernel_height * step_width * sizeof(void*),
          .indirect_input_height_stride = step_height * sizeof(void*),
          .input_offset = (size_t) ((uintptr_t) input - (uintptr_t) convolution_op->last_input),
          .input_batch_stride = (input_height * input_width * convolution_op->input_pixel_stride) << log2_input_element_size,
          .packed_weights = convolution_op->packed_weights,
          .output = convolution_op->output,
          .output_batch_stride = (output_height * output_width * convolution_op->output_pixel_stride) << log2_output_element_size,
          .output_height_stride = (output_width * convolution_op->output_pixel_stride) << log2_output_element_size,
          .output_width = output_width,
          .groups = groups,
          .zero = convolution_op->zero_buffer,
          .output_increment = (convolution_op->output_pixel_stride - groups) << log2_output_element_size,
          .unipass_ukernel = convolution_op->ukernel.dwconv.unipass_function,
      };
      memcpy(&convolution_op->context.dwconv.params, &convolution_op->params, sizeof(convolution_op->context.dwconv.params));

      convolution_op->compute.type = xnn_parallelization_type_2d;
      convolution_op->compute.task_2d = (pthreadpool_task_2d_t) xnn_compute_dwconv_unipass;
      convolution_op->compute.range[0] = batch_size;
      convolution_op->compute.range[1] = output_height;
      convolution_op->state = xnn_run_state_ready;

      return xnn_status_success;
    }
    case xnn_ukernel_type_vmulcaddc:
    {
      const size_t batch_output_size = batch_size * convolution_op->output_height * convolution_op->output_width;

      convolution_op->context.vmulcaddc = (struct vmulcaddc_context) {
          .n = convolution_op->groups << log2_input_element_size,
          .x = input,
          .x_stride = convolution_op->input_pixel_stride << log2_input_element_size,
          .w = convolution_op->packed_weights,
          .y = output,
          .y_stride = convolution_op->output_pixel_stride << log2_output_element_size,
          .ukernel = convolution_op->ukernel.vmulcaddc.function,
      };
      memcpy(&convolution_op->context.vmulcaddc.params, &convolution_op->params, sizeof(convolution_op->context.vmulcaddc.params));

      size_t mc = batch_output_size;
      if (num_threads > 1) {
        const size_t target_tiles_per_thread = 5;
        const size_t max_mc = divide_round_up(batch_output_size, num_threads * target_tiles_per_thread);
        if (max_mc < mc) {
          const uint32_t mr = convolution_op->ukernel.vmulcaddc.mr;
          mc = min(mc, divide_round_up(mc, max_mc * mr) * mr);
        }
      }
      convolution_op->compute.type = xnn_parallelization_type_1d_tile_1d;
      convolution_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_vmulcaddc;
      convolution_op->compute.range[0] = batch_output_size;
      convolution_op->compute.tile[0] = mc;
      convolution_op->state = xnn_run_state_ready;

      return xnn_status_success;
    }
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_setup_convolution2d_nhwc_qu8(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const uint8_t* input,
    uint8_t* output,
    pthreadpool_t threadpool)
{
  if (convolution_op->type != xnn_operator_type_convolution_nhwc_qu8) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qu8),
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_convolution2d_nhwc(
    convolution_op,
    batch_size, input_height, input_width,
    input, output,
    XNN_INIT_FLAG_QU8,
    0 /* log2(sizeof(input element)) = log2(sizeof(uint8_t)) */,
    0 /* log2(sizeof(filter element)) = log2(sizeof(uint8_t)) */,
    sizeof(int32_t) /* sizeof(extra weights elements) */,
    0 /* log2(sizeof(output element)) = log2(sizeof(uint8_t)) */,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_convolution2d_nhwc_qs8(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const int8_t* input,
    int8_t* output,
    pthreadpool_t threadpool)
{
  if (convolution_op->type != xnn_operator_type_convolution_nhwc_qs8) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qs8),
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_convolution2d_nhwc(
    convolution_op,
    batch_size, input_height, input_width,
    input, output,
    XNN_INIT_FLAG_QS8,
    0 /* log2(sizeof(input element)) = log2(sizeof(int8_t)) */,
    0 /* log2(sizeof(filter element)) = log2(sizeof(int8_t)) */,
    sizeof(int32_t) /* sizeof(extra weights elements) */,
    0 /* log2(sizeof(output element)) = log2(sizeof(int8_t)) */,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_convolution2d_nhwc_qc8(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const int8_t* input,
    int8_t* output,
    pthreadpool_t threadpool)
{
  if (convolution_op->type != xnn_operator_type_convolution_nhwc_qc8) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8),
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_convolution2d_nhwc(
    convolution_op,
    batch_size, input_height, input_width,
    input, output,
    XNN_INIT_FLAG_QC8,
    0 /* log2(sizeof(input element)) = log2(sizeof(int8_t)) */,
    0 /* log2(sizeof(filter element)) = log2(sizeof(int8_t)) */,
    sizeof(int32_t) + sizeof(float) /* sizeof(extra weights elements) */,
    0 /* log2(sizeof(output element)) = log2(sizeof(int8_t)) */,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_convolution2d_nhwc_f16(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  if (convolution_op->type != xnn_operator_type_convolution_nhwc_f16) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f16),
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_convolution2d_nhwc(
    convolution_op,
    batch_size, input_height, input_width,
    input, output,
    XNN_INIT_FLAG_F16,
    1 /* log2(sizeof(input element)) = log2(sizeof(uint16_t)) */,
    1 /* log2(sizeof(filter element)) = log2(sizeof(uint16_t)) */,
    sizeof(uint16_t) /* sizeof(extra weights elements) */,
    1 /* log2(sizeof(output element)) = log2(sizeof(uint16_t)) */,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_convolution2d_nhwc_f32(
    xnn_operator_t convolution_op,
    size_t batch_size,
    size_t input_height,
    size_t input_width,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  if (convolution_op->type != xnn_operator_type_convolution_nhwc_f32) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f32),
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_invalid_parameter;
  }

  return setup_convolution2d_nhwc(
    convolution_op,
    batch_size, input_height, input_width,
    input, output,
    XNN_INIT_FLAG_F32,
    2 /* log2(sizeof(input element)) = log2(sizeof(float)) */,
    2 /* log2(sizeof(filter element)) = log2(sizeof(float)) */,
    sizeof(float) /* sizeof(extra weights elements) */,
    2 /* log2(sizeof(output element)) = log2(sizeof(float)) */,
    pthreadpool_get_threads_count(threadpool));
}
