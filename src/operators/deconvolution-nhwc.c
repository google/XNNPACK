// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/cache.h"
#include "xnnpack/common.h"
#include "xnnpack/compute.h"
#include "xnnpack/config.h"
#include "xnnpack/indirection.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microkernel-type.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/operator.h"
#include "xnnpack/pack.h"
#include "xnnpack/params.h"
#include "pthreadpool.h"

#ifndef XNN_ENABLE_GEMM_M_SPECIALIZATION
#error "XNN_ENABLE_GEMM_M_SPECIALIZATION is not defined"
#endif

static enum xnn_status create_deconvolution2d_nhwc(
    uint32_t output_padding_top,
    uint32_t output_padding_right,
    uint32_t output_padding_bottom,
    uint32_t output_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    const void* kernel,
    const void* bias,
    uint32_t flags,
    uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size,
    uint32_t bias_element_size,
    xnn_pack_conv_goki_w_fn pack_conv_goki_w,
    xnn_pack_deconv_goki_w_fn pack_deconv_goki_w,
    const void* packing_params,
    int input_padding_byte,
    int packed_weights_padding_byte,
    size_t extra_weights_bytes,
    xnn_init_qs8_qc8w_scale_params_fn init_scale_params,
    const float* scale_params,
    xnn_init_qs8_qc8w_scale_params_fn init_kernel_scale_params,
    const float* kernel_scale_params,
    const void* params,
    size_t params_size,
    const struct xnn_gemm_config* gemm_config,
    const struct gemm_fused_ukernels* gemm_ukernels,
    enum xnn_operator_type operator_type,
    bool dynamic_quantization,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* deconvolution_op_out)
{
  xnn_operator_t deconvolution_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
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

  if (stride_width == 0 || stride_height == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " stride: stride dimensions must be non-zero",
      xnn_operator_type_to_string(operator_type), stride_width, stride_height);
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
  if (input_pixel_stride < input_channels) {
    xnn_log_error(
      "failed to create %s operator with input pixel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(operator_type),
      input_pixel_stride, groups, group_input_channels);
    goto error;
  }

  const size_t output_channels = groups * group_output_channels;
  if (output_pixel_stride < output_channels) {
    xnn_log_error(
      "failed to create %s operator with output pixel stride of %zu: "
      "stride must be at least as large as the number of output channels (%" PRIu32 "x%zu)",
      xnn_operator_type_to_string(operator_type),
      output_pixel_stride, groups, group_output_channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  deconvolution_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (deconvolution_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  deconvolution_op->weights_cache = weights_cache;

  const uint32_t mr = gemm_config->mr;
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;

  const uint32_t n_stride = round_up(group_output_channels, nr);
  const uint32_t k_stride = round_up_po2(group_input_channels, kr * sr);
  const uint32_t kernel_size = kernel_height * kernel_width;
  enum xnn_microkernel_type ukernel_type = xnn_microkernel_type_igemm;
  size_t packed_group_weights_size =
      ((kernel_size * k_stride << log2_filter_element_size) + bias_element_size + extra_weights_bytes) * n_stride;
  if (max(stride_height, stride_width) > 1 && max(dilation_height, dilation_width) == 1 && stride_width <= kernel_width && stride_height <= kernel_height) {
    ukernel_type = xnn_microkernel_type_subconv2d;
    const size_t subkernels = stride_height * stride_width;
    packed_group_weights_size = n_stride *
      (((kernel_size * k_stride) << log2_filter_element_size) + (bias_element_size + extra_weights_bytes) * subkernels);
    const size_t subconvolution_buffer_size = sizeof(struct subconvolution_params) * subkernels;
    deconvolution_op->subconvolution_buffer = xnn_allocate_zero_memory(subconvolution_buffer_size);
    if (deconvolution_op->subconvolution_buffer == NULL) {
      xnn_log_error(
        "failed to allocate %zu bytes for %s operator subconvolution buffer",
        subconvolution_buffer_size, xnn_operator_type_to_string(operator_type));
      goto error;
    }
  }
  const size_t aligned_total_weights_size = round_up_po2(packed_group_weights_size * groups, XNN_ALLOCATION_ALIGNMENT);
  void* weights_ptr = xnn_get_pointer_to_write_weights(
      deconvolution_op, aligned_total_weights_size, packed_weights_padding_byte);
  if (weights_ptr == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator packed weights",
      aligned_total_weights_size, xnn_operator_type_to_string(operator_type));
    goto error;
  }
  xnn_log_debug("allocated %zu bytes for packed weights in %s operator",
    aligned_total_weights_size, xnn_operator_type_to_string(operator_type));
  switch (ukernel_type) {
    case xnn_microkernel_type_igemm:
      pack_conv_goki_w(
        groups, group_output_channels, kernel_size, group_input_channels,
        nr, kr, sr,
        kernel, bias, /*scale=*/NULL, weights_ptr,
        nr * extra_weights_bytes,
        packing_params);
      break;
    case xnn_microkernel_type_subconv2d:
      pack_deconv_goki_w(
        groups, group_output_channels, kernel_height, kernel_width, group_input_channels,
        stride_height, stride_width,
        nr, kr, sr,
        kernel, bias, /*scale=*/NULL, weights_ptr, nr * extra_weights_bytes, deconvolution_op->subconvolution_buffer,
        packing_params);
      // We assume that the first subconvolution param weights point to the start of the weights, this is used to check
      // if the weights cache has moved.
      assert(deconvolution_op->subconvolution_buffer->weights == weights_ptr);
      break;
    default:
      XNN_UNREACHABLE;
  }

  if (ukernel_type == xnn_microkernel_type_subconv2d) {
    struct subconvolution_params* subconvolution_params = deconvolution_op->subconvolution_buffer;
    for (size_t offset_y = 0; offset_y < stride_height; offset_y++) {
      for (size_t offset_x = 0; offset_x < stride_width; offset_x++) {
        const size_t subkernel_height = divide_round_up(kernel_height - offset_y, stride_height);
        const size_t subkernel_width = divide_round_up(kernel_width - offset_x, stride_width);
        const size_t subkernel_size = subkernel_height * subkernel_width;

        subconvolution_params->indirection_x_stride = sizeof(void*) * subkernel_size;
        subconvolution_params->w_stride = extra_weights_bytes + bias_element_size + ((k_stride * subkernel_size) << log2_filter_element_size);
        subconvolution_params++;
      }
    }

    if (kernel_scale_params != NULL) {
      assert(init_kernel_scale_params != NULL);

      const float* kernel_scale_params_ptr = kernel_scale_params;
      for (uint32_t group = 0; group < groups; group++) {
        struct subconvolution_params* subconvolution_params = deconvolution_op->subconvolution_buffer;
        for (size_t sh = 0; sh < stride_height; sh++) {
          for (size_t sw = 0; sw < stride_width; sw++) {
            void* group_weights =
                (void*)((uintptr_t) subconvolution_params->weights + group * packed_group_weights_size + bias_element_size * gemm_config->nr);
            const size_t subkernel_height = divide_round_up(kernel_height - sh, stride_height);
            const size_t subkernel_width = divide_round_up(kernel_width - sw, stride_width);
            const size_t subkernel_size = subkernel_height * subkernel_width;
            group_weights = (void*) ((uintptr_t) group_weights + gemm_config->nr * ((subkernel_size * k_stride << log2_filter_element_size)));
            const size_t weights_stride =
                (subkernel_size * k_stride << log2_filter_element_size) + bias_element_size + extra_weights_bytes;
            init_kernel_scale_params(
                group_output_channels, gemm_config->nr, gemm_config->nr,
                gemm_config->nr * weights_stride, gemm_config->nr * weights_stride, 0,
                kernel_scale_params_ptr, group_weights);
            subconvolution_params++;
          }
        }
        kernel_scale_params_ptr += group_output_channels;
      }
    }

    if (scale_params != NULL) {
      assert(init_scale_params != NULL);

      const float* scale_params_ptr = scale_params;
      for (uint32_t group = 0; group < groups; group++) {
        struct subconvolution_params* subconvolution_params = deconvolution_op->subconvolution_buffer;
        for (size_t sh = 0; sh < stride_height; sh++) {
          for (size_t sw = 0; sw < stride_width; sw++) {
            void* group_weights =
                (void*)((uintptr_t) subconvolution_params->weights + group * packed_group_weights_size + bias_element_size * gemm_config->nr);
            const size_t subkernel_height = divide_round_up(kernel_height - sh, stride_height);
            const size_t subkernel_width = divide_round_up(kernel_width - sw, stride_width);
            const size_t subkernel_size = subkernel_height * subkernel_width;
            group_weights = (void*) ((uintptr_t) group_weights + gemm_config->nr * ((subkernel_size * k_stride << log2_filter_element_size)));
            if (kernel_scale_params != NULL) {
              group_weights = (void*) ((uintptr_t) group_weights + gemm_config->nr * sizeof(float));
            }
            const size_t weights_stride =
                (subkernel_size * k_stride << log2_filter_element_size) + bias_element_size + extra_weights_bytes;
            init_scale_params(
                group_output_channels, gemm_config->nr, gemm_config->nr,
                gemm_config->nr * weights_stride, gemm_config->nr * weights_stride, 0,
                scale_params_ptr, group_weights);
            subconvolution_params++;
          }
        }
        scale_params_ptr += group_output_channels;
      }
    }
  } else {
    if (kernel_scale_params != NULL) {
      assert(init_kernel_scale_params != NULL);

      void* group_weights =
          (void*)((uintptr_t) weights_ptr +
                  gemm_config->nr * ((kernel_size * k_stride << log2_filter_element_size) + bias_element_size));
      const size_t weights_stride =
          (kernel_size * k_stride << log2_filter_element_size) + bias_element_size + extra_weights_bytes;
      for (uint32_t group = 0; group < groups; group++) {
        init_kernel_scale_params(
            group_output_channels, gemm_config->nr, gemm_config->nr,
            gemm_config->nr * weights_stride, gemm_config->nr * weights_stride, 0,
            kernel_scale_params, group_weights);
        kernel_scale_params += group_output_channels;
        group_weights = (void*) ((uintptr_t) group_weights + n_stride * weights_stride);
      }
    }

    if (scale_params != NULL) {
      assert(init_scale_params != NULL);

      void* group_weights =
          (void*)((uintptr_t) weights_ptr +
                  gemm_config->nr * ((kernel_size * k_stride << log2_filter_element_size) + bias_element_size));
      if (kernel_scale_params != NULL) {
        group_weights = (void*) ((uintptr_t) group_weights + gemm_config->nr * sizeof(float));
      }
      const size_t weights_stride =
          (kernel_size * k_stride << log2_filter_element_size) + bias_element_size + extra_weights_bytes;
      for (uint32_t group = 0; group < groups; group++) {
        init_scale_params(
            group_output_channels, gemm_config->nr, gemm_config->nr,
            gemm_config->nr * weights_stride, gemm_config->nr * weights_stride, 0,
            scale_params, group_weights);
        scale_params += group_output_channels;
        group_weights = (void*) ((uintptr_t) group_weights + n_stride * weights_stride);
      }
    }
  }

  if (use_weights_cache(deconvolution_op)) {
    struct xnn_weights_cache_look_up_key cache_key;
    cache_key.seed = groups ^ group_input_channels ^ group_output_channels ^ kernel_size ^ nr ^ kr ^ sr ^ ukernel_type;
    cache_key.kernel = kernel;
    cache_key.bias = bias;
    deconvolution_op->packed_weights.offset = xnn_look_up_or_insert_weights_cache(
        deconvolution_op->weights_cache, &cache_key, weights_ptr, aligned_total_weights_size);
  }

  const size_t zero_size = (k_stride << log2_input_element_size) + XNN_EXTRA_BYTES;
  deconvolution_op->zero_size = zero_size;
  deconvolution_op->zero_buffer = xnn_allocate_simd_memory(zero_size);
  if (deconvolution_op->zero_buffer == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator zero padding",
      zero_size, xnn_operator_type_to_string(operator_type));
    goto error;
  }
  memset(deconvolution_op->zero_buffer, input_padding_byte, zero_size);

  deconvolution_op->padding_top = output_padding_top;
  deconvolution_op->padding_right = output_padding_right;
  deconvolution_op->padding_bottom = output_padding_bottom;
  deconvolution_op->padding_left = output_padding_left;

  deconvolution_op->kernel_height = kernel_height;
  deconvolution_op->kernel_width = kernel_width;
  deconvolution_op->stride_height = stride_height;
  deconvolution_op->stride_width = stride_width;
  deconvolution_op->dilation_height = dilation_height;
  deconvolution_op->dilation_width = dilation_width;
  deconvolution_op->groups = groups;
  deconvolution_op->group_input_channels = group_input_channels;
  deconvolution_op->group_output_channels = group_output_channels;
  deconvolution_op->input_pixel_stride = input_pixel_stride;
  deconvolution_op->output_pixel_stride = output_pixel_stride;

  memcpy(&deconvolution_op->params, params, params_size);
  deconvolution_op->type = operator_type;
  deconvolution_op->ukernel.type = ukernel_type;
  deconvolution_op->ukernel.igemm = (struct xnn_ukernel_igemm) {
    .mr = mr,
    .nr = nr,
    .kr = kr,
    .sr = sr,
  };

  assert(XNN_MAX_MR >= mr);
  for (size_t i = 0; i < mr; i++) {
    deconvolution_op->ukernel.igemm.gemm_cases[i] = gemm_ukernels->gemm[i];
    deconvolution_op->ukernel.igemm.igemm_cases[i] = gemm_ukernels->igemm[i];
  }

  deconvolution_op->state = xnn_run_state_invalid;

  *deconvolution_op_out = deconvolution_op;
  return xnn_status_success;

error:
  xnn_delete_operator(deconvolution_op);
  return status;
}

enum xnn_status create_deconvolution2d_nhwc_qs8_qc8w(
    enum xnn_operator_type operator_type,
    uint32_t output_padding_top,
    uint32_t output_padding_right,
    uint32_t output_padding_bottom,
    uint32_t output_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
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
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* deconvolution_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(operator_type), input_scale);
    return xnn_status_invalid_parameter;
  }

  float* requantization_scale = xnn_allocate_simd_memory(groups * group_output_channels * sizeof(float));
  if (requantization_scale == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator packed weights",
      groups * group_output_channels * sizeof(float),
      xnn_operator_type_to_string(operator_type));
    return xnn_status_out_of_memory;
  }
  for (size_t output_channel = 0; output_channel < groups * group_output_channels; output_channel++) {
    requantization_scale[output_channel] = kernel_scale[output_channel];
    if (requantization_scale[output_channel] >= 256.0f) {
      xnn_log_error(
        "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale in output channel #%zu: "
        "requantization scale %.7g is greater or equal to 256.0",
        xnn_operator_type_to_string(operator_type),
        input_scale, kernel_scale[output_channel], output_scale,
        output_channel, requantization_scale[output_channel]);

      xnn_release_simd_memory(requantization_scale);
      return xnn_status_unsupported_parameter;
    }
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(operator_type), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(operator_type), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  assert(gemm_config != NULL);

  union xnn_qs8_qc8w_conv_minmax_params params;
  if XNN_LIKELY(gemm_config->init.qs8_qc8w != NULL) {
    gemm_config->init.qs8_qc8w(&params,
      output_zero_point, output_min, output_max);
  }
  const struct xnn_qs8_packing_params packing_params = {
    .input_zero_point = input_zero_point,
  };
  enum xnn_status status = create_deconvolution2d_nhwc(
    output_padding_top, output_padding_right, output_padding_bottom, output_padding_left,
    kernel_height, kernel_width,
    stride_height, stride_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_pixel_stride, output_pixel_stride,
    kernel, bias, flags,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*bias_element_size=*/sizeof(int32_t),
    (xnn_pack_conv_goki_w_fn) gemm_config->pack_igemm_goki,
    (xnn_pack_deconv_goki_w_fn) gemm_config->pack_deconv_goki,
    &packing_params, input_zero_point /* input padding byte */, 0 /* packed weights padding byte */,
    /*extra_weights_bytes=*/sizeof(float),
    /*init_scale_params=*/xnn_init_qs8_qc8w_scale_fp32_params,
    /*scale_params=*/requantization_scale,
    /*init_kernel_scale_params=*/NULL,
    /*kernel_scale_params=*/NULL,
    &params, sizeof(params),
    gemm_config, &gemm_config->minmax,
    operator_type,
    /*dynamic_quantization=*/false,
    /*weights_cache=*/weights_cache,
    deconvolution_op_out);

  xnn_release_simd_memory(requantization_scale);
  return status;
}

enum xnn_status xnn_create_deconvolution2d_nhwc_qs8_qc8w(
    uint32_t output_padding_top,
    uint32_t output_padding_right,
    uint32_t output_padding_bottom,
    uint32_t output_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
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
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* deconvolution_op_out)
{
  float* requantization_scale = xnn_allocate_simd_memory(groups * group_output_channels * sizeof(float));
  if (requantization_scale == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator packed weights",
      groups * group_output_channels * sizeof(float),
      xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qs8_qc8w));
    return xnn_status_out_of_memory;
  }

  for (size_t output_channel = 0; output_channel < groups * group_output_channels; output_channel++) {
    requantization_scale[output_channel] = input_scale * kernel_scale[output_channel] / output_scale;
    if (requantization_scale[output_channel] >= 256.0f) {
      xnn_log_error(
        "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale in output channel #%zu: "
        "requantization scale %.7g is greater or equal to 256.0",
        xnn_operator_type_to_string(xnn_operator_type_fully_connected_nc_qs8_qc8w),
        input_scale, kernel_scale[output_channel], output_scale,
        output_channel, requantization_scale[output_channel]);

      xnn_release_simd_memory(requantization_scale);
      return xnn_status_unsupported_parameter;
    }
  }

  enum xnn_status status =  create_deconvolution2d_nhwc_qs8_qc8w(
      xnn_operator_type_deconvolution_nhwc_qs8_qc8w,
    output_padding_top,
    output_padding_right,
    output_padding_bottom,
    output_padding_left,
    kernel_height,
    kernel_width,
    stride_height,
    stride_width,
    dilation_height,
    dilation_width,
    groups,
    group_input_channels,
    group_output_channels,
    input_pixel_stride,
    output_pixel_stride,
    input_zero_point,
    input_scale,
    requantization_scale,
    kernel,
    bias,
    output_zero_point,
    output_scale,
    output_min,
    output_max,
    flags,
    code_cache,
    weights_cache, deconvolution_op_out);
  xnn_release_simd_memory(requantization_scale);
  return status;
}

enum xnn_status xnn_create_deconvolution2d_nhwc_qs8(
    uint32_t output_padding_top,
    uint32_t output_padding_right,
    uint32_t output_padding_bottom,
    uint32_t output_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
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
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* deconvolution_op_out)
{
  const float requantization_scale = input_scale * kernel_scale / output_scale;
  if (requantization_scale >= 256.0f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "requantization scale %.7g is greater or equal to 256.0",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qs8),
      input_scale, kernel_scale, output_scale, requantization_scale);
    return xnn_status_unsupported_parameter;
  }
  float* duplicated_requantization_scale = xnn_allocate_simd_memory(groups * group_output_channels * sizeof(float));
  if (duplicated_requantization_scale == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator packed weights",
      groups * group_output_channels * sizeof(float),
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8));
    return xnn_status_out_of_memory;
  }
  for (size_t output_channel = 0; output_channel < groups * group_output_channels; output_channel++) {
    duplicated_requantization_scale[output_channel] = requantization_scale;
  }

  enum xnn_status status = create_deconvolution2d_nhwc_qs8_qc8w(
      xnn_operator_type_deconvolution_nhwc_qs8,
    output_padding_top,
    output_padding_right,
    output_padding_bottom,
    output_padding_left,
    kernel_height,
    kernel_width,
    stride_height,
    stride_width,
    dilation_height,
    dilation_width,
    groups,
    group_input_channels,
    group_output_channels,
    input_pixel_stride,
    output_pixel_stride,
    input_zero_point,
    input_scale,
    duplicated_requantization_scale,
    kernel,
    bias,
    output_zero_point,
    output_scale,
    output_min,
    output_max,
    flags,
    code_cache,
    weights_cache, deconvolution_op_out
    );
  xnn_release_simd_memory(duplicated_requantization_scale);

  return status;
}

enum xnn_status xnn_create_deconvolution2d_nhwc_qu8(
    uint32_t output_padding_top,
    uint32_t output_padding_right,
    uint32_t output_padding_bottom,
    uint32_t output_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
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
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* deconvolution_op_out)
{
  if (input_scale <= 0.0f || !isnormal(input_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qu8), input_scale);
    return xnn_status_invalid_parameter;
  }

  if (kernel_scale <= 0.0f || !isnormal(kernel_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g kernel scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qu8), kernel_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_scale <= 0.0f || !isnormal(output_scale)) {
    xnn_log_error(
      "failed to create %s operator with %.7g output scale: scale must be finite, normalized, and positive",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qu8), output_scale);
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qu8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const float requantization_scale = input_scale * kernel_scale / output_scale;
  if (requantization_scale >= 256.0f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "requantization scale %.7g is greater or equal to 256.0",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qu8),
      input_scale, kernel_scale, output_scale, requantization_scale);
    return xnn_status_unsupported_parameter;
  }

  const struct xnn_gemm_config* gemm_config = xnn_init_qu8_gemm_config();
  assert(gemm_config != NULL);

  union xnn_qu8_conv_minmax_params params;
  if XNN_LIKELY(gemm_config->init.qu8 != NULL) {
    gemm_config->init.qu8(&params,
      kernel_zero_point, requantization_scale, output_zero_point, output_min, output_max);
  }
  const struct xnn_qu8_packing_params packing_params = {
    .input_zero_point = input_zero_point,
    .kernel_zero_point = kernel_zero_point,
  };
  return create_deconvolution2d_nhwc(
    output_padding_top, output_padding_right, output_padding_bottom, output_padding_left,
    kernel_height, kernel_width,
    stride_height, stride_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_pixel_stride, output_pixel_stride,
    kernel, bias, flags,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    /*bias_element_size=*/sizeof(int32_t),
    (xnn_pack_conv_goki_w_fn) xnn_pack_qu8_conv_goki_w,
    (xnn_pack_deconv_goki_w_fn) xnn_pack_qu8_deconv_goki_w,
    &packing_params, input_zero_point /* input padding byte */, kernel_zero_point /* packed weights padding byte */,
    /*extra_weights_bytes=*/0,
    /*init_scale_params=*/NULL,
    /*scale_params=*/NULL,
    /*init_kernel_scale_params=*/NULL,
    /*kernel_scale_params=*/NULL,
    &params, sizeof(params),
    gemm_config, &gemm_config->minmax,
    xnn_operator_type_deconvolution_nhwc_qu8,
    /*dynamic_quantization=*/false,
    /*weights_cache=*/weights_cache,
    deconvolution_op_out);
}

enum xnn_status xnn_create_deconvolution2d_nhwc_f16(
    uint32_t output_padding_top,
    uint32_t output_padding_right,
    uint32_t output_padding_bottom,
    uint32_t output_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    const void* kernel,
    const void* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* deconvolution_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f16));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f16));
    return xnn_status_invalid_parameter;
  }

  const uint16_t output_min_as_half = fp16_ieee_from_fp32_value(output_min);
  const uint16_t output_max_as_half = fp16_ieee_from_fp32_value(output_max);
  output_min = fp16_ieee_to_fp32_value(output_min_as_half);
  output_max = fp16_ieee_to_fp32_value(output_max_as_half);
  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f16), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f16));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const bool linear_activation = (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_config->linear.gemm[gemm_config->mr - 1].function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  union xnn_f16_minmax_params params;
  if XNN_LIKELY(gemm_config->init.f16 != NULL) {
    gemm_config->init.f16(&params, output_min_as_half, output_max_as_half);
  }

  xnn_pack_conv_goki_w_fn pack_conv_goki_w = (xnn_pack_conv_goki_w_fn) xnn_pack_f16_conv_goki_w;
  xnn_pack_deconv_goki_w_fn pack_deconv_goki_w = (xnn_pack_deconv_goki_w_fn) xnn_pack_f16_deconv_goki_w;
  if (flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
    pack_conv_goki_w = (xnn_pack_conv_goki_w_fn) xnn_pack_f32_to_f16_conv_goki_w;
    pack_deconv_goki_w = (xnn_pack_deconv_goki_w_fn) xnn_pack_f32_to_f16_deconv_goki_w;
  }

  return create_deconvolution2d_nhwc(
    output_padding_top, output_padding_right, output_padding_bottom, output_padding_left,
    kernel_height, kernel_width,
    stride_height, stride_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_pixel_stride, output_pixel_stride,
    kernel, bias, flags,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*bias_element_size=*/sizeof(uint16_t),
    pack_conv_goki_w,
    pack_deconv_goki_w,
    NULL /* packing params */, 0 /* input padding byte */, 0 /* packed weights padding byte */,
    /*extra_weights_bytes=*/0,
    /*init_scale_params=*/NULL,
    /*scale_params=*/NULL,
    /*init_kernel_scale_params=*/NULL,
    /*kernel_scale_params=*/NULL,
    &params, sizeof(params),
    gemm_config, gemm_ukernels,
    xnn_operator_type_deconvolution_nhwc_f16,
    /*dynamic_quantization=*/false,
    /*weights_cache=*/weights_cache,
    deconvolution_op_out);
}

enum xnn_status xnn_create_deconvolution2d_nhwc_qd8_f32_qc8w(
    uint32_t output_padding_top,
    uint32_t output_padding_right,
    uint32_t output_padding_bottom,
    uint32_t output_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    const float* kernel_scale,
    const int8_t* kernel,
    const float* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* deconvolution_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qd8_f32_qc8w));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qd8_f32_qc8w));
    return xnn_status_invalid_parameter;
  }
  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qd8_f32_qc8w), output_min, output_max);
    return xnn_status_invalid_parameter;
  }
  const struct xnn_qs8_packing_params packing_params = { .input_zero_point = 1, };

  const struct xnn_gemm_config* gemm_config = xnn_init_qd8_f32_qc8w_gemm_config();
  assert(gemm_config != NULL);

  union xnn_f32_minmax_params params;
  if XNN_LIKELY(gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&params, output_min, output_max);
  }

  enum xnn_status status = create_deconvolution2d_nhwc(
    output_padding_top, output_padding_right, output_padding_bottom, output_padding_left,
    kernel_height, kernel_width,
    stride_height, stride_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_pixel_stride, output_pixel_stride,
    kernel, /*bias=*/NULL, flags,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*bias_element_size=*/sizeof(float),
    (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w,
    (xnn_pack_deconv_goki_w_fn) xnn_pack_qs8_deconv_goki_w,
    /*packing_params=*/&packing_params,
    /*input_padding_byte=*/0,
    /*packed_weights_padding_byte=*/0,
    /*extra_weights_bytes=*/sizeof(float) * 2,
    xnn_init_qs8_qc8w_scale_fp32_params, bias,
    xnn_init_qs8_qc8w_scale_fp32_params, kernel_scale,
    &params, sizeof(params),
    gemm_config, &gemm_config->minmax,
    xnn_operator_type_deconvolution_nhwc_qd8_f32_qc8w,
    /*dynamic_quantization=*/true,
    /*weights_cache=*/weights_cache,
    deconvolution_op_out);

  return status;
}

enum xnn_status xnn_create_deconvolution2d_nhwc_f32(
    uint32_t output_padding_top,
    uint32_t output_padding_right,
    uint32_t output_padding_bottom,
    uint32_t output_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_pixel_stride,
    size_t output_pixel_stride,
    const float* kernel,
    const float* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* deconvolution_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f32));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f32));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f32), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f32));
    return xnn_status_unsupported_hardware;
  }

  const struct xnn_gemm_config* gemm_nr2_config = xnn_init_f32_gemm_nr2_config();
  if (gemm_nr2_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f32));
    return xnn_status_unsupported_hardware;
  }

  if (gemm_config->nr > group_output_channels) {
    // Default micro-kernel is suboptimal. Try to find a better micro-kernel.
    if (gemm_nr2_config->minmax.igemm[gemm_nr2_config->mr - 1].function[XNN_UARCH_DEFAULT] != NULL) {
      gemm_config = gemm_nr2_config;
    }
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  const bool linear_activation = (output_max == INFINITY) && (output_min == -output_max);
  if (linear_activation && gemm_config->linear.gemm[gemm_config->mr - 1].function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  union xnn_f32_minmax_params params;
  if XNN_LIKELY(gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&params, output_min, output_max);
  }

  return create_deconvolution2d_nhwc(
    output_padding_top, output_padding_right, output_padding_bottom, output_padding_left,
    kernel_height, kernel_width,
    stride_height, stride_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_pixel_stride, output_pixel_stride,
    kernel, bias, flags,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*bias_element_size=*/sizeof(float),
    (xnn_pack_conv_goki_w_fn) xnn_pack_f32_conv_goki_w,
    (xnn_pack_deconv_goki_w_fn) xnn_pack_f32_deconv_goki_w,
    NULL /* packing params */, 0 /* input padding byte */, 0 /* packed weights padding byte */,
    /*extra_weights_bytes=*/0,
    /*init_scale_params=*/NULL,
    /*scale_params=*/NULL,
    /*init_kernel_scale_params=*/NULL,
    /*kernel_scale_params=*/NULL,
    &params, sizeof(params),
    gemm_config, gemm_ukernels,
    xnn_operator_type_deconvolution_nhwc_f32,
    /*dynamic_quantization=*/false,
    /*weights_cache=*/weights_cache,
    deconvolution_op_out);
}

static enum xnn_status reshape_conv_path(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  uint32_t log2_input_element_size,
  uint32_t log2_filter_element_size,
  uint32_t extra_weights_element_size,
  uint32_t log2_output_element_size,
  bool dynamic_quantization,
  const void* params,
  size_t params_size,
  size_t num_threads)
{
  assert(deconvolution_op->ukernel.type == xnn_microkernel_type_igemm);

  const size_t input_height = deconvolution_op->input_height;
  const size_t input_width = deconvolution_op->input_width;
  const size_t output_height = deconvolution_op->output_height;
  const size_t output_width = deconvolution_op->output_width;
  const size_t kernel_height = deconvolution_op->kernel_height;
  const size_t kernel_width = deconvolution_op->kernel_width;
  const size_t kernel_size = kernel_height * kernel_width;

  const size_t groups = deconvolution_op->groups;
  const size_t output_size = output_height * output_width;
  size_t mr = deconvolution_op->ukernel.igemm.mr;
  const uint32_t nr = deconvolution_op->ukernel.igemm.nr;

  struct xnn_hmp_igemm_ukernel* igemm_cases = deconvolution_op->ukernel.igemm.igemm_cases;
  #if XNN_ENABLE_GEMM_M_SPECIALIZATION
    mr = xnn_get_heuristic_mr_igemm(output_size, mr, nr, igemm_cases);
  #else
    if (output_size == 1 && igemm_cases[0].function[XNN_UARCH_DEFAULT] != NULL) {
      mr = 1;
    }
  #endif

  struct xnn_hmp_igemm_ukernel igemm_ukernel = igemm_cases[mr - 1];

  const size_t tiled_output_size = round_up(output_size, mr);
  const size_t indirection_buffer_size = sizeof(void*) * kernel_size * tiled_output_size;

  if (input_height != deconvolution_op->last_input_height ||
      input_width != deconvolution_op->last_input_width)
  {
    const void** indirection_buffer = (const void**) xnn_reallocate_memory(deconvolution_op->indirection_buffer, indirection_buffer_size);
    if (indirection_buffer == NULL) {
      xnn_log_error(
        "failed to allocate %zu bytes for %s operator indirection buffer",
        indirection_buffer_size, xnn_operator_type_to_string(deconvolution_op->type));
      return xnn_status_out_of_memory;
    }
    deconvolution_op->indirection_buffer = indirection_buffer;
    xnn_log_debug("allocated %zu bytes for indirection buffer in %s operator",
      indirection_buffer_size, xnn_operator_type_to_string(deconvolution_op->type));

    // Set a dummy input first, the actual input offset is calculated in setup when we have the input pointer.
    // This offset must be aligned properly because inputs and input offsets need to be aligned.
    deconvolution_op->input = (void*) ((uintptr_t) deconvolution_op->zero_buffer + XNN_ALLOCATION_ALIGNMENT);
    deconvolution_op->last_input = deconvolution_op->input;
    deconvolution_op->last_input_height = input_height;
    deconvolution_op->last_input_width = input_width;

    xnn_indirection_init_deconv2d(deconvolution_op, mr, log2_input_element_size);
  }

  const size_t group_input_channels = deconvolution_op->group_input_channels;
  const size_t group_output_channels = deconvolution_op->group_output_channels;

  const size_t w_stride = extra_weights_element_size +
    (round_up_po2(group_input_channels, deconvolution_op->ukernel.igemm.kr * deconvolution_op->ukernel.igemm.sr) * kernel_size << log2_filter_element_size);
  deconvolution_op->context.igemm = (struct igemm_context){
    .ks = kernel_size,
    .ks_scaled = kernel_size * mr * sizeof(void*),
    .kc = group_input_channels << log2_input_element_size,
    .w_stride = w_stride,
    .indirect_a = deconvolution_op->indirection_buffer,
    .zero = deconvolution_op->zero_buffer,
    .packed_w = packed_weights(deconvolution_op),
    .cm_stride = deconvolution_op->output_pixel_stride << log2_output_element_size,
    .cn_stride = nr << log2_output_element_size,
    .ga_stride = group_input_channels << log2_input_element_size,
    .gw_stride = w_stride * round_up(group_output_channels, nr),
    .gc_stride = group_output_channels << log2_output_element_size,
    .ba_stride = input_height * input_width * deconvolution_op->input_pixel_stride << log2_input_element_size,
    .bc_stride = output_size * deconvolution_op->output_pixel_stride << log2_output_element_size,
    .log2_csize = log2_output_element_size,
    .ukernel = igemm_ukernel,
  };
  memcpy(&deconvolution_op->context.igemm.params, params, params_size);

  size_t nc = group_output_channels;
  if (num_threads > 1) {
    const size_t num_other_tiles = groups * batch_size * divide_round_up(output_size, mr);
    const size_t target_tiles_per_thread = 5;
    const size_t max_nc = divide_round_up(group_output_channels * num_other_tiles, num_threads * target_tiles_per_thread);
    if (max_nc < nc) {
      nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
    }
  }

  size_t igemm_compute_index = 0;
  if (dynamic_quantization) {
    deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_1d;
    deconvolution_op->compute[igemm_compute_index].task_1d = (pthreadpool_task_1d_t) xnn_compute_dq_zero_buffer_igemm;
    deconvolution_op->compute[igemm_compute_index].range[0] = batch_size;
    ++igemm_compute_index;
  }
  if (groups == 1) {
    #if XNN_MAX_UARCH_TYPES > 1
      if (xnn_is_hmp_igemm_ukernel(igemm_ukernel)) {
        if (batch_size > 1) {
          deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_3d_tile_2d_with_uarch;
          if (dynamic_quantization) {
            deconvolution_op->compute[igemm_compute_index].task_3d_tile_2d_with_id = (pthreadpool_task_3d_tile_2d_with_id_t) xnn_compute_batch_hmp_dqigemm;
          } else {
            deconvolution_op->compute[igemm_compute_index].task_3d_tile_2d_with_id = (pthreadpool_task_3d_tile_2d_with_id_t) xnn_compute_batch_hmp_igemm;
          }
        } else {
          deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_2d_tile_2d_with_uarch;
          if (dynamic_quantization) {
            deconvolution_op->compute[igemm_compute_index].task_2d_tile_2d_with_id = (pthreadpool_task_2d_tile_2d_with_id_t) xnn_compute_hmp_dqigemm;
          } else {
            deconvolution_op->compute[igemm_compute_index].task_2d_tile_2d_with_id = (pthreadpool_task_2d_tile_2d_with_id_t) xnn_compute_hmp_igemm;
          }
        }
      } else {
        if (batch_size > 1) {
          deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_3d_tile_2d;
          if (dynamic_quantization) {
            deconvolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_batch_dqigemm;
          } else {
            deconvolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_batch_igemm;
          }
        } else {
          deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_2d_tile_2d;
          if (dynamic_quantization) {
            deconvolution_op->compute[igemm_compute_index].task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_dqigemm;
          } else {
            deconvolution_op->compute[igemm_compute_index].task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_igemm;
          }
        }
      }
    #else
      if (batch_size > 1) {
        deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_3d_tile_2d;
        if (dynamic_quantization) {
          deconvolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_batch_dqigemm;
        } else {
          deconvolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_batch_igemm;
        }
      } else {
        deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_2d_tile_2d;
        if (dynamic_quantization) {
          deconvolution_op->compute[igemm_compute_index].task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_dqigemm;
        } else {
          deconvolution_op->compute[igemm_compute_index].task_2d_tile_2d = (pthreadpool_task_2d_tile_2d_t) xnn_compute_igemm;
        }
      }
    #endif
    if (batch_size > 1) {
      deconvolution_op->compute[igemm_compute_index].range[0] = batch_size;
      deconvolution_op->compute[igemm_compute_index].range[1] = output_size;
      deconvolution_op->compute[igemm_compute_index].range[2] = group_output_channels;
    } else {
      deconvolution_op->compute[igemm_compute_index].range[0] = output_size;
      deconvolution_op->compute[igemm_compute_index].range[1] = group_output_channels;
    }
    deconvolution_op->compute[igemm_compute_index].tile[0] = mr;
    deconvolution_op->compute[igemm_compute_index].tile[1] = nc;
  } else {
    #if XNN_MAX_UARCH_TYPES > 1
      if (xnn_is_hmp_igemm_ukernel(igemm_ukernel)) {
        if (batch_size > 1) {
          deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_4d_tile_2d_with_uarch;
          if (dynamic_quantization) {
            deconvolution_op->compute[igemm_compute_index].task_4d_tile_2d_with_id = (pthreadpool_task_4d_tile_2d_with_id_t) xnn_compute_hmp_grouped_batch_dqigemm;
          } else {
            deconvolution_op->compute[igemm_compute_index].task_4d_tile_2d_with_id = (pthreadpool_task_4d_tile_2d_with_id_t) xnn_compute_hmp_grouped_batch_igemm;
          }
        } else {
          deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_3d_tile_2d_with_uarch;
          if (dynamic_quantization) {
            deconvolution_op->compute[igemm_compute_index].task_3d_tile_2d_with_id = (pthreadpool_task_3d_tile_2d_with_id_t) xnn_compute_hmp_grouped_dqigemm;
          } else {
            deconvolution_op->compute[igemm_compute_index].task_3d_tile_2d_with_id = (pthreadpool_task_3d_tile_2d_with_id_t) xnn_compute_hmp_grouped_igemm;
          }
        }
      } else {
        if (batch_size > 1) {
          deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_4d_tile_2d;
          if (dynamic_quantization) {
            deconvolution_op->compute[igemm_compute_index].task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_grouped_batch_dqigemm;
          } else {
            deconvolution_op->compute[igemm_compute_index].task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_grouped_batch_igemm;
          }
        } else {
          deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_3d_tile_2d;
          if (dynamic_quantization) {
            deconvolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_dqigemm;
          } else {
            deconvolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_igemm;
          }
        }
      }
    #else
      if (batch_size > 1) {
        deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_4d_tile_2d;
        if (dynamic_quantization) {
          deconvolution_op->compute[igemm_compute_index].task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_grouped_batch_dqigemm;
        } else {
          deconvolution_op->compute[igemm_compute_index].task_4d_tile_2d = (pthreadpool_task_4d_tile_2d_t) xnn_compute_grouped_batch_igemm;
        }
      } else {
        deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_3d_tile_2d;
        if (dynamic_quantization) {
          deconvolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_dqigemm;
        } else {
          deconvolution_op->compute[igemm_compute_index].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_grouped_igemm;
        }
      }
    #endif
    if (batch_size > 1) {
      deconvolution_op->compute[igemm_compute_index].range[0] = batch_size;
      deconvolution_op->compute[igemm_compute_index].range[1] = groups;
      deconvolution_op->compute[igemm_compute_index].range[2] = output_size;
      deconvolution_op->compute[igemm_compute_index].range[3] = group_output_channels;
    } else {
      deconvolution_op->compute[igemm_compute_index].range[0] = groups;
      deconvolution_op->compute[igemm_compute_index].range[1] = output_size;
      deconvolution_op->compute[igemm_compute_index].range[2] = group_output_channels;
    }
    deconvolution_op->compute[igemm_compute_index].tile[0] = mr;
    deconvolution_op->compute[igemm_compute_index].tile[1] = nc;
  }
  deconvolution_op->state = xnn_run_state_needs_setup;
  return xnn_status_success;
}

static enum xnn_status reshape_subconv2d_path(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  uint32_t log2_input_element_size,
  uint32_t log2_filter_element_size,
  uint32_t extra_weights_element_size,
  uint32_t log2_output_element_size,
  bool dynamic_quantization,
  const void* params,
  size_t params_size,
  size_t num_threads)
{
  assert(deconvolution_op->ukernel.type == xnn_microkernel_type_subconv2d);

  const size_t input_height = deconvolution_op->input_height;
  const size_t input_width = deconvolution_op->input_width;
  const size_t output_height = deconvolution_op->output_height;
  const size_t output_width = deconvolution_op->output_width;
  const size_t kernel_height = deconvolution_op->kernel_height;
  const size_t kernel_width = deconvolution_op->kernel_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t stride_height = deconvolution_op->stride_height;
  const size_t stride_width = deconvolution_op->stride_width;
  const size_t output_height_positions = divide_round_up(output_height, stride_height);
  const size_t output_width_positions = divide_round_up(output_width, stride_width);

  const size_t groups = deconvolution_op->groups;
  const size_t output_size = output_height * output_width;
  uint32_t mr = deconvolution_op->ukernel.igemm.mr;
  const uint32_t nr = deconvolution_op->ukernel.igemm.nr;
  bool use_gemm = deconvolution_op->ukernel.subtype == xnn_microkernel_type_gemm;
  #if XNN_ENABLE_GEMM_M_SPECIALIZATION
    if (use_gemm) {
      mr = xnn_get_heuristic_mr_gemm(
          batch_size, mr, nr, deconvolution_op->ukernel.igemm.gemm_cases);
    } else {
      mr = xnn_get_heuristic_mr_igemm(
          batch_size, mr, nr, deconvolution_op->ukernel.igemm.igemm_cases);
    }
  #endif

  const size_t input_pixel_stride = deconvolution_op->input_pixel_stride << log2_input_element_size;
  const size_t output_pixel_stride = deconvolution_op->output_pixel_stride << log2_output_element_size;

  const bool any_size_change =
    input_height != deconvolution_op->last_input_height ||
    input_width != deconvolution_op->last_input_width ||
    output_height != deconvolution_op->last_output_height ||
    output_width != deconvolution_op->last_output_width ||
    mr != deconvolution_op->last_mr;

  if (deconvolution_op->weights_cache != NULL) {
    void* packed_weights_ptr = packed_weights(deconvolution_op);
    struct subconvolution_params* subconvolution_params = deconvolution_op->subconvolution_buffer;
    if (packed_weights_ptr != subconvolution_params->weights) {
      // Weights cache moved, update all weights pointer.
      const ptrdiff_t diff = (uintptr_t) packed_weights_ptr - (uintptr_t) subconvolution_params->weights;
      for (size_t offset_y = 0; offset_y < stride_height; offset_y++) {
        for (size_t offset_x = 0; offset_x < stride_width; offset_x++) {
          subconvolution_params->weights = (void*) ((uintptr_t) subconvolution_params->weights + diff);
          ++subconvolution_params;
        }
      }
    }
  }

  if (any_size_change) {
    // Use dummy output pointer for now, it will be set during setup.
    void* output = NULL;
    // Initialize subconvolution parameters which depend on output dimensions or MR.
    struct subconvolution_params* subconvolution_params = deconvolution_op->subconvolution_buffer;
    const size_t modulo_padding_top = deconvolution_op->padding_top % stride_height;
    const size_t modulo_padding_left = deconvolution_op->padding_left % stride_width;
    for (size_t offset_y = 0; offset_y < stride_height; offset_y++) {
      for (size_t offset_x = 0; offset_x < stride_width; offset_x++) {
        const size_t output_x_start = subtract_modulo(offset_x, modulo_padding_left, stride_width);
        const size_t output_y_start = subtract_modulo(offset_y, modulo_padding_top, stride_height);
        subconvolution_params->scaled_kernel_size = mr * subconvolution_params->indirection_x_stride;
        subconvolution_params->slice_width = divide_round_up(output_width - output_x_start, stride_width);
        subconvolution_params->slice_height = divide_round_up(output_height - output_y_start, stride_height);
        subconvolution_params->output =
          (void*) ((uintptr_t) output + ((output_y_start * output_width + output_x_start) * output_pixel_stride));
        ++subconvolution_params;
      }
    }
    deconvolution_op->last_output = output;
  }

  if (any_size_change) {
    if (!use_gemm) {
      const size_t indirection_buffer_size = sizeof(void*) *
        kernel_size * output_height * stride_width * round_up(divide_round_up(output_width, stride_width), mr);

      const void** indirection_buffer =
        (const void**) xnn_reallocate_memory(deconvolution_op->indirection_buffer, indirection_buffer_size);
      if (indirection_buffer == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator indirection buffer",
          indirection_buffer_size, xnn_operator_type_to_string(deconvolution_op->type));
        return xnn_status_out_of_memory;
      }
      deconvolution_op->indirection_buffer = indirection_buffer;
      xnn_log_debug("allocated %zu bytes for indirection buffer in %s operator",
                    indirection_buffer_size, xnn_operator_type_to_string(deconvolution_op->type));

      // Set a dummy input first, the actual input offset is calculated in setup when we have the input pointer.
      // This offset must be aligned properly because inputs and input offsets need to be aligned.
      deconvolution_op->input = (void*) ((uintptr_t) deconvolution_op->zero_buffer + XNN_ALLOCATION_ALIGNMENT);
      xnn_indirection_init_subconv2d(deconvolution_op, mr, log2_input_element_size);
    }

    deconvolution_op->last_input = deconvolution_op->input;
    deconvolution_op->last_input_height = input_height;
    deconvolution_op->last_input_width = input_width;
    deconvolution_op->last_output_height = output_height;
    deconvolution_op->last_output_width = output_width;
    deconvolution_op->last_mr = mr;
  }

  const size_t group_input_channels = deconvolution_op->group_input_channels;
  const size_t group_output_channels = deconvolution_op->group_output_channels;
  const uint32_t kr = deconvolution_op->ukernel.igemm.kr;
  const uint32_t sr = deconvolution_op->ukernel.igemm.sr;
  const size_t w_stride = stride_height * stride_width * extra_weights_element_size +
    (round_up_po2(group_input_channels, kr * sr) * kernel_size << log2_filter_element_size);
  if (use_gemm) {
    struct xnn_hmp_gemm_ukernel* gemm_cases = deconvolution_op->ukernel.igemm.gemm_cases;
    deconvolution_op->context.subgemm = (struct subgemm_context) {
        .subconvolution_params = deconvolution_op->subconvolution_buffer,
        .kc = group_input_channels << log2_input_element_size,
        .ax_stride = input_pixel_stride,
        .ay_stride = input_width * input_pixel_stride,
        .cx_stride = stride_width * output_pixel_stride,
        .cy_stride = stride_height * output_width * output_pixel_stride,
        .cn_stride = nr << log2_output_element_size,
        .ga_stride = group_input_channels << log2_input_element_size,
        .gw_stride = w_stride * round_up(group_output_channels, nr),
        .gc_stride = group_output_channels << log2_output_element_size,
        .ba_stride = input_height * input_width * input_pixel_stride,
        .bc_stride = output_size * output_pixel_stride,
        .log2_csize = log2_output_element_size,
        .ukernel = gemm_cases[mr - 1],
    };
    memcpy(&deconvolution_op->context.subgemm.params, params, params_size);
  } else {
    struct xnn_hmp_igemm_ukernel* igemm_cases = deconvolution_op->ukernel.igemm.igemm_cases;
    deconvolution_op->context.subconv = (struct subconv_context) {
        .subconvolution_params = deconvolution_op->subconvolution_buffer,
        .kc = group_input_channels << log2_input_element_size,
        .zero = deconvolution_op->zero_buffer,
        .cx_stride = stride_width * output_pixel_stride,
        .cy_stride = stride_height * output_width * output_pixel_stride,
        .cn_stride = nr << log2_output_element_size,
        .ga_stride = group_input_channels << log2_input_element_size,
        .gw_stride = w_stride * round_up(group_output_channels, nr),
        .gc_stride = group_output_channels << log2_output_element_size,
        .ba_stride = input_height * input_width * input_pixel_stride,
        .bc_stride = output_size * output_pixel_stride,
        .log2_csize = log2_output_element_size,
        .ukernel = igemm_cases[mr - 1],
    };
    memcpy(&deconvolution_op->context.subconv.params, params, params_size);
  }

  size_t nc = group_output_channels;
  if (num_threads > 1) {
    const size_t num_other_tiles = groups * stride_height * stride_width *
      output_height_positions * divide_round_up(output_width_positions, mr);
    const size_t target_tiles_per_thread = 5;
    const size_t max_nc = divide_round_up(group_output_channels * num_other_tiles, num_threads * target_tiles_per_thread);
    if (max_nc < nc) {
      nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
    }
  }

  size_t igemm_compute_index = 0;
  if (dynamic_quantization) {
    assert(deconvolution_op->ukernel.subtype == xnn_microkernel_type_igemm);

    deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_1d;
    deconvolution_op->compute[igemm_compute_index].task_1d = (pthreadpool_task_1d_t) xnn_compute_dq_zero_buffer_subconv;
    deconvolution_op->compute[igemm_compute_index].range[0] = batch_size;
    ++igemm_compute_index;
  }
  if (groups == 1) {
    deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_5d_tile_2d;
    if (use_gemm) {
      deconvolution_op->compute[igemm_compute_index].task_5d_tile_2d = (pthreadpool_task_5d_tile_2d_t) xnn_compute_subgemm2d;
    } else {
      if (dynamic_quantization) {
        deconvolution_op->compute[igemm_compute_index].task_5d_tile_2d = (pthreadpool_task_5d_tile_2d_t) xnn_compute_dqsubconv2d;
      } else {
        deconvolution_op->compute[igemm_compute_index].task_5d_tile_2d = (pthreadpool_task_5d_tile_2d_t) xnn_compute_subconv2d;
      }
    }
    deconvolution_op->compute[igemm_compute_index].range[0] = batch_size;
    deconvolution_op->compute[igemm_compute_index].range[1] = stride_height * stride_width;
    deconvolution_op->compute[igemm_compute_index].range[2] = output_height_positions;
    deconvolution_op->compute[igemm_compute_index].range[3] = output_width_positions;
    deconvolution_op->compute[igemm_compute_index].range[4] = group_output_channels;
    deconvolution_op->compute[igemm_compute_index].tile[0] = mr;
    deconvolution_op->compute[igemm_compute_index].tile[1] = nc;
  } else {
    deconvolution_op->compute[igemm_compute_index].type = xnn_parallelization_type_6d_tile_2d;
    if (use_gemm) {
      deconvolution_op->compute[igemm_compute_index].task_6d_tile_2d = (pthreadpool_task_6d_tile_2d_t) xnn_compute_grouped_subgemm2d;
    } else {
      if (dynamic_quantization) {
        deconvolution_op->compute[igemm_compute_index].task_6d_tile_2d = (pthreadpool_task_6d_tile_2d_t) xnn_compute_grouped_dqsubconv2d;
      } else {
        deconvolution_op->compute[igemm_compute_index].task_6d_tile_2d = (pthreadpool_task_6d_tile_2d_t) xnn_compute_grouped_subconv2d;
      }
    }
    deconvolution_op->compute[igemm_compute_index].range[0] = batch_size;
    deconvolution_op->compute[igemm_compute_index].range[1] = groups;
    deconvolution_op->compute[igemm_compute_index].range[2] = stride_height * stride_width;
    deconvolution_op->compute[igemm_compute_index].range[3] = output_height_positions;
    deconvolution_op->compute[igemm_compute_index].range[4] = output_width_positions;
    deconvolution_op->compute[igemm_compute_index].range[5] = group_output_channels;
    deconvolution_op->compute[igemm_compute_index].tile[0] = mr;
    deconvolution_op->compute[igemm_compute_index].tile[1] = nc;
  }

  deconvolution_op->state = xnn_run_state_needs_setup;
  return xnn_status_success;
}

static enum xnn_status reshape_deconvolution2d_nhwc(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  uint32_t log2_input_element_size,
  uint32_t log2_filter_element_size,
  uint32_t extra_weights_element_size,
  uint32_t log2_output_element_size,
  bool dynamic_quantization,
  const void* params,
  size_t params_size,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool)
{
  deconvolution_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(deconvolution_op->type));
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu input: input dimensions must be non-zero",
      xnn_operator_type_to_string(deconvolution_op->type), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (adjustment_height >= deconvolution_op->stride_height) {
    xnn_log_error(
      "failed to reshape %s operator with %" PRIu32 " height adjustment: "
      "height adjustment must be smaller than height stride (%" PRIu32 ")",
      xnn_operator_type_to_string(deconvolution_op->type), adjustment_height, deconvolution_op->stride_height);
    return xnn_status_invalid_parameter;
  }

  if (adjustment_width >= deconvolution_op->stride_width) {
    xnn_log_error(
      "failed to reshape %s operator with %" PRIu32 " width adjustment: "
      "width adjustment must be smaller than width stride (%" PRIu32 ")",
      xnn_operator_type_to_string(deconvolution_op->type), adjustment_width, deconvolution_op->stride_width);
    return xnn_status_invalid_parameter;
  }

  if (batch_size == 0) {
    deconvolution_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  deconvolution_op->batch_size = batch_size;
  deconvolution_op->input_height = input_height;
  deconvolution_op->input_width = input_width;

  deconvolution_op->output_height = xnn_compute_deconvolution_output_dimension(
      input_height, deconvolution_op->padding_top + deconvolution_op->padding_bottom,
      adjustment_height, deconvolution_op->kernel_height, deconvolution_op->dilation_height, deconvolution_op->stride_height);
  deconvolution_op->output_width = xnn_compute_deconvolution_output_dimension(
      input_width, deconvolution_op->padding_left + deconvolution_op->padding_right,
      adjustment_width, deconvolution_op->kernel_width, deconvolution_op->dilation_width, deconvolution_op->stride_width);

  if (output_height_out != NULL) {
    *output_height_out = deconvolution_op->output_height;
  }
  if (output_width_out != NULL) {
    *output_width_out = deconvolution_op->output_width;
  }

  const size_t num_threads = pthreadpool_get_threads_count(threadpool);
  switch (deconvolution_op->ukernel.type) {
    case xnn_microkernel_type_igemm:
      return reshape_conv_path(
        deconvolution_op,
        batch_size,
        log2_input_element_size, log2_filter_element_size, extra_weights_element_size, log2_output_element_size, dynamic_quantization,
        params, params_size, num_threads);
    case xnn_microkernel_type_subconv2d:
    {
      const size_t mr = deconvolution_op->ukernel.igemm.mr;
      const bool no_padding = (deconvolution_op->padding_top | deconvolution_op->padding_right | deconvolution_op->padding_bottom | deconvolution_op->padding_left) == 0;
      const bool no_adjustment = (adjustment_height | adjustment_width) == 0;
      const bool use_gemm = no_padding && no_adjustment &&
        deconvolution_op->kernel_height == deconvolution_op->stride_height &&
        deconvolution_op->kernel_width == deconvolution_op->stride_width &&
        deconvolution_op->ukernel.igemm.gemm_cases[mr - 1].function[XNN_UARCH_DEFAULT] != NULL &&
        !dynamic_quantization;
      if (use_gemm) {
        deconvolution_op->ukernel.subtype = xnn_microkernel_type_gemm;
      } else {
        deconvolution_op->ukernel.subtype = xnn_microkernel_type_igemm;
      }
      return reshape_subconv2d_path(
        deconvolution_op,
        batch_size,
        log2_input_element_size, log2_filter_element_size, extra_weights_element_size, log2_output_element_size, dynamic_quantization,
        params, params_size, num_threads);
    }
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_reshape_deconvolution2d_nhwc_qs8(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool)
{
  if (deconvolution_op->type != xnn_operator_type_deconvolution_nhwc_qs8) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qs8),
      xnn_operator_type_to_string(deconvolution_op->type));
    return xnn_status_invalid_parameter;
  }

  return reshape_deconvolution2d_nhwc(
    deconvolution_op,
    batch_size, input_height, input_width,
    adjustment_height, adjustment_width,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*extra_weights_element_size=*/sizeof(int32_t) + sizeof(float),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*dynamic_quantization=*/false,
    &deconvolution_op->params.qs8_qc8w_conv_minmax, sizeof(deconvolution_op->params.qs8_qc8w_conv_minmax),
    output_height_out, output_width_out,
    threadpool);
}

enum xnn_status xnn_reshape_deconvolution2d_nhwc_qs8_qc8w(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool)
{
  if (deconvolution_op->type != xnn_operator_type_deconvolution_nhwc_qs8_qc8w) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qs8_qc8w),
      xnn_operator_type_to_string(deconvolution_op->type));
    return xnn_status_invalid_parameter;
  }

  return reshape_deconvolution2d_nhwc(
    deconvolution_op,
    batch_size, input_height, input_width,
    adjustment_height, adjustment_width,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*extra_weights_element_size=*/sizeof(int32_t) + sizeof(float),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*dynamic_quantization=*/false,
    &deconvolution_op->params.qs8_qc8w_conv_minmax, sizeof(deconvolution_op->params.qs8_qc8w_conv_minmax),
    output_height_out, output_width_out,
    threadpool);
}

enum xnn_status xnn_reshape_deconvolution2d_nhwc_qu8(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool)
{
  if (deconvolution_op->type != xnn_operator_type_deconvolution_nhwc_qu8) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qu8),
      xnn_operator_type_to_string(deconvolution_op->type));
    return xnn_status_invalid_parameter;
  }

  return reshape_deconvolution2d_nhwc(
    deconvolution_op,
    batch_size, input_height, input_width,
    adjustment_height, adjustment_width,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    /*extra_weights_element_size=*/sizeof(int32_t),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    /*dynamic_quantization=*/false,
    &deconvolution_op->params.qu8_conv_minmax, sizeof(deconvolution_op->params.qu8_conv_minmax),
    output_height_out, output_width_out,
    threadpool);
}

enum xnn_status xnn_reshape_deconvolution2d_nhwc_f16(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool)
{
  if (deconvolution_op->type != xnn_operator_type_deconvolution_nhwc_f16) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f16),
      xnn_operator_type_to_string(deconvolution_op->type));
    return xnn_status_invalid_parameter;
  }

  return reshape_deconvolution2d_nhwc(
    deconvolution_op,
    batch_size, input_height, input_width,
    adjustment_height, adjustment_width,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*extra_weights_element_size=*/sizeof(uint16_t),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*dynamic_quantization=*/false,
    &deconvolution_op->params.f16_minmax, sizeof(deconvolution_op->params.f16_minmax),
    output_height_out, output_width_out,
    threadpool);
}

enum xnn_status xnn_reshape_deconvolution2d_nhwc_qd8_f32_qc8w(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool)
{
  if (deconvolution_op->type != xnn_operator_type_deconvolution_nhwc_qd8_f32_qc8w) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_qd8_f32_qc8w),
      xnn_operator_type_to_string(deconvolution_op->type));
    return xnn_status_invalid_parameter;
  }

  if (deconvolution_op->zero_buffers) {
    for (size_t i = 1; i < batch_size; ++i) {
      xnn_release_simd_memory(deconvolution_op->zero_buffers[i]);
    }
  }

  deconvolution_op->zero_buffers = xnn_reallocate_memory(deconvolution_op->zero_buffers, batch_size * sizeof(void*));
  deconvolution_op->zero_buffers[0] = deconvolution_op->zero_buffer;
  for (size_t i = 1; i < batch_size; ++i) {
    deconvolution_op->zero_buffers[i] = xnn_allocate_simd_memory(deconvolution_op->zero_size);
  }

  return reshape_deconvolution2d_nhwc(
    deconvolution_op,
    batch_size, input_height, input_width,
    adjustment_height, adjustment_width,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*extra_weights_element_size=*/sizeof(int32_t) + sizeof(float) * 2,
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*dynamic_quantization=*/true,
    &deconvolution_op->params.f32_minmax, sizeof(deconvolution_op->params.f32_minmax),
    output_height_out, output_width_out,
    threadpool);
}

enum xnn_status xnn_reshape_deconvolution2d_nhwc_f32(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool)
{
  if (deconvolution_op->type != xnn_operator_type_deconvolution_nhwc_f32) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(xnn_operator_type_deconvolution_nhwc_f32),
      xnn_operator_type_to_string(deconvolution_op->type));
    return xnn_status_invalid_parameter;
  }

  return reshape_deconvolution2d_nhwc(
    deconvolution_op,
    batch_size, input_height, input_width,
    adjustment_height, adjustment_width,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_filter_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*extra_weights_element_size=*/sizeof(float),
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*dynamic_quantization=*/false,
    &deconvolution_op->params.f32_minmax, sizeof(deconvolution_op->params.f32_minmax),
    output_height_out, output_width_out,
    threadpool);
}

static enum xnn_status setup_conv_path(
  xnn_operator_t deconvolution_op,
  const void* input,
  void* output)
{
  assert(deconvolution_op->ukernel.type == xnn_microkernel_type_igemm);

  deconvolution_op->context.igemm.a_offset = (size_t) ((uintptr_t) input - (uintptr_t) deconvolution_op->last_input);
  deconvolution_op->context.igemm.c = deconvolution_op->output;
  deconvolution_op->context.igemm.zero_size = deconvolution_op->zero_size;
  deconvolution_op->context.igemm.zero_buffers = deconvolution_op->zero_buffers;
  deconvolution_op->context.igemm.quantization_params = deconvolution_op->quantization_params;

  deconvolution_op->state = xnn_run_state_ready;
  return xnn_status_success;
}

static enum xnn_status setup_subconv2d_path(
  xnn_operator_t deconvolution_op,
  const void* input,
  void* output)
{
  assert(deconvolution_op->ukernel.type == xnn_microkernel_type_subconv2d);

  bool use_gemm = deconvolution_op->ukernel.subtype == xnn_microkernel_type_gemm;
  const size_t stride_height = deconvolution_op->stride_height;
  const size_t stride_width = deconvolution_op->stride_width;

  if (output != deconvolution_op->last_output) {
    struct subconvolution_params* subconvolution_params = deconvolution_op->subconvolution_buffer;
    for (size_t offset_y = 0; offset_y < stride_height; offset_y++) {
      for (size_t offset_x = 0; offset_x < stride_width; offset_x++) {
        subconvolution_params->output = (void*) ((uintptr_t) subconvolution_params->output +
                                                 ((uintptr_t) output - (uintptr_t) deconvolution_op->last_output));
        ++subconvolution_params;
      }
    }
    deconvolution_op->last_output = output;
  }

  if (use_gemm) {
    deconvolution_op->context.subgemm.a = input;
  } else {
    deconvolution_op->context.subconv.a_offset = (size_t) ((uintptr_t) input - (uintptr_t) deconvolution_op->last_input);
    deconvolution_op->context.subconv.zero_size = deconvolution_op->zero_size;
    deconvolution_op->context.subconv.zero_buffers = deconvolution_op->zero_buffers;
    deconvolution_op->context.subconv.quantization_params = deconvolution_op->quantization_params;
  }

  deconvolution_op->state = xnn_run_state_ready;
  return xnn_status_success;
}

static enum xnn_status setup_deconvolution2d_nhwc(
  xnn_operator_t deconvolution_op,
  enum xnn_operator_type expected_operator_type,
  const void* input,
  const void* quantization_params,
  void* output)
{
  if (deconvolution_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(deconvolution_op->type));
    return xnn_status_invalid_parameter;
  }

  if (deconvolution_op->weights_cache != NULL && !xnn_weights_cache_is_finalized(deconvolution_op->weights_cache)) {
    xnn_log_error("failed to setup %s operator: weights cache is not finalized",
                  xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_state;
  }

  switch (deconvolution_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(deconvolution_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  deconvolution_op->input = input;
  deconvolution_op->output = output;
  deconvolution_op->quantization_params = quantization_params;

  switch (deconvolution_op->ukernel.type) {
    case xnn_microkernel_type_igemm:
      return setup_conv_path(deconvolution_op, input, output);
    case xnn_microkernel_type_subconv2d:
    {
      return setup_subconv2d_path(deconvolution_op, input, output);
    }
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_setup_deconvolution2d_nhwc_qs8(
    xnn_operator_t deconvolution_op,
    const int8_t* input,
    int8_t* output)
{
  return setup_deconvolution2d_nhwc(deconvolution_op, xnn_operator_type_deconvolution_nhwc_qs8, input, /*quantization_params=*/NULL, output);
}

enum xnn_status xnn_setup_deconvolution2d_nhwc_qs8_qc8w(
    xnn_operator_t deconvolution_op,
    const int8_t* input,
    int8_t* output)
{
  return setup_deconvolution2d_nhwc(deconvolution_op, xnn_operator_type_deconvolution_nhwc_qs8_qc8w, input, /*quantization_params=*/NULL, output);
}

enum xnn_status xnn_setup_deconvolution2d_nhwc_qu8(
    xnn_operator_t deconvolution_op,
    const uint8_t* input,
    uint8_t* output)
{
  return setup_deconvolution2d_nhwc(deconvolution_op, xnn_operator_type_deconvolution_nhwc_qu8, input, /*quantization_params=*/NULL, output);
}

enum xnn_status xnn_setup_deconvolution2d_nhwc_f16(
    xnn_operator_t deconvolution_op,
    const void* input,
    void* output)
{
  return setup_deconvolution2d_nhwc(deconvolution_op, xnn_operator_type_deconvolution_nhwc_f16, input, /*quantization_params=*/NULL, output);
}

enum xnn_status xnn_setup_deconvolution2d_nhwc_qd8_f32_qc8w(
    xnn_operator_t deconvolution_op,
    const int8_t* input,
    float* output,
    const struct xnn_dynamic_quantization_params* quantization_params)
{
  return setup_deconvolution2d_nhwc(deconvolution_op, xnn_operator_type_deconvolution_nhwc_qd8_f32_qc8w, input, quantization_params, output);
}

enum xnn_status xnn_setup_deconvolution2d_nhwc_f32(
    xnn_operator_t deconvolution_op,
    const float* input,
    float* output)
{
  return setup_deconvolution2d_nhwc(deconvolution_op, xnn_operator_type_deconvolution_nhwc_f32, input, /*quantization_params=*/NULL, output);
}
