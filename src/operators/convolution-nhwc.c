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
#include <xnnpack/cache.h>
#include <xnnpack/common.h>
#include <xnnpack/compute.h>
#include <xnnpack/indirection.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>
#include <xnnpack/operator-utils.h>
#include <xnnpack/pack.h>
#include <xnnpack/params.h>
#include <xnnpack/post-operation.h>
#include <xnnpack/microparams-init.h>

#ifndef XNN_ENABLE_GEMM_M_SPECIALIZATION
#error "XNN_ENABLE_GEMM_M_SPECIALIZATION is not defined"
#endif

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
  const struct dwconv_parameters* best_ukernel = NULL;
  while (num_ukernels-- != 0) {
    // Find the smallest primary_tile that is at least as big as kernel_size.
    if (ukernel->primary_tile >= kernel_size) {
      if (best_ukernel == NULL || ukernel->primary_tile < best_ukernel->primary_tile) {
        best_ukernel = ukernel;
      }
    }
    ukernel++;
  }
  return best_ukernel;
}

#if XNN_PLATFORM_JIT
static inline uintptr_t cached_code_at_offset(xnn_operator_t op, size_t offset)
{
  return (uintptr_t)op->code_cache->cache.code.start + offset;
}

static size_t get_generated_gemm(
    struct xnn_hmp_gemm_codegen generators,
    const struct jit_gemm_params *jit_gemm_params,
    size_t mr,
    size_t group_output_channels,
    size_t nr,
    size_t group_input_channels,
    size_t log2_input_element_size,
    struct xnn_code_cache* code_cache)
{
  size_t offset = XNN_CACHE_NOT_FOUND;
  xnn_jit_gemm_code_generator_fn generator = generators.function[XNN_UARCH_DEFAULT];
  if (generator == NULL) {
    goto error;
  }

  enum xnn_status status = xnn_status_success;

  status = xnn_reserve_code_memory(&code_cache->cache.code, XNN_DEFAULT_MICROKERNEL_SIZE);
  if (xnn_status_success != status) {
    xnn_log_error("failed to ensure sufficient space in the code buffer for a microkernel");
    goto error;
  }

  const size_t old_size = code_cache->cache.code.size;
  void* old_code = (uint8_t*) code_cache->cache.code.start + old_size;
  status = generator(&code_cache->cache.code, mr, group_output_channels % nr,
                     group_input_channels << log2_input_element_size,
                     jit_gemm_params);

  if (xnn_status_success != status) {
    xnn_log_error("failed to generate GEMM microkernel");
    goto error;
  }

  const size_t new_size = code_cache->cache.code.size;
  return xnn_get_or_insert_code_cache(code_cache, old_code, new_size - old_size);

error:
  return offset;
}

static void generate_gemms_up_to_max_mr(
    size_t max_mr,
    struct gemm_codegens generators,
    const struct jit_gemm_params *jit_gemm_params,
    size_t group_output_channels,
    size_t nr,
    size_t group_input_channels,
    size_t log2_input_element_size,
    xnn_operator_t convolution_op)
{
  assert(XNN_MAX_MR >= max_mr);
  if (convolution_op->code_cache == NULL) {
    return;
  }
  for (size_t mr = 1; mr <= max_mr; mr++) {
    // Get smallest generator that is >= mr.
    size_t smallest_mr = mr;
    while (generators.gemm[smallest_mr - 1].function[XNN_UARCH_DEFAULT] == NULL && smallest_mr <= max_mr) {
      smallest_mr++;
    }
    xnn_log_debug("using generator for mr %zu to generate gemm of mr %zu", smallest_mr, mr);
    convolution_op->ukernel.gemm.gemm_cases[mr - 1].generated_code_offset[XNN_UARCH_DEFAULT] =
      get_generated_gemm(generators.gemm[smallest_mr - 1], jit_gemm_params, mr, group_output_channels, nr,
                         group_input_channels, log2_input_element_size, convolution_op->code_cache);
  }
}

static size_t get_generated_igemm(
    struct xnn_hmp_igemm_codegen generators,
    const struct jit_gemm_params *jit_gemm_params,
    size_t group_output_channels,
    size_t nr,
    size_t group_input_channels,
    size_t log2_input_element_size,
    size_t kernel_size,
    size_t mr,
    struct xnn_code_cache* code_cache)
{
  size_t offset = XNN_CACHE_NOT_FOUND;
  xnn_jit_igemm_code_generator_fn generator = generators.function[XNN_UARCH_DEFAULT];
  if (generator == NULL) {
    goto error;
  }
  enum xnn_status status = xnn_status_success;

  status = xnn_reserve_code_memory(&code_cache->cache.code, XNN_DEFAULT_MICROKERNEL_SIZE);
  if (xnn_status_success != status) {
    xnn_log_error("failed to ensure sufficient space in code buffer for microkernel");
    goto error;
  }

  const size_t old_size = code_cache->cache.code.size;
  void* old_code = (uint8_t*) code_cache->cache.code.start + old_size;
  status = generator(&code_cache->cache.code, mr, group_output_channels % nr,
                     group_input_channels << log2_input_element_size,
                     kernel_size * mr * sizeof(void*), jit_gemm_params);
  if (status != xnn_status_success) {
    xnn_log_error("failed to generate IGEMM microkernel");
    goto error;
  }

  const size_t new_size = code_cache->cache.code.size;
  return xnn_get_or_insert_code_cache(code_cache, old_code, new_size - old_size);

error:
  return offset;
}

static void generate_igemms_up_to_max_mr(
    size_t max_mr,
    struct gemm_codegens generators,
    const struct jit_gemm_params *jit_gemm_params,
    size_t group_output_channels,
    size_t nr,
    size_t group_input_channels,
    size_t log2_input_element_size,
    size_t kernel_size,
    xnn_operator_t convolution_op)
{
  assert(XNN_MAX_MR >= max_mr);
  if (convolution_op->code_cache == NULL) {
    return;
  }
  for (size_t mr = 1; mr <= max_mr; mr++) {
    // Get smallest generator that is >= mr.
    size_t smallest_mr = mr;
    while (generators.igemm[smallest_mr - 1].function[XNN_UARCH_DEFAULT] == NULL && smallest_mr <= max_mr) {
      smallest_mr++;
    }
    xnn_log_debug("using generator for mr %zu to generate igemm of mr %zu", smallest_mr, mr);
    convolution_op->ukernel.igemm.igemm_cases[mr - 1].generated_code_offset[XNN_UARCH_DEFAULT] =
      get_generated_igemm(generators.igemm[smallest_mr - 1], jit_gemm_params, group_output_channels, nr,
                          group_input_channels, log2_input_element_size, kernel_size, mr,
                          convolution_op->code_cache);
  }
}
#endif  // XNN_PLATFORM_JIT

static enum xnn_status create_vmulcaddc_path(
    uint32_t groups,
    const void* kernel,
    const void* bias,
    uint32_t log2_filter_element_size,
    uint32_t bias_element_size,
    xnn_pack_vmulcaddc_w_fn pack_vmulcaddc_w,
    const void* packing_params,
    int packed_weights_padding_byte,
    const void* vmulcaddc_params,
    size_t vmulcaddc_params_size,
    const struct vmulcaddc_parameters* vmulcaddc_parameters,
    enum xnn_operator_type operator_type,
    xnn_operator_t convolution_op)
{
  assert(vmulcaddc_parameters != NULL);
  assert(vmulcaddc_params != NULL);

  enum xnn_status status = xnn_status_out_of_memory;

  const size_t c_stride = round_up_po2(groups, vmulcaddc_parameters->channel_tile);
  const size_t packed_weights_size = ((UINT32_C(1) << log2_filter_element_size) + bias_element_size) * c_stride;
  size_t aligned_total_weights_size = round_up_po2(packed_weights_size, XNN_ALLOCATION_ALIGNMENT);
  void* weights_ptr = xnn_get_pointer_to_write_weights(
      convolution_op, aligned_total_weights_size, packed_weights_padding_byte);
  if (weights_ptr == NULL) {
    xnn_log_error("failed to reserve or allocated %zu bytes for %s operator vmulcaddc packed weights",
                  aligned_total_weights_size, xnn_operator_type_to_string(operator_type));
    goto error;
  }
  xnn_log_debug("allocated %zu bytes for packed weights in %s operator",
                aligned_total_weights_size, xnn_operator_type_to_string(operator_type));

  pack_vmulcaddc_w(groups, vmulcaddc_parameters->channel_tile, kernel, bias, weights_ptr, packing_params);

  if (use_weights_cache(convolution_op)) {
    convolution_op->packed_weights.offset = xnn_get_or_insert_weights_cache(
        convolution_op->weights_cache, weights_ptr, aligned_total_weights_size);
  }

  memcpy(&convolution_op->params, vmulcaddc_params, vmulcaddc_params_size);

  convolution_op->ukernel.vmulcaddc = (struct xnn_ukernel_vmulcaddc) {
    .function = vmulcaddc_parameters->ukernel,
        .mr = vmulcaddc_parameters->row_tile,
  };
  return xnn_status_success;

error:
  return status;
}

static enum xnn_status create_dwconv_path(
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t groups,
    const void* kernel,
    const void* bias,
    uint32_t flags,
    uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size,
    uint32_t bias_element_size,
    xnn_pack_dwconv_hwg_w_fn pack_dwconv_hwg_w,
    xnn_pack_dwconv_ghw_w_fn pack_dwconv_ghw_w,
    const void* packing_params,
    int packed_weights_padding_byte,
    size_t extra_weights_bytes,
    xnn_init_qc8_scale_params_fn init_scale_params,
    const float* scale_params,
    const void* dwconv_params,
    size_t dwconv_params_size,
    const struct dwconv_parameters* dwconv_ukernel,
    bool linear_activation,
    enum xnn_operator_type operator_type,
    size_t* zero_size,
    xnn_operator_t convolution_op)
{
  assert(dwconv_ukernel != NULL);
  enum xnn_status status = xnn_status_out_of_memory;
  const uint8_t primary_tile = dwconv_ukernel->primary_tile;
  assert(primary_tile >= kernel_height * kernel_width);

  const size_t c_stride = round_up_po2(groups, dwconv_ukernel->channel_tile);
  const size_t packed_weights_size =
      ((primary_tile << log2_filter_element_size) + bias_element_size + extra_weights_bytes) * c_stride;
  size_t aligned_total_weights_size = round_up_po2(packed_weights_size, XNN_ALLOCATION_ALIGNMENT);
  void* weights_ptr = xnn_get_pointer_to_write_weights(
      convolution_op, aligned_total_weights_size, packed_weights_padding_byte);
  if (weights_ptr == NULL) {
    xnn_log_error("failed to reserve or allocated %zu bytes for %s operator dwconv packed weights",
                  aligned_total_weights_size, xnn_operator_type_to_string(operator_type));
    goto error;
  }
  xnn_log_debug("allocated %zu bytes for packed weights in %s operator",
                aligned_total_weights_size, xnn_operator_type_to_string(operator_type));

  memcpy(&convolution_op->params, dwconv_params, dwconv_params_size);

  if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
    pack_dwconv_hwg_w(
        dwconv_ukernel->primary_tile,
        kernel_height, kernel_width,
        groups, dwconv_ukernel->channel_tile,
        kernel, bias, weights_ptr,
        dwconv_ukernel->channel_tile * extra_weights_bytes,
        packing_params);
  } else {
    pack_dwconv_ghw_w(
        dwconv_ukernel->primary_tile,
        kernel_height, kernel_width,
        groups, dwconv_ukernel->channel_tile,
        kernel, bias, weights_ptr,
        dwconv_ukernel->channel_tile * extra_weights_bytes,
        packing_params);
  }

  if (scale_params != NULL) {
    assert(init_scale_params != NULL);

    init_scale_params(
        /*channels=*/groups,
        /*channels_tile=*/dwconv_ukernel->channel_tile,
        /*stride=*/dwconv_ukernel->channel_tile *
            ((primary_tile << log2_filter_element_size) + bias_element_size + extra_weights_bytes),
        /*scale=*/scale_params,
        /*packed_w=*/
        (void*)((uintptr_t)weights_ptr +
                dwconv_ukernel->channel_tile * ((primary_tile << log2_filter_element_size) + bias_element_size)));
  }

  if (use_weights_cache(convolution_op)) {
    convolution_op->packed_weights.offset = xnn_get_or_insert_weights_cache(
        convolution_op->weights_cache, weights_ptr, aligned_total_weights_size);
  }

  const union dwconv_fused_ukernels* ukernels = &dwconv_ukernel->minmax;
  if (linear_activation && dwconv_ukernel->linear.unipass != NULL) {
    ukernels = &dwconv_ukernel->linear;
  }
  convolution_op->ukernel.dwconv = (struct xnn_ukernel_dwconv) {
    .unipass_fn = ukernels->unipass,
    .primary_tile = dwconv_ukernel->primary_tile,
    .middle_tile = dwconv_ukernel->middle_tile,
    .last_tile = dwconv_ukernel->last_tile,
  };

  *zero_size = XNN_EXTRA_BYTES + (c_stride << log2_input_element_size);
  return xnn_status_success;
error:
  return status;
}

static enum xnn_status create_gemm_or_igemm(
    enum xnn_microkernel_type ukernel_type,
    uint32_t kernel_size,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    const void* kernel,
    const void* bias,
    uint32_t flags,
    uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size,
    uint32_t bias_element_size,
    xnn_pack_gemm_goi_w_fn pack_gemm_goi_w,
    xnn_pack_conv_kgo_w_fn pack_conv_kgo_w,
    xnn_pack_conv_goki_w_fn pack_conv_goki_w,
    const void* packing_params,
    int packed_weights_padding_byte,
    size_t extra_weights_bytes,
    xnn_init_qc8_scale_params_fn init_scale_params,
    const float* scale_params,
    const void* gemm_params,
    size_t gemm_params_size,
    const struct gemm_parameters* gemm_parameters,
    const struct jit_gemm_params* jit_gemm_params,
    bool linear_activation,
    bool relu_activation,
    enum xnn_operator_type operator_type,
    size_t num_post_operations,
    void* post_operation_params,
    xnn_operator_t convolution_op,
    size_t* zero_size)
{
  enum xnn_status status = xnn_status_out_of_memory;
  const uint32_t nr = gemm_parameters->nr;
  const uint32_t kr = UINT32_C(1) << gemm_parameters->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_parameters->log2_sr;
  const size_t n_stride = round_up(group_output_channels, nr);
  const size_t k_stride = round_up_po2(group_input_channels, kr * sr);

  const size_t packed_group_weights_size =
      ((kernel_size * k_stride << log2_filter_element_size) + bias_element_size + extra_weights_bytes) * n_stride;
  const size_t aligned_total_weights_size = round_up_po2(packed_group_weights_size * groups, XNN_ALLOCATION_ALIGNMENT);
  void* weights_ptr = xnn_get_pointer_to_write_weights(
      convolution_op, aligned_total_weights_size, packed_weights_padding_byte);
  if (weights_ptr == NULL) {
    xnn_log_error("failed to reserve or allocated %zu bytes for %s operator gemm packed weights",
                  aligned_total_weights_size, xnn_operator_type_to_string(operator_type));
    goto error;
  }
  xnn_log_debug("allocated %zu bytes for packed weights in %s operator",
                aligned_total_weights_size, xnn_operator_type_to_string(operator_type));

  memcpy(&convolution_op->params, gemm_params, gemm_params_size);
  convolution_op->num_post_operation_params = num_post_operations;
  convolution_op->post_operation_params = post_operation_params;

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_parameters->minmax;
  const uint32_t mr = gemm_parameters->mr;
  if (linear_activation && gemm_parameters->linear.gemm[mr - 1].function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_parameters->linear;
  } else if (relu_activation && gemm_parameters->relu.gemm[mr - 1].function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_parameters->relu;
  }
  switch (ukernel_type) {
    case xnn_microkernel_type_gemm:
      pack_gemm_goi_w(
          groups, group_output_channels, group_input_channels,
          nr, kr, sr,
          kernel, bias, weights_ptr, gemm_parameters->nr * extra_weights_bytes, packing_params);
      convolution_op->ukernel.gemm = (struct xnn_ukernel_gemm) {
        .mr = mr,
            .nr = nr,
            .kr = kr,
            .sr = sr,
      };

      assert(XNN_MAX_MR >= mr);
      for (size_t i = 0; i < mr; i++) {
        convolution_op->ukernel.gemm.gemm_cases[i] = gemm_ukernels->gemm[i];
      }

#if XNN_PLATFORM_JIT
      generate_gemms_up_to_max_mr(
          mr, gemm_parameters->generator, jit_gemm_params, group_output_channels, nr,
          group_input_channels, log2_input_element_size, convolution_op);
#endif  // XNN_PLATFORM_JIT

      break;
    case xnn_microkernel_type_igemm:
      if (flags & XNN_FLAG_DEPTHWISE_CONVOLUTION) {
        pack_conv_kgo_w(
            groups, group_output_channels, kernel_size,
            nr, kr, sr,
            kernel, bias, weights_ptr, gemm_parameters->nr * extra_weights_bytes, packing_params);
      } else {
        pack_conv_goki_w(
            groups, group_output_channels, kernel_size, group_input_channels,
            nr, kr, sr,
            kernel, bias, weights_ptr, gemm_parameters->nr * extra_weights_bytes, packing_params);
      }
      convolution_op->ukernel.igemm = (struct xnn_ukernel_igemm) {
        .mr = mr,
            .nr = nr,
            .kr = kr,
            .sr = sr,
      };

      assert(XNN_MAX_MR >= mr);
      for (size_t i = 0; i < mr; i++) {
        convolution_op->ukernel.igemm.igemm_cases[i] = gemm_ukernels->igemm[i];
      }

#if XNN_PLATFORM_JIT
      generate_igemms_up_to_max_mr(
          mr, gemm_parameters->generator, jit_gemm_params, group_output_channels, nr,
          group_input_channels, log2_input_element_size, kernel_size, convolution_op);
#endif  // XNN_PLATFORM_JIT

      break;
    default:
      XNN_UNREACHABLE;
  }

  if (scale_params != NULL) {
    assert(init_scale_params != NULL);

    void* group_weights =
        (void*)((uintptr_t)weights_ptr +
                gemm_parameters->nr * ((kernel_size * k_stride << log2_filter_element_size) + bias_element_size));
    const size_t weights_stride =
        (kernel_size * k_stride << log2_filter_element_size) + bias_element_size + extra_weights_bytes;
    for (uint32_t group = 0; group < groups; group++) {
      init_scale_params(
          group_output_channels, gemm_parameters->nr,
          gemm_parameters->nr * weights_stride,
          scale_params, group_weights);
      scale_params += group_output_channels;
      group_weights = (void*) ((uintptr_t) group_weights + n_stride * weights_stride);
    }
  }

  if (use_weights_cache(convolution_op)) {
    convolution_op->packed_weights.offset = xnn_get_or_insert_weights_cache(
        convolution_op->weights_cache, weights_ptr, aligned_total_weights_size);
  }

  *zero_size = XNN_EXTRA_BYTES + (k_stride << log2_input_element_size);
  return xnn_status_success;

error:
  return status;
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
    xnn_pack_vmulcaddc_w_fn pack_vmulcaddc_w,
    xnn_pack_dwconv_hwg_w_fn pack_dwconv_hwg_w,
    xnn_pack_dwconv_ghw_w_fn pack_dwconv_ghw_w,
    xnn_pack_gemm_goi_w_fn pack_gemm_goi_w,
    xnn_pack_conv_kgo_w_fn pack_conv_kgo_w,
    xnn_pack_conv_goki_w_fn pack_conv_goki_w,
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
    struct jit_gemm_params* jit_gemm_params,
    bool linear_activation,
    bool relu_activation,
    uint32_t datatype_init_flags,
    enum xnn_operator_type operator_type,
    size_t num_post_operations,
    void* post_operation_params,
    xnn_caches_t caches,
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

  if (caches != NULL) {
    convolution_op->weights_cache = caches->weights_cache;
    convolution_op->code_cache = caches->code_cache;
  }

  const size_t kernel_size = kernel_height * kernel_width;

  enum xnn_microkernel_type ukernel_type = xnn_microkernel_type_default;
  const bool unit_subsampling = (subsampling_width | subsampling_height) == 1;
  if (group_input_channels == 1 && group_output_channels == 1 && kernel_size == 1 && unit_subsampling && !any_padding && vmulcaddc_parameters != NULL) {
    ukernel_type = xnn_microkernel_type_vmulcaddc;
  } else if (group_input_channels == 1 && group_output_channels == 1 && dwconv_ukernel != NULL)
  {
    ukernel_type = xnn_microkernel_type_dwconv;
  } else if (kernel_size == 1 && unit_subsampling && !any_padding) {
    ukernel_type = xnn_microkernel_type_gemm;
  } else {
    ukernel_type = xnn_microkernel_type_igemm;
  }
  assert(ukernel_type != xnn_microkernel_type_default);

  if (num_post_operations != 0 && (ukernel_type != xnn_microkernel_type_gemm && ukernel_type != xnn_microkernel_type_igemm)) {
    xnn_log_error(
        "convolution with post operations not support for these parameters: "
        "kernel_size: %zu unit_subsampling: %d padding: %d, ukernel_type: %d",
        kernel_size, unit_subsampling, any_padding, ukernel_type);
    goto error;
  }

  size_t zero_size = 0;
  switch (ukernel_type) {
    case xnn_microkernel_type_vmulcaddc:
    {
      status = create_vmulcaddc_path(
          groups, kernel, bias, log2_filter_element_size, bias_element_size,
          pack_vmulcaddc_w, packing_params, packed_weights_padding_byte,
          vmulcaddc_params, vmulcaddc_params_size, vmulcaddc_parameters,
          operator_type, convolution_op);
      if (status != xnn_status_success) {
        goto error;
      }
      break;
    }
    case xnn_microkernel_type_dwconv:
    {
      status = create_dwconv_path(
          kernel_height, kernel_width,
          groups, kernel, bias, flags,
          log2_input_element_size, log2_filter_element_size, bias_element_size,
          pack_dwconv_hwg_w, pack_dwconv_ghw_w, packing_params, packed_weights_padding_byte, extra_weights_bytes,
          init_scale_params, scale_params,
          dwconv_params, dwconv_params_size, dwconv_ukernel,
          linear_activation, operator_type, &zero_size, convolution_op);
      if (status != xnn_status_success) {
        goto error;
      }
      break;
    }
    case xnn_microkernel_type_gemm:
    case xnn_microkernel_type_igemm:
    {
      status = create_gemm_or_igemm(
          ukernel_type, kernel_size,
          groups, group_input_channels, group_output_channels,
          kernel, bias, flags,
          log2_input_element_size, log2_filter_element_size, bias_element_size,
          pack_gemm_goi_w, pack_conv_kgo_w, pack_conv_goki_w, packing_params,
          packed_weights_padding_byte, extra_weights_bytes,
          init_scale_params, scale_params,
          gemm_params, gemm_params_size, gemm_parameters, jit_gemm_params,
          linear_activation, relu_activation,
          operator_type,
          num_post_operations, post_operation_params,
          convolution_op,
          &zero_size);
      if (status != xnn_status_success) {
        goto error;
      }
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
    xnn_caches_t caches,
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
  if (requantization_scale >= 256.0f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "requantization scale %.7g is greater or equal to 256.0",
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
    /*log2_input_element_size=*/0,  // log2(sizeof(uint8_t))
    /*log2_filter_element_size=*/0,  // log2(sizeof(uint8_t))
    /*bias_element_size=*/sizeof(int32_t),
    (xnn_pack_vmulcaddc_w_fn) NULL,
    (xnn_pack_dwconv_hwg_w_fn) xnn_pack_qu8_dwconv_hwg_w,
    (xnn_pack_dwconv_ghw_w_fn) xnn_pack_qu8_dwconv_ghw_w,
    (xnn_pack_gemm_goi_w_fn) xnn_pack_qu8_gemm_goi_w,
    (xnn_pack_conv_kgo_w_fn) xnn_pack_qu8_conv_kgo_w,
    (xnn_pack_conv_goki_w_fn) xnn_pack_qu8_conv_goki_w,
    /*packing_params=*/&packing_params,
    /*input_padding_byte=*/input_zero_point,
    /*packed_weights_padding_byte=*/kernel_zero_point,
    /*extra_weights_bytes=*/0,
    /*init_scale_params=*/NULL,
    /*scale_params=*/NULL,
    /*gemm_params=*/&gemm_params,
    /*gemm_params_size=*/sizeof(gemm_params),
    /*dwconv_params=*/&dwconv_params,
    /*dwconv_params_size=*/sizeof(dwconv_params),
    /*vmulcaddc_params=*/NULL,
    /*vmulcaddc_params_size=*/0,
    /*gemm_parameters=*/&xnn_params.qu8.gemm,
    /*dwconv_ukernel=*/dwconv_ukernel,
    /*vmulcaddc_parameters=*/NULL,
    /*jit_gemm_params=*/NULL,
    /*linear_activation=*/false,
    /*relu_activation=*/false,
    /*datatype_init_flags=*/XNN_INIT_FLAG_QU8,
    /*operator_type=*/xnn_operator_type_convolution_nhwc_qu8,
    /*num_post_operations=*/0,
    /*post_operation_params=*/NULL,
    /*caches=*/caches,
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
    xnn_caches_t caches,
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
  if (requantization_scale >= 256.0f) {
    xnn_log_error(
      "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale: "
      "requantization scale %.7g is greater or equal to 256.0",
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
    /*log2_input_element_size=*/0,  // log2(sizeof(int8_t))
    /*log2_filter_element_size=*/0,  // log2(sizeof(int8_t))
    /*bias_element_size=*/sizeof(int32_t),
    (xnn_pack_vmulcaddc_w_fn) NULL,
    (xnn_pack_dwconv_hwg_w_fn) xnn_pack_qs8_dwconv_hwg_w,
    (xnn_pack_dwconv_ghw_w_fn) xnn_pack_qs8_dwconv_ghw_w,
    (xnn_pack_gemm_goi_w_fn) xnn_pack_qs8_gemm_goi_w,
    (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w,
    (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w,
    /*packing_params=*/&packing_params,
    /*input_padding_byte=*/input_zero_point,
    /*packed_weights_padding_byte=*/0,
    /*extra_weights_bytes=*/0,
    /*init_scale_params=*/NULL,
    /*scale_params=*/NULL,
    /*gemm_params=*/&gemm_params,
    /*gemm_params_size=*/sizeof(gemm_params),
    /*dwconv_params=*/&dwconv_params,
    /*dwconv_params_size=*/sizeof(dwconv_params),
    /*vmulcaddc_params=*/NULL,
    /*vmulcaddc_params_size=*/0,
    /*gemm_parameters=*/&xnn_params.qs8.gemm,
    /*dwconv_ukernel=*/dwconv_ukernel,
    /*vmulcaddc_parameters=*/NULL,
    /*jit_gemm_params=*/NULL,
    /*linear_activation=*/false,
    /*relu_activation=*/false,
    /*datatype_init_flags=*/XNN_INIT_FLAG_QS8,
    /*operator_type=*/xnn_operator_type_convolution_nhwc_qs8,
    /*num_post_operations=*/0,
    /*post_operation_params=*/NULL,
    /*caches=*/caches,
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
    xnn_caches_t caches,
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
    if (requantization_scale[output_channel] >= 256.0f) {
      xnn_log_error(
        "failed to create %s operator with %.7g input scale, %.7g kernel scale, and %.7g output scale in output channel #%zu: "
        "requantization scale %.7g is greater or equal to 256.0",
        xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_qc8),
        input_scale, kernel_scale[output_channel], output_scale,
        output_channel, requantization_scale[output_channel]);
      return xnn_status_unsupported_parameter;
    }
  }

  const struct xnn_qs8_packing_params packing_params = { .input_zero_point = input_zero_point, };

  union xnn_qc8_conv_minmax_params gemm_params;
  if XNN_LIKELY(xnn_params.qc8.gemm.init.qc8 != NULL) {
    xnn_params.qc8.gemm.init.qc8(&gemm_params,
      output_zero_point, output_min, output_max);
  }

  union xnn_qc8_conv_minmax_params dwconv_params;
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
    /*log2_input_element_size=*/0,  // log2(sizeof(int8_t))
    /*log2_filter_element_size=*/0,  // log2(sizeof(int8_t))
    /*bias_element_size=*/sizeof(int32_t),
    (xnn_pack_vmulcaddc_w_fn) NULL,
    (xnn_pack_dwconv_hwg_w_fn) xnn_pack_qs8_dwconv_hwg_w,
    (xnn_pack_dwconv_ghw_w_fn) xnn_pack_qs8_dwconv_ghw_w,
    (xnn_pack_gemm_goi_w_fn) xnn_pack_qs8_gemm_goi_w,
    (xnn_pack_conv_kgo_w_fn) xnn_pack_qs8_conv_kgo_w,
    (xnn_pack_conv_goki_w_fn) xnn_pack_qs8_conv_goki_w,
    /*packing_params=*/&packing_params,
    /*input_padding_byte=*/input_zero_point,
    /*packed_weights_padding_byte=*/0,
    /*extra_weights_bytes=*/sizeof(float),
    /*init_scale_params=*/xnn_init_qc8_scale_fp32_params,
    /*scale_params=*/requantization_scale,
    /*gemm_params=*/&gemm_params,
    /*gemm_params_size=*/sizeof(gemm_params),
    /*dwconv_params=*/&dwconv_params,
    /*dwconv_params_size=*/sizeof(dwconv_params),
    /*vmulcaddc_params=*/NULL,
    /*vmulcaddc_params_size=*/0,
    /*gemm_parameters=*/&xnn_params.qc8.gemm,
    /*dwconv_ukernel=*/dwconv_ukernel,
    /*vmulcaddc_parameters=*/NULL,
    /*jit_gemm_params=*/NULL,
    /*linear_activation=*/false,
    /*relu_activation=*/false,
    /*datatype_init_flags=*/XNN_INIT_FLAG_QC8,
    /*operator_type=*/xnn_operator_type_convolution_nhwc_qc8,
    /*num_post_operations=*/0,
    /*post_operation_params=*/NULL,
    /*caches=*/caches,
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
    xnn_caches_t caches,
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

  union xnn_f16_minmax_params gemm_params;
  if XNN_LIKELY(xnn_params.f16.gemm.init.f16 != NULL) {
    xnn_params.f16.gemm.init.f16(&gemm_params,
      fp16_output_min, fp16_output_max);
  }

  union xnn_f16_minmax_params dwconv_params;
  const struct dwconv_parameters* dwconv_ukernel =
    find_dwconv_ukernel(kernel_height * kernel_width, xnn_params.f16.dwconv, XNN_MAX_F16_DWCONV_UKERNELS);
  if XNN_LIKELY(dwconv_ukernel != NULL) {
    dwconv_ukernel->init.f16(&dwconv_params, fp16_output_min, fp16_output_max);
  }

  union xnn_f16_minmax_params vmulcaddc_params;
  if XNN_LIKELY(xnn_params.f16.vmulcaddc.init.f16 != NULL) {
    xnn_params.f16.vmulcaddc.init.f16(&vmulcaddc_params, fp16_output_min, fp16_output_max);
  }

  xnn_pack_vmulcaddc_w_fn pack_vmulcaddc_w = (xnn_pack_vmulcaddc_w_fn) xnn_pack_f16_vmulcaddc_w;
  xnn_pack_dwconv_hwg_w_fn pack_dwconv_hwg_w = (xnn_pack_dwconv_hwg_w_fn) xnn_pack_f16_dwconv_hwg_w;
  xnn_pack_dwconv_ghw_w_fn pack_dwconv_ghw_w = (xnn_pack_dwconv_ghw_w_fn) xnn_pack_f16_dwconv_ghw_w;
  xnn_pack_gemm_goi_w_fn pack_gemm_goi_w = (xnn_pack_gemm_goi_w_fn) xnn_pack_f16_gemm_goi_w;
  xnn_pack_conv_kgo_w_fn pack_conv_kgo_w = (xnn_pack_conv_kgo_w_fn) xnn_pack_f16_conv_kgo_w;
  xnn_pack_conv_goki_w_fn pack_conv_goki_w = (xnn_pack_conv_goki_w_fn) xnn_pack_f16_conv_goki_w;
  if (flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
    pack_vmulcaddc_w = (xnn_pack_vmulcaddc_w_fn) xnn_pack_f32_to_f16_vmulcaddc_w;
    pack_dwconv_hwg_w = (xnn_pack_dwconv_hwg_w_fn) xnn_pack_f32_to_f16_dwconv_hwg_w;
    pack_dwconv_ghw_w = (xnn_pack_dwconv_ghw_w_fn) xnn_pack_f32_to_f16_dwconv_ghw_w;
    pack_gemm_goi_w = (xnn_pack_gemm_goi_w_fn) xnn_pack_f32_to_f16_gemm_goi_w;
    pack_conv_kgo_w = (xnn_pack_conv_kgo_w_fn) xnn_pack_f32_to_f16_conv_kgo_w;
    pack_conv_goki_w = (xnn_pack_conv_goki_w_fn) xnn_pack_f32_to_f16_conv_goki_w;
  }

  return create_convolution2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    kernel_height, kernel_width,
    subsampling_height, subsampling_width,
    dilation_height, dilation_width,
    groups, group_input_channels, group_output_channels,
    input_channel_stride, output_channel_stride,
    kernel, bias, flags,
    /*log2_input_element_size=*/1,  // log2(sizeof(uint16_t))
    /*log2_filter_element_size=*/1,  // log2(sizeof(uint16_t))
    /*bias_element_size=*/sizeof(uint16_t),
    pack_vmulcaddc_w,
    pack_dwconv_hwg_w,
    pack_dwconv_ghw_w,
    pack_gemm_goi_w,
    pack_conv_kgo_w,
    pack_conv_goki_w,
    /*packing_params=*/NULL,
    /*input_padding_byte=*/0,
    /*packed_weights_padding_byte=*/0,
    /*extra_weights_bytes=*/0,
    /*init_scale_params=*/NULL,
    /*scale_params=*/NULL,
    /*gemm_params=*/&gemm_params,
    /*gemm_params_size=*/sizeof(gemm_params),
    /*dwconv_params=*/&dwconv_params,
    /*dwconv_params_size=*/sizeof(dwconv_params),
    /*vmulcaddc_params=*/&vmulcaddc_params,
    /*vmulcaddc_params_size=*/sizeof(vmulcaddc_params),
    /*gemm_parameters=*/&xnn_params.f16.gemm,
    /*dwconv_ukernel=*/dwconv_ukernel,
    /*vmulcaddc_parameters=*/&xnn_params.f16.vmulcaddc,
    /*jit_gemm_params=*/NULL,
    /*linear_activation=*/false,
    /*relu_activation=*/false,
    /*datatype_init_flags=*/XNN_INIT_FLAG_F16,
    /*operator_type=*/xnn_operator_type_convolution_nhwc_f16,
    /*num_post_operations=*/0,
    /*post_operation_params=*/NULL,
    /*caches=*/caches,
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
    xnn_caches_t caches,
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

  const struct gemm_parameters* gemm_parameters = &xnn_params.f32.gemm;
  if (gemm_parameters->nr > group_output_channels) {
    // Default micro-kernel is suboptimal. Try to find a better micro-kernel.

    if (xnn_params.f32.gemm2.minmax.igemm[gemm_parameters->mr].function[XNN_UARCH_DEFAULT] != NULL) {
      gemm_parameters = &xnn_params.f32.gemm2;
    }
  }

  union xnn_f32_minmax_params gemm_params;
  if XNN_LIKELY(gemm_parameters->init.f32 != NULL) {
    gemm_parameters->init.f32(&gemm_params, output_min, output_max);
  }

  struct jit_gemm_params jit_gemm_params = {
    .f32_minmax = {
      .min = output_min,
      .max = output_max
    }
  };

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
    /*log2_input_element_size=*/2,  // log2(sizeof(float))
    /*log2_filter_element_size=*/2,  // log2(sizeof(float))
    /*bias_element_size=*/sizeof(float),
    (xnn_pack_vmulcaddc_w_fn) xnn_pack_f32_vmulcaddc_w,
    (xnn_pack_dwconv_hwg_w_fn) xnn_pack_f32_dwconv_hwg_w,
    (xnn_pack_dwconv_ghw_w_fn) xnn_pack_f32_dwconv_ghw_w,
    (xnn_pack_gemm_goi_w_fn) xnn_pack_f32_gemm_goi_w,
    (xnn_pack_conv_kgo_w_fn) xnn_pack_f32_conv_kgo_w,
    (xnn_pack_conv_goki_w_fn) xnn_pack_f32_conv_goki_w,
    /*packing_params=*/NULL,
    /*input_padding_byte=*/0,
    /*packed_weights_padding_byte=*/0,
    /*extra_weights_bytes=*/0,
    /*init_scale_params=*/NULL,
    /*scale_params=*/NULL,
    /*gemm_params=*/&gemm_params,
    /*gemm_params_size=*/sizeof(gemm_params),
    /*dwconv_params=*/&dwconv_params,
    /*dwconv_params_size=*/sizeof(dwconv_params),
    /*vmulcaddc_params=*/&vmulcaddc_params,
    /*vmulcaddc_params_size=*/sizeof(vmulcaddc_params),
    /*gemm_parameters=*/gemm_parameters,
    /*dwconv_ukernel=*/dwconv_ukernel,
    /*vmulcaddc_parameters=*/&xnn_params.f32.vmulcaddc,
    /*jit_gemm_params=*/&jit_gemm_params,
    /*linear_activation=*/linear_activation,
    /*relu_activation=*/relu_activation,
    /*datatype_init_flags=*/XNN_INIT_FLAG_F32,
    /*operator_type=*/xnn_operator_type_convolution_nhwc_f32,
    /*num_post_operations=*/0,
    /*post_operation_params=*/NULL,
    /*caches=*/caches,
    convolution_op_out);
}

enum xnn_status xnn_create_fused_convolution2d_nhwc_f32(
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
    size_t num_post_operations,
    struct xnn_post_operation* post_operations,
    uint32_t flags,
    xnn_caches_t caches,
    xnn_operator_t* convolution_op_out)
{
  #if !XNN_ENABLE_JIT
    xnn_log_error(
      "failed to create %s operator: convolution with post operations available only if JIT is enabled",
      xnn_operator_type_to_string(xnn_operator_type_convolution_nhwc_f32));
    return xnn_status_invalid_parameter;
  #endif

  // Convolution is specified with linear activation, any clamping should be specified as a post operator.
  const float output_max = INFINITY;
  const float output_min = -INFINITY;

  struct jit_gemm_params jit_gemm_params = {
    .f32_minmax = {
      .min = output_min,
      .max = output_max
    },
    .num_post_operations = num_post_operations,
    .post_operations = post_operations,
  };

  char* post_operation_params = allocate_and_initialize_post_operation_params(num_post_operations, post_operations);

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
    /*log2_input_element_size=*/2,  // log2(sizeof(float))
    /*log2_filter_element_size=*/2,  // log2(sizeof(float))
    /*bias_element_size=*/sizeof(float),
    (xnn_pack_vmulcaddc_w_fn) xnn_pack_f32_vmulcaddc_w,
    (xnn_pack_dwconv_hwg_w_fn) xnn_pack_f32_dwconv_hwg_w,
    (xnn_pack_dwconv_ghw_w_fn) xnn_pack_f32_dwconv_ghw_w,
    (xnn_pack_gemm_goi_w_fn) xnn_pack_f32_gemm_goi_w,
    (xnn_pack_conv_kgo_w_fn) xnn_pack_f32_conv_kgo_w,
    (xnn_pack_conv_goki_w_fn) xnn_pack_f32_conv_goki_w,
    /*packing_params=*/NULL,
    /*input_padding_byte=*/0,
    /*packed_weights_padding_byte=*/0,
    /*extra_weights_bytes=*/0,
    /*init_scale_params=*/NULL,
    /*scale_params=*/NULL,
    /*gemm_params=*/(void*) &gemm_params,
    /*gemm_params_size=*/sizeof(gemm_params),
    /*dwconv_params=*/&dwconv_params,
    /*dwconv_params_size=*/sizeof(dwconv_params),
    /*vmulcaddc_params=*/&vmulcaddc_params,
    /*vmulcaddc_params_size=*/sizeof(vmulcaddc_params),
    /*gemm_parameters=*/&xnn_params.f32.gemm,
    /*dwconv_ukernel=*/dwconv_ukernel,
    /*vmulcaddc_parameters=*/&xnn_params.f32.vmulcaddc,
    /*jit_gemm_params=*/&jit_gemm_params,
    /*linear_activation=*/true,
    /*relu_activation=*/false,
    /*datatype_init_flags=*/XNN_INIT_FLAG_F32,
    /*operator_type=*/xnn_operator_type_convolution_nhwc_f32,
    /*num_post_operations=*/num_post_operations,
    /*post_operation_params=*/post_operation_params,
    /*caches=*/caches,
    convolution_op_out);
}

static enum xnn_status setup_gemm(
    xnn_operator_t convolution_op,
    uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size,
    uint32_t extra_weights_elements_size,
    uint32_t log2_output_element_size,
    size_t num_threads)
{
  // Convolution maps directly to GEMM and doesn't use indirection buffer.
  const size_t batch_size = convolution_op->batch_size;

  const size_t output_height = convolution_op->output_height;
  const size_t output_width = convolution_op->output_width;
  const size_t output_size = output_height * output_width;
  const size_t batch_output_size = batch_size * output_size;

  const size_t groups = convolution_op->groups;
  const size_t group_input_channels = convolution_op->group_input_channels;
  const size_t w_stride = extra_weights_elements_size +
    (round_up_po2(group_input_channels, convolution_op->ukernel.gemm.kr * convolution_op->ukernel.gemm.sr) << log2_filter_element_size);
  const size_t group_output_channels = convolution_op->group_output_channels;

  uint32_t mr = convolution_op->ukernel.gemm.mr;
  const uint32_t nr = convolution_op->ukernel.gemm.nr;
  struct xnn_hmp_gemm_ukernel *gemm_cases = convolution_op->ukernel.gemm.gemm_cases;

  #if XNN_ENABLE_GEMM_M_SPECIALIZATION
    mr = xnn_get_heuristic_mr_gemm(batch_output_size, mr, nr, gemm_cases, convolution_op->code_cache != NULL);
  #else
    if (batch_output_size == 1 && gemm_cases[0].function[XNN_UARCH_DEFAULT] != NULL) {
      mr = 1;
    }
  #endif

  #if XNN_PLATFORM_JIT
    if (convolution_op->code_cache != NULL) {
      const size_t jit_code_offset = gemm_cases[mr - 1].generated_code_offset[XNN_UARCH_DEFAULT];
      if (jit_code_offset != XNN_CACHE_NOT_FOUND) {
        gemm_cases[mr - 1].function[XNN_UARCH_DEFAULT] =
            (xnn_gemm_ukernel_fn) cached_code_at_offset(convolution_op, jit_code_offset);
        // TODO(zhin): different code generators for different uarch.
        #if XNN_MAX_UARCH_TYPES > 1
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            gemm_cases[mr - 1].function[i] =
                (xnn_gemm_ukernel_fn) cached_code_at_offset(convolution_op, jit_code_offset);
          }
        #endif
      }
    }
  #endif  // XNN_PLATFORM_JIT
  struct xnn_hmp_gemm_ukernel gemm_ukernel = gemm_cases[mr - 1];

  convolution_op->context.gemm = (struct gemm_context) {
      .k_scaled = group_input_channels << log2_input_element_size,
      .a = convolution_op->input,
      .a_stride = convolution_op->input_pixel_stride << log2_input_element_size,
      .packed_w = packed_weights(convolution_op),
      .w_stride = w_stride,
      .wg_stride = w_stride * round_up(group_output_channels, nr),
      .c = convolution_op->output,
      .cm_stride = convolution_op->output_pixel_stride << log2_output_element_size,
      .cn_stride = nr << log2_output_element_size,
      .cg_stride = group_output_channels << log2_output_element_size,
      .log2_csize = log2_output_element_size,
      .ukernel = gemm_ukernel,
  };
  memcpy(&convolution_op->context.gemm.params, &convolution_op->params, sizeof(convolution_op->context.gemm.params));
  if (convolution_op->num_post_operation_params == 0) {
    convolution_op->context.gemm.fused_params = &convolution_op->context.gemm.params;
  } else {
    convolution_op->context.gemm.fused_params = convolution_op->post_operation_params;
  }

  #if XNN_TEST_MODE
    const size_t nc = nr;
  #else
    size_t nc = group_output_channels;
    if (num_threads > 1) {
      const size_t num_other_tiles = groups * divide_round_up(batch_output_size, mr);
      const size_t target_tiles_per_thread = 5;
      const size_t max_nc = divide_round_up(group_output_channels * num_other_tiles, num_threads * target_tiles_per_thread);
      if (max_nc < nc) {
        nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
      }
    }
  #endif
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

static enum xnn_status setup_igemm(
    xnn_operator_t convolution_op,
    uint32_t log2_input_element_size,
    uint32_t log2_filter_element_size,
    uint32_t extra_weights_elements_size,
    uint32_t log2_output_element_size,
    size_t num_threads)
{
  const size_t batch_size = convolution_op->batch_size;
  const size_t input_height = convolution_op->input_height;
  const size_t input_width = convolution_op->input_width;
  const size_t groups = convolution_op->groups;
  const size_t kernel_height = convolution_op->kernel_height;
  const size_t kernel_width = convolution_op->kernel_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t output_height = convolution_op->output_height;
  const size_t output_width = convolution_op->output_width;
  const size_t output_size = output_height * output_width;

  uint32_t mr = convolution_op->ukernel.igemm.mr;
  const uint32_t nr = convolution_op->ukernel.igemm.nr;
  struct xnn_hmp_igemm_ukernel* igemm_cases = convolution_op->ukernel.igemm.igemm_cases;

  #if XNN_ENABLE_GEMM_M_SPECIALIZATION
    mr = xnn_get_heuristic_mr_igemm(output_size, mr, nr, igemm_cases, convolution_op->code_cache != NULL);
  #else
    if (output_size == 1 && igemm_cases[0].function[XNN_UARCH_DEFAULT] != NULL) {
      mr = 1;
    }
  #endif

  #if XNN_PLATFORM_JIT
    if (convolution_op->code_cache != NULL) {
      const size_t jit_code_offset = igemm_cases[mr - 1].generated_code_offset[XNN_UARCH_DEFAULT];
      if (jit_code_offset != XNN_CACHE_NOT_FOUND) {
        igemm_cases[mr - 1].function[XNN_UARCH_DEFAULT] =
            (xnn_igemm_ukernel_fn) cached_code_at_offset(convolution_op, jit_code_offset);
        // TODO(zhin): different code generators for different uarch.
        #if XNN_MAX_UARCH_TYPES > 1
          for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
            igemm_cases[mr - 1].function[i] =
                (xnn_igemm_ukernel_fn) cached_code_at_offset(convolution_op, jit_code_offset);
          }
        #endif
      }
    }
  #endif  // XNN_PLATFORM_JIT
  struct xnn_hmp_igemm_ukernel igemm_ukernel = igemm_cases[mr - 1];

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
    convolution_op->last_input = convolution_op->input;
    convolution_op->last_input_height = input_height;
    convolution_op->last_input_width = input_width;
    xnn_log_debug("allocated %zu bytes for indirection buffer in %s operator",
      indirection_buffer_size, xnn_operator_type_to_string(convolution_op->type));

    xnn_indirection_init_conv2d(convolution_op, mr, log2_input_element_size);
  }

  const size_t group_input_channels = convolution_op->group_input_channels;
  const size_t w_stride = extra_weights_elements_size +
    (round_up_po2(group_input_channels, convolution_op->ukernel.igemm.kr * convolution_op->ukernel.igemm.sr) * kernel_size << log2_filter_element_size);
  const size_t group_output_channels = convolution_op->group_output_channels;
  convolution_op->context.igemm = (struct igemm_context) {
      .ks = kernel_size,
      .ks_scaled = kernel_size * mr * sizeof(void*),
      .kc = group_input_channels << log2_input_element_size,
      .w_stride = w_stride,
      .indirect_a = convolution_op->indirection_buffer,
      .a_offset = (size_t) ((uintptr_t) convolution_op->input - (uintptr_t) convolution_op->last_input),
      .zero = convolution_op->zero_buffer,
      .packed_w = packed_weights(convolution_op),
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

  #if XNN_TEST_MODE
    const size_t nc = nr;
  #else
    size_t nc = group_output_channels;
    if (num_threads > 1) {
      const size_t num_other_tiles = groups * batch_size * divide_round_up(output_size, mr);
      const size_t target_tiles_per_thread = 5;
      const size_t max_nc = divide_round_up(group_output_channels * num_other_tiles, num_threads * target_tiles_per_thread);
      if (max_nc < nc) {
        nc = min(nc, divide_round_up(nc, max_nc * nr) * nr);
      }
    }
  #endif
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

static enum xnn_status setup_dwconv(
    xnn_operator_t convolution_op,
    uint32_t log2_input_element_size,
    uint32_t log2_output_element_size,
    size_t num_threads)
{
  const size_t input_height = convolution_op->input_height;
  const size_t input_width = convolution_op->input_width;
  const size_t kernel_height = convolution_op->kernel_height;
  const size_t kernel_width = convolution_op->kernel_width;
  const size_t kernel_size = kernel_height * kernel_width;
  const size_t output_height = convolution_op->output_height;
  const size_t output_width = convolution_op->output_width;
  const size_t step_width = convolution_op->dilation_width == 1 ?
      min(convolution_op->stride_width, kernel_width) : kernel_width;
  const size_t step_height = kernel_size + (output_width - 1) * step_width * kernel_height;
  const size_t primary_tile = convolution_op->ukernel.dwconv.primary_tile;
  if (input_height != convolution_op->last_input_height || input_width != convolution_op->last_input_width) {
    // Micro-kernel will read (primary_tile - kernel_size) elements after the end of indirection buffer.
    const size_t indirection_buffer_size =
      sizeof(void*) * (primary_tile - kernel_size + output_height * step_height);

    const void** indirection_buffer =
      (const void**) xnn_reallocate_memory(convolution_op->indirection_buffer, indirection_buffer_size);
    if (indirection_buffer == NULL) {
      xnn_log_error("failed to allocate %zu bytes for %s operator indirection buffer",
        indirection_buffer_size, xnn_operator_type_to_string(convolution_op->type));
      return xnn_status_out_of_memory;
    }
    convolution_op->indirection_buffer = indirection_buffer;
    xnn_log_debug("allocated %zu bytes for indirection buffer in %s operator",
      indirection_buffer_size, xnn_operator_type_to_string(convolution_op->type));

    #if XNN_TEST_MODE
      memset(convolution_op->indirection_buffer, 0, indirection_buffer_size);
    #endif

    xnn_indirection_init_dwconv2d(convolution_op, step_height, step_width, primary_tile, log2_input_element_size);

    #if XNN_TEST_MODE
      for (size_t i = 0; i < indirection_buffer_size / sizeof(void*); i++) {
        // Indirection initialization should have set all indirection pointers, make sure none of them are NULL.
        assert(convolution_op->indirection_buffer[i] != NULL);
      }
    #endif

    convolution_op->last_input = convolution_op->input;
    convolution_op->last_input_height = input_height;
    convolution_op->last_input_width = input_width;
  }

  const size_t groups = convolution_op->groups;
  convolution_op->context.dwconv = (struct dwconv_context) {
      .indirect_input = convolution_op->indirection_buffer,
      .indirect_input_width_stride = kernel_height * step_width * sizeof(void*),
      .indirect_input_height_stride = step_height * sizeof(void*),
      .input_offset = (size_t) ((uintptr_t) convolution_op->input - (uintptr_t) convolution_op->last_input),
      .input_batch_stride = (input_height * input_width * convolution_op->input_pixel_stride) << log2_input_element_size,
      .packed_weights = packed_weights(convolution_op),
      .output = convolution_op->output,
      .output_batch_stride = (output_height * output_width * convolution_op->output_pixel_stride) << log2_output_element_size,
      .output_height_stride = (output_width * convolution_op->output_pixel_stride) << log2_output_element_size,
      .output_width = output_width,
      .groups = groups,
      .zero = convolution_op->zero_buffer,
      .output_increment = (convolution_op->output_pixel_stride - groups) << log2_output_element_size,
      .unipass_ukernel = convolution_op->ukernel.dwconv.unipass_fn,
  };
  memcpy(&convolution_op->context.dwconv.params, &convolution_op->params, sizeof(convolution_op->context.dwconv.params));

  convolution_op->compute.type = xnn_parallelization_type_2d;
  convolution_op->compute.task_2d = (pthreadpool_task_2d_t) xnn_compute_dwconv_unipass;
  convolution_op->compute.range[0] = convolution_op->batch_size;
  convolution_op->compute.range[1] = output_height;
  convolution_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

static enum xnn_status setup_vmulcaddc(
  xnn_operator_t convolution_op,
  uint32_t log2_input_element_size,
  uint32_t log2_output_element_size,
  size_t num_threads)
{
  const size_t batch_output_size = convolution_op->batch_size * convolution_op->output_height * convolution_op->output_width;

  convolution_op->context.vmulcaddc = (struct vmulcaddc_context) {
    .n = convolution_op->groups << log2_input_element_size,
    .x = convolution_op->input,
    .x_stride = convolution_op->input_pixel_stride << log2_input_element_size,
    .w = packed_weights(convolution_op),
    .y = convolution_op->output,
    .y_stride = convolution_op->output_pixel_stride << log2_output_element_size,
    .ukernel = convolution_op->ukernel.vmulcaddc.function,
  };
  memcpy(&convolution_op->context.vmulcaddc.params, &convolution_op->params,
         sizeof(convolution_op->context.vmulcaddc.params));

#if XNN_TEST_MODE
  const size_t mc = convolution_op->ukernel.vmulcaddc.mr;
#else
  size_t mc = batch_output_size;
  if (num_threads > 1) {
    const size_t target_tiles_per_thread = 5;
    const size_t max_mc = divide_round_up(batch_output_size, num_threads * target_tiles_per_thread);
    if (max_mc < mc) {
      const uint32_t mr = convolution_op->ukernel.vmulcaddc.mr;
      mc = min(mc, divide_round_up(mc, max_mc * mr) * mr);
    }
  }
#endif
  convolution_op->compute.type = xnn_parallelization_type_1d_tile_1d;
  convolution_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_vmulcaddc;
  convolution_op->compute.range[0] = batch_output_size;
  convolution_op->compute.tile[0] = mc;
  convolution_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

static enum xnn_status setup_convolution2d_nhwc(
  xnn_operator_t convolution_op,
  enum xnn_operator_type expected_operator_type,
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
  if (convolution_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_invalid_parameter;
  }
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

  if (convolution_op->weights_cache != NULL && !xnn_weights_cache_is_finalized(convolution_op->weights_cache)) {
    xnn_log_error("failed to setup %s operator: weights cache is not finalized",
      xnn_operator_type_to_string(convolution_op->type));
    return xnn_status_invalid_state;
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
    convolution_op->output_height = xnn_compute_convolution_output_dimension(
        convolution_op->padding_top + input_height + convolution_op->padding_bottom,
        convolution_op->kernel_height,
        convolution_op->dilation_height,
        convolution_op->stride_height);
    convolution_op->output_width = xnn_compute_convolution_output_dimension(
        convolution_op->padding_left + input_width + convolution_op->padding_right,
        convolution_op->kernel_width,
        convolution_op->dilation_width,
        convolution_op->stride_width);
  }
  convolution_op->output = output;

  switch (convolution_op->ukernel.type) {
    case xnn_microkernel_type_gemm:
      return setup_gemm(
          convolution_op,
          log2_input_element_size, log2_filter_element_size, extra_weights_elements_size, log2_output_element_size,
          num_threads);
    case xnn_microkernel_type_igemm:
      return setup_igemm(
          convolution_op,
          log2_input_element_size, log2_filter_element_size, extra_weights_elements_size, log2_output_element_size,
          num_threads);
    case xnn_microkernel_type_dwconv:
      return setup_dwconv(
          convolution_op,
          log2_input_element_size, log2_output_element_size,
          num_threads);
    case xnn_microkernel_type_vmulcaddc:
      return setup_vmulcaddc(
          convolution_op,
          log2_input_element_size, log2_output_element_size,
          num_threads);
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
  return setup_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_qu8,
    batch_size, input_height, input_width,
    input, output,
    XNN_INIT_FLAG_QU8,
    /*log2_input_element_size=*/0,  // log2(sizeof(uint8_t))
    /*log2_filter_element_size=*/0,  // log2(sizeof(uint8_t))
    /*extra_weights_elements_size=*/sizeof(int32_t),
    /*log2_output_element_size=*/0,  // log2(sizeof(uint8_t))
    /*num_threads=*/pthreadpool_get_threads_count(threadpool));
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
  return setup_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_qs8,
    batch_size, input_height, input_width,
    input, output,
    XNN_INIT_FLAG_QS8,
    /*log2_input_element_size=*/0,  // log2(sizeof(int8_t))
    /*log2_filter_element_size=*/0,  // log2(sizeof(int8_t))
    /*extra_weights_elements_size=*/sizeof(int32_t),
    /*log2_output_element_size=*/0,  // log2(sizeof(int8_t))
    /*num_threads=*/pthreadpool_get_threads_count(threadpool));
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
  return setup_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_qc8,
    batch_size, input_height, input_width,
    input, output,
    XNN_INIT_FLAG_QC8,
    /*log2_input_element_size=*/0,  // log2(sizeof(int8_t))
    /*log2_filter_element_size=*/0,  // log2(sizeof(int8_t))
    /*extra_weights_elements_size=*/sizeof(int32_t) + sizeof(float),
    /*log2_output_element_size=*/0,  // log2(sizeof(int8_t))
    /*num_threads=*/pthreadpool_get_threads_count(threadpool));
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
  return setup_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_f16,
    batch_size, input_height, input_width,
    input, output,
    XNN_INIT_FLAG_F16,
    /*log2_input_element_size=*/1,  // log2(sizeof(uint16_t))
    /*log2_filter_element_size=*/1,  // log2(sizeof(uint16_t))
    /*extra_weights_elements_size=*/sizeof(uint16_t),
    /*log2_output_element_size=*/1,  // log2(sizeof(uint16_t))
    /*num_threads=*/pthreadpool_get_threads_count(threadpool));
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
  return setup_convolution2d_nhwc(
    convolution_op, xnn_operator_type_convolution_nhwc_f32,
    batch_size, input_height, input_width,
    input, output,
    XNN_INIT_FLAG_F32,
    /*log2_input_element_size=*/2,  // log2(sizeof(float))
    /*log2_filter_element_size=*/2,  // log2(sizeof(float))
    /*extra_weights_elements_size=*/sizeof(float),
    /*log2_output_element_size=*/2,  // log2(sizeof(float))
    /*num_threads=*/pthreadpool_get_threads_count(threadpool));
}
