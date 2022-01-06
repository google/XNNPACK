// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/operator.h>
#include <xnnpack/log.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/params.h>
#include <xnnpack/compute.h>


void xnn_compute_grouped_gemm(
    const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t group_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t k_scaled  = context->k_scaled;
  const size_t a_stride  = context->a_stride;
  const size_t cm_stride = context->cm_stride;

  context->ukernel.function[XNN_UARCH_DEFAULT](
      mr_block_size,
      nr_block_size,
      k_scaled,
      (const void*) ((uintptr_t) context->a + mr_block_start * a_stride + group_index * k_scaled),
      a_stride,
      (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride + group_index * context->wg_stride),
      (void*) ((uintptr_t) context->c + mr_block_start * cm_stride + (nr_block_start << context->log2_csize) + group_index * context->cg_stride),
      cm_stride,
      context->cn_stride,
      &context->params);
}

void xnn_compute_gemm(
    const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t a_stride  = context->a_stride;
  const size_t cm_stride = context->cm_stride;

  context->ukernel.function[XNN_UARCH_DEFAULT](
      mr_block_size,
      nr_block_size,
      context->k_scaled,
      (const void*) ((uintptr_t) context->a + mr_block_start * a_stride),
      a_stride,
      (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride),
      (void*) ((uintptr_t) context->c + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
      cm_stride,
      context->cn_stride,
      &context->params);
}

void xnn_compute_spmm(
    const struct spmm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t mr_block_start,
    size_t mr_block_size)
{
  context->ukernel(
      mr_block_size,
      context->n,
      (const void*) ((uintptr_t) context->input + batch_index * context->batched_input_stride + mr_block_start),
      context->nonzero_weights,
      context->input_increments,
      context->output_channel_nonzeros,
      (void*) ((uintptr_t) context->output + batch_index * context->batched_output_stride + mr_block_start),
      context->scaled_m,
      &context->params);
}

void xnn_compute_grouped_batch_igemm(
    const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t group_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t ks        = context->ks;
  const size_t cm_stride = context->cm_stride;

  context->ukernel.function[XNN_UARCH_DEFAULT](
      mr_block_size,
      nr_block_size,
      context->kc,
      context->ks_scaled,
      (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
      (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride + group_index * context->gw_stride),
      (void*) ((uintptr_t) context->c + group_index * context->gc_stride + batch_index * context->bc_stride + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
      cm_stride,
      context->cn_stride,
      context->a_offset + group_index * context->ga_stride + batch_index * context->ba_stride,
      context->zero,
      &context->params);
}

void xnn_compute_grouped_igemm(
    const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t group_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t ks        = context->ks;
  const size_t cm_stride = context->cm_stride;

  context->ukernel.function[XNN_UARCH_DEFAULT](
      mr_block_size,
      nr_block_size,
      context->kc,
      context->ks_scaled,
      (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
      (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride + group_index * context->gw_stride),
      (void*) ((uintptr_t) context->c + group_index * context->gc_stride + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
      cm_stride,
      context->cn_stride,
      context->a_offset + group_index * context->ga_stride,
      context->zero,
      &context->params);
}

void xnn_compute_batch_igemm(
    const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t ks        = context->ks;
  const size_t cm_stride = context->cm_stride;

  context->ukernel.function[XNN_UARCH_DEFAULT](
      mr_block_size,
      nr_block_size,
      context->kc,
      context->ks_scaled,
      (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
      (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride),
      (void*) ((uintptr_t) context->c + batch_index * context->bc_stride + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
      cm_stride,
      context->cn_stride,
      context->a_offset + batch_index * context->ba_stride,
      context->zero,
      &context->params);
}

void xnn_compute_igemm(
    const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t ks        = context->ks;
  const size_t cm_stride = context->cm_stride;

  context->ukernel.function[XNN_UARCH_DEFAULT](
      mr_block_size,
      nr_block_size,
      context->kc,
      context->ks_scaled,
      (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
      (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride),
      (void*) ((uintptr_t) context->c + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
      cm_stride,
      context->cn_stride,
      context->a_offset,
      context->zero,
      &context->params);
}

void xnn_compute_grouped_subgemm2d(
      const struct subgemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t group_index,
      size_t subkernel_index,
      size_t slice_y,
      size_t slice_x_start,
      size_t nc_block_start,
      size_t slice_x_max,
      size_t nc_block_size)
{
  const struct subconvolution_params* subconvolution_params = &context->subconvolution_params[subkernel_index];

  if XNN_UNLIKELY(slice_y >= subconvolution_params->slice_height) {
    return;
  }

  const size_t slice_width = subconvolution_params->slice_width;
  if XNN_UNLIKELY(slice_x_start >= slice_width) {
    return;
  }
  const size_t slice_x_size = min(slice_x_max, slice_width - slice_x_start);

  const size_t ax_stride = context->ax_stride;
  const size_t cx_stride = context->cx_stride;
  context->ukernel.function[XNN_UARCH_DEFAULT](
      slice_x_size,
      nc_block_size,
      context->kc,
      (const void*) ((uintptr_t) context->a + group_index * context->ga_stride + slice_y * context->ay_stride + slice_x_start * ax_stride + batch_index * context->ba_stride),
      ax_stride,
      (const void*) ((uintptr_t) subconvolution_params->weights + nc_block_start * subconvolution_params->w_stride + group_index * context->gw_stride),
      (void*) ((uintptr_t) subconvolution_params->output + group_index * context->gc_stride + slice_y * context->cy_stride + slice_x_start * cx_stride + batch_index * context->bc_stride + (nc_block_start << context->log2_csize)),
      cx_stride,
      context->cn_stride,
      &context->params);
}

void xnn_compute_subgemm2d(
      const struct subgemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t subkernel_index,
      size_t slice_y,
      size_t slice_x_start,
      size_t nc_block_start,
      size_t slice_x_max,
      size_t nc_block_size)
{
  const struct subconvolution_params* subconvolution_params = &context->subconvolution_params[subkernel_index];

  if XNN_UNLIKELY(slice_y >= subconvolution_params->slice_height) {
    return;
  }

  const size_t slice_width = subconvolution_params->slice_width;
  if XNN_UNLIKELY(slice_x_start >= slice_width) {
    return;
  }
  const size_t slice_x_size = min(slice_x_max, slice_width - slice_x_start);

  const size_t ax_stride = context->ax_stride;
  const size_t cx_stride = context->cx_stride;
  context->ukernel.function[XNN_UARCH_DEFAULT](
      slice_x_size,
      nc_block_size,
      context->kc,
      (const void*) ((uintptr_t) context->a + slice_y * context->ay_stride + slice_x_start * ax_stride + batch_index * context->ba_stride),
      ax_stride,
      (const void*) ((uintptr_t) subconvolution_params->weights + nc_block_start * subconvolution_params->w_stride),
      (void*) ((uintptr_t) subconvolution_params->output + slice_y * context->cy_stride + slice_x_start * cx_stride + batch_index * context->bc_stride + (nc_block_start << context->log2_csize)),
      cx_stride,
      context->cn_stride,
      &context->params);
}

void xnn_compute_grouped_subconv2d(
      const struct subconv_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t group_index,
      size_t subkernel_index,
      size_t slice_y,
      size_t slice_x_start,
      size_t nc_block_start,
      size_t slice_x_max,
      size_t nc_block_size)
{
  const struct subconvolution_params* subconvolution_params = &context->subconvolution_params[subkernel_index];

  if XNN_UNLIKELY(slice_y >= subconvolution_params->slice_height) {
    return;
  }

  const size_t slice_width = subconvolution_params->slice_width;
  if XNN_UNLIKELY(slice_x_start >= slice_width) {
    return;
  }
  const size_t slice_x_size = min(slice_x_max, slice_width - slice_x_start);

  const size_t cx_stride = context->cx_stride;
  context->ukernel.function[XNN_UARCH_DEFAULT](
      slice_x_size,
      nc_block_size,
      context->kc,
      subconvolution_params->scaled_kernel_size,
      (const void**) ((uintptr_t) subconvolution_params->indirection_buffer + slice_y * subconvolution_params->indirection_y_stride + slice_x_start * subconvolution_params->indirection_x_stride),
      (const void*) ((uintptr_t) subconvolution_params->weights + nc_block_start * subconvolution_params->w_stride + group_index * context->gw_stride),
      (void*) ((uintptr_t) subconvolution_params->output + group_index * context->gc_stride + slice_y * context->cy_stride + slice_x_start * cx_stride + batch_index * context->bc_stride + (nc_block_start << context->log2_csize)),
      cx_stride,
      context->cn_stride,
      context->a_offset + group_index * context->ga_stride + batch_index * context->ba_stride,
      context->zero,
      &context->params);
}

void xnn_compute_subconv2d(
      const struct subconv_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t subkernel_index,
      size_t slice_y,
      size_t slice_x_start,
      size_t nc_block_start,
      size_t slice_x_max,
      size_t nc_block_size)
{
  const struct subconvolution_params* subconvolution_params = &context->subconvolution_params[subkernel_index];

  if XNN_UNLIKELY(slice_y >= subconvolution_params->slice_height) {
    return;
  }

  const size_t slice_width = subconvolution_params->slice_width;
  if XNN_UNLIKELY(slice_x_start >= slice_width) {
    return;
  }
  const size_t slice_x_size = min(slice_x_max, slice_width - slice_x_start);

  const size_t cx_stride = context->cx_stride;
  context->ukernel.function[XNN_UARCH_DEFAULT](
      slice_x_size,
      nc_block_size,
      context->kc,
      subconvolution_params->scaled_kernel_size,
      (const void**) ((uintptr_t) subconvolution_params->indirection_buffer + slice_y * subconvolution_params->indirection_y_stride + slice_x_start * subconvolution_params->indirection_x_stride),
      (const void*) ((uintptr_t) subconvolution_params->weights + nc_block_start * subconvolution_params->w_stride),
      (void*) ((uintptr_t) subconvolution_params->output + slice_y * context->cy_stride + slice_x_start * cx_stride + batch_index * context->bc_stride + (nc_block_start << context->log2_csize)),
      cx_stride,
      context->cn_stride,
      context->a_offset + batch_index * context->ba_stride,
      context->zero,
      &context->params);
}

void xnn_compute_conv2d_hwc2chw(
      const struct conv2d_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t output_y_start,
      size_t output_y_slice)
{
  context->hwc2chw_ukernel(
      context->input_height,
      context->input_width,
      output_y_start,
      output_y_start + output_y_slice,
      (const void*) ((uintptr_t) context->input + batch_index * context->input_batch_stride),
      context->zero,
      context->packed_weights,
      (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride),
      context->input_padding_top,
      context->output_channels,
      context->output_height_stride,
      context->output_channel_stride,
      &context->params);
}

void xnn_compute_dwconv_unipass(
    const struct dwconv_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input + output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  context->unipass_ukernel(
    context->groups, context->output_width,
    indirect_input, context->packed_weights, output,
    context->indirect_input_width_stride, context->output_increment,
    input_offset, context->zero,
    &context->params);
}

void xnn_compute_dwconv2d_chw(
    const struct dwconv2d_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t channel)
{
  context->chw_ukernel(
    context->input_height,
    context->input_width,
    (const void*) ((uintptr_t) context->input + channel * context->input_channel_stride + batch_index * context->input_batch_stride),
    (const void*) ((uintptr_t) context->packed_weights + channel * context->weights_channel_stride),
    context->zero,
    (void*) ((uintptr_t) context->output + channel * context->output_channel_stride + batch_index * context->output_batch_stride),
    context->input_padding_top,
    &context->params);
}

void xnn_compute_depthtospace2d_hwc_contiguous(
    const struct depthtospace2d_hwc_context* context,
    size_t batch_input_y,
    size_t input_x,
    size_t block_y)
{
  const size_t input_width = context->input_width;
  const size_t elements = context->elements;
  const void* input = (const void*) ((uintptr_t) context->input +
    (batch_input_y * input_width + input_x) * context->input_width_stride + block_y * elements);
  void* output = (void*) ((uintptr_t) context->output +
    ((batch_input_y * context->block_size + block_y) * input_width + input_x) * elements);

  context->ukernel(
    elements,
    input,
    output,
    NULL);
}

void xnn_compute_depthtospace2d_hwc_strided(
    const struct depthtospace2d_hwc_context* context,
    size_t batch_input_y,
    size_t input_x,
    size_t block_y,
    size_t block_x)
{
  const size_t block_size = context->block_size;
  const size_t elements = context->elements;
  const void* input = (const void*) ((uintptr_t) context->input +
    batch_input_y * context->input_height_stride + input_x * context->input_width_stride + (block_y * block_size + block_x) * elements);
  void* output = (void*) ((uintptr_t) context->output +
    (batch_input_y * block_size + block_y) * context->output_height_stride +
    (input_x * block_size + block_x) * context->output_width_stride);

  context->ukernel(
    elements,
    input,
    output,
    NULL);
}

void xnn_compute_depthtospace2d_chw2hwc(
    const struct depthtospace2d_chw2hwc_context* context,
    size_t batch_index)
{
  context->ukernel(
    context->output_channels,
    context->input_height,
    context->input_width,
    context->block_size,
    (const void*) ((uintptr_t) context->input + batch_index * context->input_batch_stride),
    (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride),
    context->output_channel_stride);
}

void xnn_compute_argmax_pooling_unipass(
    const struct argmax_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input = (const void**) ((uintptr_t) context->indirect_input +
    output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);
  uint32_t* index = (uint32_t*) ((uintptr_t) context->index +
    batch_index * context->index_batch_stride + output_y * context->index_height_stride);

  context->unipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, output, index,
    context->input_increment, context->output_increment);
}

void xnn_compute_argmax_pooling_multipass(
    const struct argmax_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input = (const void**) ((uintptr_t) context->indirect_input +
    output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);
  uint32_t* index = (uint32_t*) ((uintptr_t) context->index +
    batch_index * context->index_batch_stride + output_y * context->index_height_stride);

  void* multipass_accumulation_buffer = XNN_SIMD_ALLOCA(context->channels * sizeof(float) + XNN_EXTRA_BYTES);
  void* multipass_index_buffer = XNN_SIMD_ALLOCA(context->channels * sizeof(uint32_t) + XNN_EXTRA_BYTES);

  context->multipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, multipass_accumulation_buffer, multipass_index_buffer, output, index,
    context->input_increment, context->output_increment);
}

void xnn_compute_max_pooling(
    const struct max_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input = (const void**) ((uintptr_t) context->indirect_input +
    output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  context->ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, output,
    context->input_increment, context->output_increment,
    &context->params);
}

void xnn_compute_unpooling(
    const struct unpooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t input_y,
    size_t input_x)
{
  const void* input = (const void*) ((uintptr_t) context->input +
      input_y * context->input_height_stride + input_x * context->input_width_stride);
  const uint32_t* index = (const uint32_t*) ((uintptr_t) context->index +
      input_y * context->index_height_stride + input_x * context->index_width_stride);
  void** indirect_output =
    (void**) ((uintptr_t) context->indirect_output +
      input_y * context->indirect_output_height_stride + input_x * context->indirect_output_width_stride);

  context->ukernel(
    context->pooling_size,
    context->channels,
    context->fill_value,
    input, index, indirect_output);
}

void xnn_compute_average_pooling_unipass(
    const struct average_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input + output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  context->unipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, context->zero, output,
    context->input_increment, context->output_increment,
    &context->params);
}

void xnn_compute_average_pooling_multipass(
    const struct average_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input + output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  void* multipass_buffer =
    XNN_SIMD_ALLOCA(context->channels * sizeof(int32_t) + XNN_EXTRA_BYTES * sizeof(int32_t) / sizeof(uint8_t));

  context->multipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, context->zero, multipass_buffer, output,
    context->input_increment, context->output_increment,
    &context->params);
}

void xnn_compute_pixelwise_average_pooling_unipass(
    const struct pixelwise_average_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input + output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  const void* pixelwise_buffer =
    (const void*) ((uintptr_t) context->pixelwise_buffer + output_y * context->pixelwise_buffer_height_stride);
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  context->unipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, context->zero, pixelwise_buffer, output,
    context->input_increment, context->output_increment,
    &context->params);
}

void xnn_compute_pixelwise_average_pooling_multipass(
    const struct pixelwise_average_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input + output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  const void* pixelwise_buffer =
    (const void*) ((uintptr_t) context->pixelwise_buffer + output_y * context->pixelwise_buffer_height_stride);
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  void* multipass_buffer = XNN_SIMD_ALLOCA(context->channels * sizeof(int32_t) + XNN_EXTRA_BYTES * sizeof(int32_t) / sizeof(uint8_t));

  context->multipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, context->zero, pixelwise_buffer, multipass_buffer, output,
    context->input_increment, context->output_increment,
    &context->params);
}

void xnn_compute_global_average_pooling_nwc_unipass(
    const struct global_average_pooling_nwc_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index)
{
  const void* input =
    (const void*) ((uintptr_t) context->input + batch_index * context->input_batch_stride);
  void* output =
    (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride);

  context->unipass_ukernel(
    context->input_elements,
    context->channels,
    input,
    context->input_pixel_stride,
    context->zero,
    output,
    &context->params);
}

void xnn_compute_global_average_pooling_nwc_multipass(
    const struct global_average_pooling_nwc_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index)
{
  const void* input =
    (const void*) ((uintptr_t) context->input + batch_index * context->input_batch_stride);
  void* output =
    (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride);

  void* multipass_buffer =
    XNN_SIMD_ALLOCA(context->channels * sizeof(int32_t) + XNN_EXTRA_BYTES * sizeof(int32_t) / sizeof(uint8_t));

  context->multipass_ukernel(
    context->input_elements,
    context->channels,
    input,
    context->input_pixel_stride,
    context->zero,
    multipass_buffer,
    output,
    &context->params);
}

void xnn_compute_global_average_pooling_ncw(
    const struct global_average_pooling_ncw_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t channels_start,
    size_t channels_slice)
{
  const void* input = (const void*) ((uintptr_t) context->input +
    channels_start * context->input_channel_stride + batch_index * context->input_batch_stride);
  void* output = (void*) ((uintptr_t) context->output +
    channels_start * context->output_channel_stride + batch_index * context->output_batch_stride);

  context->ukernel(
    context->input_elements,
    channels_slice,
    input,
    output,
    &context->params);
}

void xnn_compute_resize_bilinear(
    const struct resize_bilinear_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t pixel_start,
    size_t pixel_range)
{
  void* output =
    (void*) ((uintptr_t) context->output + pixel_start * context->output_pixel_stride + batch_index * context->output_batch_stride);

  context->ukernel(
    pixel_range,
    context->scaled_channels,
    context->indirect_input + pixel_start * 4,
    context->input_offset + batch_index * context->input_batch_stride,
    (const void*) ((uintptr_t) context->packed_weights + (pixel_start << context->log2_wsize)),
    output,
    context->output_pixel_stride - context->scaled_channels);
}

void xnn_compute_resize_bilinear_chw(
    const struct resize_bilinear_chw_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t channel_start,
    size_t channel_range)
{
  void* output =
    (void*) ((uintptr_t) context->output + channel_start * context->output_channel_stride + batch_index * context->output_batch_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride + channel_start * context->input_channel_stride;

  context->ukernel(
    context->output_pixels,
    channel_range,
    context->indirect_input,
    input_offset,
    context->packed_weights,
    output,
    context->input_channel_stride);
}

void xnn_compute_prelu(
    const struct prelu_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_start,
    size_t batch_range)
{
  const size_t x_stride = context->x_stride;
  const size_t y_stride = context->y_stride;
  const void* x = (const void*) ((uintptr_t) context->x + x_stride * batch_start);
  void* y = (void*) ((uintptr_t) context->y + y_stride * batch_start);

  context->ukernel(batch_range, context->n, x, x_stride, context->w, y, y_stride);
}

void xnn_compute_pad_5d(
    const struct pad_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j, size_t k, size_t l, size_t m)
{
  const void* input = (const void*) ((uintptr_t) context->input +
    i * context->input_stride[4] + j * context->input_stride[3] + k * context->input_stride[2] + l * context->input_stride[1] + m * context->input_stride[0]);
  void* output = (void*) ((uintptr_t) context->output +
    i * context->output_stride[4] + j * context->output_stride[3] + k * context->output_stride[2] + l * context->output_stride[1] + m * context->output_stride[0]);

  const size_t i_padding = context->pre_paddings[5];
  const size_t j_padding = context->pre_paddings[4];
  const size_t k_padding = context->pre_paddings[3];
  const size_t l_padding = context->pre_paddings[2];
  const size_t m_padding = context->pre_paddings[1];

  const size_t i_size = context->input_size[5];
  const size_t j_size = context->input_size[4];
  const size_t k_size = context->input_size[3];
  const size_t l_size = context->input_size[2];
  const size_t m_size = context->input_size[1];

  if XNN_LIKELY(i - i_padding < i_size && j - j_padding < j_size && k - k_padding < k_size &&
                l - l_padding < l_size && m - m_padding < m_size)
  {
    context->pad_ukernel(
      1 /* rows */,
      context->input_size[0], context->pre_paddings[0], context->post_paddings[0],
      input, 0 /* input stride */, output, 0 /* output stride */,
      context->padding_value);
  } else {
    context->fill_ukernel(1 /* rows */, context->output_size[0], output, 0 /* output stride */, context->padding_value);
  }
}

void xnn_compute_elementwise_binary_5d(
    const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j, size_t k, size_t l, size_t m)
{
  const void* a = (const void*) ((uintptr_t) context->a +
    i * context->a_stride[0] + j * context->a_stride[1] + k * context->a_stride[2] + l * context->a_stride[3] + m * context->a_stride[4]);
  const void* b = (const void*) ((uintptr_t) context->b +
    i * context->b_stride[0] + j * context->b_stride[1] + k * context->b_stride[2] + l * context->b_stride[3] + m * context->b_stride[4]);
  void* y = (void*) ((uintptr_t) context->y +
    i * context->y_stride[0] + j * context->y_stride[1] + k * context->y_stride[2] + l * context->y_stride[3] + m * context->y_stride[4]);
  context->ukernel(context->elements, a, b, y, &context->params);
}

void xnn_compute_channel_shuffle_fixed(
    const struct channel_shuffle_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t index)
{
  const void* x = (const void*) ((uintptr_t) context->x + index * context->x_stride);
  void* y = (void*) ((uintptr_t) context->y + index * context->y_stride);

  context->fixed_ukernel(context->n, x, y);
}

void xnn_compute_channel_shuffle_variable(
    const struct channel_shuffle_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t index)
{
  const void* x = (const void*) ((uintptr_t) context->x + index * context->x_stride);
  void* y = (void*) ((uintptr_t) context->y + index * context->y_stride);

  context->variable_ukernel(context->n, context->m, x, y);
}

void xnn_compute_lut_strided(
    const struct lut_strided_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index)
{
  const void* x = (const void*) ((uintptr_t) context->x + context->x_stride * batch_index);
  void* y = (void*) ((uintptr_t) context->y + context->y_stride * batch_index);

  context->ukernel(context->n, x, y, context->t);
}

void xnn_compute_lut_contiguous(
    const struct lut_contiguous_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t offset,
    size_t size)
{
  const void* x = (const void*) ((uintptr_t) context->x + offset);
  void* y = (void*) ((uintptr_t) context->y + offset);

  context->ukernel(size, x, y, context->t);
}

void xnn_compute_univector_strided(
    const struct univector_strided_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t batch_range)
{
  const size_t x_stride = context->x_stride;
  const size_t y_stride = context->y_stride;

  const void* x = (const void*) ((uintptr_t) context->x + x_stride * batch_index);
  void* y = (void*) ((uintptr_t) context->y + y_stride * batch_index);
  do {
    context->ukernel(context->n, x, y, &context->params);
    x = (const void*) ((uintptr_t) x + x_stride);
    y = (void*) ((uintptr_t) y + y_stride);
  } while (--batch_range != 0);
}

void xnn_compute_univector_contiguous(
    const struct univector_contiguous_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t offset,
    size_t size)
{
  const uint32_t log2_xsize = context->log2_xsize;
  const uint32_t log2_ysize = context->log2_ysize;
  const void* x = (const void*) ((uintptr_t) context->x + offset);
  void* y = (void*) ((uintptr_t) context->y + ((offset >> log2_xsize) << log2_ysize));
  context->ukernel(size, x, y, &context->params);
}

void xnn_compute_u8_softmax(
    const struct u8_softmax_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index)
{
  const uint8_t* x = (const uint8_t*) ((uintptr_t) context->x + context->x_stride * batch_index);
  uint8_t* y = (uint8_t*) ((uintptr_t) context->y + context->y_stride * batch_index);
  const size_t n = context->n;

  uint8_t x_max = 0;
  context->rmax_ukernel(n, x, &x_max);
  const size_t adjustment = x_max ^ 255;
  const uint32_t* t = (const uint32_t*) context->t + adjustment;
  context->lut_norm_ukernel(n, x, t, y);
}

void xnn_compute_f32_three_pass_softmax(
    const struct f32_three_pass_softmax_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index)
{
  const float* x = (const float*) ((uintptr_t) context->x + context->x_stride * batch_index);
  float* y = (float*) ((uintptr_t) context->y + context->y_stride * batch_index);
  const size_t n = context->n;

  // First pass: reduce-max
  float x_max;
  context->rmax_ukernel(n, x, &x_max);

  // Second pass: reduce-add & store exp(x-x_max)
  float y_sum;
  context->raddstoreexpminusmax_ukernel(n, x, &x_max, y, &y_sum, &context->expminus_params);

  // Third pass: scale y
  const float y_scale = 1.0f / y_sum;
  context->vmulc_ukernel(n, y, &y_scale, y, &context->minmax_params);
}

void xnn_compute_vmulcaddc(
    const struct vmulcaddc_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_start,
    size_t batch_size)
{
  const size_t x_stride = context->x_stride;
  const size_t y_stride = context->y_stride;

  const void* x = (const void*) ((uintptr_t) context->x + x_stride * batch_start);
  void* y = (void*) ((uintptr_t) context->y + y_stride * batch_start);

  context->ukernel(
    batch_size,
    context->n,
    x, x_stride,
    context->w,
    y, y_stride,
    &context->params);
}

#if XNN_MAX_UARCH_TYPES > 1
  void xnn_compute_hmp_grouped_gemm(
      const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      uint32_t uarch_index,
      size_t group_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size)
  {
    const size_t k_scaled  = context->k_scaled;
    const size_t a_stride  = context->a_stride;
    const size_t cm_stride = context->cm_stride;

    context->ukernel.function[uarch_index](
        mr_block_size,
        nr_block_size,
        k_scaled,
        (const void*) ((uintptr_t) context->a + mr_block_start * a_stride + group_index * k_scaled),
        a_stride,
        (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride + group_index * context->wg_stride),
        (void*) ((uintptr_t) context->c + mr_block_start * cm_stride + (nr_block_start << context->log2_csize) + group_index * context->cg_stride),
        cm_stride,
        context->cn_stride,
        &context->params);
  }

  void xnn_compute_hmp_gemm(
      const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      uint32_t uarch_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size)
  {
    const size_t a_stride  = context->a_stride;
    const size_t cm_stride = context->cm_stride;

    context->ukernel.function[uarch_index](
        mr_block_size,
        nr_block_size,
        context->k_scaled,
        (const void*) ((uintptr_t) context->a + mr_block_start * a_stride),
        a_stride,
        (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride),
        (void*) ((uintptr_t) context->c + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
        cm_stride,
        context->cn_stride,
        &context->params);
  }

  void xnn_compute_hmp_grouped_batch_igemm(
      const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      uint32_t uarch_index,
      size_t batch_index,
      size_t group_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size)
  {
    const size_t ks        = context->ks;
    const size_t cm_stride = context->cm_stride;

    context->ukernel.function[uarch_index](
        mr_block_size,
        nr_block_size,
        context->kc,
        context->ks_scaled,
        (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
        (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride + group_index * context->gw_stride),
        (void*) ((uintptr_t) context->c + group_index * context->gc_stride + batch_index * context->bc_stride + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
        cm_stride,
        context->cn_stride,
        context->a_offset + group_index * context->ga_stride + batch_index * context->ba_stride,
        context->zero,
        &context->params);
  }

  void xnn_compute_hmp_grouped_igemm(
      const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      uint32_t uarch_index,
      size_t group_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size)
  {
    const size_t ks        = context->ks;
    const size_t cm_stride = context->cm_stride;

    context->ukernel.function[uarch_index](
        mr_block_size,
        nr_block_size,
        context->kc,
        context->ks_scaled,
        (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
        (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride + group_index * context->gw_stride),
        (void*) ((uintptr_t) context->c + group_index * context->gc_stride + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
        cm_stride,
        context->cn_stride,
        context->a_offset + group_index * context->ga_stride,
        context->zero,
        &context->params);
  }

  void xnn_compute_batch_hmp_igemm(
      const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      uint32_t uarch_index,
      size_t batch_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size)
  {
    const size_t ks        = context->ks;
    const size_t cm_stride = context->cm_stride;

    context->ukernel.function[uarch_index](
        mr_block_size,
        nr_block_size,
        context->kc,
        context->ks_scaled,
        (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
        (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride),
        (void*) ((uintptr_t) context->c + batch_index * context->bc_stride + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
        cm_stride,
        context->cn_stride,
        context->a_offset + batch_index * context->ba_stride,
        context->zero,
        &context->params);
  }

  void xnn_compute_hmp_igemm(
      const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      uint32_t uarch_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size)
  {
    const size_t ks        = context->ks;
    const size_t cm_stride = context->cm_stride;

    context->ukernel.function[uarch_index](
        mr_block_size,
        nr_block_size,
        context->kc,
        context->ks_scaled,
        (const void**) ((uintptr_t) context->indirect_a + mr_block_start * ks * sizeof(void*)),
        (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride),
        (void*) ((uintptr_t) context->c + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
        cm_stride,
        context->cn_stride,
        context->a_offset,
        context->zero,
        &context->params);
  }
#endif  // XNN_MAX_UARCH_TYPES > 1

enum xnn_status xnn_run_operator(xnn_operator_t op, pthreadpool_t threadpool)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to run operator: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }
  switch (op->state) {
    case xnn_run_state_invalid:
      xnn_log_error("failed to run operator: operator was not successfully setup");
      return xnn_status_invalid_state;
    case xnn_run_state_ready:
      break;
    case xnn_run_state_skip:
      return xnn_status_success;
  }

  uint32_t flags = PTHREADPOOL_FLAG_DISABLE_DENORMALS;
  if (op->flags & XNN_FLAG_YIELD_WORKERS) {
    flags |= PTHREADPOOL_FLAG_YIELD_WORKERS;
  }
  switch (op->compute.type) {
    case xnn_parallelization_type_invalid:
      break;
    case xnn_parallelization_type_1d:
      assert(op->compute.range[0] != 0);
      pthreadpool_parallelize_1d(
          threadpool,
          op->compute.task_1d,
          &op->context,
          op->compute.range[0],
          flags);
      break;
    case xnn_parallelization_type_1d_tile_1d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.tile[0] != 0);
      pthreadpool_parallelize_1d_tile_1d(
          threadpool,
          op->compute.task_1d_tile_1d,
          &op->context,
          op->compute.range[0],
          op->compute.tile[0],
          flags);
      break;
    case xnn_parallelization_type_2d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      pthreadpool_parallelize_2d(
          threadpool,
          op->compute.task_2d,
          &op->context,
          op->compute.range[0], op->compute.range[1],
          flags);
      break;
    case xnn_parallelization_type_2d_tile_1d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.tile[0] != 0);
      pthreadpool_parallelize_2d_tile_1d(
          threadpool,
          op->compute.task_2d_tile_1d,
          &op->context,
          op->compute.range[0], op->compute.range[1],
          op->compute.tile[0],
          flags);
      break;
    case xnn_parallelization_type_2d_tile_2d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_2d_tile_2d(
          threadpool,
          op->compute.task_2d_tile_2d,
          &op->context,
          op->compute.range[0], op->compute.range[1],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
    case xnn_parallelization_type_3d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      pthreadpool_parallelize_3d(
          threadpool,
          op->compute.task_3d,
          &op->context,
          op->compute.range[0], op->compute.range[1], op->compute.range[2],
          flags);
      break;
    case xnn_parallelization_type_3d_tile_2d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_3d_tile_2d(
          threadpool,
          op->compute.task_3d_tile_2d,
          &op->context,
          op->compute.range[0], op->compute.range[1], op->compute.range[2],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
    case xnn_parallelization_type_4d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.range[3] != 0);
      pthreadpool_parallelize_4d(
          threadpool,
          op->compute.task_4d,
          &op->context,
          op->compute.range[0], op->compute.range[1], op->compute.range[2], op->compute.range[3],
          flags);
      break;
    case xnn_parallelization_type_4d_tile_2d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.range[3] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_4d_tile_2d(
          threadpool,
          op->compute.task_4d_tile_2d,
          &op->context,
          op->compute.range[0], op->compute.range[1], op->compute.range[2], op->compute.range[3],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
    case xnn_parallelization_type_5d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.range[3] != 0);
      assert(op->compute.range[4] != 0);
      pthreadpool_parallelize_5d(
          threadpool,
          op->compute.task_5d,
          &op->context,
          op->compute.range[0], op->compute.range[1], op->compute.range[2], op->compute.range[3], op->compute.range[4],
          flags);
      break;
    case xnn_parallelization_type_5d_tile_2d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.range[3] != 0);
      assert(op->compute.range[4] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_5d_tile_2d(
          threadpool,
          op->compute.task_5d_tile_2d,
          &op->context,
          op->compute.range[0], op->compute.range[1], op->compute.range[2], op->compute.range[3], op->compute.range[4],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
    case xnn_parallelization_type_6d_tile_2d:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.range[3] != 0);
      assert(op->compute.range[4] != 0);
      assert(op->compute.range[5] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_6d_tile_2d(
          threadpool,
          op->compute.task_6d_tile_2d,
          &op->context,
          op->compute.range[0], op->compute.range[1], op->compute.range[2], op->compute.range[3], op->compute.range[4], op->compute.range[5],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
#if XNN_MAX_UARCH_TYPES > 1
    case xnn_parallelization_type_2d_tile_2d_with_uarch:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_2d_tile_2d_with_uarch(
          threadpool,
          op->compute.task_2d_tile_2d_with_id,
          &op->context,
          0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
          op->compute.range[0], op->compute.range[1],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
    case xnn_parallelization_type_3d_tile_2d_with_uarch:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_3d_tile_2d_with_uarch(
          threadpool,
          op->compute.task_3d_tile_2d_with_id,
          &op->context,
          0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
          op->compute.range[0], op->compute.range[1], op->compute.range[2],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
    case xnn_parallelization_type_4d_tile_2d_with_uarch:
      assert(op->compute.range[0] != 0);
      assert(op->compute.range[1] != 0);
      assert(op->compute.range[2] != 0);
      assert(op->compute.range[3] != 0);
      assert(op->compute.tile[0] != 0);
      assert(op->compute.tile[1] != 0);
      pthreadpool_parallelize_4d_tile_2d_with_uarch(
          threadpool,
          op->compute.task_4d_tile_2d_with_id,
          &op->context,
          0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
          op->compute.range[0], op->compute.range[1], op->compute.range[2], op->compute.range[3],
          op->compute.tile[0], op->compute.tile[1],
          flags);
      break;
#endif  // XNN_MAX_UARCH_TYPES > 1
    default:
      XNN_UNREACHABLE;
  }
  return xnn_status_success;
}
