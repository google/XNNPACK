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

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/compute.h"
#include "xnnpack/config-types.h"
#include "xnnpack/indirection.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microkernel-type.h"
#include "xnnpack/microparams.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/packq.h"
#include "xnnpack/quantization.h"
#include "pthreadpool.h"

void xnn_compute_transposec_2d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t tile_i,
    size_t tile_j)
{
  const size_t ld_input = context->input_stride[1];
  const size_t ld_output = context->output_stride[0];
  context->const_size_ukernel(
      (const void*) ((uintptr_t) context->x + i * context->input_stride[0] + j * context->input_stride[1]),
      (void*) ((uintptr_t) context->y + j * context->output_stride[1] + i * context->output_stride[0]),
      ld_input,
      ld_output,
      tile_i,
      tile_j);
}

void xnn_compute_transposec_3d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t tile_j,
    size_t tile_k)
{
  const size_t ld_input = context->input_stride[2];
  const size_t ld_output = context->output_stride[1];
  const void* x = (const void*) ((uintptr_t) context->x +
                                 i * context->input_stride[0] + j * context->input_stride[1] + k * context->input_stride[2]);
  void* y = (void*) ((uintptr_t) context->y + i * context->output_stride[0] + j * context->output_stride[1] +
                     k * context->output_stride[2]);

  context->const_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      tile_j,
      tile_k);
}

void xnn_compute_transposec_4d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t tile_k,
    size_t tile_l)
{
  const size_t ld_input = context->input_stride[3];
  const size_t ld_output = context->output_stride[2];
  const void* x = (const void*) ((uintptr_t)context->x + i * context->input_stride[0] + j * context->input_stride[1] +
                                 k * context->input_stride[2] + l * context->input_stride[3]);
  void* y = (void*) ((uintptr_t)context->y + i * context->output_stride[0] + j * context->output_stride[1] +
                     k * context->output_stride[2] + l * context->output_stride[3]);

  context->const_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      tile_k,
      tile_l);
}

void xnn_compute_transposec_5d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t m,
    size_t tile_l,
    size_t tile_m)
{
  const size_t ld_input = context->input_stride[4];
  const size_t ld_output = context->output_stride[3];
  const void* x = (const void*)((uintptr_t)context->x + i * context->input_stride[0] + j * context->input_stride[1] +
                                 k * context->input_stride[2] + l * context->input_stride[3] + m * context->input_stride[4]);
  void* y = (void*)((uintptr_t)context->y + i * context->output_stride[0] + j * context->output_stride[1] +
                     k * context->output_stride[2] + l * context->output_stride[3] + m * context->output_stride[4]);

  context->const_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      tile_l,
      tile_m);
}

void xnn_compute_transposec_6d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t m,
    size_t n,
    size_t tile_m,
    size_t tile_n)
{
  const size_t ld_input = context->input_stride[5];
  const size_t ld_output = context->output_stride[4];
  const void* x = (const void*)((uintptr_t)context->x + i * context->input_stride[0] + j * context->input_stride[1] +
                                 k * context->input_stride[2] + l * context->input_stride[3] +
                                 m * context->input_stride[4] + n * context->input_stride[5]);
  void* y = (void*)((uintptr_t)context->y + i * context->output_stride[0] + j * context->output_stride[1] +
                     k * context->output_stride[2] + l * context->output_stride[3] + m * context->output_stride[4] +
                     n * context->output_stride[5]);

  context->const_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      tile_m,
      tile_n);
}

void xnn_compute_transposev_2d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t tile_i,
    size_t tile_j)
{
  const size_t element_size = context->output_stride[1];
  const size_t ld_input = context->input_stride[1];
  const size_t ld_output = context->output_stride[0];
  const void* x = (const void*) ((uintptr_t) context->x +
                                 i * context->input_stride[0] + j * context->input_stride[1]);
  void* y = (void*) ((uintptr_t) context->y + context->output_stride[1] * j + i * context->output_stride[0]);

  context->variable_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      context->input_stride[0],
      context->output_stride[1],
      element_size,
      tile_i,
      tile_j);
}

void xnn_compute_transposev_3d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t tile_j,
    size_t tile_k)
{
  const size_t element_size = context->output_stride[2];
  const size_t ld_input = context->input_stride[2];
  const size_t ld_output = context->output_stride[1];
  const void* x = (const void*)((uintptr_t)context->x + i * context->input_stride[0] + j * context->input_stride[1] +
                                 k * context->input_stride[2]);
  void* y = (void*)((uintptr_t)context->y + i * context->output_stride[0] + j * context->output_stride[1] +
                     k * context->output_stride[2]);

  context->variable_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      context->input_stride[1],
      context->output_stride[2],
      element_size,
      tile_j,
      tile_k);
}

void xnn_compute_transposev_4d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t tile_k,
    size_t tile_l)
{
  const size_t element_size = context->output_stride[3];
  const size_t ld_input = context->input_stride[3];
  const size_t ld_output = context->output_stride[2];
  const void* x = (const void*)((uintptr_t)context->x + i * context->input_stride[0] + j * context->input_stride[1] +
                                 k * context->input_stride[2] + l * context->input_stride[3]);
  void* y = (void*)((uintptr_t)context->y + context->output_stride[3] * l + i * context->output_stride[0] +
                     j * context->output_stride[1] + k * context->output_stride[2]);

  context->variable_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      context->input_stride[2],
      context->output_stride[3],
      element_size,
      tile_k,
      tile_l);
}

void xnn_compute_transposev_5d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t m,
    size_t tile_l,
    size_t tile_m)
{
  const size_t element_size = context->output_stride[4];
  const size_t ld_input = context->input_stride[4];
  const size_t ld_output = context->output_stride[3];
  const void* x = (const void*)((uintptr_t)context->x + i * context->input_stride[0] + j * context->input_stride[1] +
                                 k * context->input_stride[2] + l * context->input_stride[3] + m * context->input_stride[4]);
  void* y = (void*)((uintptr_t)context->y + context->output_stride[4] * m + i * context->output_stride[0] +
                     j * context->output_stride[1] + k * context->output_stride[2] + l * context->output_stride[3]);

  context->variable_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      context->input_stride[3],
      context->output_stride[4],
      element_size,
      tile_l,
      tile_m);
}

void xnn_compute_transposev_6d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t m,
    size_t n,
    size_t tile_m,
    size_t tile_n)
{
  const size_t element_size = context->output_stride[5];
  const size_t ld_input = context->input_stride[5];
  const size_t ld_output = context->output_stride[4];
  const void* x = (const void*)((uintptr_t)context->x + i * context->input_stride[0] + j * context->input_stride[1] +
                                 k * context->input_stride[2] + l * context->input_stride[3] +
                                 m * context->input_stride[4] + n * context->input_stride[5]);
  void* y = (void*)((uintptr_t)context->y + context->output_stride[5] * n + i * context->output_stride[0] +
                     j * context->output_stride[1] + k * context->output_stride[2] + l * context->output_stride[3] +
                     m * context->output_stride[4]);

  context->variable_size_ukernel(
      x,
      y,
      ld_input,
      ld_output,
      context->input_stride[4],
      context->output_stride[5],
      element_size,
      tile_m,
      tile_n);
}

void xnn_compute_packw_gemm_gio(
    const struct packw_gemm_gio_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t n_block_start,
    size_t n_block_size)
{
  const void* kernel = (const void*) ((uintptr_t) context->kernel + n_block_start * context->n_stride);
  const void* bias = context->bias;
  if (bias != NULL) {
    bias = (const void*) ((uintptr_t) bias + n_block_start * context->b_stride);
  }
  void* packed_weights = (void*) ((uintptr_t) context->packed_weights + n_block_start * context->w_stride);

  context->packw_gemm_gio(
      /*groups=*/1, n_block_size, context->kc, context->nr, context->kr,
      context->sr, context->k_stride_elements, kernel, bias, /*scale=*/NULL,
      packed_weights, /*extra_bytes=*/0, /*params=*/NULL);
}

void xnn_compute_batched_packw_gemm_gio(
    const struct packw_gemm_gio_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t n_block_start,
    size_t n_block_size)
{
  const void* kernel = (const void*) ((uintptr_t) context->kernel + n_block_start * context->n_stride +
                                      batch_index * context->gk_stride);
  const void* bias = context->bias;
  if (bias != NULL) {
    bias = (const void*) ((uintptr_t) bias + n_block_start * context->b_stride +
                          batch_index * context->gb_stride);
  }
  void* packed_weights = (void*) ((uintptr_t) context->packed_weights + n_block_start * context->w_stride +
                                  batch_index * context->gc_stride);

  context->packw_gemm_gio(
      /*groups=*/1, n_block_size, context->kc, context->nr, context->kr,
      context->sr, context->k_stride_elements, kernel, bias, /*scale=*/NULL,
      packed_weights, /*extra_bytes=*/0, /*params=*/NULL);
}

void xnn_compute_packw_gemm_goi(
    const struct packw_gemm_goi_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t n_block_start,
    size_t n_block_size)
{
  const void* kernel = (const void*) ((const uintptr_t) context->kernel + context->k_stride * n_block_start);
  const void* bias = context->bias;
  if (bias != NULL) {
    bias = (const void*) ((const uintptr_t) bias + (n_block_start * context->b_stride));
  }
  void* packed_weights = (void*) ((uintptr_t) context->packed_weights + context->w_stride * n_block_start);

  context->packw_gemm_goi(
      /*groups=*/1, n_block_size, context->kc, context->nr, context->kr,
      context->sr, kernel, bias, /*scale=*/NULL, packed_weights,
      /*extra_bytes=*/0, /*params=*/NULL);
}

void xnn_compute_batched_packw_gemm_goi(
    const struct packw_gemm_goi_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t n_block_start,
    size_t n_block_size)
{
  const void* kernel = (const void*) ((uintptr_t) context->kernel + context->k_stride * n_block_start +
                                      batch_index * context->gk_stride);
  const void* bias = context->bias;
  if (bias != NULL) {
    bias = (const void*) ((uintptr_t) bias + n_block_start * context->b_stride +
                          batch_index * context->gb_stride);
  }
  void* packed_weights = (void*) ((uintptr_t) context->packed_weights + context->w_stride * n_block_start +
                                  batch_index * context->gc_stride);

  context->packw_gemm_goi(
      /*groups=*/1, n_block_size, context->kc, context->nr, context->kr,
      context->sr, kernel, bias, /*scale=*/NULL, packed_weights,
      /*extra_bytes=*/0, /*params=*/NULL);
}

void xnn_compute_hmp_grouped_gemm(
    const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    uint32_t uarch_index, size_t group_index, size_t mr_block_start,
    size_t nr_block_start, size_t mr_block_size, size_t nr_block_size) {
  const size_t k_scaled  = context->k_scaled;
  const size_t a_stride  = context->a_stride;
  const size_t cm_stride = context->cm_stride;
  const size_t num_batch_dims = context->num_batch_dims;
  const size_t group_index_c = group_index;

  // Compute the group index offsets into A and B.
  size_t group_index_a = 0;
  size_t group_index_b = 0;
  for (int k = 0; k < num_batch_dims; k++) {
    // Extract the kth batch index from the group_index.
    const size_t index = group_index / context->batch_strides_c[k];
    group_index %= context->batch_strides_c[k];

    // Compute the corresponding kth group index offsets into A and B.
    group_index_a = (index % context->batch_dims_a[k]) +
                    context->batch_dims_a[k] * group_index_a;
    group_index_b = (index % context->batch_dims_b[k]) +
                    context->batch_dims_b[k] * group_index_b;
  }
  if (context->quantization_params != NULL) {
    // If the effective `mr_block_size` is smaller than the kernel's `mr`,
    // create a padded copy of the dynamic quantization params.
    const struct xnn_qd8_quantization_params* quantization_params =
        &context->quantization_params[group_index_a * context->gq_stride +
                                      mr_block_start];
    struct xnn_qd8_quantization_params padded_quantization_params[XNN_MAX_MR];
    if (mr_block_size < context->mr) {
      memcpy(padded_quantization_params, quantization_params,
             mr_block_size * sizeof(struct xnn_qd8_quantization_params));
      for (size_t i = mr_block_size; i < context->mr; i++) {
        padded_quantization_params[i] =
            padded_quantization_params[mr_block_size - 1];
      }
      quantization_params = padded_quantization_params;
    };

    context->dq_ukernel.function[uarch_index](
        mr_block_size, nr_block_size, k_scaled,
        (const void*)((uintptr_t)context->a + mr_block_start * a_stride +
                      group_index_a * context->ga_stride),
        a_stride,
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride +
                      group_index_b * context->gw_stride),
        (void*)((uintptr_t)context->c + mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize) +
                group_index_c * context->gc_stride),
        cm_stride, context->cn_stride, &context->params, quantization_params);
  } else {
    context->ukernel.function[uarch_index](
        mr_block_size, nr_block_size, k_scaled,
        (const void*)((uintptr_t)context->a + mr_block_start * a_stride +
                      group_index_a * context->ga_stride),
        a_stride,
        (const void*)((uintptr_t)context->packed_w +
                      nr_block_start * context->w_stride +
                      group_index_b * context->gw_stride),
        (void*)((uintptr_t)context->c + mr_block_start * cm_stride +
                (nr_block_start << context->log2_csize) +
                group_index_c * context->gc_stride),
        cm_stride, context->cn_stride, &context->params);
  }
}

void xnn_compute_grouped_gemm(
    const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t group_index, size_t mr_block_start, size_t nr_block_start,
    size_t mr_block_size, size_t nr_block_size) {
  xnn_compute_hmp_grouped_gemm(context, XNN_UARCH_DEFAULT, group_index,
                               mr_block_start, nr_block_start, mr_block_size,
                               nr_block_size);
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
      context->fused_params);
}

void xnn_compute_dqgemm(
    const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t a_stride  = context->a_stride;
  const size_t cm_stride = context->cm_stride;

  context->dq_ukernel.function[XNN_UARCH_DEFAULT](
      mr_block_size,
      nr_block_size,
      context->k_scaled,
      (const void*) ((uintptr_t) context->a + mr_block_start * a_stride),
      a_stride,
      (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride),
      (void*) ((uintptr_t) context->c + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
      cm_stride,
      context->cn_stride,
      context->fused_params,
      (const void*) ((uintptr_t) &context->quantization_params[mr_block_start]));
}

void xnn_compute_hmp_qp8gemm(
    const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    uint32_t uarch_index, size_t mr_block_start, size_t nr_block_start,
    size_t mr_block_size, size_t nr_block_size) {
  const size_t a_offset = xnn_x8_packq_f32qp8_packed_offset(
      mr_block_start, context->k_scaled, context->mr, context->kr, context->sr);
  const size_t cm_stride = context->cm_stride;

  context->qp8_ukernel.function[uarch_index](
      mr_block_size, nr_block_size, context->k_scaled,
      (const void*)((uintptr_t)context->a + a_offset),
      (const void*)((uintptr_t)context->packed_w +
                    nr_block_start * context->w_stride),
      (void*)((uintptr_t)context->c + mr_block_start * cm_stride +
              (nr_block_start << context->log2_csize)),
      cm_stride,
      /*dst_stride_col=*/sizeof(float), context->fused_params);
}

void xnn_compute_qp8gemm(
    const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t mr_block_start, size_t nr_block_start, size_t mr_block_size,
    size_t nr_block_size) {
  xnn_compute_hmp_qp8gemm(context, XNN_UARCH_DEFAULT, mr_block_start,
                          nr_block_start, mr_block_size, nr_block_size);
}

void xnn_compute_hmp_dqgemm_bl(
    const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    uint32_t uarch_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t a_stride  = context->a_stride;
  const size_t cm_stride = context->cm_stride;

  context->dq_bl_ukernel.function[uarch_index](
      mr_block_size,
      nr_block_size,
      context->k_scaled,
      (const void*) ((uintptr_t) context->a + mr_block_start * a_stride),
      a_stride,
      (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride),
      (void*) ((uintptr_t) context->c + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
      cm_stride,
      context->cn_stride,
      context->fused_params,
     (const void*) ((uintptr_t) &context->quantization_params[mr_block_start]));
}

void xnn_compute_dqgemm_bl(
    const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  return xnn_compute_hmp_dqgemm_bl(context, /*uarch_index=*/0, mr_block_start, nr_block_start, mr_block_size, nr_block_size);
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

void xnn_compute_dq_zero_buffer_igemm(
    const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index
    ) {
  memset(context->zero_buffers[batch_index], context->quantization_params[batch_index].zero_point, context->zero_size);
}

void xnn_compute_dq_zero_buffer_subconv(
    const struct subconv_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index
    ) {
  memset(context->zero_buffers[batch_index], context->quantization_params[batch_index].zero_point, context->zero_size);
}

void xnn_compute_grouped_batch_dqigemm(
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

  context->dq_ukernel.function[XNN_UARCH_DEFAULT](
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
      context->zero_buffers[batch_index],
      &context->params,
      (const void*) ((uintptr_t) &context->quantization_params[batch_index]));
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

void xnn_compute_grouped_dqigemm(
    const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t group_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t ks        = context->ks;
  const size_t cm_stride = context->cm_stride;

  context->dq_ukernel.function[XNN_UARCH_DEFAULT](
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
      context->zero_buffers[0],
      &context->params,
      (const void*) ((uintptr_t) context->quantization_params));
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

void xnn_compute_batch_dqigemm(
    const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t ks        = context->ks;
  const size_t cm_stride = context->cm_stride;

  context->dq_ukernel.function[XNN_UARCH_DEFAULT](
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
      context->zero_buffers[batch_index],
      &context->params,
      (const void*) ((uintptr_t) &context->quantization_params[batch_index]));
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

void xnn_compute_dqigemm(
    const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t mr_block_start,
    size_t nr_block_start,
    size_t mr_block_size,
    size_t nr_block_size)
{
  const size_t ks        = context->ks;
  const size_t cm_stride = context->cm_stride;

  context->dq_ukernel.function[XNN_UARCH_DEFAULT](
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
      context->zero_buffers[0],
      &context->params,
      (const void*) ((uintptr_t) &context->quantization_params[/*mr_block_start=*/0]));
}
// `output_tile_start` should be a multiple of igemm.mr (tile size).
void xnn_compute_conv2d_igemm_indirection(
    const struct conv2d_igemm_indirection_init_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t output_tile_start,
    size_t output_tile_size)
{
  xnn_indirection_init_conv2d(
    output_tile_size,
    output_tile_start,
    output_tile_start + output_tile_size,
    context->indirection_buffer,
    context->input,
    context->zero_buffer,
    context->input_pixel_stride,
    context->input_height, context->input_width,
    context->output_height, context->output_width,
    context->kernel_height, context->kernel_width,
    context->stride_height, context->stride_width,
    context->dilation_height, context->dilation_width,
    context->input_padding_top, context->input_padding_left);
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

void xnn_compute_grouped_dqsubconv2d(
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
  context->dq_ukernel.function[XNN_UARCH_DEFAULT](
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
      context->zero_buffers[batch_index],
      &context->params,
      (const void*) ((uintptr_t) &context->quantization_params[batch_index]));
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

void xnn_compute_dqsubconv2d(
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
  context->dq_ukernel.function[XNN_UARCH_DEFAULT](
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
      context->zero_buffers[batch_index],
      &context->params,
      (const void*) ((uintptr_t) &context->quantization_params[batch_index]));
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

void xnn_compute_dwconv_indirection(
  const struct dwconv_indirection_init_context context[restrict XNN_MIN_ELEMENTS(1)],
  size_t output_y_start,
  size_t output_y_tile)
{
  xnn_indirection_init_dwconv2d(
    output_y_start,
    output_y_start + output_y_tile,
    context->indirection_buffer,
    context->input,
    context->input_pixel_stride,
    context->zero_buffer,
    context->input_height, context->input_width,
    context->output_height, context->output_width,
    context->kernel_height, context->kernel_width,
    context->stride_height, context->stride_width,
    context->dilation_height, context->dilation_width,
    context->input_padding_top, context->input_padding_left,
    context->step_height, context->step_width, context->tile_size);
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

void xnn_compute_dwconv_multipass(
    const struct dwconv_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input + output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);
  void* multipass_buffer =
      (void*) ((uintptr_t) context->multipass_buffer + (batch_index * context->output_height + output_y) *
               context->buffer_size);

  context->multipass_ukernel(
    context->groups, context->output_width, indirect_input, context->packed_weights, output,
    context->indirect_input_width_stride, context->output_increment, input_offset, context->zero, context->kernel_size,
    multipass_buffer, &context->params);
}

void xnn_compute_dwconv_multipass_with_thread(
    const struct dwconv_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t thread_index,
    size_t batch_index,
    size_t output_y)
{
  const void** indirect_input =
    (const void**) ((uintptr_t) context->indirect_input + output_y * context->indirect_input_height_stride);
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride;
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);
  void* multipass_buffer = (void*) ((uintptr_t) context->multipass_buffer + thread_index * context->buffer_size);

  context->multipass_ukernel(
    context->groups, context->output_width, indirect_input, context->packed_weights, output,
    context->indirect_input_width_stride, context->output_increment, input_offset, context->zero, context->kernel_size,
    multipass_buffer, &context->params);
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

  void* multipass_accumulation_buffer =
    (void*) ((uintptr_t) context->multipass_buffer + (batch_index * context->output_height + output_y) *
      context->accumulation_and_index_buffer_size);
  void* multipass_index_buffer =
    (void*) ((uintptr_t) multipass_accumulation_buffer + context->accumulation_buffer_size);

  context->multipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, multipass_accumulation_buffer, multipass_index_buffer, output, index,
    context->input_increment, context->output_increment);
}

void xnn_compute_argmax_pooling_multipass_with_thread(
    const struct argmax_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t thread_index,
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

  void* multipass_accumulation_buffer = (void*) (
    (uintptr_t) context->multipass_buffer + thread_index * context->accumulation_and_index_buffer_size);
  void* multipass_index_buffer = (void*) ((uintptr_t) multipass_accumulation_buffer + context->accumulation_buffer_size);

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
  // Min to clamp the large output y to indirect_top_height:
  // - top section will have indirect_y be the original output_y
  // - compressed and bottom section indirect_y will be indirect_top_height (i.e. y of compressed section)
  // doz calculates the additional y values needed for bottom section:
  // - top and compressed section will be 0, since their output_y + 1 will be <= indirect_bot_start.
  // - bottom section will start at 1 (since output_y == indirect_bot_start).
  // Since we only have 1 compressed row, adding these 2 values will give us the corrected indirect_y for all sections.
  const size_t indirect_y = min(output_y, context->indirect_top_height) + doz(output_y + 1, context->indirect_bot_start);
  const void** indirect_input = (void*) ((uintptr_t) context->indirect_input + indirect_y * context->indirect_input_height_stride);

  // For top section, output_y == indirect_y (since there is no compression), so the first term is 0 (no input offset).
  // For bottom section, output_y >= indirect_bot_start, so the second term becomes 0 (no input offset).
  // For the middle section, output_y - indirect_y is the y of the row within the compressed section (i.e. first
  // compressed row will be 0, second will be 1). Second term is 1 since output_y < indirect_bot_start.
  const size_t input_offset_for_compressed_section =
      (output_y - indirect_y) * (output_y < context->indirect_bot_start) * context->input_y_stride;
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride + input_offset_for_compressed_section;

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
  // Refer to xnn_compute_average_pooling_unipass for documentation on these terms.
  const size_t indirect_y = min(output_y, context->indirect_top_height) + doz(output_y + 1, context->indirect_bot_start);
  const void** indirect_input = (void*) ((uintptr_t) context->indirect_input + indirect_y * context->indirect_input_height_stride);
  const size_t input_offset_for_compressed_section =
      (output_y - indirect_y) * (output_y < context->indirect_bot_start) * context->input_y_stride;
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride + input_offset_for_compressed_section;

  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);
  void* multipass_buffer = (void*) ((uintptr_t) context->multipass_buffer +
    (batch_index * context->multipass_batch_stride) + output_y * context->multipass_pixel_stride);

  context->multipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, context->zero,
    multipass_buffer,
    output,
    context->input_increment, context->output_increment,
    &context->params);
}

void xnn_compute_average_pooling_multipass_with_thread(
    const struct average_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t thread_index,
    size_t batch_index,
    size_t output_y)
{
  // Refer to xnn_compute_average_pooling_unipass for documentation on these terms.
  const size_t indirect_y = min(output_y, context->indirect_top_height) + doz(output_y + 1, context->indirect_bot_start);
  const void** indirect_input = (void*) ((uintptr_t) context->indirect_input + indirect_y * context->indirect_input_height_stride);
  const size_t input_offset_for_compressed_section =
      (output_y - indirect_y) * (output_y < context->indirect_bot_start) * context->input_y_stride;
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride + input_offset_for_compressed_section;

  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  context->multipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, context->zero,
    (void*) ((uintptr_t) context->multipass_buffer + thread_index * context->multipass_pixel_stride),
    output,
    context->input_increment, context->output_increment,
    &context->params);
}

void xnn_compute_pixelwise_average_pooling_unipass(
    const struct pixelwise_average_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t output_y)
{
  // Refer to xnn_compute_average_pooling_unipass for documentation on these terms.
  const size_t indirect_y = min(output_y, context->indirect_top_height) + doz(output_y + 1, context->indirect_bot_start);
  const void** indirect_input = (void*) ((uintptr_t) context->indirect_input + indirect_y * context->indirect_input_height_stride);
  const size_t input_offset_for_compressed_section =
      (output_y - indirect_y) * (output_y < context->indirect_bot_start) * context->input_y_stride;
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride + input_offset_for_compressed_section;

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
  // Refer to xnn_compute_average_pooling_unipass for documentation on these terms.
  const size_t indirect_y = min(output_y, context->indirect_top_height) + doz(output_y + 1, context->indirect_bot_start);
  const void** indirect_input = (void*) ((uintptr_t) context->indirect_input + indirect_y * context->indirect_input_height_stride);
  const size_t input_offset_for_compressed_section =
      (output_y - indirect_y) * (output_y < context->indirect_bot_start) * context->input_y_stride;
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride + input_offset_for_compressed_section;

  const void* pixelwise_buffer =
    (const void*) ((uintptr_t) context->pixelwise_buffer + output_y * context->pixelwise_buffer_height_stride);
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);
  void* multipass_buffer = (void*) ((uintptr_t) context->multipass_buffer +
    batch_index * context->multipass_batch_stride + output_y * context->multipass_pixel_stride);

  context->multipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, context->zero, pixelwise_buffer,
    multipass_buffer,
    output,
    context->input_increment, context->output_increment,
    &context->params);
}

void xnn_compute_pixelwise_average_pooling_multipass_with_thread(
    const struct pixelwise_average_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t thread_index,
    size_t batch_index,
    size_t output_y)
{
  // Refer to xnn_compute_average_pooling_unipass for documentation on these terms.
  const size_t indirect_y = min(output_y, context->indirect_top_height) + doz(output_y + 1, context->indirect_bot_start);
  const void** indirect_input = (void*) ((uintptr_t) context->indirect_input + indirect_y * context->indirect_input_height_stride);
  const size_t input_offset_for_compressed_section =
      (output_y - indirect_y) * (output_y < context->indirect_bot_start) * context->input_y_stride;
  const size_t input_offset = context->input_offset + batch_index * context->input_batch_stride + input_offset_for_compressed_section;

  const void* pixelwise_buffer =
    (const void*) ((uintptr_t) context->pixelwise_buffer + output_y * context->pixelwise_buffer_height_stride);
  void* output = (void*) ((uintptr_t) context->output +
    batch_index * context->output_batch_stride + output_y * context->output_height_stride);

  context->multipass_ukernel(
    context->output_width, context->pooling_size, context->channels,
    indirect_input, input_offset, context->zero, pixelwise_buffer,
    (void*) ((uintptr_t) context->multipass_buffer + thread_index * context->multipass_pixel_stride),
    output,
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
    (void*) ((uintptr_t) context->multipass_buffer + batch_index * context->multipass_batch_stride);

  assert(context->multipass_buffer != NULL);

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

void xnn_compute_global_average_pooling_nwc_multipass_with_thread(
    const struct global_average_pooling_nwc_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t thread_index,
    size_t batch_index)
{
  const void* input =
    (const void*) ((uintptr_t) context->input + batch_index * context->input_batch_stride);
  void* output =
    (void*) ((uintptr_t) context->output + batch_index * context->output_batch_stride);
  void* multipass_buffer =
    (void*) ((uintptr_t) context->multipass_buffer + thread_index * context->multipass_batch_stride);

  assert(context->multipass_buffer != NULL);

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

void xnn_compute_resize_bilinear_indirection(
    const struct resize_bilinear_nhwc_indirection_init_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t output_y_start,
    size_t output_y_tile)
{
  void* buffer = context->buffer;

  context->indirection_init(
    output_y_start,
    output_y_start + output_y_tile,
    context->input_pixel_stride,
    context->input_height, context->input_width,
    context->output_height, context->output_width,
    context->input,
    /*indirection_buffer==*/(const void**) ((uintptr_t) buffer + context->indirect_input_offset),
    /*packed_weights=*/(void*) buffer,
    context->align_corners, context->tensorflow_legacy_mode);
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

void xnn_compute_scaled_dot_product_attention(
  const struct scaled_dot_product_attention_context context[restrict XNN_MIN_ELEMENTS(1)],
  size_t batch_index,
  size_t head_index,
  size_t tokens_start,
  size_t tokens_block_size)
{
  const size_t query_key_scaled_channels = context->query_key_scaled_channels;
  const size_t query_tile_offset =
    batch_index * context->query_batch_stride + head_index * context->query_head_stride +
    tokens_start * query_key_scaled_channels;
  const size_t key_value_tokens_scaled = context->key_value_tokens_scaled;
  const size_t key_value_tokens_start_scaled = tokens_start * key_value_tokens_scaled;
  const size_t cn_stride = context->cn_stride;
  const void* scaled_query = (void*) ((uintptr_t) context->scaled_query + query_tile_offset);
  const void* minmax_params = &context->minmax_params;

  {
    uintptr_t query = (uintptr_t) context->query + query_tile_offset;
    uintptr_t query_scaled_current = (uintptr_t) scaled_query;
    // Q_scaled = Q * Scale (along channels). Q and Q_scaled have dimensions [tokens_block_size, query_key_channels].
    size_t i = tokens_block_size;
    do {
      context->vmul_ukernel(
        /*batch=*/query_key_scaled_channels,
        /*input_x=*/(const void*) query,
        /*input_y=*/context->scale,
        /*output=*/(void*) query_scaled_current,
        /*params=*/minmax_params);
      query += query_key_scaled_channels;
      query_scaled_current += query_key_scaled_channels;
    } while (--i != 0);
  }

  const size_t logits_batch_offset =
      batch_index * context->logits_batch_stride + head_index * context->logits_head_stride;
  void* const logits =
    (void*) ((uintptr_t) context->logits_buffer + logits_batch_offset + key_value_tokens_start_scaled);

  {
    void* key = (void*) ((uintptr_t) context->key +
                         batch_index * context->key_batch_stride +
                         head_index * context->key_head_stride);
    // S = GEMM(Q_scaled, K^t). S is [tokens_block_size, key_value_tokens].
    context->gemm_ukernel.function[XNN_UARCH_DEFAULT](
      /*mr=*/tokens_block_size,
      /*nr=*/context->key_value_tokens,
      /*k=*/query_key_scaled_channels,
      /*a=*/scaled_query,
      /*a_stride=*/query_key_scaled_channels,
      /*w=*/(void*) key,
      /*c=*/(void*) (uintptr_t) logits,
      /*cm_stride=*/key_value_tokens_scaled,
      /*cn_stride=*/cn_stride,
      /*params=*/minmax_params);
  }

  {
    const size_t tokens_block_size_scaled = tokens_block_size * key_value_tokens_scaled;
    struct attention_logits_cap logits_cap = context->logits_cap;
    if (logits_cap.type == xnn_attention_logits_cap_type_tanh) {
      // (Optional) S = TanH(S/Cap) * Cap. Overwrites buffer.
      context->vmulc_ukernel(
        /*batch=*/tokens_block_size_scaled,
        /*input_x=*/logits,
        /*input_y=*/&logits_cap.cap_reciprocal,
        /*output=*/logits,
        /*params=*/minmax_params);
      context->vtanh_ukernel(
        /*batch=*/tokens_block_size_scaled,
        /*input=*/logits,
        /*output=*/logits,
        /*params=*/&context->tanh_params);
      context->vmulc_ukernel(
        /*batch=*/tokens_block_size_scaled,
        /*input_x=*/logits,
        /*input_y=*/&logits_cap.cap,
        /*output=*/logits,
        /*params=*/minmax_params);
    }

    // S = S + Mask. Mask has dimensions [query_tokens, key_value_tokens].
    // Mask. Overwrites buffer.
    context->vadd_ukernel(
      /*batch=*/tokens_block_size_scaled,
      /*input_x=*/logits,
      /*input_y=*/(void*) ((uintptr_t) context->mask + key_value_tokens_start_scaled),
      /*output=*/logits,
      /*params=*/minmax_params);
  }

  // P = Softmax(S). P has dimensions [tokens_block_size, key_value_tokens].
  {
    void* logits_row = logits;
    size_t i = tokens_block_size;
    do {
      // Skip initialization of locals as they will be written to immediately.
      float rowmax;
      context->rmax_ukernel(
        /*batch=*/key_value_tokens_scaled,
        /*input=*/logits_row,
        /*output=*/&rowmax,
        /*params=*/&context->rmax_params);

      float rowsum;
      context->raddstoreexpminusmax_ukernel(
        /*batch=*/key_value_tokens_scaled,
        /*input=*/logits_row,
        /*max=*/&rowmax,
        /*output=*/logits_row,
        /*sum=*/&rowsum,
        /*params=*/&context->expminus_params);

      float rowscale;
      context->compute_reciprocal(
        /*input=*/&rowsum,
        /*output=*/&rowscale);

      context->vmulc_ukernel(
        /*batch=*/key_value_tokens_scaled,
        /*input_x=*/logits_row,
        /*input_y=*/&rowscale,
        /*output=*/logits_row,
        /*params=*/minmax_params);

      logits_row = (void*) ((uintptr_t) logits_row + key_value_tokens_scaled);
    } while (--i != 0);
  }

  {
    void* value = (void*) ((uintptr_t) context->value +
                           batch_index * context->value_batch_stride +
                           head_index * context->value_head_stride);
    const size_t output_tile_offset =
      batch_index * context->output_batch_stride + head_index * context->output_head_stride +
      tokens_start * context->value_scaled_channels;
    // O = GEMM(P, V). O has dimension [tokens_block_size, value_channels].
    context->gemm_ukernel.function[XNN_UARCH_DEFAULT](
        /*mr=*/tokens_block_size,
        /*nc=*/context->value_channels,
        /*kc=*/key_value_tokens_scaled,
        /*a=*/logits,
        /*a_stride=*/key_value_tokens_scaled,
        /*w=*/value,
        /*c=*/(void*) ((uintptr_t) context->output + output_tile_offset),
        /*cm_stride=*/context->value_scaled_channels,
        /*cn_stride=*/cn_stride,
        /*params=*/minmax_params);
  }
}

void xnn_compute_scaled_dot_product_attention_with_thread(
  const struct scaled_dot_product_attention_context context[restrict XNN_MIN_ELEMENTS(1)],
  size_t thread_index,
  size_t batch_index,
  size_t head_index,
  size_t tokens_start,
  size_t tokens_block_size)
{
  const size_t query_key_scaled_channels = context->query_key_scaled_channels;
  const size_t query_tile_offset =
    batch_index * context->query_batch_stride + head_index * context->query_head_stride +
    tokens_start * query_key_scaled_channels;
  const size_t key_value_tokens_scaled = context->key_value_tokens_scaled;
  const size_t key_value_tokens_start_scaled = tokens_start * key_value_tokens_scaled;
  const size_t cn_stride = context->cn_stride;
  const void* scaled_query =
    (void*) ((uintptr_t) context->scaled_query + thread_index * context->scaled_query_thread_stride);
  const void* minmax_params = &context->minmax_params;

  {
    uintptr_t query = (uintptr_t) context->query + query_tile_offset;
    uintptr_t query_scaled_current = (uintptr_t) scaled_query;
    // Q_scaled = Q * Scale (along channels). Q and Q_scaled have dimensions [tokens_block_size, query_key_channels].
    size_t i = tokens_block_size;
    do {
      context->vmul_ukernel(
        /*batch=*/query_key_scaled_channels,
        /*input_x=*/(const void*) query,
        /*input_y=*/context->scale,
        /*output=*/(void*) query_scaled_current,
        /*params=*/minmax_params);
      query += query_key_scaled_channels;
      query_scaled_current += query_key_scaled_channels;
    } while (--i != 0);
  }

  void* const logits = (void*) ((uintptr_t) context->logits_buffer + thread_index * context->logits_thread_stride);

  {
    void* key = (void*) ((uintptr_t) context->key +
                         batch_index * context->key_batch_stride +
                         head_index * context->key_head_stride);
    // S = GEMM(Q_scaled, K^t). S is [tokens_block_size, key_value_tokens].
    context->gemm_ukernel.function[XNN_UARCH_DEFAULT](
      /*mr=*/tokens_block_size,
      /*nr=*/context->key_value_tokens,
      /*k=*/query_key_scaled_channels,
      /*a=*/scaled_query,
      /*a_stride=*/query_key_scaled_channels,
      /*w=*/(void*) key,
      /*c=*/(void*) (uintptr_t) logits,
      /*cm_stride=*/key_value_tokens_scaled,
      /*cn_stride=*/cn_stride,
      /*params=*/minmax_params);
  }

  {
    const size_t tokens_block_size_scaled = tokens_block_size * key_value_tokens_scaled;
    struct attention_logits_cap logits_cap = context->logits_cap;
    if (logits_cap.type == xnn_attention_logits_cap_type_tanh) {
      // (Optional) S = TanH(S/Cap) * Cap. Overwrites buffer.
      context->vmulc_ukernel(
        /*batch=*/tokens_block_size_scaled,
        /*input_x=*/logits,
        /*input_y=*/&logits_cap.cap_reciprocal,
        /*output=*/logits,
        /*params=*/minmax_params);
      context->vtanh_ukernel(
        /*batch=*/tokens_block_size_scaled,
        /*input=*/logits,
        /*output=*/logits,
        /*params=*/&context->tanh_params);
      context->vmulc_ukernel(
        /*batch=*/tokens_block_size_scaled,
        /*input_x=*/logits,
        /*input_y=*/&logits_cap.cap,
        /*output=*/logits,
        /*params=*/minmax_params);
    }

    // S = S + Mask. Mask has dimensions [query_tokens, key_value_tokens].
    // Mask. Overwrites buffer.
    context->vadd_ukernel(
      /*batch=*/tokens_block_size_scaled,
      /*input_x=*/logits,
      /*input_y=*/(void*) ((uintptr_t) context->mask + key_value_tokens_start_scaled),
      /*output=*/logits,
      /*params=*/minmax_params);
  }

  // P = Softmax(S). P has dimensions [tokens_block_size, key_value_tokens].
  {
    void* logits_row = logits;
    size_t i = tokens_block_size;
    do {
      // Skip initialization of locals as they will be written to immediately.
      float rowmax;
      context->rmax_ukernel(
        /*batch=*/key_value_tokens_scaled,
        /*input=*/logits_row,
        /*output=*/&rowmax,
        /*params=*/&context->rmax_params);

      float rowsum;
      context->raddstoreexpminusmax_ukernel(
        /*batch=*/key_value_tokens_scaled,
        /*input=*/logits_row,
        /*max=*/&rowmax,
        /*output=*/logits_row,
        /*sum=*/&rowsum,
        /*params=*/&context->expminus_params);

      float rowscale;
      context->compute_reciprocal(
        /*input=*/&rowsum,
        /*output=*/&rowscale);

      context->vmulc_ukernel(
        /*batch=*/key_value_tokens_scaled,
        /*input_x=*/logits_row,
        /*input_y=*/&rowscale,
        /*output=*/logits_row,
        /*params=*/minmax_params);

      logits_row = (void*) ((uintptr_t) logits_row + key_value_tokens_scaled);
    } while (--i != 0);
  }

  {
    void* value = (void*) ((uintptr_t) context->value +
                           batch_index * context->value_batch_stride +
                           head_index * context->value_head_stride);
    const size_t output_tile_offset =
      batch_index * context->output_batch_stride + head_index * context->output_head_stride +
      tokens_start * context->value_scaled_channels;
    // O = GEMM(P, V). O has dimension [tokens_block_size, value_channels].
    context->gemm_ukernel.function[XNN_UARCH_DEFAULT](
        /*mr=*/tokens_block_size,
        /*nc=*/context->value_channels,
        /*kc=*/key_value_tokens_scaled,
        /*a=*/logits,
        /*a_stride=*/key_value_tokens_scaled,
        /*w=*/value,
        /*c=*/(void*) ((uintptr_t) context->output + output_tile_offset),
        /*cm_stride=*/context->value_scaled_channels,
        /*cn_stride=*/cn_stride,
        /*params=*/minmax_params);
  }
}

void xnn_compute_slice_1d(
    const struct slice_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i)
{
  const void* input = (const void*) ((uintptr_t) context->input + i * context->input_stride[0]);
  void* output = (void*) ((uintptr_t) context->output + i * context->output_stride[0]);

  context->ukernel(context->contiguous_size, input, output, NULL);
}

void xnn_compute_slice_2d(
    const struct slice_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j)
{
  const void* input =
      (const void*) ((uintptr_t) context->input +
                     i * context->input_stride[1] +
                     j * context->input_stride[0]);
  void* output =
      (void*) ((uintptr_t) context->output + i * context->output_stride[1] + j * context->output_stride[0]);

  context->ukernel(context->contiguous_size, input, output, NULL);
}

void xnn_compute_slice_3d(
    const struct slice_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j, size_t k)
{
  const void* input =
      (const void*) ((uintptr_t) context->input +
                     i * context->input_stride[2] +
                     j * context->input_stride[1] +
                     k * context->input_stride[0]);
  void* output =
      (void*) ((uintptr_t) context->output + i * context->output_stride[2] +
               j * context->output_stride[1] + k * context->output_stride[0]);

  context->ukernel(context->contiguous_size, input, output, NULL);
}

void xnn_compute_slice_4d(
    const struct slice_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j, size_t k, size_t l)
{
  const void* input =
      (const void*) ((uintptr_t) context->input +
                     i * context->input_stride[3] +
                     j * context->input_stride[2] +
                     k * context->input_stride[1] +
                     l * context->input_stride[0]);
  void* output =
      (void*) ((uintptr_t) context->output + i * context->output_stride[3] +
               j * context->output_stride[2] + k * context->output_stride[1] + l * context->output_stride[0]);

  context->ukernel(context->contiguous_size, input, output, NULL);
}

void xnn_compute_slice_5d(
    const struct slice_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j, size_t k, size_t l, size_t m)
{
  const void* input =
      (const void* ) ((uintptr_t) context->input +
                      i * context->input_stride[4] +
                      j * context->input_stride[3] +
                      k * context->input_stride[2] +
                      l * context->input_stride[1] +
                      m * context->input_stride[0]);
  void* output =
      (void*) ((uintptr_t) context->output + i * context->output_stride[4] +
               j * context->output_stride[3] + k * context->output_stride[2] +
               l * context->output_stride[1] + m * context->output_stride[0]);

  context->ukernel(context->contiguous_size, input, output, NULL);
}

void xnn_compute_elementwise_binary_1d_tile(
    const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t offset,
    size_t size)
{
  size_t a_offset = ((context->a_stride[4] == 0 ? 0 : offset));
  size_t b_offset = ((context->b_stride[4] == 0 ? 0 : offset));
  const void* a = (const void*) ((uintptr_t) context->a + a_offset);
  const void* b = (const void*) ((uintptr_t) context->b + b_offset);
  void* y = (void*) ((uintptr_t) context->y + offset);
  context->ukernel(size, a, b, y, &context->params);
}

void xnn_compute_elementwise_binary_1d(
    const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i)
{
  const void* a = (const void*) ((uintptr_t) context->a + i * context->a_stride[4]);
  const void* b = (const void*) ((uintptr_t) context->b + i * context->b_stride[4]);
  void* y = (void*) ((uintptr_t) context->y + i * context->y_stride[4]);
  context->ukernel(context->elements, a, b, y, &context->params);
}

void xnn_compute_elementwise_binary_2d(
    const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j)
{
  const void* a = (const void*) ((uintptr_t) context->a + i * context->a_stride[3] + j * context->a_stride[4]);
  const void* b = (const void*) ((uintptr_t) context->b + i * context->b_stride[3] + j * context->b_stride[4]);
  void* y = (void*) ((uintptr_t) context->y + i * context->y_stride[3] + j * context->y_stride[4]);
  context->ukernel(context->elements, a, b, y, &context->params);
}

void xnn_compute_elementwise_binary_3d(
    const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j, size_t k)
{
  const void* a = (const void*) ((uintptr_t) context->a +
    i * context->a_stride[2] + j * context->a_stride[3] + k * context->a_stride[4]);
  const void* b = (const void*) ((uintptr_t) context->b +
    i * context->b_stride[2] + j * context->b_stride[3] + k * context->b_stride[4]);
  void* y = (void*) ((uintptr_t) context->y +
    i * context->y_stride[2] + j * context->y_stride[3] + k * context->y_stride[4]);
  context->ukernel(context->elements, a, b, y, &context->params);
}

void xnn_compute_elementwise_binary_4d(
    const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t i, size_t j, size_t k, size_t l)
{
  const void* a = (const void*) ((uintptr_t) context->a +
    i * context->a_stride[1] + j * context->a_stride[2] + k * context->a_stride[3] + l * context->a_stride[4]);
  const void* b = (const void*) ((uintptr_t) context->b +
    i * context->b_stride[1] + j * context->b_stride[2] + k * context->b_stride[3] + l * context->b_stride[4]);
  void* y = (void*) ((uintptr_t) context->y +
    i * context->y_stride[1] + j * context->y_stride[2] + k * context->y_stride[3] + l * context->y_stride[4]);
  context->ukernel(context->elements, a, b, y, &context->params);
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

void xnn_compute_contiguous_reduce(
    const struct reduce_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t output_idx0,
    size_t output_idx1,
    size_t output_idx2,
    size_t output1_block_size,
    size_t output2_block_size)
{
  assert(output1_block_size == 1);
  const size_t* input_stride = context->input_stride;
  const size_t* output_stride = context->output_stride;

  // input dimensions 1, 3 & 5 are reduced so the entirety of these dimensions
  // are processed so their indices are always 0.
  size_t input_offset = input_stride[0] * output_idx0 + input_stride[2] * output_idx1
      + input_stride[4] * output_idx2;
  size_t output_offset = (output_stride[0] * output_idx0 + output_stride[1] * output_idx1
                          + output_stride[2] * output_idx2) * context->output_element_size;
  size_t workspace_offset = (output_stride[0] * output_idx0 + output_stride[1] * output_idx1
                             + output_stride[2] * output_idx2) * context->accumulation_element_size;
  int input_shape1 = context->input_shape[1];
  int input_shape3 = context->input_shape[3];

  void* output_ptr = NULL;
  if (context->workspace) {
    output_ptr = context->workspace;
  } else {
    output_ptr = context->output;
  }
  void* output = (void*) ((uintptr_t) output_ptr + workspace_offset);
  // Rsum microkernels accumulate into the output buffer.
  memset(output, 0, context->accumulation_element_size * output2_block_size);

  // Input dimension 1 is reduced.
  for (size_t i = 0; i < input_shape1; ++i) {
    const void* input = (const void*) ((uintptr_t) context->input + input_offset);
    // Input dimension 3 is reduced.
    for (size_t j = 0; j < input_shape3; ++j) {
      const void* input_row = input;
      // output2_block_size output elements are written.
      for (size_t k = 0; k < output2_block_size; ++k) {
        // The microkernel reduces input dimension 5.
        context->ukernel.rsum(context->channels, input_row, output, &context->params);
        // input_stride[4] is the number of bytes of input which have been
        // processed by the microkernel call.
        input_row = (const void*) ((uintptr_t) input_row + input_stride[4]);
        // Increment output pointer by the number of output bytes which have
        // been written.
        output = (void*) ((uintptr_t) output + context->accumulation_element_size);
      }
      // Reset the output pointer.
      output = (void*) ((uintptr_t) output_ptr + workspace_offset);
      // Iterating over input_shape[3].
      input = (const void*) ((uintptr_t) input + input_stride[3]);
    }
    // Iterating over input_shape[1].
    input_offset += input_stride[1];
  }
  // Convert to output datatype if accumulation type != output type.
  if (context->workspace) {
    const void* workspace_ptr = (void*) ((uintptr_t) context->workspace + workspace_offset);
    output_ptr = (void*) ((uintptr_t) context->output + output_offset);
    context->cvt_ukernel(context->accumulation_element_size * output2_block_size, workspace_ptr,
                         output_ptr, /*params=*/NULL);
  }
}

void xnn_compute_discontiguous_reduce(
    const struct reduce_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t output_idx0,
    size_t output_idx1,
    size_t output_idx2,
    size_t output1_block_size,
    size_t output2_block_size)
{
  assert(output1_block_size == 1);
  const size_t* input_stride = context->input_stride;
  const size_t* output_stride = context->output_stride;

  // input dimensions 0, 2 & 4 are reduced so the entirety of these dimensions
  // are processed so their indices are always 0.
  size_t input_offset = input_stride[1] * output_idx0 + input_stride[3] * output_idx1 + input_stride[5] * output_idx2;
  size_t output_offset = (output_stride[0] * output_idx0 + output_stride[1] * output_idx1
                          + output_stride[2] * output_idx2) * context->output_element_size;
  size_t workspace_offset = (output_stride[0] * output_idx0 + output_stride[1] * output_idx1
                             + output_stride[2] * output_idx2) * context->accumulation_element_size;
  int input_shape0 = context->input_shape[0];
  int input_shape2 = context->input_shape[2];

  void* output_ptr = NULL;
  if (context->workspace) {
    output_ptr = context->workspace;
  } else {
    output_ptr = context->output;
  }
  void* output = (void*) ((uintptr_t) output_ptr + workspace_offset);
  // RDsum microkernels accumulate into the output buffer.
  memset(output, 0, context->accumulation_element_size * output2_block_size);

  // Input dimension 0 is reduced.
  for (size_t i = 0; i < input_shape0; ++i) {
    const void* input = (const void*) ((uintptr_t) context->input + input_offset);
    // Input dimension 2 is reduced.
    for (size_t j = 0; j < input_shape2; ++j) {
      const void* input_row = input;
      // The microkernel reduces input dimension 4 and iterates over output_block_size elements of dimension 5.
      context->ukernel.rdsum(context->channels, output2_block_size, input_row, input_stride[4],
                             context->zero, output, &context->params);
      // input_stride[4] is the number of bytes of input which have been
      // processed by the microkernel call.
      input_row = (const void*) ((uintptr_t) input_row + input_stride[4]);
      // Reset the output pointer.
      output = (void*) ((uintptr_t) output_ptr + workspace_offset);
      // Iterating over input_shape[2].
      input = (const void*) ((uintptr_t) input + input_stride[2]);
    }
    // Iterating over input_shape[0].
    input_offset += input_stride[0];
  }
  // Convert to output datatype if accumulation type != output type.
  if (context->workspace) {
    const void* workspace_ptr = (void*) ((uintptr_t) context->workspace + workspace_offset);
    output_ptr = (void*) ((uintptr_t) context->output + output_offset);
    context->cvt_ukernel(context->accumulation_element_size * output2_block_size, workspace_ptr,
                         output_ptr, /*params=*/NULL);
  }
}

void xnn_compute_pad_qd8_params(
    const struct f32_qd8_convert_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index)
{
  const size_t batch_size = context->batch_size;
  for (size_t i = 0; i < XNN_EXTRA_QUANTIZATION_PARAMS; ++i) {
    context->quantization_params[batch_size + i].zero_point = context->quantization_params[batch_size - 1].zero_point;
    context->quantization_params[batch_size + i].inv_scale = context->quantization_params[batch_size - 1].inv_scale;
  }
}

void xnn_compute_f16_qd8_convert(
    const struct f16_qd8_convert_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index)
{
  const size_t x_stride = context->x_stride;
  const size_t y_stride = context->y_stride;
  const size_t n        = context->n;
  const void* input = (const void*) ((uintptr_t) context->x + x_stride * batch_index);
  void* output = (void*) ((uintptr_t) context->y + y_stride * batch_index);

  uint16_t minmax[2];
  context->rminmax_ukernel(n, input, minmax, &context->params);
  uint16_t f16_scale;
  context->quantization_params[batch_index] = xnn_f16_qd8_asymmetric_quantization_params(minmax[0], minmax[1], &f16_scale);

  union xnn_f16_qs8_cvt_params params;
  context->init_params(&params, f16_scale, context->quantization_params[batch_index].zero_point, INT8_MIN, INT8_MAX);
  context->convert_ukernel(n, input, output, &params);
}

void xnn_compute_f32_qd8_convert(
    const struct f32_qd8_convert_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index)
{
  const size_t x_stride = context->x_stride;
  const size_t y_stride = context->y_stride;
  const size_t n        = context->n;
  const void* input = (const void*) ((uintptr_t) context->x + x_stride * batch_index);
  void* output = (void*) ((uintptr_t) context->y + y_stride * batch_index);

  float minmax[2];
  context->rminmax_ukernel(n, input, minmax, &context->params);
  context->quantization_params[batch_index] = xnn_f32_qd8_asymmetric_quantization_params(minmax[0], minmax[1]);

  union xnn_f32_qs8_cvt_params params;
  context->init_params(&params, 1.0f / context->quantization_params[batch_index].inv_scale, context->quantization_params[batch_index].zero_point, INT8_MIN, INT8_MAX);
  context->convert_ukernel(n, input, output, &params);
}

void xnn_compute_f32_qp8_convert(
    const struct f32_qp8_convert_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t m_idx_start) {
  const float* lhs = (const float*)((const char*)context->lhs +
                                    m_idx_start * context->lhs_stride);
  int8_t* lhs_packed =
      context->lhs_packed +
      xnn_x8_packq_f32qp8_packed_offset(m_idx_start, context->k, context->mr,
                                        context->kr, context->sr);

  context->packq_ukernel(/*m=*/1, context->k, context->mr, context->kr,
                         context->sr, m_idx_start, lhs, context->lhs_stride,
                         lhs_packed);
}

void xnn_compute_u8_softmax(
    const struct u8_softmax_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index)
{
  const uint8_t* x = (const uint8_t*) ((uintptr_t) context->x + context->x_stride * batch_index);
  uint8_t* y = (uint8_t*) ((uintptr_t) context->y + context->y_stride * batch_index);
  const size_t n = context->n;

  uint8_t x_max = 0;
  context->rmax_ukernel(n, x, &x_max, /*params=*/NULL);
  const size_t adjustment = x_max ^ 255;
  const uint32_t* t = (const uint32_t*) context->t + adjustment;
  context->lut_norm_ukernel(n, x, t, y);
}

void xnn_compute_floating_point_softmax(
    const struct floating_point_softmax_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index)
{
  const void* x = (const void*) ((uintptr_t) context->x + context->x_stride * batch_index);
  void* y = (void*) ((uintptr_t) context->y + context->y_stride * batch_index);
  const size_t n = context->n;

  // First pass: reduce-max
  union {
    float as_float;
    uint16_t as_half;
  } x_max;
  context->rmax_ukernel(n, x, &x_max, &context->rmax_params);

  // Second pass: reduce-add & store exp(x-x_max)
  union {
    float as_float;
    uint16_t as_half;
  } y_sum;
  context->raddstoreexpminusmax_ukernel(n, x, &x_max, y, &y_sum, &context->expminus_params);

  // Third pass: scale y
  union {
    float as_float;
    uint16_t as_half;
  } y_scale;
  context->compute_reciprocal(&y_sum, &y_scale);
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

void xnn_compute_rope(
    const struct rope_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t head_index,
    size_t sequence_index)
{
  const size_t scaled_channels = context->scaled_channels;
  const size_t offset = batch_index * context->batch_stride + head_index * context->head_stride + sequence_index * context->sequence_stride;
  const void* input = (const void*) ((uintptr_t) context->input + offset);
  const void* weights = (const void*) ((uintptr_t) context->weights + sequence_index * (scaled_channels + scaled_channels));
  void* output = (void*) ((uintptr_t) context->output + offset);

  context->vcmul(
    scaled_channels,
    input, weights, output,
    NULL);
}

#if XNN_MAX_UARCH_TYPES > 1
void xnn_compute_hmp_gemm(
    const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
    uint32_t uarch_index, size_t mr_block_start, size_t nr_block_start,
    size_t mr_block_size, size_t nr_block_size) {
  const size_t a_stride = context->a_stride;
  const size_t cm_stride = context->cm_stride;

  context->ukernel.function[uarch_index](
      mr_block_size, nr_block_size, context->k_scaled,
      (const void*)((uintptr_t)context->a + mr_block_start * a_stride),
      a_stride,
      (const void*)((uintptr_t)context->packed_w +
                    nr_block_start * context->w_stride),
      (void*)((uintptr_t)context->c + mr_block_start * cm_stride +
              (nr_block_start << context->log2_csize)),
      cm_stride, context->cn_stride, context->fused_params);
  }

  void xnn_compute_hmp_dqgemm(
      const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      uint32_t uarch_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size)
  {
    const size_t a_stride  = context->a_stride;
    const size_t cm_stride = context->cm_stride;

    context->dq_ukernel.function[uarch_index](
        mr_block_size,
        nr_block_size,
        context->k_scaled,
        (const void*) ((uintptr_t) context->a + mr_block_start * a_stride),
        a_stride,
        (const void*) ((uintptr_t) context->packed_w + nr_block_start * context->w_stride),
        (void*) ((uintptr_t) context->c + mr_block_start * cm_stride + (nr_block_start << context->log2_csize)),
        cm_stride,
        context->cn_stride,
        context->fused_params,
       (const void*) ((uintptr_t) &context->quantization_params[mr_block_start]));
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

  void xnn_compute_hmp_grouped_batch_dqigemm(
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

    context->dq_ukernel.function[uarch_index](
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
        context->zero_buffers[batch_index],
        &context->params,
        (const void*) ((uintptr_t) &context->quantization_params[batch_index]));
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

  void xnn_compute_hmp_grouped_dqigemm(
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

    context->dq_ukernel.function[uarch_index](
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
        context->zero_buffers[0],
        &context->params,
        (const void*) ((uintptr_t) context->quantization_params));
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

  void xnn_compute_batch_hmp_dqigemm(
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

    context->dq_ukernel.function[uarch_index](
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
        context->zero_buffers[batch_index],
        &context->params,
        (const void*) ((uintptr_t) &context->quantization_params[batch_index]));
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

  void xnn_compute_hmp_dqigemm(
      const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      uint32_t uarch_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size)
  {
    const size_t ks        = context->ks;
    const size_t cm_stride = context->cm_stride;

    context->dq_ukernel.function[uarch_index](
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
        context->zero_buffers[0],
        &context->params,
        (const void*) ((uintptr_t) context->quantization_params));
  }

void xnn_compute_hmp_scaled_dot_product_attention(
  const struct scaled_dot_product_attention_context context[restrict XNN_MIN_ELEMENTS(1)],
  uint32_t uarch_index,
  size_t batch_index,
  size_t head_index,
  size_t tokens_start,
  size_t tokens_block_size)
{
  const size_t query_key_scaled_channels = context->query_key_scaled_channels;
  const size_t query_tile_offset =
    batch_index * context->query_batch_stride + head_index * context->query_head_stride +
    tokens_start * query_key_scaled_channels;
  const size_t key_value_tokens_scaled = context->key_value_tokens_scaled;
  const size_t key_value_tokens_start_scaled = tokens_start * key_value_tokens_scaled;
  const size_t cn_stride = context->cn_stride;
  const void* scaled_query = (void*) ((uintptr_t) context->scaled_query + query_tile_offset);
  const void* minmax_params = &context->minmax_params;

  {
    uintptr_t query = (uintptr_t) context->query + query_tile_offset;
    uintptr_t query_scaled_current = (uintptr_t) scaled_query;
    // Q_scaled = Q * Scale (along channels). Q and Q_scaled have dimensions [tokens_block_size, query_key_channels].
    size_t i = tokens_block_size;
    do {
      context->vmul_ukernel(
        /*batch=*/query_key_scaled_channels,
        /*input_x=*/(const void*) query,
        /*input_y=*/context->scale,
        /*output=*/(void*) query_scaled_current,
        /*params=*/minmax_params);
      query += query_key_scaled_channels;
      query_scaled_current += query_key_scaled_channels;
    } while (--i != 0);
  }

  const size_t logits_batch_offset =
    batch_index * context->logits_batch_stride + head_index * context->logits_head_stride;
  void* const logits =
    (void*) (((uintptr_t) context->logits_buffer + logits_batch_offset + key_value_tokens_start_scaled));

  {
    void* key = (void*) ((uintptr_t) context->key +
                         batch_index * context->key_batch_stride +
                         head_index * context->key_head_stride);
    // S = GEMM(Q_scaled, K^t). S is [tokens_block_size, key_value_tokens].
    context->gemm_ukernel.function[uarch_index](
      /*mr=*/tokens_block_size,
      /*nr=*/context->key_value_tokens,
      /*k=*/query_key_scaled_channels,
      /*a=*/scaled_query,
      /*a_stride=*/query_key_scaled_channels,
      /*w=*/(void*) key,
      /*c=*/(void*) (uintptr_t) logits,
      /*cm_stride=*/key_value_tokens_scaled,
      /*cn_stride=*/cn_stride,
      /*params=*/minmax_params);
  }

  {
    const size_t tokens_block_size_scaled = tokens_block_size * key_value_tokens_scaled;
    struct attention_logits_cap logits_cap = context->logits_cap;
    if (logits_cap.type == xnn_attention_logits_cap_type_tanh) {
      // (Optional) S = TanH(S/Cap) * Cap. Overwrites buffer.
      context->vmulc_ukernel(
        /*batch=*/tokens_block_size_scaled,
        /*input_x=*/logits,
        /*input_y=*/&logits_cap.cap_reciprocal,
        /*output=*/logits,
        /*params=*/minmax_params);
      context->vtanh_ukernel(
        /*batch=*/tokens_block_size_scaled,
        /*input=*/logits,
        /*output=*/logits,
        /*params=*/&context->tanh_params);
      context->vmulc_ukernel(
        /*batch=*/tokens_block_size_scaled,
        /*input_x=*/logits,
        /*input_y=*/&logits_cap.cap,
        /*output=*/logits,
        /*params=*/minmax_params);
    }

    // S = S + Mask. Mask has dimensions [query_tokens, key_value_tokens].
    // Mask. Overwrites buffer.
    context->vadd_ukernel(
      /*batch=*/tokens_block_size_scaled,
      /*input_x=*/logits,
      /*input_y=*/(void*) ((uintptr_t) context->mask + key_value_tokens_start_scaled),
      /*output=*/logits,
      /*params=*/minmax_params);
  }

  // P = Softmax(S). P has dimensions [tokens_block_size, key_value_tokens].
  {
    void* logits_row = logits;
    size_t i = tokens_block_size;
    do {
      // Skip initialization of locals as they will be written to immediately.
      float rowmax;
      context->rmax_ukernel(
        /*batch=*/key_value_tokens_scaled,
        /*input=*/logits_row,
        /*output=*/&rowmax,
        /*params=*/&context->rmax_params);

      float rowsum;
      context->raddstoreexpminusmax_ukernel(
        /*batch=*/key_value_tokens_scaled,
        /*input=*/logits_row,
        /*max=*/&rowmax,
        /*output=*/logits_row,
        /*sum=*/&rowsum,
        /*params=*/&context->expminus_params);

      float rowscale;
      context->compute_reciprocal(
        /*input=*/&rowsum,
        /*output=*/&rowscale);

      context->vmulc_ukernel(
        /*batch=*/key_value_tokens_scaled,
        /*input_x=*/logits_row,
        /*input_y=*/&rowscale,
        /*output=*/logits_row,
        /*params=*/minmax_params);

      logits_row = (void*) ((uintptr_t) logits_row + key_value_tokens_scaled);
    } while (--i != 0);
  }

  {
    void* value = (void*) ((uintptr_t) context->value +
                           batch_index * context->value_batch_stride +
                           head_index * context->value_head_stride);
    const size_t output_tile_offset =
      batch_index * context->output_batch_stride + head_index * context->output_head_stride +
      tokens_start * context->value_scaled_channels;
    // O = GEMM(P, V). O has dimension [tokens_block_size, value_channels].
    context->gemm_ukernel.function[uarch_index](
        /*mr=*/tokens_block_size,
        /*nc=*/context->value_channels,
        /*kc=*/key_value_tokens_scaled,
        /*a=*/logits,
        /*a_stride=*/key_value_tokens_scaled,
        /*w=*/value,
        /*c=*/(void*) ((uintptr_t) context->output + output_tile_offset),
        /*cm_stride=*/context->value_scaled_channels,
        /*cn_stride=*/cn_stride,
        /*params=*/minmax_params);
  }
}

void xnn_compute_hmp_scaled_dot_product_attention_with_thread(
  const struct scaled_dot_product_attention_context context[restrict XNN_MIN_ELEMENTS(1)],
  uint32_t uarch_index,
  size_t thread_index,
  size_t batch_index,
  size_t head_index,
  size_t tokens_start,
  size_t tokens_block_size)
{
  const size_t query_key_scaled_channels = context->query_key_scaled_channels;
  const size_t query_tile_offset =
    batch_index * context->query_batch_stride + head_index * context->query_head_stride +
    tokens_start * query_key_scaled_channels;
  const size_t key_value_tokens_scaled = context->key_value_tokens_scaled;
  const size_t key_value_tokens_start_scaled = tokens_start * key_value_tokens_scaled;
  const size_t cn_stride = context->cn_stride;
  const void* scaled_query =
    (void*) ((uintptr_t) context->scaled_query + thread_index * context->scaled_query_thread_stride);
  const void* minmax_params = &context->minmax_params;

  {
    uintptr_t query = (uintptr_t) context->query + query_tile_offset;
    uintptr_t query_scaled_current = (uintptr_t) scaled_query;
    // Q_scaled = Q * Scale (along channels). Q and Q_scaled have dimensions [tokens_block_size, query_key_channels].
    size_t i = tokens_block_size;
    do {
      context->vmul_ukernel(
        /*batch=*/query_key_scaled_channels,
        /*input_x=*/(const void*) query,
        /*input_y=*/context->scale,
        /*output=*/(void*) query_scaled_current,
        /*params=*/minmax_params);
      query += query_key_scaled_channels;
      query_scaled_current += query_key_scaled_channels;
    } while (--i != 0);
  }

  void* const logits = (void*) ((uintptr_t) context->logits_buffer + thread_index * context->logits_thread_stride);

  {
    void* key = (void*) ((uintptr_t) context->key +
                         batch_index * context->key_batch_stride +
                         head_index * context->key_head_stride);
    // S = GEMM(Q_scaled, K^t). S is [tokens_block_size, key_value_tokens].
    context->gemm_ukernel.function[uarch_index](
      /*mr=*/tokens_block_size,
      /*nr=*/context->key_value_tokens,
      /*k=*/query_key_scaled_channels,
      /*a=*/scaled_query,
      /*a_stride=*/query_key_scaled_channels,
      /*w=*/(void*) key,
      /*c=*/(void*) (uintptr_t) logits,
      /*cm_stride=*/key_value_tokens_scaled,
      /*cn_stride=*/cn_stride,
      /*params=*/minmax_params);
  }

  {
    const size_t tokens_block_size_scaled = tokens_block_size * key_value_tokens_scaled;
    struct attention_logits_cap logits_cap = context->logits_cap;
    if (logits_cap.type == xnn_attention_logits_cap_type_tanh) {
      // (Optional) S = TanH(S/Cap) * Cap. Overwrites buffer.
      context->vmulc_ukernel(
        /*batch=*/tokens_block_size_scaled,
        /*input_x=*/logits,
        /*input_y=*/&logits_cap.cap_reciprocal,
        /*output=*/logits,
        /*params=*/minmax_params);
      context->vtanh_ukernel(
        /*batch=*/tokens_block_size_scaled,
        /*input=*/logits,
        /*output=*/logits,
        /*params=*/&context->tanh_params);
      context->vmulc_ukernel(
        /*batch=*/tokens_block_size_scaled,
        /*input_x=*/logits,
        /*input_y=*/&logits_cap.cap,
        /*output=*/logits,
        /*params=*/minmax_params);
    }

    // S = S + Mask. Mask has dimensions [query_tokens, key_value_tokens].
    // Mask. Overwrites buffer.
    context->vadd_ukernel(
      /*batch=*/tokens_block_size_scaled,
      /*input_x=*/logits,
      /*input_y=*/(void*) ((uintptr_t) context->mask + key_value_tokens_start_scaled),
      /*output=*/logits,
      /*params=*/minmax_params);
  }

  // P = Softmax(S). P has dimensions [tokens_block_size, key_value_tokens].
  {
    void* logits_row = logits;
    size_t i = tokens_block_size;
    do {
      // Skip initialization of locals as they will be written to immediately.
      float rowmax;
      context->rmax_ukernel(
        /*batch=*/key_value_tokens_scaled,
        /*input=*/logits_row,
        /*output=*/&rowmax,
        /*params=*/&context->rmax_params);

      float rowsum;
      context->raddstoreexpminusmax_ukernel(
        /*batch=*/key_value_tokens_scaled,
        /*input=*/logits_row,
        /*max=*/&rowmax,
        /*output=*/logits_row,
        /*sum=*/&rowsum,
        /*params=*/&context->expminus_params);

      float rowscale;
      context->compute_reciprocal(
        /*input=*/&rowsum,
        /*output=*/&rowscale);

      context->vmulc_ukernel(
        /*batch=*/key_value_tokens_scaled,
        /*input_x=*/logits_row,
        /*input_y=*/&rowscale,
        /*output=*/logits_row,
        /*params=*/minmax_params);

      logits_row = (void*) ((uintptr_t) logits_row + key_value_tokens_scaled);
    } while (--i != 0);
  }

  {
    void* value = (void*) ((uintptr_t) context->value +
                           batch_index * context->value_batch_stride +
                           head_index * context->value_head_stride);
    const size_t output_tile_offset =
      batch_index * context->output_batch_stride + head_index * context->output_head_stride +
      tokens_start * context->value_scaled_channels;
    // O = GEMM(P, V). O has dimension [tokens_block_size, value_channels].
    context->gemm_ukernel.function[uarch_index](
        /*mr=*/tokens_block_size,
        /*nc=*/context->value_channels,
        /*kc=*/key_value_tokens_scaled,
        /*a=*/logits,
        /*a_stride=*/key_value_tokens_scaled,
        /*w=*/value,
        /*c=*/(void*) ((uintptr_t) context->output + output_tile_offset),
        /*cm_stride=*/context->value_scaled_channels,
        /*cn_stride=*/cn_stride,
        /*params=*/minmax_params);
  }
}
#endif  // XNN_MAX_UARCH_TYPES > 1

enum xnn_status xnn_run_operator(xnn_operator_t op, pthreadpool_t threadpool)
{
  return xnn_run_operator_with_index(op, 0, 0, threadpool);
}

enum xnn_status xnn_run_operator_with_index(
  xnn_operator_t op,
  size_t opdata_index,
  size_t operator_object_index,
  pthreadpool_t threadpool)
{
  switch (op->state) {
    case xnn_run_state_invalid:
      xnn_log_error("failed to run operator: operator was not successfully setup");
      return xnn_status_invalid_state;
    case xnn_run_state_ready:
      xnn_log_debug("running operator %zu:%zu (%s %s)", opdata_index,
                    operator_object_index,
                    xnn_operator_type_to_string(op->type),
                    xnn_microkernel_type_to_string(op->ukernel.type));
      break;
    case xnn_run_state_skip:
      xnn_log_debug("skip running operator %zu:%zu (%s %s)", opdata_index,
                    operator_object_index,
                    xnn_operator_type_to_string(op->type),
                    xnn_microkernel_type_to_string(op->ukernel.type));
      return xnn_status_success;
    case xnn_run_state_needs_setup:
      xnn_log_error(
        "failed to run operator %zu:%zu (%s %s): operator has been reshaped but not yet setup", opdata_index,
        operator_object_index, xnn_operator_type_to_string(op->type), xnn_microkernel_type_to_string(op->ukernel.type));
      return xnn_status_invalid_state;
  }

  uint32_t flags = PTHREADPOOL_FLAG_DISABLE_DENORMALS;
  if (op->flags & XNN_FLAG_YIELD_WORKERS) {
    flags |= PTHREADPOOL_FLAG_YIELD_WORKERS;
  }
  for (size_t i = 0; i < XNN_MAX_COMPUTE_INVOCATIONS; i++) {
    switch (op->compute[i].type) {
      case xnn_parallelization_type_invalid:
        break;
      case xnn_parallelization_type_1d:
        assert(op->compute[i].range[0] != 0);
        pthreadpool_parallelize_1d(
            threadpool,
            op->compute[i].task_1d,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0],
            flags);
        break;
      case xnn_parallelization_type_1d_with_thread:
        assert(op->compute[i].range[0] != 0);
        pthreadpool_parallelize_1d_with_thread(
            threadpool,
            op->compute[i].task_1d_with_thread,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0],
            flags);
        break;
      case xnn_parallelization_type_1d_tile_1d:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].tile[0] != 0);
        pthreadpool_parallelize_1d_tile_1d(
            threadpool,
            op->compute[i].task_1d_tile_1d,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0],
            op->compute[i].tile[0],
            flags);
        break;
      case xnn_parallelization_type_2d:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        pthreadpool_parallelize_2d(
            threadpool,
            op->compute[i].task_2d,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0], op->compute[i].range[1],
            flags);
        break;
      case xnn_parallelization_type_2d_with_thread:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        pthreadpool_parallelize_2d_with_thread(
            threadpool,
            op->compute[i].task_2d_with_thread,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0], op->compute[i].range[1],
            flags);
        break;
      case xnn_parallelization_type_2d_tile_1d:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].tile[0] != 0);
        pthreadpool_parallelize_2d_tile_1d(
            threadpool,
            op->compute[i].task_2d_tile_1d,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0], op->compute[i].range[1],
            op->compute[i].tile[0],
            flags);
        break;
      case xnn_parallelization_type_2d_tile_2d:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].tile[0] != 0);
        assert(op->compute[i].tile[1] != 0);
        pthreadpool_parallelize_2d_tile_2d(
            threadpool,
            op->compute[i].task_2d_tile_2d,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0], op->compute[i].range[1],
            op->compute[i].tile[0], op->compute[i].tile[1],
            flags);
        break;
      case xnn_parallelization_type_3d:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].range[2] != 0);
        pthreadpool_parallelize_3d(
            threadpool,
            op->compute[i].task_3d,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0], op->compute[i].range[1], op->compute[i].range[2],
            flags);
        break;
      case xnn_parallelization_type_3d_tile_1d:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].range[2] != 0);
        assert(op->compute[i].tile[0] != 0);
        pthreadpool_parallelize_3d_tile_1d(
            threadpool,
            op->compute[i].task_3d_tile_1d,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0], op->compute[i].range[1], op->compute[i].range[2],
            op->compute[i].tile[0],
            flags);
        break;
      case xnn_parallelization_type_3d_tile_1d_with_thread:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].range[2] != 0);
        assert(op->compute[i].tile[0] != 0);
        pthreadpool_parallelize_3d_tile_1d_with_thread(
            threadpool,
            op->compute[i].task_3d_tile_1d_with_thread,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0], op->compute[i].range[1], op->compute[i].range[2],
            op->compute[i].tile[0],
            flags);
        break;
      case xnn_parallelization_type_3d_tile_2d:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].range[2] != 0);
        assert(op->compute[i].tile[0] != 0);
        assert(op->compute[i].tile[1] != 0);
        pthreadpool_parallelize_3d_tile_2d(
            threadpool,
            op->compute[i].task_3d_tile_2d,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0], op->compute[i].range[1], op->compute[i].range[2],
            op->compute[i].tile[0], op->compute[i].tile[1],
            flags);
        break;
      case xnn_parallelization_type_4d:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].range[2] != 0);
        assert(op->compute[i].range[3] != 0);
        pthreadpool_parallelize_4d(
            threadpool,
            op->compute[i].task_4d,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0], op->compute[i].range[1], op->compute[i].range[2], op->compute[i].range[3],
            flags);
        break;
      case xnn_parallelization_type_4d_tile_2d:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].range[2] != 0);
        assert(op->compute[i].range[3] != 0);
        assert(op->compute[i].tile[0] != 0);
        assert(op->compute[i].tile[1] != 0);
        pthreadpool_parallelize_4d_tile_2d(
            threadpool,
            op->compute[i].task_4d_tile_2d,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0], op->compute[i].range[1], op->compute[i].range[2], op->compute[i].range[3],
            op->compute[i].tile[0], op->compute[i].tile[1],
            flags);
        break;
      case xnn_parallelization_type_5d:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].range[2] != 0);
        assert(op->compute[i].range[3] != 0);
        assert(op->compute[i].range[4] != 0);
        pthreadpool_parallelize_5d(
            threadpool,
            op->compute[i].task_5d,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0], op->compute[i].range[1], op->compute[i].range[2], op->compute[i].range[3],
              op->compute[i].range[4],
            flags);
        break;
      case xnn_parallelization_type_5d_tile_2d:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].range[2] != 0);
        assert(op->compute[i].range[3] != 0);
        assert(op->compute[i].range[4] != 0);
        assert(op->compute[i].tile[0] != 0);
        assert(op->compute[i].tile[1] != 0);
        pthreadpool_parallelize_5d_tile_2d(
            threadpool,
            op->compute[i].task_5d_tile_2d,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0], op->compute[i].range[1], op->compute[i].range[2], op->compute[i].range[3],
              op->compute[i].range[4],
            op->compute[i].tile[0], op->compute[i].tile[1],
            flags);
        break;
      case xnn_parallelization_type_6d_tile_2d:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].range[2] != 0);
        assert(op->compute[i].range[3] != 0);
        assert(op->compute[i].range[4] != 0);
        assert(op->compute[i].range[5] != 0);
        assert(op->compute[i].tile[0] != 0);
        assert(op->compute[i].tile[1] != 0);
        pthreadpool_parallelize_6d_tile_2d(
            threadpool,
            op->compute[i].task_6d_tile_2d,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            op->compute[i].range[0], op->compute[i].range[1], op->compute[i].range[2], op->compute[i].range[3],
              op->compute[i].range[4], op->compute[i].range[5],
            op->compute[i].tile[0], op->compute[i].tile[1],
            flags);
        break;
  #if XNN_MAX_UARCH_TYPES > 1
      case xnn_parallelization_type_2d_tile_1d_with_uarch:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].tile[0] != 0);
        pthreadpool_parallelize_2d_tile_1d_with_uarch(
            threadpool,
            op->compute[i].task_2d_tile_1d_with_id,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
            op->compute[i].range[0], op->compute[i].range[1],
            op->compute[i].tile[0],
            flags);
        break;
      case xnn_parallelization_type_2d_tile_2d_with_uarch:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].tile[0] != 0);
        assert(op->compute[i].tile[1] != 0);
        pthreadpool_parallelize_2d_tile_2d_with_uarch(
            threadpool,
            op->compute[i].task_2d_tile_2d_with_id,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
            op->compute[i].range[0], op->compute[i].range[1],
            op->compute[i].tile[0], op->compute[i].tile[1],
            flags);
        break;
      case xnn_parallelization_type_3d_tile_1d_with_uarch:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].range[2] != 0);
        assert(op->compute[i].tile[0] != 0);
        pthreadpool_parallelize_3d_tile_1d_with_uarch(
            threadpool,
            op->compute[i].task_3d_tile_1d_with_id,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
            op->compute[i].range[0], op->compute[i].range[1], op->compute[i].range[2],
            op->compute[i].tile[0],
            flags);
        break;
      case xnn_parallelization_type_3d_tile_1d_with_uarch_with_thread:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].range[2] != 0);
        assert(op->compute[i].tile[0] != 0);
        pthreadpool_parallelize_3d_tile_1d_with_uarch_with_thread(
            threadpool,
            op->compute[i].task_3d_tile_1d_with_id_with_thread,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
            op->compute[i].range[0], op->compute[i].range[1], op->compute[i].range[2],
            op->compute[i].tile[0],
            flags);
        break;
      case xnn_parallelization_type_3d_tile_2d_with_uarch:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].range[2] != 0);
        assert(op->compute[i].tile[0] != 0);
        assert(op->compute[i].tile[1] != 0);
        pthreadpool_parallelize_3d_tile_2d_with_uarch(
            threadpool,
            op->compute[i].task_3d_tile_2d_with_id,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
            op->compute[i].range[0], op->compute[i].range[1], op->compute[i].range[2],
            op->compute[i].tile[0], op->compute[i].tile[1],
            flags);
        break;
      case xnn_parallelization_type_4d_tile_2d_with_uarch:
        assert(op->compute[i].range[0] != 0);
        assert(op->compute[i].range[1] != 0);
        assert(op->compute[i].range[2] != 0);
        assert(op->compute[i].range[3] != 0);
        assert(op->compute[i].tile[0] != 0);
        assert(op->compute[i].tile[1] != 0);
        pthreadpool_parallelize_4d_tile_2d_with_uarch(
            threadpool,
            op->compute[i].task_4d_tile_2d_with_id,
            (void*) ((uintptr_t) &op->context + op->compute[i].context_offset),
            0 /* default uarch index */, XNN_MAX_UARCH_TYPES - 1,
            op->compute[i].range[0], op->compute[i].range[1], op->compute[i].range[2], op->compute[i].range[3],
            op->compute[i].tile[0], op->compute[i].tile[1],
            flags);
        break;
  #endif  // XNN_MAX_UARCH_TYPES > 1
      default:
        XNN_UNREACHABLE;
    }
  }
  return xnn_status_success;
}
