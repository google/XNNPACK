
// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/config-types.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/packw.h"
#include "xnnpack/pack.h"
#include "xnnpack/unaligned.h"
#include "pthreadpool.h"
#include <stdio.h>

struct packw_gemm_goi_bl_context {
  // Number of input channels.
  size_t kc;
  // Number of output channels the GEMM is optimized for.
  size_t nr;
  size_t kr;
  size_t sr;
  size_t bl;
  // Pointer to kernel.
  const void* kernel;
  // Stride, in bytes, between each N of the kernel.
  size_t k_stride;
  // Pointer to bias.
  const void* bias;
  // Stride, in bytes, between each bias.
  size_t b_stride;
  // Output pointer to write packed kernel and bias.
  void* packed_weights;
  // Stride, in bytes, between each packed kernel and bias.
  size_t w_stride;
  // scales
  const void* scales;
  // Stride, in bytes between each N scales.
  size_t s_stride;
  // extra bytes
  size_t extra_bytes_n;
  // extra bytes
  size_t extra_bytes_bl;

  // Strides used for batched packw.
  // Stride, in bytes, between each group of kernel
  size_t gk_stride;
  // Stride, in bytes, between each group of bias.
  size_t gb_stride;
  // Stride, in bytes, between each group of packed weights.
  size_t gc_stride;

  // Packing params passed to the packing microkernel.
  const void *params;

  // Microkernel to preform packing.
  xnn_packw_gemm_goi_bl_ukernel_fn packw_gemm_goi_bl;
};

void xnn_compute_packw_gemm_goi_bl(
    const struct packw_gemm_goi_bl_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t n_block_start,
    size_t n_block_size)
{
  const void* kernel = (const void*) ((const uintptr_t) context->kernel + context->k_stride * n_block_start);
  const void* bias = context->bias;
  if (bias != NULL) {
    bias = (const void*) ((const uintptr_t) bias + (n_block_start * context->b_stride));
  }
  const void* scales = (const void*) ((const uintptr_t)context->scales + (n_block_start * context->s_stride));

  void* packed_weights = (void*) ((uintptr_t) context->packed_weights + context->w_stride * n_block_start);

  context->packw_gemm_goi_bl(
      /*groups=*/1, 
      /*nc=*/n_block_size, 
      /*kc=*/context->kc, 
      /*nr=*/context->nr, 
      /*kr=*/context->kr,
      /*sr=*/context->sr, 
      /*bl=*/context->bl,
      /*k=*/kernel, 
      /*bias=*/bias, 
      /*scale=*/scales, 
      /*packed_weights=*/packed_weights,
      /*extra_bytes=*/context->extra_bytes_bl, 
      /*extra_bytes_n=*/context->extra_bytes_n,
      /*params=*/context->params);
    
}

void xnn_multithread_qb4_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t block_size, size_t k_stride, const void* accumulator_init, const void* weights,
    xnn_init_scale_params_fn init_extra_data0_fn, const void* extra_data0,
    size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    xnn_packw_gemm_goi_bl_ukernel_fn packw_gemm_goi_bl, const void* params, pthreadpool_t threadpool) 
{
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const size_t planes = gemm_config->planes;

  const size_t extra_bytes_bl = sizeof(uint16_t);
  const size_t extra_bytes_n = sizeof(uint32_t);
  const size_t num_blocks = input_channels / block_size;
  const size_t num_bytes_ksum = 4;

  struct packw_gemm_goi_bl_context context= (struct packw_gemm_goi_bl_context) {
    .kc = input_channels,
    .nr = nr,
    .kr = kr,
    .sr = sr,
    .bl = block_size,
    .kernel = weights,
    .k_stride = k_stride,
    .bias = accumulator_init,
    .b_stride = sizeof(int32_t),
    .packed_weights = packed_weights_ptr,
    .w_stride = k_stride + extra_bytes_n + num_blocks * extra_bytes_bl + num_bytes_ksum,
    .scales = extra_data1,
    .s_stride = sizeof(xnn_bfloat16) * num_blocks,
    .extra_bytes_bl = nr * extra_bytes_bl,
    .extra_bytes_n = nr * extra_bytes_n,
    .params = params,
    .packw_gemm_goi_bl = packw_gemm_goi_bl,
  };

  uint32_t pthreadpool_flags = PTHREADPOOL_FLAG_DISABLE_DENORMALS;
  pthreadpool_parallelize_1d_tile_1d(
    threadpool,
    (pthreadpool_task_1d_tile_1d_t) xnn_compute_packw_gemm_goi_bl,
    (void*) ((uintptr_t) &context),
    output_channels,
    nr, 
    pthreadpool_flags
  );
}

void xnn_pack_qb4_x16c8_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t block_size, size_t k_stride, const void* accumulator_init, const void* weights,
    xnn_init_scale_params_fn init_extra_data0_fn, const void* extra_data0,
    size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params, pthreadpool_t threadpool) {
  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    // No packing ukernel for gio
    xnn_pack_qb4_weights_and_biases(
      flags, gemm_config, input_channels, output_channels, groups,
      block_size, k_stride, accumulator_init, weights, init_extra_data0_fn,
      extra_data0, extra_data0_element_size, init_extra_data1_fn, extra_data1,
      extra_data1_element_size, packed_weights_ptr, params, threadpool);
  } else {
    xnn_multithread_qb4_weights_and_biases(
      flags, gemm_config, input_channels, output_channels, groups,
      block_size, k_stride, accumulator_init, weights, init_extra_data0_fn,
      extra_data0, extra_data0_element_size, init_extra_data1_fn, extra_data1,
      extra_data1_element_size, packed_weights_ptr, 
      (xnn_packw_gemm_goi_bl_ukernel_fn) xnn_qb4_packw_gemm_goi_ukernel_x16c8__scalar,
      params, threadpool);
  }
}

void xnn_pack_qb4_x16c4_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t block_size, size_t k_stride, const void* accumulator_init, const void* weights,
    xnn_init_scale_params_fn init_extra_data0_fn, const void* extra_data0,
    size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params, pthreadpool_t threadpool) {
  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    // No packing ukernel for gio
    xnn_pack_qb4_weights_and_biases(
      flags, gemm_config, input_channels, output_channels, groups,
      block_size, k_stride, accumulator_init, weights, init_extra_data0_fn,
      extra_data0, extra_data0_element_size, init_extra_data1_fn, extra_data1,
      extra_data1_element_size, packed_weights_ptr, params, threadpool);
  } else {
    xnn_multithread_qb4_weights_and_biases(
      flags, gemm_config, input_channels, output_channels, groups,
      block_size, k_stride, accumulator_init, weights, init_extra_data0_fn,
      extra_data0, extra_data0_element_size, init_extra_data1_fn, extra_data1,
      extra_data1_element_size, packed_weights_ptr, 
      (xnn_packw_gemm_goi_bl_ukernel_fn) xnn_qb4_packw_gemm_goi_ukernel_x16c4__scalar,
      params, threadpool);
  }
}
