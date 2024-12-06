
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

void xnn_pack_qb4_x16c8_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t block_size, size_t k_stride, const void* accumulator_init, const void* weights,
    xnn_init_scale_params_fn init_extra_data0_fn, const void* extra_data0,
    size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    // No packing ukernel for gio
    return xnn_pack_qb4_weights_and_biases(
      flags, gemm_config, input_channels, output_channels, groups,
      block_size, k_stride, accumulator_init, weights, init_extra_data0_fn,
      extra_data0, extra_data0_element_size, init_extra_data1_fn, extra_data1,
      extra_data1_element_size, packed_weights_ptr, params);
  }
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const size_t planes = gemm_config->planes;

  const size_t extra_bytes_bl = sizeof(uint16_t);
  const size_t extra_bytes_n = sizeof(uint32_t);

  xnn_qb4_packw_gemm_goi_ukernel_x16c8__scalar(
    /*g=*/groups, 
    /*nc=*/output_channels, 
    /*kc=*/input_channels,
    /*nr=*/nr, 
    /*kr=*/kr, 
    /*sr=*/sr,
    /*bl=*/block_size,
    /*k=*/(const uint8_t*)weights, 
    /*bias=*/(const int32_t*)accumulator_init,
    /*scale=*/(const xnn_bfloat16*)extra_data1,
    /*packed_weights=*/(int8_t*)packed_weights_ptr,
    /*extra_bytes_bl=*/nr * extra_bytes_bl,
    /*extra_bytes_n=*/nr * extra_bytes_n,
    /*params*/(const struct xnn_qs8_qc4w_packing_params *)params);
}

void xnn_pack_qb4_x16c4_weights_and_biases(
    uint32_t flags, const struct xnn_gemm_config* gemm_config,
    size_t input_channels, size_t output_channels, size_t groups,
    size_t block_size, size_t k_stride, const void* accumulator_init, const void* weights,
    xnn_init_scale_params_fn init_extra_data0_fn, const void* extra_data0,
    size_t extra_data0_element_size,
    xnn_init_scale_params_fn init_extra_data1_fn, const void* extra_data1,
    size_t extra_data1_element_size, void* packed_weights_ptr,
    const void* params) {
  if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    // No packing ukernel for gio
    return xnn_pack_qb4_weights_and_biases(
      flags, gemm_config, input_channels, output_channels, groups,
      block_size, k_stride, accumulator_init, weights, init_extra_data0_fn,
      extra_data0, extra_data0_element_size, init_extra_data1_fn, extra_data1,
      extra_data1_element_size, packed_weights_ptr, params);
  }
  const uint32_t nr = gemm_config->nr;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;
  const size_t planes = gemm_config->planes;

  const size_t extra_bytes_bl = sizeof(uint16_t);
  const size_t extra_bytes_n = sizeof(uint32_t);

  xnn_qb4_packw_gemm_goi_ukernel_x16c4__scalar(
    /*g=*/groups, 
    /*nc=*/output_channels, 
    /*kc=*/input_channels,
    /*nr=*/nr, 
    /*kr=*/kr, 
    /*sr=*/sr,
    /*bl=*/block_size,
    /*k=*/(const uint8_t*)weights, 
    /*bias=*/(const int32_t*)accumulator_init,
    /*scale=*/(const xnn_bfloat16*)extra_data1,
    /*packed_weights=*/(int8_t*)packed_weights_ptr,
    /*extra_bytes_bl=*/nr * extra_bytes_bl,
    /*extra_bytes_n=*/nr * extra_bytes_n,
    /*params*/(const struct xnn_qs8_qc4w_packing_params *)params);
}
