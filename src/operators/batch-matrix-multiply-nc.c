// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/cache.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/compute.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microkernel-type.h"
#include "src/xnnpack/microkernel-utils.h"
#include "src/xnnpack/microparams-init.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/pack.h"
#include "src/xnnpack/packq.h"
#include "src/xnnpack/params.h"
#include <pthreadpool.h>

enum xnn_status create_batch_matrix_multiply_nc(
    uint32_t flags, const void* params, size_t params_size,
    const struct xnn_gemm_config* gemm_config,
    const struct gemm_fused_ukernels* gemm_ukernels,
    enum xnn_operator_type operator_type,
    xnn_operator_t* batch_matrix_multiply_op_out) {
  xnn_operator_t batch_matrix_multiply_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_out_of_memory;
  batch_matrix_multiply_op =
      xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (batch_matrix_multiply_op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct xnn_operator),
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }

  memcpy(&batch_matrix_multiply_op->params, params, params_size);
  batch_matrix_multiply_op->type = operator_type;
  batch_matrix_multiply_op->flags = flags;
  batch_matrix_multiply_op->gemm_config = gemm_config;

  const size_t mr = gemm_config->mr;
  batch_matrix_multiply_op->ukernel.type = xnn_microkernel_type_gemm;
  batch_matrix_multiply_op->ukernel.gemm = (struct xnn_ukernel_gemm){
      .mr = mr,
      .mr_packed = gemm_config->mr_packed,
      .nr = gemm_config->nr,
      .kr = UINT32_C(1) << gemm_config->log2_kr,
      .sr = UINT32_C(1) << gemm_config->log2_sr,
  };

  assert(mr <= XNN_MAX_MR);
  for (size_t i = 0; i < mr; i++) {
    batch_matrix_multiply_op->ukernel.gemm.gemm_cases[i] =
        gemm_ukernels->gemm[i];
  }
  if (batch_matrix_multiply_op->flags & XNN_FLAG_TRANSPOSE_B) {
    batch_matrix_multiply_op->ukernel.gemm.packw_gemm_goi =
        gemm_config->pack_gemm_goi;
  } else {
    batch_matrix_multiply_op->ukernel.gemm.packw_gemm_gio =
        gemm_config->pack_gemm_gio;
  }

  batch_matrix_multiply_op->state = xnn_run_state_invalid;

  *batch_matrix_multiply_op_out = batch_matrix_multiply_op;
  return xnn_status_success;

error:
  xnn_delete_operator(batch_matrix_multiply_op);
  return status;
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_f16(
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f16_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(
            xnn_operator_type_batch_matrix_multiply_nc_f16));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  if (gemm_config->linear.gemm[gemm_config->mr - 1]
          .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f16_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f16 != NULL) {
    gemm_config->init.f16(&params, xnn_float16_from_float(-INFINITY),
                          xnn_float16_from_float(INFINITY));
  }

  return create_batch_matrix_multiply_nc(
      flags, &params, sizeof(params), gemm_config, gemm_ukernels,
      xnn_operator_type_batch_matrix_multiply_nc_f16,
      batch_matrix_multiply_op_out);
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_pf16(
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_pf16_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(
            xnn_operator_type_batch_matrix_multiply_nc_pf16));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  if (gemm_config->linear.gemm[gemm_config->mr - 1]
          .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f16_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f16 != NULL) {
    gemm_config->init.f16(&params, xnn_float16_from_float(-INFINITY),
                          xnn_float16_from_float(INFINITY));
  }

  return create_batch_matrix_multiply_nc(
      flags, &params, sizeof(params), gemm_config, gemm_ukernels,
      xnn_operator_type_batch_matrix_multiply_nc_pf16,
      batch_matrix_multiply_op_out);
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_bf16_f32(
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_bf16_f32_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(
            xnn_operator_type_batch_matrix_multiply_nc_bf16_f32));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  if (gemm_config->linear.gemm[gemm_config->mr - 1]
          .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f32_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&params, -INFINITY, INFINITY);
  }

  return create_batch_matrix_multiply_nc(
      flags, &params, sizeof(params), gemm_config, gemm_ukernels,
      xnn_operator_type_batch_matrix_multiply_nc_bf16_f32,
      batch_matrix_multiply_op_out);
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_f32(
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(
            xnn_operator_type_batch_matrix_multiply_nc_f32));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  if (gemm_config->linear.gemm[gemm_config->mr - 1]
          .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f32_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&params, -INFINITY, INFINITY);
  }

  return create_batch_matrix_multiply_nc(
      flags, &params, sizeof(params), gemm_config, gemm_ukernels,
      xnn_operator_type_batch_matrix_multiply_nc_f32,
      batch_matrix_multiply_op_out);
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_pf32(
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_pf32_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(
            xnn_operator_type_batch_matrix_multiply_nc_pf32));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  if (gemm_config->linear.gemm[gemm_config->mr - 1]
          .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f32_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&params, -INFINITY, INFINITY);
  }

  return create_batch_matrix_multiply_nc(
      flags, &params, sizeof(params), gemm_config, gemm_ukernels,
      xnn_operator_type_batch_matrix_multiply_nc_pf32,
      batch_matrix_multiply_op_out);
}

enum xnn_status create_batch_matrix_multiply_nc_fx_const_weights(
    size_t batch_size_b, size_t k, size_t n, const void* data_b,
    size_t log2_kernel_element_size, size_t bias_element_size, uint32_t flags,
    xnn_operator_t* batch_matrix_multiply_op_out) {
  xnn_operator_t batch_matrix_multiply_op = *batch_matrix_multiply_op_out;
  batch_matrix_multiply_op->context.gemm.const_weights = true;
  const struct xnn_gemm_config* gemm_config =
      batch_matrix_multiply_op->gemm_config;

  // Check if we've already cached the packed data for `B`.
  uint32_t cache_seed = murmur_hash3(
      &batch_matrix_multiply_op->context.gemm.gemm.gemm,
      sizeof(batch_matrix_multiply_op->context.gemm.gemm.gemm), k * n);
  if (batch_matrix_multiply_op->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    cache_seed = ~cache_seed;
  }
  size_t cache_offset = XNN_CACHE_NOT_FOUND;
  struct xnn_weights_cache_look_up_key cache_key;
  cache_key.seed = cache_seed;
  cache_key.kernel = data_b;
  cache_key.bias = NULL;
  if (use_weights_cache(batch_matrix_multiply_op)) {
    cache_offset = xnn_weights_cache_look_up(
        batch_matrix_multiply_op->weights_cache, &cache_key);
  }

  // Compute the shape and size of the packed data.
  const uint32_t kr = batch_matrix_multiply_op->ukernel.gemm.kr;
  const uint32_t sr = batch_matrix_multiply_op->ukernel.gemm.sr;
  const size_t k_stride = round_up_po2(k, kr * sr);
  const size_t weights_stride =
      gemm_config->packed_stride_weights_and_biases
          ? gemm_config->packed_stride_weights_and_biases(
                gemm_config, k, /*block_size=*/k_stride, k_stride,
                /*extra_bytes=*/0)
          : (k_stride << log2_kernel_element_size) + bias_element_size;
  batch_matrix_multiply_op->weights_stride = weights_stride;

  // If the packed data has not been cached, pack and cache it.
  if (cache_offset == XNN_CACHE_NOT_FOUND) {
    const uint32_t nr = batch_matrix_multiply_op->ukernel.gemm.nr;
    const size_t n_stride = round_up(n, nr);
    const size_t packed_size = batch_size_b * n_stride * weights_stride;
    const size_t aligned_size =
        round_up_po2(packed_size, XNN_ALLOCATION_ALIGNMENT);

    // Allocate the packed weights.
    void* packed_data = xnn_get_pointer_to_write_weights(
        batch_matrix_multiply_op, aligned_size);
    if (packed_data == NULL) {
      xnn_log_error(
          "failed to allocate %zu bytes for %s operator packed weights",
          packed_size,
          xnn_operator_type_to_string_v2(batch_matrix_multiply_op));
      return xnn_status_out_of_memory;
    }
    xnn_log_debug(
        "allocated %zu bytes for packed weights in %s operator (ptr=%p)",
        aligned_size,
        xnn_operator_type_to_string_v2(batch_matrix_multiply_op),
        packed_data);

    // Pack the weights.
    if (gemm_config->pack_weights_and_biases) {
      gemm_config->pack_weights_and_biases(flags, gemm_config, k, n,
                                           /*groups=*/batch_size_b,
                                           /*unused_block_size=*/0,
                                           /*kstride=*/k_stride,
                                           /*accumulator_init=*/NULL,
                                           /*weights=*/data_b,
                                           /*int_extra_data0_fn=*/NULL,
                                           /*extra_data0=*/NULL,
                                           /*extra_data0_size=*/0,
                                           /*init_extra_data1_fn=*/
                                           NULL,
                                           /*extra_data1=*/NULL,
                                           /*extra_data1_size=*/0,
                                           /*packed_weights_ptr=*/packed_data,
                                           /*packing_params=*/NULL);
    } else {
      if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
        batch_matrix_multiply_op->ukernel.gemm.packw_gemm_goi(
            /*groups=*/batch_size_b, n, k, nr, kr, sr, data_b,
            /*bias=*/NULL, /*scale=*/NULL, packed_data,
            /*extra_bytes=*/0, /*packing_params=*/NULL);
      } else {
        batch_matrix_multiply_op->ukernel.gemm.packw_gemm_gio(
            /*groups=*/batch_size_b, n, k, nr, kr, sr, n, data_b,
            /*bias=*/NULL,
            /*scale=*/NULL, packed_data,
            /*extra_bytes=*/0, /*packing_params=*/NULL);
      }
    }

    // Cache the weights.
    if (use_weights_cache(batch_matrix_multiply_op)) {
      batch_matrix_multiply_op->packed_weights.offset =
          xnn_look_up_or_insert_weights_cache(
              batch_matrix_multiply_op->weights_cache, &cache_key, packed_data,
              aligned_size);
    }

  } else {
    // Retrieve the packed weights from the cache entry.
    batch_matrix_multiply_op->packed_weights.offset = cache_offset;
  }

  return xnn_status_success;
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_f16_const_weights(
    size_t batch_size_b, size_t k, size_t n, const void* data_b, uint32_t flags,
    xnn_operator_t* batch_matrix_multiply_op_out) {
  const enum xnn_status status = xnn_create_batch_matrix_multiply_nc_f16(
      flags, batch_matrix_multiply_op_out);
  if (status != xnn_status_success) {
    return status;
  }

  return create_batch_matrix_multiply_nc_fx_const_weights(
      batch_size_b, k, n, data_b,
      /*log2_kernel_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*bias_element_size=*/sizeof(xnn_float16), flags,
      batch_matrix_multiply_op_out);
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_pf16_const_weights(
    size_t batch_size_b, size_t k, size_t n, const xnn_float16* data_b,
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op_out) {
  const enum xnn_status status = xnn_create_batch_matrix_multiply_nc_pf16(
      flags, batch_matrix_multiply_op_out);
  if (status != xnn_status_success) {
    return status;
  }

  return create_batch_matrix_multiply_nc_fx_const_weights(
      batch_size_b, k, n, data_b,
      /*log2_kernel_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*bias_element_size=*/sizeof(xnn_float16), flags,
      batch_matrix_multiply_op_out);
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_f32_const_weights(
    size_t batch_size_b, size_t k, size_t n, const float* data_b,
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op_out) {
  const enum xnn_status status = xnn_create_batch_matrix_multiply_nc_f32(
      flags, batch_matrix_multiply_op_out);
  if (status != xnn_status_success) {
    return status;
  }

  return create_batch_matrix_multiply_nc_fx_const_weights(
      batch_size_b, k, n, data_b,
      /*log2_kernel_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*bias_element_size=*/sizeof(float), flags, batch_matrix_multiply_op_out);
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_pf32_const_weights(
    size_t batch_size_b, size_t k, size_t n, const float* data_b,
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op_out) {
  const enum xnn_status status = xnn_create_batch_matrix_multiply_nc_pf32(
      flags, batch_matrix_multiply_op_out);
  if (status != xnn_status_success) {
    return status;
  }

  return create_batch_matrix_multiply_nc_fx_const_weights(
      batch_size_b, k, n, data_b,
      /*log2_kernel_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*bias_element_size=*/sizeof(float), flags, batch_matrix_multiply_op_out);
}

enum xnn_status create_batch_matrix_multiply_nc_qx8_f32_qc8w(
    size_t batch_size_b, size_t k, size_t n, const int8_t* data_b,
    const float* scale_b, uint32_t flags,
    const struct xnn_gemm_config* gemm_config,
    enum xnn_operator_type expected_operator_type,
    xnn_operator_t* batch_matrix_multiply_op_out) {
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  if (gemm_config->linear.gemm[gemm_config->mr - 1]
          .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->linear;
  }

  struct xnn_f32_minmax_params params;
  if XNN_LIKELY (gemm_config->init.f32 != NULL) {
    gemm_config->init.f32(&params, -INFINITY, INFINITY);
  }

  enum xnn_status status = create_batch_matrix_multiply_nc(
      flags, &params, sizeof(params), gemm_config, gemm_ukernels,
      expected_operator_type, batch_matrix_multiply_op_out);
  if (status != xnn_status_success) {
    return status;
  }
  xnn_operator_t batch_matrix_multiply_op = *batch_matrix_multiply_op_out;

  // Check if we've already cached the packed data for `B`.
  uint32_t cache_seed = murmur_hash3(
      &batch_matrix_multiply_op->context.gemm.gemm.gemm,
      sizeof(batch_matrix_multiply_op->context.gemm.gemm.gemm), k * n);
  if (batch_matrix_multiply_op->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    cache_seed = ~cache_seed;
  }
  size_t cache_offset = XNN_CACHE_NOT_FOUND;
  struct xnn_weights_cache_look_up_key cache_key;
  cache_key.seed = cache_seed;
  cache_key.kernel = data_b;
  cache_key.bias = NULL;
  if (use_weights_cache(batch_matrix_multiply_op)) {
    cache_offset = xnn_weights_cache_look_up(
        batch_matrix_multiply_op->weights_cache, &cache_key);
  }

  const uint32_t kr = batch_matrix_multiply_op->ukernel.gemm.kr;
  const uint32_t sr = batch_matrix_multiply_op->ukernel.gemm.sr;
  const size_t extra_bytes = 2 * sizeof(float);
  const size_t k_stride = round_up_po2(k, kr * sr);
  const size_t weights_stride =
      gemm_config->packed_stride_weights_and_biases
          ? gemm_config->packed_stride_weights_and_biases(
                gemm_config, k, /*unused_blocksize=*/0, k_stride, extra_bytes)
          : (k_stride << XNN_LOG2_SIZEOF_INT8_T) + extra_bytes +
                sizeof(int32_t);
  batch_matrix_multiply_op->weights_stride = weights_stride;

  // If the packed data has not been cached, pack and cache it.
  if (cache_offset == XNN_CACHE_NOT_FOUND) {
    const uint32_t nr = batch_matrix_multiply_op->ukernel.gemm.nr;
    const size_t n_stride = round_up(n, nr);
    const size_t packed_size = batch_size_b * n_stride * weights_stride;
    const size_t aligned_size =
        round_up_po2(packed_size, XNN_ALLOCATION_ALIGNMENT);

    void* packed_data = xnn_get_pointer_to_write_weights(
        batch_matrix_multiply_op, aligned_size);
    if (packed_data == NULL) {
      xnn_log_error(
          "failed to allocate %zu bytes for %s operator packed weights",
          packed_size,
          xnn_operator_type_to_string_v2(batch_matrix_multiply_op));
      return xnn_status_out_of_memory;
    }
    xnn_log_debug(
        "allocated %zu bytes for packed weights in %s operator (ptr=%p)",
        aligned_size,
        xnn_operator_type_to_string_v2(batch_matrix_multiply_op),
        packed_data);
    if (extra_bytes > 0) {
      // TODO(b/402602597): We shouldn't need this initialization.
      memset(packed_data, 0, aligned_size);
    }

    if (gemm_config->pack_weights_and_biases) {
      const struct xnn_qs8_qc8w_packing_params pack_gemm_params = {
          /*input_zero_point=*/1, 1.0f};
      gemm_config->pack_weights_and_biases(
          batch_matrix_multiply_op->flags ^ XNN_FLAG_TRANSPOSE_WEIGHTS,
          gemm_config, /*input_channels=*/k,
          /*output_channels=*/n,
          /*groups=*/batch_size_b,
          /*unused_block_size=*/0,
          /*k_stride=*/k_stride,
          /*accumulator_init=*/NULL,
          /*weights=*/data_b,
          /*int_extra_data0_fn=*/
          (xnn_init_scale_params_fn)xnn_init_qs8_qc8w_scale_fp32_params,
          /*extra_data0=*/NULL,
          /*extra_data0_size=*/sizeof(float),
          /*init_extra_data1_fn=*/
          (xnn_init_scale_params_fn)xnn_init_qs8_qc8w_scale_fp32_params,
          /*extra_data1=*/scale_b,
          /*extra_data1_size=*/sizeof(float),
          /*packed_weights_ptr=*/packed_data, &pack_gemm_params);
    } else {
      const struct xnn_qs8_packing_params pack_gemm_params = {
          /*input_zero_point=*/1};
      if (batch_matrix_multiply_op->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
        batch_matrix_multiply_op->ukernel.gemm.packw_gemm_goi(
            /*groups=*/batch_size_b, n, k, nr, kr, sr, data_b, /*b=*/NULL,
            /*scale=*/NULL, packed_data, nr * extra_bytes, &pack_gemm_params);
      } else {
        batch_matrix_multiply_op->ukernel.gemm.packw_gemm_gio(
            /*groups=*/batch_size_b, n, k, nr, kr, sr, n, data_b, /*b=*/NULL,
            /*scale=*/NULL, packed_data, nr * extra_bytes, &pack_gemm_params);
      }

      if (scale_b != NULL) {
        for (size_t batch = 0; batch < batch_size_b; batch++) {
          void* packed_data_batch =
              (void*)((char*)packed_data + batch * n_stride * weights_stride);
          void* weights = (void*)((uintptr_t)packed_data_batch +
                                  nr * ((k_stride << XNN_LOG2_SIZEOF_INT8_T) +
                                        sizeof(int32_t)));
          xnn_init_qs8_qc8w_scale_fp32_params(n, nr, nr * weights_stride,
                                              &scale_b[batch * n], weights);
        }
      }
    }

    if (use_weights_cache(batch_matrix_multiply_op)) {
      batch_matrix_multiply_op->packed_weights.offset =
          xnn_look_up_or_insert_weights_cache(
              batch_matrix_multiply_op->weights_cache, &cache_key, packed_data,
              aligned_size);
    }

  } else {
    batch_matrix_multiply_op->packed_weights.offset = cache_offset;
  }

  return xnn_status_success;
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_qd8_f32_qc8w(
    size_t batch_size_b, size_t k, size_t n, const int8_t* data_b,
    const float* scale_b, uint32_t flags,
    xnn_operator_t* batch_matrix_multiply_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qd8_f32_qc8w_gemm_config();
  return create_batch_matrix_multiply_nc_qx8_f32_qc8w(
      batch_size_b, k, n, data_b, scale_b, flags, gemm_config,
      xnn_operator_type_batch_matrix_multiply_nc_qd8_f32_qc8w,
      batch_matrix_multiply_op_out);
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_qp8_f32_qc8w(
    size_t batch_size_b, size_t k, size_t n, const int8_t* data_b,
    const float* scale_b, uint32_t flags,
    xnn_operator_t* batch_matrix_multiply_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qp8_f32_qc8w_gemm_config();
  return create_batch_matrix_multiply_nc_qx8_f32_qc8w(
      batch_size_b, k, n, data_b, scale_b, flags, gemm_config,
      xnn_operator_type_batch_matrix_multiply_nc_qp8_f32_qc8w,
      batch_matrix_multiply_op_out);
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_qdu8_f32_qc8w(
    size_t batch_size_b, size_t k, size_t n, const int8_t* data_b,
    const float* scale_b, uint32_t flags,
    xnn_operator_t* batch_matrix_multiply_op_out) {
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qdu8_f32_qc8w_gemm_config();
  return create_batch_matrix_multiply_nc_qx8_f32_qc8w(
      batch_size_b, k, n, data_b, scale_b, flags, gemm_config,
      xnn_operator_type_batch_matrix_multiply_nc_qdu8_f32_qc8w,
      batch_matrix_multiply_op_out);
}

static enum xnn_status reshape_batch_matrix_multiply_nc(
    xnn_operator_t batch_matrix_multiply_op,
    enum xnn_operator_type expected_operator_type, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, size_t* workspace_alignment,
    uint32_t log2_input_a_element_size, uint32_t log2_input_b_element_size,
    uint32_t bias_element_size, uint32_t w_stride_extra_bytes,
    uint32_t log2_output_element_size, const void* params, size_t params_size,
    size_t num_threads) {
  if (batch_matrix_multiply_op->type != expected_operator_type) {
    xnn_log_error(
        "failed to setup operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(batch_matrix_multiply_op));
    return xnn_status_invalid_parameter;
  }
  batch_matrix_multiply_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
                  xnn_operator_type_to_string_v2(batch_matrix_multiply_op));
    return xnn_status_uninitialized;
  }

  if (m == 0) {
    xnn_log_error(
        "failed to reshape %s operator with %zu rows: number of rows must be "
        "non-zero",
        xnn_operator_type_to_string_v2(batch_matrix_multiply_op), m);
    return xnn_status_invalid_parameter;
  }

  if (k == 0) {
    xnn_log_error(
        "failed to reshape %s operator with %zu columns: number of columns "
        "must be non-zero",
        xnn_operator_type_to_string_v2(batch_matrix_multiply_op), k);
    return xnn_status_invalid_parameter;
  }

  if (n == 0) {
    xnn_log_error(
        "failed to reshape %s operator with %zu columns: number of columns "
        "must be non-zero",
        xnn_operator_type_to_string_v2(batch_matrix_multiply_op), n);
    return xnn_status_invalid_parameter;
  }

  // Compute the batch sizes of the A and B tensors.
  size_t batch_dims_c[XNN_MAX_TENSOR_DIMS];
  size_t batch_size_a = 1;
  size_t batch_size_b = 1;
  size_t batch_size_c = 1;
  for (int k = 0; k < num_batch_dims; k++) {
    batch_dims_c[k] = max(batch_dims_a[k], batch_dims_b[k]);
    batch_size_a *= batch_dims_a[k];
    batch_size_b *= batch_dims_b[k];
    batch_size_c *= batch_dims_c[k];
  }

  // Compute the stride for each batch dimension of the output C.
  size_t batch_strides_c[XNN_MAX_TENSOR_DIMS];
  if (num_batch_dims > 0) {
    batch_strides_c[num_batch_dims - 1] = 1;
    for (int k = (int)num_batch_dims - 2; k >= 0; k--) {
      batch_strides_c[k] = batch_strides_c[k + 1] * batch_dims_c[k + 1];
    }
  }

  if (batch_size_c == 0) {
    batch_matrix_multiply_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  // Fail if the batch sizes for `A` and `B` are not compatible.
  for (int k = 0; k < num_batch_dims; k++) {
    if ((batch_dims_a[k] != 1 && batch_dims_c[k] != batch_dims_a[k]) ||
        (batch_dims_b[k] != 1 && batch_dims_c[k] != batch_dims_b[k])) {
      xnn_log_error(
          "failed to reshape %s operator with incompatible %i-th batch "
          "dimensions %zu and %zu: batch dimensions must be equal or "
          "broadcastable",
          xnn_operator_type_to_string_v2(batch_matrix_multiply_op), k,
          batch_dims_a[k], batch_dims_b[k]);
      return xnn_status_invalid_parameter;
    }
  }

  const uint32_t nr = batch_matrix_multiply_op->ukernel.gemm.nr;
  const uint32_t kr = batch_matrix_multiply_op->ukernel.gemm.kr;
  const uint32_t sr = batch_matrix_multiply_op->ukernel.gemm.sr;

  uint32_t mr = batch_matrix_multiply_op->ukernel.gemm.mr;
  struct xnn_hmp_gemm_ukernel* gemm_cases =
      batch_matrix_multiply_op->ukernel.gemm.gemm_cases;

  if (m == 1 && batch_matrix_multiply_op->ukernel.gemm.gemm_cases[0]
                        .function[XNN_UARCH_DEFAULT] != NULL) {
    mr = 1;
  }

  const uint32_t mr_packed =
      m > 1 ? batch_matrix_multiply_op->ukernel.gemm.mr_packed : 1;

  assert(mr != 0 && mr <= XNN_MAX_MR);
  struct xnn_hmp_gemm_ukernel gemm_ukernel = gemm_cases[mr - 1];

  struct compute_parameters* gemm_compute =
      &batch_matrix_multiply_op->compute[0];

  switch (batch_matrix_multiply_op->type) {
    case xnn_operator_type_batch_matrix_multiply_nc_qd8_f32_qc8w:
    case xnn_operator_type_batch_matrix_multiply_nc_qdu8_f32_qc8w:
    case xnn_operator_type_batch_matrix_multiply_nc_qp8_f32_qc8w:
      // Nothing to do here, the `B` matrix has already been packed.
      break;

    case xnn_operator_type_batch_matrix_multiply_nc_bf16_f32:
    case xnn_operator_type_batch_matrix_multiply_nc_f16:
    case xnn_operator_type_batch_matrix_multiply_nc_f32:
    case xnn_operator_type_batch_matrix_multiply_nc_pf16:
    case xnn_operator_type_batch_matrix_multiply_nc_pf32: {
      // Do nothing if the weights don't need to be packed.
      if (batch_matrix_multiply_op->context.gemm.const_weights) {
        break;
      }

      gemm_compute = &batch_matrix_multiply_op->compute[1];

      const size_t n_stride = round_up(n, nr);
      const size_t k_stride = round_up_po2(k, kr * sr);
      const struct xnn_gemm_config* gemm_config =
          batch_matrix_multiply_op->gemm_config;
      const size_t weights_stride =
          gemm_config->packed_stride_weights_and_biases
              ? gemm_config->packed_stride_weights_and_biases(
                    gemm_config, k, /*block_size=*/k_stride, k_stride,
                    /*extra_bytes=*/0)
              : (k_stride << log2_input_b_element_size) + bias_element_size;
      const size_t input_b_batch_stride = n_stride * weights_stride;

      // Store the computed weights stride in the op for later use.
      batch_matrix_multiply_op->weights_stride = weights_stride;

      // Compute the required workspace size.
      if (workspace_size != NULL) {
        *workspace_size = batch_size_b * input_b_batch_stride;
      }
      if (workspace_alignment != NULL) {
        *workspace_alignment = XNN_ALLOCATION_ALIGNMENT;
      }

      if (batch_matrix_multiply_op->flags & XNN_FLAG_TRANSPOSE_B) {
        assert(batch_matrix_multiply_op->ukernel.gemm.packw_gemm_goi != NULL ||
               gemm_config->pack_weights_and_biases);
        batch_matrix_multiply_op->context.gemm.packw_gemm_goi =
            (struct packw_gemm_goi_context){
                .kc = k,
                .nr = nr,
                .kr = kr,
                .sr = sr,
                .k_stride = k << log2_input_b_element_size,
                .bias = NULL,
                .b_stride = bias_element_size,
                .w_stride = weights_stride,
                .packw_gemm_goi =
                    batch_matrix_multiply_op->ukernel.gemm.packw_gemm_goi,
                .gk_stride = n * (k << log2_input_b_element_size),
                .gb_stride = n * bias_element_size,
                .gc_stride = input_b_batch_stride,
                .pack_weights_and_biases = gemm_config->pack_weights_and_biases,
                .gemm_config = gemm_config,
            };
        batch_matrix_multiply_op->compute[0].task_2d_tile_1d_dynamic =
            (pthreadpool_task_2d_tile_1d_dynamic_t)
                xnn_compute_batched_packw_gemm_goi;
        batch_matrix_multiply_op->compute[0].context_offset =
            offsetof(struct xnn_operator, context.gemm.packw_gemm_goi) -
            offsetof(struct xnn_operator, context);
      } else {
        assert(batch_matrix_multiply_op->ukernel.gemm.packw_gemm_gio != NULL ||
               gemm_config->pack_weights_and_biases);
        batch_matrix_multiply_op->context.gemm.packw_gemm_gio =
            (struct packw_gemm_gio_context){
                .n_stride = 1 << log2_input_b_element_size,
                .k_stride_elements = n,
                .kc = k,
                .nr = nr,
                .kr = kr,
                .sr = sr,
                .bias = NULL,
                .b_stride = bias_element_size,
                .w_stride = weights_stride,
                .packw_gemm_gio =
                    batch_matrix_multiply_op->ukernel.gemm.packw_gemm_gio,
                .gk_stride = k * (n << log2_input_b_element_size),
                .gb_stride = n * bias_element_size,
                .gc_stride = input_b_batch_stride,
                .pack_weights_and_biases = gemm_config->pack_weights_and_biases,
                .gemm_config = gemm_config,
            };
        batch_matrix_multiply_op->compute[0].task_2d_tile_1d_dynamic =
            (pthreadpool_task_2d_tile_1d_dynamic_t)
                xnn_compute_batched_packw_gemm_gio;
        batch_matrix_multiply_op->compute[0].context_offset =
            offsetof(struct xnn_operator, context.gemm.packw_gemm_gio) -
            offsetof(struct xnn_operator, context);
      }
      batch_matrix_multiply_op->compute[0].type =
          xnn_parallelization_type_2d_tile_1d_dynamic;
      batch_matrix_multiply_op->compute[0].range[0] = batch_size_b;
      batch_matrix_multiply_op->compute[0].range[1] = n;
      batch_matrix_multiply_op->compute[0].tile[0] = nr;
      break;
    }
    default:
      XNN_UNREACHABLE;
  }

  const bool packed_lhs =
      (batch_matrix_multiply_op->type ==
       xnn_operator_type_batch_matrix_multiply_nc_qp8_f32_qc8w) ||
      (batch_matrix_multiply_op->type ==
       xnn_operator_type_batch_matrix_multiply_nc_pf16) ||
      (batch_matrix_multiply_op->type ==
       xnn_operator_type_batch_matrix_multiply_nc_pf32);

  const size_t k_scaled = k << log2_input_a_element_size;
  const size_t a_stride =
      packed_lhs ? xnn_x8_packq_f32qp8_packed_offset(mr, k, mr, kr, sr)
                 : k_scaled;

  batch_matrix_multiply_op->context.gemm.gemm.gemm = (struct gemm_context){
      .k_scaled = k_scaled,
      .a_stride = a_stride,
      .ga_stride = m * k_scaled,
      .w_stride = batch_matrix_multiply_op->weights_stride,
      .gw_stride = batch_matrix_multiply_op->weights_stride * round_up(n, nr),
      .cm_stride = n << log2_output_element_size,
      .cn_stride = nr << log2_output_element_size,
      .gc_stride = (m * n) << log2_output_element_size,
      .log2_csize = log2_output_element_size,
      .gq_stride = m,
      .num_batch_dims = num_batch_dims,
      .mr = mr,
      .kr = kr,
      .sr = sr,
      .ukernel = gemm_ukernel,
      .kc = k,
  };

  if (packed_lhs) {
    switch (batch_matrix_multiply_op->type) {
      case xnn_operator_type_batch_matrix_multiply_nc_qp8_f32_qc8w:
        batch_matrix_multiply_op->context.gemm.gemm.gemm.packed_lh_offset_fn =
            xnn_x8_packq_f32qp8_packed_offset;
        batch_matrix_multiply_op->context.gemm.gemm.gemm.ga_stride =
            xnn_x8_packq_f32qp8_packed_size(m, k, mr_packed, kr, sr);
        break;
      case xnn_operator_type_batch_matrix_multiply_nc_pf16:
        batch_matrix_multiply_op->context.gemm.gemm.gemm.packed_lh_offset_fn =
            xnn_init_x16_pack_lh_config()->offset_fn;
        batch_matrix_multiply_op->context.gemm.gemm.gemm.ga_stride =
            xnn_init_x16_pack_lh_config()->size_fn(m, k, mr_packed, kr, sr);
        break;
      case xnn_operator_type_batch_matrix_multiply_nc_pf32:
        batch_matrix_multiply_op->context.gemm.gemm.gemm.packed_lh_offset_fn =
            xnn_init_x32_pack_lh_config()->offset_fn;
        batch_matrix_multiply_op->context.gemm.gemm.gemm.ga_stride =
            xnn_init_x32_pack_lh_config()->size_fn(m, k, mr_packed, kr, sr);
        break;
      default:
        XNN_UNREACHABLE;
    }
  }

  // Copy the batch dimensions into the `gemm_context` struct since we don't
  // know if these pointers will be valid by the time the GEMM is actually
  // called.
  memcpy(batch_matrix_multiply_op->context.gemm.gemm.gemm.batch_dims_a,
         batch_dims_a, sizeof(size_t) * num_batch_dims);
  memcpy(batch_matrix_multiply_op->context.gemm.gemm.gemm.batch_dims_b,
         batch_dims_b, sizeof(size_t) * num_batch_dims);
  memcpy(batch_matrix_multiply_op->context.gemm.gemm.gemm.batch_strides_c,
         batch_strides_c, sizeof(size_t) * num_batch_dims);
  memcpy(&batch_matrix_multiply_op->context.gemm.gemm.gemm.params, params,
         params_size);
  batch_matrix_multiply_op->context.gemm.gemm.gemm.fused_params =
      &batch_matrix_multiply_op->context.gemm.gemm.gemm.params;

  // Compute the optimal tile size for this GEMM.
  const size_t nc = xnn_gemm_best_tile_size(
      /*num_groups=*/batch_size_c, m, n,
      /*m_stride=*/batch_matrix_multiply_op->context.gemm.gemm.gemm.a_stride,
      /*n_stride=*/batch_matrix_multiply_op->context.gemm.gemm.gemm.w_stride,
      /*cm_stride=*/batch_matrix_multiply_op->context.gemm.gemm.gemm.cm_stride,
      /*cn_stride=*/1 << log2_output_element_size, mr, nr, num_threads);

#if XNN_MAX_UARCH_TYPES > 1
  if (xnn_is_hmp_gemm_ukernel(gemm_ukernel)) {
    gemm_compute->type = xnn_parallelization_type_3d_tile_2d_dynamic_with_uarch;
    if (packed_lhs) {
      gemm_compute->task_3d_tile_2d_dynamic_with_id =
          (pthreadpool_task_3d_tile_2d_dynamic_with_id_t)
              xnn_compute_hmp_grouped_qp8gemm;
    } else {
      gemm_compute->task_3d_tile_2d_dynamic_with_id =
          (pthreadpool_task_3d_tile_2d_dynamic_with_id_t)
              xnn_compute_hmp_grouped_gemm;
    }
  } else {
    gemm_compute->type = xnn_parallelization_type_3d_tile_2d_dynamic;
    if (packed_lhs) {
      gemm_compute->task_3d_tile_2d_dynamic =
          (pthreadpool_task_3d_tile_2d_dynamic_t)xnn_compute_grouped_qp8gemm;
    } else {
      gemm_compute->task_3d_tile_2d_dynamic =
          (pthreadpool_task_3d_tile_2d_dynamic_t)xnn_compute_grouped_gemm;
    }
  }
#else
  gemm_compute->type = xnn_parallelization_type_3d_tile_2d_dynamic;
  if (packed_lhs) {
    gemm_compute->task_3d_tile_2d_dynamic =
        (pthreadpool_task_3d_tile_2d_dynamic_t)xnn_compute_grouped_qp8gemm;
  } else {
    gemm_compute->task_3d_tile_2d_dynamic =
        (pthreadpool_task_3d_tile_2d_dynamic_t)xnn_compute_grouped_gemm;
  }
#endif
  gemm_compute->range[0] = batch_size_c;
  gemm_compute->range[2] = m;
  gemm_compute->range[1] = n;
  gemm_compute->tile[1] = mr;
  gemm_compute->tile[0] = nc;
  batch_matrix_multiply_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_f16(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, size_t* workspace_alignment,
    pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_f16,
      num_batch_dims, batch_dims_a, batch_dims_b, m, k, n, workspace_size,
      workspace_alignment,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*bias_element_size=*/sizeof(uint16_t),
      /*w_stride_extra_bytes=*/0,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
      &batch_matrix_multiply_op->params.f16_minmax,
      sizeof(batch_matrix_multiply_op->params.f16_minmax),
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_pf16(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, size_t* workspace_alignment,
    pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_pf16,
      num_batch_dims, batch_dims_a, batch_dims_b, m, k, n, workspace_size,
      workspace_alignment,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*bias_element_size=*/sizeof(uint16_t),
      /*w_stride_extra_bytes=*/0,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
      &batch_matrix_multiply_op->params.f16_minmax,
      sizeof(batch_matrix_multiply_op->params.f16_minmax),
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_bf16_f32(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, size_t* workspace_alignment,
    pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_bf16_f32,
      num_batch_dims, batch_dims_a, batch_dims_b, m, k, n, workspace_size,
      workspace_alignment,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*bias_element_size=*/sizeof(float),
      /*w_stride_extra_bytes=*/0,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &batch_matrix_multiply_op->params.f32_minmax,
      sizeof(batch_matrix_multiply_op->params.f32_minmax),
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_f32(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, size_t* workspace_alignment,
    pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_f32,
      num_batch_dims, batch_dims_a, batch_dims_b, m, k, n, workspace_size,
      workspace_alignment,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*bias_element_size=*/sizeof(float),
      /*w_stride_extra_bytes=*/0,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &batch_matrix_multiply_op->params.f32_minmax,
      sizeof(batch_matrix_multiply_op->params.f32_minmax),
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_pf32(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, size_t* workspace_alignment,
    pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_pf32,
      num_batch_dims, batch_dims_a, batch_dims_b, m, k, n, workspace_size,
      workspace_alignment,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*bias_element_size=*/sizeof(float),
      /*w_stride_extra_bytes=*/0,
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &batch_matrix_multiply_op->params.f32_minmax,
      sizeof(batch_matrix_multiply_op->params.f32_minmax),
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_qd8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_qd8_f32_qc8w, num_batch_dims,
      batch_dims_a, batch_dims_b, m, k, n, /*workspace_size=*/NULL,
      /*workspace_alignment=*/NULL,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*bias_element_size=*/sizeof(int32_t),
      /*w_stride_extra_bytes=*/2 * sizeof(float),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &batch_matrix_multiply_op->params.f32_minmax,
      sizeof(batch_matrix_multiply_op->params.f32_minmax),
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_qp8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_qp8_f32_qc8w, num_batch_dims,
      batch_dims_a, batch_dims_b, m, k, n, /*workspace_size=*/NULL,
      /*workspace_alignment=*/NULL,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*bias_element_size=*/sizeof(int32_t),
      /*w_stride_extra_bytes=*/2 * sizeof(float),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &batch_matrix_multiply_op->params.f32_minmax,
      sizeof(batch_matrix_multiply_op->params.f32_minmax),
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_qdu8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_qdu8_f32_qc8w, num_batch_dims,
      batch_dims_a, batch_dims_b, m, k, n, /*workspace_size=*/NULL,
      /*workspace_alignment=*/NULL,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*bias_element_size=*/sizeof(int32_t),
      /*w_stride_extra_bytes=*/2 * sizeof(float),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &batch_matrix_multiply_op->params.f32_minmax,
      sizeof(batch_matrix_multiply_op->params.f32_minmax),
      pthreadpool_get_threads_count(threadpool));
}

static enum xnn_status setup_batch_matrix_multiply_nc(
    xnn_operator_t batch_matrix_multiply_op,
    enum xnn_operator_type expected_operator_type, const void* input_a,
    const struct xnn_quantization_params* quantization_params,
    const void* input_b, void* packed_weights, void* output) {
  if (batch_matrix_multiply_op->type != expected_operator_type) {
    xnn_log_error(
        "failed to setup operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(batch_matrix_multiply_op));
    return xnn_status_invalid_parameter;
  }

  switch (batch_matrix_multiply_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
          "failed to setup %s operator: operator has not been reshaped yet",
          xnn_operator_type_to_string_v2(batch_matrix_multiply_op));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different
      // pointers.
      break;
  }

  if (batch_matrix_multiply_op->flags & XNN_FLAG_TRANSPOSE_B) {
    batch_matrix_multiply_op->context.gemm.packw_gemm_goi.kernel = input_b;
    batch_matrix_multiply_op->context.gemm.packw_gemm_goi.bias = NULL;
    batch_matrix_multiply_op->context.gemm.packw_gemm_goi.packed_weights =
        packed_weights;
  } else {
    batch_matrix_multiply_op->context.gemm.packw_gemm_gio.kernel = input_b;
    batch_matrix_multiply_op->context.gemm.packw_gemm_gio.bias = NULL;
    batch_matrix_multiply_op->context.gemm.packw_gemm_gio.packed_weights =
        packed_weights;
  }

  batch_matrix_multiply_op->context.gemm.gemm.gemm.a = input_a;
  batch_matrix_multiply_op->context.gemm.gemm.gemm.packed_w = packed_weights;
  batch_matrix_multiply_op->context.gemm.gemm.gemm.c = output;
  batch_matrix_multiply_op->context.gemm.gemm.gemm.quantization_params =
      (const void*)quantization_params;

  batch_matrix_multiply_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_f16(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const void* input_a, const void* input_b, void* output) {
  return setup_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_f16,
      input_a, /*quantization_params=*/NULL, input_b,
      /*packed_weights=*/
      batch_matrix_multiply_op->context.gemm.const_weights
          ? packed_weights(batch_matrix_multiply_op)
          : workspace,
      output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_pf16(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const void* input_a, const void* input_b, void* output) {
  return setup_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_pf16,
      input_a, /*quantization_params=*/NULL, input_b,
      /*packed_weights=*/
      batch_matrix_multiply_op->context.gemm.const_weights
          ? packed_weights(batch_matrix_multiply_op)
          : workspace,
      output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_bf16_f32(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const void* input_a, const void* input_b, void* output) {
  return setup_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_bf16_f32,
      input_a, /*quantization_params=*/NULL, input_b,
      /*packed_weights=*/
      batch_matrix_multiply_op->context.gemm.const_weights
          ? packed_weights(batch_matrix_multiply_op)
          : workspace,
      output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_f32(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const float* input_a, const float* input_b, float* output) {
  return setup_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_f32,
      input_a, /*quantization_params=*/NULL, input_b,
      /*packed_weights=*/
      batch_matrix_multiply_op->context.gemm.const_weights
          ? packed_weights(batch_matrix_multiply_op)
          : workspace,
      output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_pf32(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const float* input_a, const float* input_b, float* output) {
  return setup_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_pf32,
      input_a, /*quantization_params=*/NULL, input_b,
      /*packed_weights=*/
      batch_matrix_multiply_op->context.gemm.const_weights
          ? packed_weights(batch_matrix_multiply_op)
          : workspace,
      output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_qd8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, const int8_t* input_a,
    const struct xnn_quantization_params* quantization_params, float* output) {
  return setup_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_qd8_f32_qc8w, input_a,
      quantization_params, /*input_b=*/NULL,
      packed_weights(batch_matrix_multiply_op), output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_qp8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, const int8_t* input_a,
    float* output) {
  return setup_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_qp8_f32_qc8w, input_a,
      /*quantization_params=*/NULL, /*input_b=*/NULL,
      packed_weights(batch_matrix_multiply_op), output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_qdu8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, const int8_t* input_a,
    const struct xnn_quantization_params* quantization_params, float* output) {
  return setup_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_qdu8_f32_qc8w, input_a,
      quantization_params, /*input_b=*/NULL,
      packed_weights(batch_matrix_multiply_op), output);
}
