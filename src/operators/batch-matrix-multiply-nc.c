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
#include "src/xnnpack/params.h"
#include <pthreadpool.h>

// This op has at most three `compute`s:
//  * Dynamic weights packing,
//  * Inlined LHS packing,
//  * The GEMM.
#define XNN_BATCH_MATMUL_MAX_COMPUTE_INVOCATIONS 3

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
  batch_matrix_multiply_op->compute =
      xnn_allocate_zero_memory(XNN_BATCH_MATMUL_MAX_COMPUTE_INVOCATIONS *
                               sizeof(struct compute_parameters));
  if (batch_matrix_multiply_op->compute == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct compute_parameters),
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }
  // This will be set to the appropriate value in
  // `reshape_batch_matrix_multiply_nc`.
  batch_matrix_multiply_op->num_compute_invocations = 0;

  batch_matrix_multiply_op->ukernel.gemm_ukernels =
      xnn_allocate_zero_simd_memory(sizeof(struct gemm_types));
  if (batch_matrix_multiply_op->ukernel.gemm_ukernels == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct gemm_types),
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }

  batch_matrix_multiply_op->dynamic_context.gemm =
      xnn_allocate_zero_simd_memory(sizeof(struct gemm_op_context));
  if (batch_matrix_multiply_op->dynamic_context.gemm == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct gemm_op_context),
                  xnn_operator_type_to_string(operator_type));
    goto error;
  }

  memcpy(&batch_matrix_multiply_op->params, params, params_size);
  batch_matrix_multiply_op->type = operator_type;
  batch_matrix_multiply_op->flags = flags;
  batch_matrix_multiply_op->gemm_config = gemm_config;

  const size_t mr = gemm_config->mr;
  const size_t mr_packed = gemm_config->mr_packed ? gemm_config->mr_packed : mr;
  batch_matrix_multiply_op->ukernel.type = xnn_microkernel_type_gemm;
  batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm =
      (struct xnn_ukernel_gemm){
          .mr = mr,
          .mr_packed = mr_packed,
          .nr = gemm_config->nr,
          .kr = UINT32_C(1) << gemm_config->log2_kr,
          .sr = UINT32_C(1) << gemm_config->log2_sr,
      };

  assert(mr <= XNN_MAX_MR);
  for (size_t i = 0; i < mr; i++) {
    batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.gemm_cases[i] =
        gemm_ukernels->gemm[i];
  }
  if (batch_matrix_multiply_op->flags & XNN_FLAG_TRANSPOSE_B) {
    batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.packw_gemm_goi =
        gemm_config->pack_gemm_goi;
  } else {
    batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.packw_gemm_gio =
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
  const struct xnn_gemm_config* gemm_config = xnn_init_f32_gemm_config(flags);
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

enum xnn_status create_batch_matrix_multiply_nc_const_weights(
    size_t batch_size_b, size_t k, size_t n, const void* data_b,
    size_t log2_kernel_element_size, size_t bias_element_size,
    const void* packing_params, xnn_init_scale_params_fn init_scale_params,
    const float* scale_params, size_t extra_weights_bytes, uint32_t flags,
    xnn_operator_t* batch_matrix_multiply_op_out) {
  xnn_operator_t batch_matrix_multiply_op = *batch_matrix_multiply_op_out;
  batch_matrix_multiply_op->dynamic_context.gemm->const_weights = true;
  const struct xnn_gemm_config* gemm_config =
      batch_matrix_multiply_op->gemm_config;

  // Check if we've already cached the packed data for `B`.
  uint32_t cache_seed = murmur_hash3(
      &batch_matrix_multiply_op->dynamic_context.gemm->gemm,
      sizeof(batch_matrix_multiply_op->dynamic_context.gemm->gemm), k * n);
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
  const uint32_t kr = batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.kr;
  const uint32_t sr = batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.sr;
  const size_t k_stride = round_up_po2(k, kr * sr);
  const size_t weights_stride =
      gemm_config->packed_stride_weights_and_biases
          ? gemm_config->packed_stride_weights_and_biases(
                gemm_config, k, /*block_size=*/k_stride, k_stride,
                extra_weights_bytes)
          : (k_stride << log2_kernel_element_size) + bias_element_size +
              extra_weights_bytes;
  batch_matrix_multiply_op->weights_stride = weights_stride;

  // If the packed data has not been cached, pack and cache it.
  if (cache_offset == XNN_CACHE_NOT_FOUND) {
    const uint32_t nr =
        batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.nr;
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
        aligned_size, xnn_operator_type_to_string_v2(batch_matrix_multiply_op),
        packed_data);

    // Pack the weights.
    if (gemm_config->pack_weights_and_biases) {
      gemm_config->pack_weights_and_biases(
          (flags ^ XNN_FLAG_TRANSPOSE_WEIGHTS), gemm_config, k, n,
          /*groups=*/batch_size_b, /*block_size=*/0,
          /*kstride=*/(flags & XNN_FLAG_TRANSPOSE_WEIGHTS) ? k : n,
          /*accumulator_init=*/NULL, /*weights=*/data_b,
          /*int_extra_data0_fn=*/init_scale_params,
          /*extra_data0=*/scale_params,
          /*extra_data0_size=*/extra_weights_bytes,
          /*init_extra_data1_fn=*/NULL, /*extra_data1=*/NULL,
          /*extra_data1_size=*/0, /*packed_weights_ptr=*/packed_data,
          packing_params);
    } else {
      if (flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
        batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.packw_gemm_goi(
            /*groups=*/batch_size_b, n, k, nr, kr, sr, data_b, /*bias=*/NULL,
            /*scale=*/NULL, packed_data,
            /*extra_bytes=*/nr * extra_weights_bytes, packing_params);
      } else {
        batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.packw_gemm_gio(
            /*groups=*/batch_size_b, n, k, nr, kr, sr, n, data_b, /*bias=*/NULL,
            /*scale=*/NULL, packed_data,
            /*extra_bytes=*/nr * extra_weights_bytes, packing_params);
      }

      if (scale_params != NULL) {
        assert(init_scale_params != NULL);
        for (size_t batch = 0; batch < batch_size_b; batch++) {
          void* weights_batch =
              (void*)((char*)packed_data + batch * n_stride * weights_stride);
          void* weights = (void*)((uintptr_t)weights_batch +
                                  nr * (weights_stride - extra_weights_bytes));
          init_scale_params(n, nr, nr * weights_stride, scale_params, weights);
        }
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

  return create_batch_matrix_multiply_nc_const_weights(
      batch_size_b, k, n, data_b,
      /*log2_kernel_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*bias_element_size=*/sizeof(xnn_float16),
      /*packing_params=*/NULL, /*init_scale_params=*/NULL,
      /*scale_params=*/NULL, /*extra_weights_bytes=*/0, flags,
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

  return create_batch_matrix_multiply_nc_const_weights(
      batch_size_b, k, n, data_b,
      /*log2_kernel_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*bias_element_size=*/sizeof(xnn_float16), /*packing_params=*/NULL,
      /*init_scale_params=*/NULL, /*scale_params=*/NULL,
      /*extra_weights_bytes=*/0, flags, batch_matrix_multiply_op_out);
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_f32_const_weights(
    size_t batch_size_b, size_t k, size_t n, const float* data_b,
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op_out) {
  const enum xnn_status status = xnn_create_batch_matrix_multiply_nc_f32(
      flags, batch_matrix_multiply_op_out);
  if (status != xnn_status_success) {
    return status;
  }

  return create_batch_matrix_multiply_nc_const_weights(
      batch_size_b, k, n, data_b,
      /*log2_kernel_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*bias_element_size=*/sizeof(float), /*packing_params=*/NULL,
      /*init_scale_params=*/NULL, /*scale_params=*/NULL,
      /*extra_weights_bytes=*/0, flags, batch_matrix_multiply_op_out);
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_pf32_const_weights(
    size_t batch_size_b, size_t k, size_t n, const float* data_b,
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op_out) {
  const enum xnn_status status = xnn_create_batch_matrix_multiply_nc_pf32(
      flags, batch_matrix_multiply_op_out);
  if (status != xnn_status_success) {
    return status;
  }

  return create_batch_matrix_multiply_nc_const_weights(
      batch_size_b, k, n, data_b,
      /*log2_kernel_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*bias_element_size=*/sizeof(float), /*packing_params=*/NULL,
      /*init_scale_params=*/NULL, /*scale_params=*/NULL,
      /*extra_weights_bytes=*/0, flags, batch_matrix_multiply_op_out);
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

  // We only allow static `qcint8` weights.
  batch_matrix_multiply_op->dynamic_context.gemm->const_weights = true;

  // Check if we've already cached the packed data for `B`.
  uint32_t cache_seed = murmur_hash3(
      &batch_matrix_multiply_op->dynamic_context.gemm->gemm,
      sizeof(batch_matrix_multiply_op->dynamic_context.gemm->gemm), k * n);
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

  const uint32_t kr = batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.kr;
  const uint32_t sr = batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.sr;
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
    const uint32_t nr =
        batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.nr;
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
        aligned_size, xnn_operator_type_to_string_v2(batch_matrix_multiply_op),
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
        batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.packw_gemm_goi(
            /*groups=*/batch_size_b, n, k, nr, kr, sr, data_b, /*b=*/NULL,
            /*scale=*/NULL, packed_data, nr * extra_bytes, &pack_gemm_params);
      } else {
        batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.packw_gemm_gio(
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

enum xnn_status xnn_create_batch_matrix_multiply_nc_qs8(
    int8_t output_zero_point, int8_t output_min, int8_t output_max,
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op_out) {
  const struct xnn_gemm_config* gemm_config = xnn_init_qs8_qc8w_gemm_config();
  if (gemm_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(
            xnn_operator_type_batch_matrix_multiply_nc_qs8));
    return xnn_status_unsupported_hardware;
  }

  const struct gemm_fused_ukernels* gemm_ukernels = &gemm_config->minmax;
  if (gemm_config->linear.gemm[gemm_config->mr - 1]
          .function[XNN_UARCH_DEFAULT] != NULL) {
    gemm_ukernels = &gemm_config->minmax;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%d, %d] output range:"
      " lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(
          xnn_operator_type_batch_matrix_multiply_nc_qs8), output_min,
          output_max);
    return xnn_status_invalid_parameter;
  }

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(
          xnn_operator_type_batch_matrix_multiply_nc_qs8));
    return xnn_status_uninitialized;
  }

  union xnn_qs8_qc8w_conv_minmax_params params;
  if XNN_LIKELY(gemm_config->init.qs8_qc8w != NULL) {
    gemm_config->init.qs8_qc8w(&params, output_zero_point, output_min,
                               output_max);
  }

  return create_batch_matrix_multiply_nc(
      flags, &params, sizeof(params), gemm_config, gemm_ukernels,
      xnn_operator_type_batch_matrix_multiply_nc_qs8,
      batch_matrix_multiply_op_out);
}

enum xnn_status xnn_create_batch_matrix_multiply_nc_qs8_const_weights(
    size_t batch_size_b, size_t k, size_t n, const void* data_b,
    int8_t output_zero_point, int8_t output_min, int8_t output_max,
    int8_t input_zero_point, const float* scale_params, uint32_t flags,
    xnn_operator_t* batch_matrix_multiply_op_out) {
  const enum xnn_status status = xnn_create_batch_matrix_multiply_nc_qs8(
      output_zero_point, output_min, output_max, flags,
      batch_matrix_multiply_op_out);
  if (status != xnn_status_success) {
    return status;
  }

  struct xnn_qs8_packing_params packing_params;
  packing_params.input_zero_point = input_zero_point;
  return create_batch_matrix_multiply_nc_const_weights(
      batch_size_b, k, n, data_b,
      /*log2_kernel_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*bias_element_size=*/sizeof(int32_t), &packing_params,
      /*init_scale_params=*/
        (xnn_init_scale_params_fn) xnn_init_qs8_to_qs8_qc8w_scale_fp32_params,
      /*scale_params=*/scale_params, /*extra_weights_bytes=*/sizeof(float),
      flags, batch_matrix_multiply_op_out);
}

static enum xnn_status reshape_batch_matrix_multiply_nc(
    xnn_operator_t batch_matrix_multiply_op,
    enum xnn_operator_type expected_operator_type, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, uint32_t log2_input_a_element_size,
    uint32_t log2_input_b_element_size, uint32_t bias_element_size,
    uint32_t log2_output_element_size, const void* params, size_t params_size,
    const void* packing_params, xnn_init_scale_params_fn init_scale_params,
    const float* scale_params, size_t extra_weights_bytes, size_t num_threads) {
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

  const uint32_t nr = batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.nr;
  const uint32_t kr = batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.kr;
  const uint32_t sr = batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.sr;

  uint32_t mr = batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.mr;
  struct xnn_hmp_gemm_ukernel* gemm_cases =
      batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.gemm_cases;

  if (m == 1 &&
      batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.gemm_cases[0]
              .function[XNN_UARCH_DEFAULT] != NULL) {
    mr = 1;
  }

  const uint32_t mr_packed =
      m > 1 ? batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm.mr_packed
            : 1;

  assert(mr != 0 && mr <= XNN_MAX_MR);
  struct xnn_hmp_gemm_ukernel gemm_ukernel = gemm_cases[mr - 1];

  // Clear the operator's compute data to avoid accidentally reusing values from
  // a previous reshape (this was an interesting bug to track down).
  memset(batch_matrix_multiply_op->compute, 0,
         XNN_BATCH_MATMUL_MAX_COMPUTE_INVOCATIONS *
             sizeof(struct compute_parameters));
  batch_matrix_multiply_op->num_compute_invocations = 1;
  struct compute_parameters* gemm_compute =
      &batch_matrix_multiply_op->compute[0];
  struct gemm_op_context* gemm_context =
      batch_matrix_multiply_op->dynamic_context.gemm;

  // Do nothing if the weights don't need to be packed.
  if (!gemm_context->const_weights) {
    gemm_compute++;
    batch_matrix_multiply_op->num_compute_invocations++;

    const size_t n_stride = round_up(n, nr);
    const size_t k_stride = round_up_po2(k, kr * sr);
    const struct xnn_gemm_config* gemm_config =
        batch_matrix_multiply_op->gemm_config;
    const size_t weights_stride =
        gemm_config->packed_stride_weights_and_biases
            ? gemm_config->packed_stride_weights_and_biases(
                  gemm_config, k, /*block_size=*/k_stride, k_stride,
                  extra_weights_bytes)
            : (k_stride << log2_input_b_element_size) + bias_element_size +
                extra_weights_bytes;
    const size_t input_b_batch_stride = n_stride * weights_stride;

    // Store the computed weights stride in the op for later use.
    batch_matrix_multiply_op->weights_stride = weights_stride;

    // Compute the required workspace size.
    if (workspace_size != NULL) {
      *workspace_size = batch_size_b * input_b_batch_stride;
    }
    xnn_log_debug("Requesting workspace of size %zu for packed weights.",
                  *workspace_size);

    if (batch_matrix_multiply_op->flags & XNN_FLAG_TRANSPOSE_B) {
      assert(batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm
                     .packw_gemm_goi != NULL ||
             gemm_config->pack_weights_and_biases);
      gemm_context->packw_gemm_goi = (struct packw_gemm_goi_context){
          .kc = k,
          .nr = nr,
          .kr = kr,
          .sr = sr,
          .k_stride = k << log2_input_b_element_size,
          .bias = NULL,
          .b_stride = bias_element_size,
          .w_stride = weights_stride,
          .packw_gemm_goi = batch_matrix_multiply_op->ukernel.gemm_ukernels
                                ->gemm.packw_gemm_goi,
          .gk_stride = n * (k << log2_input_b_element_size),
          .gb_stride = n * bias_element_size,
          .gc_stride = input_b_batch_stride,
          .pack_weights_and_biases = gemm_config->pack_weights_and_biases,
          .gemm_config = gemm_config,
          .params = packing_params,
          .init_scale_params = init_scale_params,
          .scale_params = scale_params,
          .scale_params_size = extra_weights_bytes,
      };
      batch_matrix_multiply_op->compute[0].task_2d_tile_1d_dynamic =
          (pthreadpool_task_2d_tile_1d_dynamic_t)
              xnn_compute_batched_packw_gemm_goi;
      batch_matrix_multiply_op->compute[0].context_offset =
          offsetof(struct gemm_op_context, packw_gemm_goi);
    } else {
      assert(batch_matrix_multiply_op->ukernel.gemm_ukernels->gemm
                     .packw_gemm_gio != NULL ||
             gemm_config->pack_weights_and_biases);
      gemm_context->packw_gemm_gio = (struct packw_gemm_gio_context){
          .n_stride = 1 << log2_input_b_element_size,
          .k_stride_elements = n,
          .kc = k,
          .nr = nr,
          .kr = kr,
          .sr = sr,
          .bias = NULL,
          .b_stride = bias_element_size,
          .w_stride = weights_stride,
          .packw_gemm_gio = batch_matrix_multiply_op->ukernel.gemm_ukernels
                                ->gemm.packw_gemm_gio,
          .gk_stride = k * (n << log2_input_b_element_size),
          .gb_stride = n * bias_element_size,
          .gc_stride = input_b_batch_stride,
          .pack_weights_and_biases = gemm_config->pack_weights_and_biases,
          .gemm_config = gemm_config,
          .params = packing_params,
          .init_scale_params = init_scale_params,
          .scale_params = scale_params,
          .scale_params_size = extra_weights_bytes,
      };
      batch_matrix_multiply_op->compute[0].task_2d_tile_1d_dynamic =
          (pthreadpool_task_2d_tile_1d_dynamic_t)
              xnn_compute_batched_packw_gemm_gio;
      batch_matrix_multiply_op->compute[0].context_offset =
          offsetof(struct gemm_op_context, packw_gemm_gio);
    }
    batch_matrix_multiply_op->compute[0].type =
        xnn_parallelization_type_2d_tile_1d_dynamic;
    batch_matrix_multiply_op->compute[0].range[0] = batch_size_b;
    batch_matrix_multiply_op->compute[0].range[1] = n;
    batch_matrix_multiply_op->compute[0].tile[0] = nr;
  }

  const struct xnn_pack_lh_config* packed_lh_config = NULL;
  bool inline_lhs_packing =
      batch_matrix_multiply_op->flags & XNN_FLAG_INLINE_LHS_PACKING;
  switch (batch_matrix_multiply_op->type) {
    case xnn_operator_type_batch_matrix_multiply_nc_qd8_f32_qc8w:
      if (inline_lhs_packing) {
        packed_lh_config = xnn_init_f32_qdint8_pack_lh_config();
      }
      break;
    case xnn_operator_type_batch_matrix_multiply_nc_qdu8_f32_qc8w:
      if (inline_lhs_packing) {
        packed_lh_config = xnn_init_f32_qduint8_pack_lh_config();
      }
      break;
    case xnn_operator_type_batch_matrix_multiply_nc_qp8_f32_qc8w:
      packed_lh_config = xnn_init_qp8_pack_lh_config();
      break;
    case xnn_operator_type_batch_matrix_multiply_nc_pf16:
      packed_lh_config = xnn_init_x16_pack_lh_config();
      break;
    case xnn_operator_type_batch_matrix_multiply_nc_pf32:
      packed_lh_config = xnn_init_x32_pack_lh_config();
      break;
    default:
      break;
  }

  // Compute the optimal tile size for this GEMM.
  const size_t nc = xnn_gemm_best_tile_size(
      /*num_groups=*/batch_size_c, m, n,
      /*m_stride=*/k << (packed_lh_config
                             ? packed_lh_config->log2_packed_element_size
                             : log2_input_a_element_size),
      /*n_stride=*/batch_matrix_multiply_op->weights_stride,
      /*cn_stride=*/1 << log2_output_element_size, mr, nr, num_threads);

  // If we are packing the LHS, provide a per-thread workspace to do so inline.
  size_t workspace_offset = 0;
  size_t ga_stride = (m * k) << log2_input_a_element_size;
  memset(&gemm_context->pack_lh, 0, sizeof(struct pack_lh_context));
  if (packed_lh_config) {
    ga_stride = packed_lh_config->size_fn(m, k, mr_packed, kr, sr);
    log2_input_a_element_size = packed_lh_config->log2_packed_element_size;
    if (inline_lhs_packing) {
      assert(workspace_size);
      const size_t per_thread_workspace_size =
          packed_lh_config->size_fn(mr, k, mr_packed, kr, sr);

      // If the batch size of the LHS is smaller than that of the output, then
      // it does not make sense to inline the LHS packing, as this would mean
      // packing the same LHS several times.
      //
      // Similarly, inlining the packing also doesn't make sense if the number
      // of threads exceeds the number of tiles that we can parallelize over.
      //
      // Finally, if the packed LHS data will not stay in cache for the duration
      // of a GEMM tile computation, it does not make sense to inline the LHS
      // packing either, as it makes more sense to loop over `m` with chunks of
      // `nc < n`.
      //
      // In any of these cases, we pack the entire LHS into the workspace in a
      // separate `compute`, just as if it were a separate op.
      const bool should_inline_lhs_packing = xnn_should_inline_lhs_packing(
          batch_matrix_multiply_op->gemm_config,
          /*m_packed_stride=*/divide_round_up(per_thread_workspace_size, mr),
          /*n_stride=*/batch_matrix_multiply_op->weights_stride,
          /*cn_stride=*/1 << log2_output_element_size, /*mc=*/m,
          /*nc=*/n);

      if (packed_lh_config->gemv_noop && mr == 1) {
        xnn_log_debug(
            "Skipping inline packing for %s with batch_size=%zu, m=%zu, n=%zu, "
            "and k=%zu since it is a no-op for GEMV.",
            xnn_operator_type_to_string(batch_matrix_multiply_op->type),
            batch_size_c, m, n, k);
      } else if (batch_size_a < batch_size_c || !should_inline_lhs_packing ||
                 num_threads * mr > round_up(batch_size_a, mr)) {
        xnn_log_debug(
            "Pre-packing LHS of %s with batch_size=%zu, m=%zu, n=%zu, and "
            "k=%zu despite request to inline because %s.",
            xnn_operator_type_to_string(batch_matrix_multiply_op->type),
            batch_size_c, m, n, k,
            (batch_size_a < batch_size_c)
                ? "broadcasting reuses rows of the lhs"
                : (!should_inline_lhs_packing
                       ? "packed lhs will likely not stay in cache"
                       : "batch size does not parallelize well over the number "
                         "of threads"));

        // Allocate a workspace for the entire LHS.
        workspace_offset =
            round_up_po2(*workspace_size, XNN_ALLOCATION_ALIGNMENT);
        *workspace_size = workspace_offset + batch_size_a * ga_stride;

        // Set up the LHS packing as a separate compute.
        gemm_context->pack_lh = (struct pack_lh_context){
            .m = m,
            .k = k,
            .mr = mr_packed,
            .kr = kr,
            .sr = sr,
            .lhs_stride = k << packed_lh_config->log2_input_element_size,
            .gi_stride = m * k << packed_lh_config->log2_input_element_size,
            .gp_stride = ga_stride,
            .packed_offset_fn = packed_lh_config->offset_fn,
            .pack_lh_ukernel = packed_lh_config->pack_lh_fn,
            .workspace_offset = workspace_offset,
        };

        struct compute_parameters* pack_compute = gemm_compute++;
        batch_matrix_multiply_op->num_compute_invocations++;

        pack_compute->context_offset =
            offsetof(struct gemm_op_context, pack_lh);
        pack_compute->type = xnn_parallelization_type_2d_tile_1d_dynamic;
        pack_compute->task_2d_tile_1d_dynamic =
            (pthreadpool_task_2d_tile_1d_dynamic_t)xnn_compute_pack_lh;
        pack_compute->range[0] = batch_size_a;
        pack_compute->range[1] = m;
        pack_compute->tile[0] = mr_packed;

        inline_lhs_packing = false;
      } else {
        xnn_log_debug(
            "Inlining LHS packing for %s with batch_size=%zu, m=%zu, n=%zu, "
            "and k=%zu.",
            xnn_operator_type_to_string(batch_matrix_multiply_op->type),
            batch_size_c, m, n, k);
        workspace_offset =
            round_up_po2(*workspace_size, XNN_ALLOCATION_ALIGNMENT);
        *workspace_size =
            workspace_offset + num_threads * per_thread_workspace_size;
        xnn_log_debug(
            "Requesting workspace of %zu x %zu bytes for LHS packing.",
            num_threads, per_thread_workspace_size);
        log2_input_a_element_size = packed_lh_config->log2_input_element_size;
        ga_stride = (m * k) << log2_input_a_element_size;
      }
    }
  }

  gemm_context->gemm = (struct gemm_context){
      .k_scaled = k << log2_input_a_element_size,
      .a_stride = k << log2_input_a_element_size,
      .ga_stride = ga_stride,
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
      .nc = n,
      .mr_packed = mr_packed,
      .packed_lh_config = packed_lh_config,
      .workspace_offset = workspace_offset,
      .dynamic_quantization =
          (batch_matrix_multiply_op->type ==
               xnn_operator_type_batch_matrix_multiply_nc_qd8_f32_qc8w ||
           batch_matrix_multiply_op->type ==
               xnn_operator_type_batch_matrix_multiply_nc_qdu8_f32_qc8w),
  };

  // Copy the batch dimensions into the `gemm_context` struct since we don't
  // know if these pointers will be valid by the time the GEMM is actually
  // called.
  memcpy(gemm_context->gemm.batch_dims_a, batch_dims_a,
         sizeof(size_t) * num_batch_dims);
  memcpy(gemm_context->gemm.batch_dims_b, batch_dims_b,
         sizeof(size_t) * num_batch_dims);
  memcpy(gemm_context->gemm.batch_strides_c, batch_strides_c,
         sizeof(size_t) * num_batch_dims);
  memcpy(&gemm_context->gemm.params, params, params_size);
  gemm_context->gemm.fused_params = &gemm_context->gemm.params;

#if XNN_MAX_UARCH_TYPES > 1
  if (xnn_is_hmp_gemm_ukernel(gemm_ukernel)) {
    if (packed_lh_config) {
      if (inline_lhs_packing) {
        gemm_compute->type =
            xnn_parallelization_type_2d_tile_1d_dynamic_with_uarch_with_thread;
        gemm_compute->task_2d_tile_1d_dynamic_with_id_with_thread =
            (pthreadpool_task_2d_tile_1d_dynamic_with_id_with_thread_t)
                xnn_compute_hmp_grouped_inline_packed_qp8gemm;
      } else {
        gemm_compute->type =
            xnn_parallelization_type_3d_tile_2d_dynamic_with_uarch;
        gemm_compute->task_3d_tile_2d_dynamic_with_id =
            (pthreadpool_task_3d_tile_2d_dynamic_with_id_t)
                xnn_compute_hmp_grouped_qp8gemm;
      }
    } else {
      gemm_compute->type =
          xnn_parallelization_type_3d_tile_2d_dynamic_with_uarch;
      gemm_compute->task_3d_tile_2d_dynamic_with_id =
          (pthreadpool_task_3d_tile_2d_dynamic_with_id_t)
              xnn_compute_hmp_grouped_gemm;
    }
  } else
#endif  // XNN_MAX_UARCH_TYPES > 1
    if (packed_lh_config) {
      if (inline_lhs_packing) {
        gemm_compute->type =
            xnn_parallelization_type_2d_tile_1d_dynamic_with_thread;
        gemm_compute->task_2d_tile_1d_dynamic_with_id =
            (pthreadpool_task_2d_tile_1d_dynamic_with_id_t)
                xnn_compute_grouped_inline_packed_qp8gemm;
      } else {
        gemm_compute->type = xnn_parallelization_type_3d_tile_2d_dynamic;
        gemm_compute->task_3d_tile_2d_dynamic =
            (pthreadpool_task_3d_tile_2d_dynamic_t)xnn_compute_grouped_qp8gemm;
      }
    } else {
      gemm_compute->type = xnn_parallelization_type_3d_tile_2d_dynamic;
      gemm_compute->task_3d_tile_2d_dynamic =
          (pthreadpool_task_3d_tile_2d_dynamic_t)xnn_compute_grouped_gemm;
    }

  if (packed_lh_config && inline_lhs_packing) {
    gemm_compute->range[0] = batch_size_c;
    gemm_compute->range[1] = m;
    gemm_compute->tile[0] = mr;
  } else {
    gemm_compute->range[0] = batch_size_c;
    gemm_compute->range[1] = n;
    gemm_compute->range[2] = m;
    gemm_compute->tile[0] = nc;
    gemm_compute->tile[1] = mr;
  }

  batch_matrix_multiply_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_f16(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_f16,
      num_batch_dims, batch_dims_a, batch_dims_b, m, k, n, workspace_size,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*bias_element_size=*/sizeof(uint16_t),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
      &batch_matrix_multiply_op->params.f16_minmax,
      sizeof(batch_matrix_multiply_op->params.f16_minmax),
      /*packing_params=*/NULL, /*init_scale_params=*/NULL,
      /*scale_params=*/NULL, /*extra_weights_bytes=*/0,
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_pf16(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_pf16,
      num_batch_dims, batch_dims_a, batch_dims_b, m, k, n, workspace_size,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*bias_element_size=*/sizeof(uint16_t),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
      &batch_matrix_multiply_op->params.f16_minmax,
      sizeof(batch_matrix_multiply_op->params.f16_minmax),
      /*packing_params=*/NULL, /*init_scale_params=*/NULL,
      /*scale_params=*/NULL, /*extra_weights_bytes=*/0,
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_bf16_f32(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_bf16_f32, num_batch_dims,
      batch_dims_a, batch_dims_b, m, k, n, workspace_size,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_HALF,
      /*bias_element_size=*/sizeof(float),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &batch_matrix_multiply_op->params.f32_minmax,
      sizeof(batch_matrix_multiply_op->params.f32_minmax),
      /*packing_params=*/NULL, /*init_scale_params=*/NULL,
      /*scale_params=*/NULL, /*extra_weights_bytes=*/0,
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_f32(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_f32,
      num_batch_dims, batch_dims_a, batch_dims_b, m, k, n, workspace_size,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*bias_element_size=*/sizeof(float),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &batch_matrix_multiply_op->params.f32_minmax,
      sizeof(batch_matrix_multiply_op->params.f32_minmax),
      /*packing_params=*/NULL, /*init_scale_params=*/NULL,
      /*scale_params=*/NULL, /*extra_weights_bytes=*/0,
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_pf32(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_pf32,
      num_batch_dims, batch_dims_a, batch_dims_b, m, k, n, workspace_size,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      /*bias_element_size=*/sizeof(float),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &batch_matrix_multiply_op->params.f32_minmax,
      sizeof(batch_matrix_multiply_op->params.f32_minmax),
      /*packing_params=*/NULL, /*init_scale_params=*/NULL,
      /*scale_params=*/NULL, /*extra_weights_bytes=*/0,
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_qs8(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, const float* scale, const void* packing_params,
    size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_qs8,
      num_batch_dims, batch_dims_a, batch_dims_b, m, k, n,
      workspace_size, /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*bias_element_size=*/sizeof(int32_t),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      &batch_matrix_multiply_op->params.qs8_qc8w_conv_minmax,
      sizeof(batch_matrix_multiply_op->params.qs8_qc8w_conv_minmax),
      packing_params,
      (xnn_init_scale_params_fn) xnn_init_qs8_to_qs8_qc8w_scale_fp32_params,
      scale, /*extra_weights_bytes=*/sizeof(float),
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_qs8_const_weights(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_qs8, num_batch_dims,
      batch_dims_a, batch_dims_b, m, k, n, /*workspace_size=*/0,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*bias_element_size=*/sizeof(int32_t),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      &batch_matrix_multiply_op->params.qs8_qc8w_conv_minmax,
      sizeof(batch_matrix_multiply_op->params.qs8_qc8w_conv_minmax),
      /*packing_params=*/NULL, /*init_scale_params=*/NULL,
      /*scale_params=*/NULL, /*extra_weights_bytes=*/0,
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_qd8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_qd8_f32_qc8w, num_batch_dims,
      batch_dims_a, batch_dims_b, m, k, n, workspace_size,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*bias_element_size=*/sizeof(int32_t),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &batch_matrix_multiply_op->params.f32_minmax,
      sizeof(batch_matrix_multiply_op->params.f32_minmax),
      /*packing_params=*/NULL, /*init_scale_params=*/NULL,
      /*scale_params=*/NULL, /*extra_weights_bytes=*/0,
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_qp8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_qp8_f32_qc8w, num_batch_dims,
      batch_dims_a, batch_dims_b, m, k, n, workspace_size,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*bias_element_size=*/sizeof(int32_t),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &batch_matrix_multiply_op->params.f32_minmax,
      sizeof(batch_matrix_multiply_op->params.f32_minmax),
      /*packing_params=*/NULL, /*init_scale_params=*/NULL,
      /*scale_params=*/NULL, /*extra_weights_bytes=*/0,
      pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_qdu8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, pthreadpool_t threadpool) {
  return reshape_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_qdu8_f32_qc8w, num_batch_dims,
      batch_dims_a, batch_dims_b, m, k, n, workspace_size,
      /*log2_input_a_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*log2_input_b_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
      /*bias_element_size=*/sizeof(int32_t),
      /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
      &batch_matrix_multiply_op->params.f32_minmax,
      sizeof(batch_matrix_multiply_op->params.f32_minmax),
      /*packing_params=*/NULL, /*init_scale_params=*/NULL,
      /*scale_params=*/NULL, /*extra_weights_bytes=*/0,
      pthreadpool_get_threads_count(threadpool));
}

static enum xnn_status setup_batch_matrix_multiply_nc(
    xnn_operator_t batch_matrix_multiply_op,
    enum xnn_operator_type expected_operator_type, const void* input_a,
    const struct xnn_quantization_params* quantization_params,
    const void* input_b, void* packed_weights, void* workspace, void* output) {
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

  struct gemm_op_context* gemm_context =
      batch_matrix_multiply_op->dynamic_context.gemm;

  if (batch_matrix_multiply_op->flags & XNN_FLAG_TRANSPOSE_B) {
    gemm_context->packw_gemm_goi.kernel = input_b;
    gemm_context->packw_gemm_goi.bias = NULL;
    gemm_context->packw_gemm_goi.packed_weights = packed_weights;
  } else {
    gemm_context->packw_gemm_gio.kernel = input_b;
    gemm_context->packw_gemm_gio.bias = NULL;
    gemm_context->packw_gemm_gio.packed_weights = packed_weights;
  }

  if (gemm_context->pack_lh.m > 0) {
    gemm_context->pack_lh.lhs = input_a;
    void* pack_lh_workspace =
        (void*)((uintptr_t)workspace + gemm_context->pack_lh.workspace_offset);
    gemm_context->pack_lh.lhs_packed = pack_lh_workspace;
    gemm_context->gemm.a = pack_lh_workspace;
  } else {
    gemm_context->gemm.a = input_a;
    gemm_context->gemm.workspace = workspace;
  }
  gemm_context->gemm.packed_w = packed_weights;
  gemm_context->gemm.c = output;
  gemm_context->gemm.quantization_params = (const void*)quantization_params;

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
      batch_matrix_multiply_op->dynamic_context.gemm->const_weights
          ? packed_weights(batch_matrix_multiply_op)
          : workspace,
      workspace, output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_pf16(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const void* input_a, const void* input_b, void* output) {
  return setup_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_pf16,
      input_a, /*quantization_params=*/NULL, input_b,
      /*packed_weights=*/
      batch_matrix_multiply_op->dynamic_context.gemm->const_weights
          ? packed_weights(batch_matrix_multiply_op)
          : workspace,
      workspace, output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_bf16_f32(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const void* input_a, const void* input_b, void* output) {
  return setup_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_bf16_f32, input_a,
      /*quantization_params=*/NULL, input_b,
      /*packed_weights=*/
      batch_matrix_multiply_op->dynamic_context.gemm->const_weights
          ? packed_weights(batch_matrix_multiply_op)
          : workspace,
      workspace, output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_f32(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const float* input_a, const float* input_b, float* output) {
  return setup_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_f32,
      input_a, /*quantization_params=*/NULL, input_b,
      /*packed_weights=*/
      batch_matrix_multiply_op->dynamic_context.gemm->const_weights
          ? packed_weights(batch_matrix_multiply_op)
          : workspace,
      workspace, output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_pf32(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const float* input_a, const float* input_b, float* output) {
  return setup_batch_matrix_multiply_nc(
      batch_matrix_multiply_op, xnn_operator_type_batch_matrix_multiply_nc_pf32,
      input_a, /*quantization_params=*/NULL, input_b,
      /*packed_weights=*/
      batch_matrix_multiply_op->dynamic_context.gemm->const_weights
          ? packed_weights(batch_matrix_multiply_op)
          : workspace,
      workspace, output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_qs8(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const int8_t* input_a, const int8_t* input_b, int8_t* output) {
  return setup_batch_matrix_multiply_nc(
    batch_matrix_multiply_op,
    xnn_operator_type_batch_matrix_multiply_nc_qs8, input_a,
    /*quantization_params=*/NULL, input_b,
    batch_matrix_multiply_op->dynamic_context.gemm->const_weights
          ? packed_weights(batch_matrix_multiply_op)
          : workspace,
    workspace, output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_qd8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const int8_t* input_a, const int8_t* input_b,
    const struct xnn_quantization_params* quantization_params, float* output) {
  return setup_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_qd8_f32_qc8w, input_a,
      quantization_params, input_b, packed_weights(batch_matrix_multiply_op),
      workspace, output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_qp8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const int8_t* input_a, const float* input_b, float* output) {
  return setup_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_qp8_f32_qc8w, input_a,
      /*quantization_params=*/NULL, input_b,
      packed_weights(batch_matrix_multiply_op), workspace, output);
}

enum xnn_status xnn_setup_batch_matrix_multiply_nc_qdu8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const int8_t* input_a, const float* input_b,
    const struct xnn_quantization_params* quantization_params, float* output) {
  return setup_batch_matrix_multiply_nc(
      batch_matrix_multiply_op,
      xnn_operator_type_batch_matrix_multiply_nc_qdu8_f32_qc8w, input_a,
      quantization_params, input_b, packed_weights(batch_matrix_multiply_op),
      workspace, output);
}
