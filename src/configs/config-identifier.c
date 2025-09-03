// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "include/xnnpack.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"

xnn_config_identifier xnn_create_config_identifier(xnn_config_name name,
                                                   uint32_t version) {
  return (xnn_config_identifier)name << 32 | version;
}

xnn_config_name xnn_get_config_name(xnn_config_identifier identifier) {
  return identifier >> 32;
}

xnn_config_name xnn_get_config_version(xnn_config_identifier identifier) {
  return identifier & 0xffffffff;
}

const struct xnn_config_common_initial_sequence*
xnn_experimental_get_test_config() {
  const struct xnn_gemm_config* f32_config =
      xnn_init_f32_gemm_config(/*flags=*/0);
  return (const struct xnn_config_common_initial_sequence*)f32_config;
}

#define XNNPACK_CHECK_CONFIG(INIT_FUNCTION)                                 \
  {                                                                         \
    const struct xnn_gemm_config* kernel_config = INIT_FUNCTION;            \
    if (kernel_config && config->identifier == kernel_config->identifier) { \
      return true;                                                          \
    }                                                                       \
  }

bool xnn_experimental_check_config_version(
    const struct xnn_config_common_initial_sequence* config) {
  if (config == NULL) {
    return false;
  }
  XNNPACK_CHECK_CONFIG(xnn_init_bf16_f32_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_f16_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_f32_gemm_config(/*flags=*/0));
  XNNPACK_CHECK_CONFIG(xnn_init_f32_gemm_nr2_config(/*flags=*/0));
  XNNPACK_CHECK_CONFIG(xnn_init_f32_igemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_f32_qc8w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_f32_qc4w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_pf16_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_pf32_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_pqs8_qc8w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qd8_f16_qb4w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qd8_f16_qc4w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qd8_f16_qc8w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qd8_f16_qc8w_igemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qd8_f32_qb4w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qd8_f32_qc4w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qd8_f32_qc8w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qp8_f32_qc4w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qp8_f32_qc8w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qp8_f32_qb4w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qdu8_f32_qc4w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qdu8_f16_qc8w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qdu8_f32_qc8w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qdu8_f32_qb4w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qdu8_f16_qc4w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qdu8_f32_qc8w_igemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qs8_qc4w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qs8_qc8w_gemm_config());
  XNNPACK_CHECK_CONFIG(xnn_init_qu8_gemm_config());
  return false;
}
