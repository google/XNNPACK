// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#include "include/experimental.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"

xnn_config_identifier xnn_create_config_identifier(xnn_config_name name,
                                                   uint32_t version) {
  struct xnn_config_identifier id = {((uint64_t)name) << 32 | version};
  return id;
}

xnn_config_name xnn_get_config_name(const xnn_config_identifier* identifier) {
  return identifier->identifier >> 32;
}

xnn_config_name xnn_get_config_version(
    const xnn_config_identifier* identifier) {
  return identifier->identifier & 0xffffffff;
}

const xnn_config_identifier* xnn_get_test_config() {
  const struct xnn_gemm_config* f32_config =
      xnn_init_f32_gemm_config(/*flags=*/0);
  return &(f32_config->identifier);
}

#define XNNPACK_CHECK_CONFIG(CONFIG_NAME, ...)                             \
  if (config_name == xnn_config_name_##CONFIG_NAME) {                      \
    const struct xnn_gemm_config* kernel_config =                          \
        xnn_init_##CONFIG_NAME##_config(__VA_ARGS__);                      \
    return kernel_config &&                                                \
           identifier->identifier == kernel_config->identifier.identifier; \
  }

bool xnn_check_config_version(const xnn_config_identifier* identifier) {
  if (identifier == NULL) {
    return false;
  }
  const xnn_config_name config_name = xnn_get_config_name(identifier);
  XNNPACK_CHECK_CONFIG(bf16_f32_gemm);
  XNNPACK_CHECK_CONFIG(f16_gemm);
  XNNPACK_CHECK_CONFIG(f32_gemm, /*flags=*/0);
  XNNPACK_CHECK_CONFIG(f32_gemm_nr2, /*flags=*/0);
  XNNPACK_CHECK_CONFIG(f32_igemm);
  XNNPACK_CHECK_CONFIG(f32_qc8w_gemm);
  XNNPACK_CHECK_CONFIG(f32_qc4w_gemm);
  XNNPACK_CHECK_CONFIG(pf16_gemm);
  XNNPACK_CHECK_CONFIG(pf32_gemm);
  XNNPACK_CHECK_CONFIG(pqs8_qc8w_gemm);
  XNNPACK_CHECK_CONFIG(qd8_f16_qb4w_gemm);
  XNNPACK_CHECK_CONFIG(qd8_f16_qc4w_gemm);
  XNNPACK_CHECK_CONFIG(qd8_f16_qc8w_gemm);
  XNNPACK_CHECK_CONFIG(qd8_f16_qc8w_igemm);
  XNNPACK_CHECK_CONFIG(qd8_f32_qb4w_gemm);
  XNNPACK_CHECK_CONFIG(qd8_f32_qc4w_gemm);
  XNNPACK_CHECK_CONFIG(qd8_f32_qc8w_gemm);
  XNNPACK_CHECK_CONFIG(qp8_f32_qc4w_gemm);
  XNNPACK_CHECK_CONFIG(qp8_f32_qc8w_gemm);
  XNNPACK_CHECK_CONFIG(qp8_f32_qb4w_gemm);
  XNNPACK_CHECK_CONFIG(qdu8_f32_qc4w_gemm);
  XNNPACK_CHECK_CONFIG(qdu8_f16_qc8w_gemm);
  XNNPACK_CHECK_CONFIG(qdu8_f32_qc8w_gemm);
  XNNPACK_CHECK_CONFIG(qdu8_f32_qb4w_gemm);
  XNNPACK_CHECK_CONFIG(qdu8_f16_qc4w_gemm);
  XNNPACK_CHECK_CONFIG(qdu8_f32_qc8w_igemm);
  XNNPACK_CHECK_CONFIG(qs8_qc4w_gemm);
  XNNPACK_CHECK_CONFIG(qs8_qc8w_gemm);
  XNNPACK_CHECK_CONFIG(qu8_gemm);
  return false;
}
