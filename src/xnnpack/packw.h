// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdint.h>
#include <stddef.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_X32_PACKW_GEMM_GOI_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(               \
      size_t g,                            \
      size_t nc,                           \
      size_t kc,                           \
      size_t nr,                           \
      size_t kr,                           \
      size_t sr,                           \
      const uint32_t* k,                   \
      const uint32_t* b,                   \
      uint32_t* packed_weights,            \
      size_t extra_bytes,                  \
      const void* params);                 \

DECLARE_X32_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x32_packw_gemm_goi_ukernel_x2__scalar)
DECLARE_X32_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x32_packw_gemm_goi_ukernel_x4__scalar)

DECLARE_X32_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x32_packw_gemm_goi_ukernel_x2__neon)
DECLARE_X32_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x32_packw_gemm_goi_ukernel_x8__neon)
DECLARE_X32_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x32_packw_gemm_goi_ukernel_x12__neon)

#ifdef __cplusplus
}  // extern "C"
#endif
