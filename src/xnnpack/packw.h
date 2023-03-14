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

struct xnn_qs8_packw_params {
  int8_t input_zero_point;
};


#define DECLARE_X8_PACKW_GEMM_GOI_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                \
      size_t batch,                                         \
      size_t dim_n,                                         \
      size_t dim_k,                                         \
      size_t nr,                                            \
      size_t kr,                                            \
      size_t sr,                                            \
      const int8_t* k,                                      \
      const int32_t* b,                                     \
      int8_t* packed_weights,                               \
      size_t extra_bytes,                                   \
      const void* params);                                  \

DECLARE_X8_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int)
DECLARE_X8_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int)

#define DECLARE_X16_PACKW_GEMM_GOI_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                 \
      size_t batch,                                          \
      size_t dim_n,                                          \
      size_t dim_k,                                          \
      size_t nr,                                             \
      size_t kr,                                             \
      size_t sr,                                             \
      const uint16_t* k,                                     \
      const uint16_t* b,                                     \
      uint16_t* packed_weights,                              \
      size_t extra_bytes,                                    \
      const void* params);                                   \

DECLARE_X16_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x16_packw_gemm_goi_ukernel_x8__scalar_int)
DECLARE_X16_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x16_packw_gemm_goi_ukernel_x16__scalar_int)

#define DECLARE_X32_PACKW_GEMM_GOI_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                 \
      size_t batch,                                          \
      size_t dim_n,                                          \
      size_t dim_k,                                          \
      size_t nr,                                             \
      size_t kr,                                             \
      size_t sr,                                             \
      const uint32_t* k,                                     \
      const uint32_t* b,                                     \
      uint32_t* packed_weights,                              \
      size_t extra_bytes,                                    \
      const void* params);                                   \

DECLARE_X32_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_int)
DECLARE_X32_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_int)
DECLARE_X32_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x32_packw_gemm_goi_ukernel_x2__scalar_float)
DECLARE_X32_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x32_packw_gemm_goi_ukernel_x4__scalar_float)

DECLARE_X32_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x32_packw_gemm_goi_ukernel_x2__neon)
DECLARE_X32_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x32_packw_gemm_goi_ukernel_x8__neon)
DECLARE_X32_PACKW_GEMM_GOI_UKERNEL_FUNCTION(xnn_x32_packw_gemm_goi_ukernel_x12__neon)


#ifdef __cplusplus
}  // extern "C"
#endif
