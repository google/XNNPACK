#pragma once

#include <stddef.h>

#include <xnnpack/params.h>
#include <xnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_F32_VSCALE_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      const float* x,                                \
      float* y,                                      \
      float c);

DECLARE_F32_VSCALE_UKERNEL_FUNCTION(xnn_f32_vscale_ukernel__avx_unroll32)
DECLARE_F32_VSCALE_UKERNEL_FUNCTION(xnn_f32_vscale_ukernel__avx512f_unroll64)


#ifdef __cplusplus
} /* extern "C" */
#endif
