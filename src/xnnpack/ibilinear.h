// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"

#ifdef __cplusplus
extern "C" {
#endif


#define XNN_UKERNEL(arch_flags, fn_name, channel_tile, pixel_tile, datatype, params_type, init_params) \
  XNN_INTERNAL void fn_name(                           \
      size_t output_pixels,                            \
      size_t channels,                                 \
      const datatype** input,                          \
      size_t input_offset,                             \
      const datatype* weights,                         \
      datatype* output,                                \
      size_t output_increment);
#include "f16-ibilinear/f16-ibilinear.h"
#include "f32-ibilinear/f32-ibilinear.h"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, fn_name, channel_tile, pixel_tile, datatype, params_type, init_params) \
  XNN_INTERNAL void fn_name(                           \
      size_t output_pixels,                            \
      size_t channels,                                 \
      const datatype** input,                          \
      size_t input_offset,                             \
      const int16_t* weights,                          \
      datatype* output,                                \
      size_t output_increment);
#include "s8-ibilinear/s8-ibilinear.h"
#include "u8-ibilinear/u8-ibilinear.h"
#undef XNN_UKERNEL

#define DECLARE_F32_IBILINEAR_CHW_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                \
      size_t output_pixels,                                 \
      size_t channels,                                      \
      const float** input,                                  \
      size_t input_offset,                                  \
      const float* weights,                                 \
      float* output,                                        \
      size_t input_increment);

DECLARE_F32_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f32_ibilinear_chw_ukernel__scalar_p1)
DECLARE_F32_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f32_ibilinear_chw_ukernel__scalar_p2)
DECLARE_F32_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f32_ibilinear_chw_ukernel__scalar_p4)

DECLARE_F32_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f32_ibilinear_chw_ukernel__wasmsimd_p4)
DECLARE_F32_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f32_ibilinear_chw_ukernel__wasmsimd_p8)

DECLARE_F32_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f32_ibilinear_chw_ukernel__neon_p4)
DECLARE_F32_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f32_ibilinear_chw_ukernel__neon_p8)
DECLARE_F32_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f32_ibilinear_chw_ukernel__neon_p16)

DECLARE_F32_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f32_ibilinear_chw_ukernel__neonfma_p4)
DECLARE_F32_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f32_ibilinear_chw_ukernel__neonfma_p8)
DECLARE_F32_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f32_ibilinear_chw_ukernel__neonfma_p16)

DECLARE_F32_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f32_ibilinear_chw_ukernel__sse_p4)
DECLARE_F32_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f32_ibilinear_chw_ukernel__sse_p8)


#define DECLARE_F16_IBILINEAR_CHW_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                \
      size_t output_pixels,                                 \
      size_t channels,                                      \
      const xnn_float16** input,                   \
      size_t input_offset,                                  \
      const xnn_float16* weights,                  \
      xnn_float16* output,                         \
      size_t input_increment);

DECLARE_F16_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p4)
DECLARE_F16_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p8)
DECLARE_F16_IBILINEAR_CHW_UKERNEL_FUNCTION(xnn_f16_ibilinear_chw_ukernel__neonfp16arith_p16)

#ifdef __cplusplus
}  // extern "C"
#endif
