// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_VUNARY_H_
#define XNNPACK_SRC_XNNPACK_VUNARY_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_BF16_UKERNEL_FUNCTION(fn_name, params_type)                   \
  XNN_INTERNAL void fn_name(size_t n, const xnn_bfloat16* x, xnn_bfloat16* y, \
                            const params_type* params);

#define DECLARE_F16_UKERNEL_FUNCTION(fn_name, params_type)                  \
  XNN_INTERNAL void fn_name(size_t n, const xnn_float16* x, xnn_float16* y, \
                            const params_type* params);

#define DECLARE_F32_UKERNEL_FUNCTION(fn_name, params_type)      \
  XNN_INTERNAL void fn_name(size_t n, const float* x, float* y, \
                            const params_type* params);

#define XNN_UKERNEL(arch_flags, fn_name, batch_tile, vector_tile, datatype, \
                    params_type, init_params)                               \
  XNN_INTERNAL void fn_name(size_t n, const int8_t* x, int8_t* y,           \
                            const struct xnn_s8_minmax_params* params);
#include "src/s8-vclamp/s8-vclamp.inc"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, fn_name, batch_tile, vector_tile, datatype, \
                    params_type, init_params)                               \
  XNN_INTERNAL void fn_name(size_t n, const uint8_t* x, uint8_t* y,         \
                            const struct xnn_u8_minmax_params* params);
#include "src/u8-vclamp/u8-vclamp.inc"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, fn_name, batch_tile, vector_tile, datatype, \
                    params_type, init_params)                               \
  DECLARE_F16_UKERNEL_FUNCTION(fn_name, params_type);
#include "src/f16-vabs/f16-vabs.inc"
#include "src/f16-vapproxgelu/f16-vapproxgelu.inc"
#include "src/f16-vclamp/f16-vclamp.inc"
#include "src/f16-vcos/f16-vcos.inc"
#include "src/f16-velu/f16-velu.inc"
#include "src/f16-vexp/f16-vexp.inc"
#include "src/f16-vgelu/f16-vgelu.inc"
#include "src/f16-vhswish/f16-vhswish.inc"
#include "src/f16-vlrelu/f16-vlrelu.inc"
#include "src/f16-vneg/f16-vneg.inc"
#include "src/f16-vrnd/f16-vrndd.inc"
#include "src/f16-vrnd/f16-vrndne.inc"
#include "src/f16-vrnd/f16-vrndu.inc"
#include "src/f16-vrnd/f16-vrndz.inc"
#include "src/f16-vrsqrt/f16-vrsqrt.inc"
#include "src/f16-vsigmoid/f16-vsigmoid.inc"
#include "src/f16-vsin/f16-vsin.inc"
#include "src/f16-vsqr/f16-vsqr.inc"
#include "src/f16-vsqrt/f16-vsqrt.inc"
#include "src/f16-vtanh/f16-vtanh.inc"
#undef XNN_UKERNEL
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, fn_name, batch_tile, vector_tile, datatype, \
                    params_type, init_params)                               \
  DECLARE_F32_UKERNEL_FUNCTION(fn_name, params_type);
#include "src/f32-vabs/f32-vabs.inc"
#include "src/f32-vapproxgelu/f32-vapproxgelu.inc"
#include "src/f32-vclamp/f32-vclamp.inc"
#include "src/f32-vcos/f32-vcos.inc"
#include "src/f32-velu/f32-velu.inc"
#include "src/f32-vexp/f32-vexp.inc"
#include "src/f32-vgelu/f32-vgelu.inc"
#include "src/f32-vhswish/f32-vhswish.inc"
#include "src/f32-vlog/f32-vlog.inc"
#include "src/f32-vlrelu/f32-vlrelu.inc"
#include "src/f32-vneg/f32-vneg.inc"
#include "src/f32-vrnd/f32-vrndd.inc"
#include "src/f32-vrnd/f32-vrndne.inc"
#include "src/f32-vrnd/f32-vrndu.inc"
#include "src/f32-vrnd/f32-vrndz.inc"
#include "src/f32-vrsqrt/f32-vrsqrt.inc"
#include "src/f32-vsigmoid/f32-vsigmoid.inc"
#include "src/f32-vsin/f32-vsin.inc"
#include "src/f32-vsqr/f32-vsqr.inc"
#include "src/f32-vsqrt/f32-vsqrt.inc"
#include "src/f32-vtanh/f32-vtanh.inc"
#undef XNN_UKERNEL
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, fn_name, batch_tile, vector_tile, datatype, \
                    params_type, init_params)                               \
  XNN_INTERNAL void fn_name(size_t n, const int8_t* input, int8_t* output,  \
                            const params_type* params);
#include "src/qs8-vlrelu/qs8-vlrelu.inc"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, fn_name, batch_tile, vector_tile, datatype,  \
                    params_type, init_params)                                \
  XNN_INTERNAL void fn_name(size_t n, const uint8_t* input, uint8_t* output, \
                            const params_type* params);
#include "src/qu8-vlrelu/qu8-vlrelu.inc"
#undef XNN_UKERNEL

#define DECLARE_XX_VUNARY_UKERNEL_FUNCTION(fn_name)           \
  XNN_INTERNAL void fn_name(size_t n, const void* x, void* y, \
                            const void* params);

DECLARE_XX_VUNARY_UKERNEL_FUNCTION(xnn_xx_copy_ukernel__scalar_memcpy)

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_VUNARY_H_
