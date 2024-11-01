// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_BF16_UKERNEL_FUNCTION(fn_name, params_type) \
  XNN_INTERNAL void fn_name(                                 \
      size_t n,                                              \
      const xnn_bfloat16* x,                                         \
      xnn_bfloat16* y,                                               \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#define DECLARE_F16_UKERNEL_FUNCTION(fn_name, params_type) \
  XNN_INTERNAL void fn_name(                                \
      size_t n,                                             \
      const xnn_float16* x,                                        \
      xnn_float16* y,                                              \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#define DECLARE_F32_UKERNEL_FUNCTION(fn_name, params_type) \
  XNN_INTERNAL void fn_name(                                \
      size_t n,                                             \
      const float* x,                                       \
      float* y,                                             \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#define XNN_UKERNEL(arch_flags, fn_name, batch_tile, vector_tile, datatype) \
  XNN_INTERNAL void fn_name(size_t n, const int8_t* x, int8_t* y,           \
                            const struct xnn_s8_minmax_params                \
                                params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "s8-vclamp/s8-vclamp.h"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, fn_name, batch_tile, vector_tile, datatype) \
  XNN_INTERNAL void fn_name(size_t n, const uint8_t* x, uint8_t* y,         \
                            const struct xnn_u8_minmax_params                \
                                params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "u8-vclamp/u8-vclamp.h"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, fn_name, batch_tile, vector_tile, datatype) \
  DECLARE_F16_UKERNEL_FUNCTION(fn_name, xnn_f16_default_params);
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, fn_name, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  DECLARE_F16_UKERNEL_FUNCTION(fn_name, params_type);
#include "f16-vabs/f16-vabs.h"
#include "f16-vclamp/f16-vclamp.h"
#include "f16-velu/f16-velu.h"
#include "f16-vhswish/f16-vhswish.h"
#include "f16-vlrelu/f16-vlrelu.h"
#include "f16-vneg/f16-vneg.h"
#include "f16-vrnd/f16-vrndd.h"
#include "f16-vrnd/f16-vrndne.h"
#include "f16-vrnd/f16-vrndu.h"
#include "f16-vrnd/f16-vrndz.h"
#include "f16-vrsqrt/f16-vrsqrt.h"
#include "f16-vsigmoid/f16-vsigmoid.h"
#include "f16-vsqr/f16-vsqr.h"
#include "f16-vsqrt/f16-vsqrt.h"
#include "f16-vtanh/f16-vtanh.h"
#undef XNN_UKERNEL
#undef XNN_UKERNEL_WITH_PARAMS

#define XNN_UKERNEL(arch_flags, fn_name, batch_tile, vector_tile, datatype) \
  DECLARE_F32_UKERNEL_FUNCTION(fn_name, xnn_f32_default_params);
#define XNN_UKERNEL_WITH_PARAMS(arch_flags, fn_name, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  DECLARE_F32_UKERNEL_FUNCTION(fn_name, params_type);
#include "f32-vabs/f32-vabs.h"
#include "f32-vclamp/f32-vclamp.h"
#include "f32-velu/f32-velu.h"
#include "f32-vexp/f32-vexp.h"
#include "f32-vgelu/f32-vgelu.h"
#include "f32-vhswish/f32-vhswish.h"
#include "f32-vlog/f32-vlog.h"
#include "f32-vlrelu/f32-vlrelu.h"
#include "f32-vneg/f32-vneg.h"
#include "f32-vrelu/f32-vrelu.h"
#include "f32-vrnd/f32-vrndd.h"
#include "f32-vrnd/f32-vrndne.h"
#include "f32-vrnd/f32-vrndu.h"
#include "f32-vrnd/f32-vrndz.h"
#include "f32-vrsqrt/f32-vrsqrt.h"
#include "f32-vsigmoid/f32-vsigmoid.h"
#include "f32-vsqr/f32-vsqr.h"
#include "f32-vsqrt/f32-vsqrt.h"
#include "f32-vtanh/f32-vtanh.h"
#undef XNN_UKERNEL
#undef XNN_UKERNEL_WITH_PARAMS

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, fn_name, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  XNN_INTERNAL void fn_name(                                                  \
      size_t n, const int8_t* input, int8_t* output,                          \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "qs8-vlrelu/qs8-vlrelu.h"
#undef XNN_UKERNEL_WITH_PARAMS

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, fn_name, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  XNN_INTERNAL void fn_name(                                                  \
      size_t n, const uint8_t* input, uint8_t* output,                        \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "qu8-vlrelu/qu8-vlrelu.h"
#undef XNN_UKERNEL_WITH_PARAMS

#define DECLARE_XX_VUNARY_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t n,                                     \
      const void* x,                                \
      void* y,                                      \
      const void* params);

DECLARE_XX_VUNARY_UKERNEL_FUNCTION(xnn_xx_copy_ukernel__scalar_memcpy)


#ifdef __cplusplus
}  // extern "C"
#endif
