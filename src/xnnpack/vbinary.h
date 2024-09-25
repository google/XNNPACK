// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  XNN_INTERNAL void ukernel(                                                  \
      size_t n, const xnn_float16* a, const xnn_float16* b, xnn_float16* y,   \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/f16-vbinary/f16-vadd.h"
#include "src/f16-vbinary/f16-vaddc.h"
#include "src/f16-vbinary/f16-vcmul.h"
#include "src/f16-vbinary/f16-vdiv.h"
#include "src/f16-vbinary/f16-vdivc.h"
#include "src/f16-vbinary/f16-vmax.h"
#include "src/f16-vbinary/f16-vmaxc.h"
#include "src/f16-vbinary/f16-vmin.h"
#include "src/f16-vbinary/f16-vminc.h"
#include "src/f16-vbinary/f16-vmul.h"
#include "src/f16-vbinary/f16-vmulc.h"
#include "src/f16-vbinary/f16-vprelu.h"
#include "src/f16-vbinary/f16-vpreluc.h"
#include "src/f16-vbinary/f16-vrpreluc.h"
#include "src/f16-vbinary/f16-vrdivc.h"
#include "src/f16-vbinary/f16-vrsubc.h"
#include "src/f16-vbinary/f16-vsqrdiff.h"
#include "src/f16-vbinary/f16-vsqrdiffc.h"
#include "src/f16-vbinary/f16-vsub.h"
#include "src/f16-vbinary/f16-vsubc.h"
#undef XNN_UKERNEL_WITH_PARAMS

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  XNN_INTERNAL void ukernel(                                                  \
      size_t n, const float* a, const float* b, float* y,                     \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/f32-vbinary/f32-vadd.h"
#include "src/f32-vbinary/f32-vaddc.h"
#include "src/f32-vbinary/f32-vcopysign.h"
#include "src/f32-vbinary/f32-vcopysignc.h"
#include "src/f32-vbinary/f32-vcmul.h"
#include "src/f32-vbinary/f32-vdiv.h"
#include "src/f32-vbinary/f32-vdivc.h"
#include "src/f32-vbinary/f32-vmax.h"
#include "src/f32-vbinary/f32-vmaxc.h"
#include "src/f32-vbinary/f32-vmin.h"
#include "src/f32-vbinary/f32-vminc.h"
#include "src/f32-vbinary/f32-vmul.h"
#include "src/f32-vbinary/f32-vmulc.h"
#include "src/f32-vbinary/f32-vprelu.h"
#include "src/f32-vbinary/f32-vpreluc.h"
#include "src/f32-vbinary/f32-vrpreluc.h"
#include "src/f32-vbinary/f32-vrcopysignc.h"
#include "src/f32-vbinary/f32-vrdivc.h"
#include "src/f32-vbinary/f32-vrsubc.h"
#include "src/f32-vbinary/f32-vsqrdiff.h"
#include "src/f32-vbinary/f32-vsqrdiffc.h"
#include "src/f32-vbinary/f32-vsub.h"
#include "src/f32-vbinary/f32-vsubc.h"
#undef XNN_UKERNEL_WITH_PARAMS

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  XNN_INTERNAL void ukernel(                                                  \
      size_t n, const uint8_t* input_a, const uint8_t* input_b,               \
      uint8_t* output,                                                        \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/qu8-vadd/qu8-vadd-minmax.h"
#include "src/qu8-vaddc/qu8-vaddc-minmax.h"
#include "src/qu8-vmul/qu8-vmul-minmax-fp32.h"
#include "src/qu8-vmul/qu8-vmul-minmax-rndnu.h"
#include "src/qu8-vmulc/qu8-vmulc-minmax-fp32.h"
#include "src/qu8-vmulc/qu8-vmulc-minmax-rndnu.h"
#undef XNN_UKERNEL_WITH_PARAMS

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  XNN_INTERNAL void ukernel(                                                  \
      size_t n, const int8_t* input_a, const int8_t* input_b, int8_t* output, \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/qs8-vadd/qs8-vadd-minmax.h"
#include "src/qs8-vaddc/qs8-vaddc-minmax.h"
#include "src/qs8-vmul/qs8-vmul-minmax-fp32.h"
#include "src/qs8-vmul/qs8-vmul-minmax-rndnu.h"
#include "src/qs8-vmulc/qs8-vmulc-minmax-fp32.h"
#include "src/qs8-vmulc/qs8-vmulc-minmax-rndnu.h"
#undef XNN_UKERNEL_WITH_PARAMS

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, \
                                datatype, params_type, init_params)           \
  XNN_INTERNAL void ukernel(                                                  \
      size_t n, const int32_t* input_a, const int32_t* input_b,               \
      int32_t* output,                                                        \
      const params_type params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);
#include "src/s32-vmul/s32-vmul.h"
#include "src/s32-vmul/s32-vmulc.h"
#undef XNN_UKERNEL_WITH_PARAMS

#ifdef __cplusplus
}  // extern "C"
#endif
