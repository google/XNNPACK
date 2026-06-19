// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_VBINARY_H_
#define XNNPACK_SRC_XNNPACK_VBINARY_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype, \
                    params_type, init_params)                               \
  XNN_INTERNAL void ukernel(size_t n, const xnn_float16* a,                 \
                            const xnn_float16* b, xnn_float16* y,           \
                            const params_type* params);
#include "src/f16-vbinary/f16-vadd.inc"
#include "src/f16-vbinary/f16-vaddc.inc"
#include "src/f16-vbinary/f16-vcmul.inc"
#include "src/f16-vbinary/f16-vdiv.inc"
#include "src/f16-vbinary/f16-vdivc.inc"
#include "src/f16-vbinary/f16-vmax.inc"
#include "src/f16-vbinary/f16-vmaxc.inc"
#include "src/f16-vbinary/f16-vmin.inc"
#include "src/f16-vbinary/f16-vminc.inc"
#include "src/f16-vbinary/f16-vmul.inc"
#include "src/f16-vbinary/f16-vmulc.inc"
#include "src/f16-vbinary/f16-vprelu.inc"
#include "src/f16-vbinary/f16-vpreluc.inc"
#include "src/f16-vbinary/f16-vrdivc.inc"
#include "src/f16-vbinary/f16-vrpreluc.inc"
#include "src/f16-vbinary/f16-vrsubc.inc"
#include "src/f16-vbinary/f16-vsqrdiff.inc"
#include "src/f16-vbinary/f16-vsqrdiffc.inc"
#include "src/f16-vbinary/f16-vsub.inc"
#include "src/f16-vbinary/f16-vsubc.inc"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype, \
                    params_type, init_params)                               \
  XNN_INTERNAL void ukernel(size_t n, const float* a, const float* b,       \
                            float* y, const params_type* params);
#include "src/f32-vbinary/f32-vadd.inc"
#include "src/f32-vbinary/f32-vaddc.inc"
#include "src/f32-vbinary/f32-vcmul.inc"
#include "src/f32-vbinary/f32-vcopysign.inc"
#include "src/f32-vbinary/f32-vcopysignc.inc"
#include "src/f32-vbinary/f32-vdiv.inc"
#include "src/f32-vbinary/f32-vdivc.inc"
#include "src/f32-vbinary/f32-vmax.inc"
#include "src/f32-vbinary/f32-vmaxc.inc"
#include "src/f32-vbinary/f32-vmin.inc"
#include "src/f32-vbinary/f32-vminc.inc"
#include "src/f32-vbinary/f32-vmul.inc"
#include "src/f32-vbinary/f32-vmulc.inc"
#include "src/f32-vbinary/f32-vprelu.inc"
#include "src/f32-vbinary/f32-vpreluc.inc"
#include "src/f32-vbinary/f32-vrcopysignc.inc"
#include "src/f32-vbinary/f32-vrdivc.inc"
#include "src/f32-vbinary/f32-vrpreluc.inc"
#include "src/f32-vbinary/f32-vrsubc.inc"
#include "src/f32-vbinary/f32-vsqrdiff.inc"
#include "src/f32-vbinary/f32-vsqrdiffc.inc"
#include "src/f32-vbinary/f32-vsub.inc"
#include "src/f32-vbinary/f32-vsubc.inc"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype, \
                    params_type, init_params)                               \
  XNN_INTERNAL void ukernel(size_t n, const uint8_t* input_a,               \
                            const uint8_t* input_b, uint8_t* output,        \
                            const params_type* params);
#include "src/qu8-vadd/qu8-vadd-minmax.inc"
#include "src/qu8-vaddc/qu8-vaddc-minmax.inc"
#include "src/qu8-vmul/qu8-vmul-minmax-fp32.inc"
#include "src/qu8-vmul/qu8-vmul-minmax-rndnu.inc"
#include "src/qu8-vmulc/qu8-vmulc-minmax-fp32.inc"
#include "src/qu8-vmulc/qu8-vmulc-minmax-rndnu.inc"
#include "src/qu8-vprelu/qu8-vprelu.inc"
#include "src/qu8-vpreluc/qu8-vpreluc.inc"
#include "src/qu8-vrpreluc/qu8-vrpreluc.inc"
#undef XNN_UKERNEL

#define XNN_UKERNEL(arch_flags, ukernel, batch_tile, vector_tile, datatype, \
                    params_type, init_params)                               \
  XNN_INTERNAL void ukernel(size_t n, const int8_t* input_a,                \
                            const int8_t* input_b, int8_t* output,          \
                            const params_type* params);
#include "src/qs8-vadd/qs8-vadd-minmax.inc"
#include "src/qs8-vaddc/qs8-vaddc-minmax.inc"
#include "src/qs8-vmul/qs8-vmul-minmax-fp32.inc"
#include "src/qs8-vmul/qs8-vmul-minmax-rndnu.inc"
#include "src/qs8-vmulc/qs8-vmulc-minmax-fp32.inc"
#include "src/qs8-vmulc/qs8-vmulc-minmax-rndnu.inc"
#include "src/qs8-vprelu/qs8-vprelu.inc"
#include "src/qs8-vpreluc/qs8-vpreluc.inc"
#include "src/qs8-vrpreluc/qs8-vrpreluc.inc"
#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_VBINARY_H_
