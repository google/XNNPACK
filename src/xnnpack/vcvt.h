// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_VCVT_H_
#define XNNPACK_SRC_XNNPACK_VCVT_H_

#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

#define XNN_UKERNEL(arch_flags, ukernel, batch_tile,          \
                                    vector_tile, type_in, type_out,           \
                                    params_type, init_params)                 \
  XNN_INTERNAL void ukernel(size_t n, const type_in* input, type_out* output, \
                            const params_type* params);
#include "src/f16-f32-vcvt/f16-f32-vcvt.inc"
#include "src/f16-qs8-vcvt/f16-qs8-vcvt.inc"
#include "src/f16-qu8-vcvt/f16-qu8-vcvt.inc"
#include "src/f32-f16-vcvt/f32-f16-vcvt.inc"
#include "src/f32-qs8-vcvt/f32-qs8-vcvt.inc"
#include "src/f32-qu8-vcvt/f32-qu8-vcvt.inc"
#include "src/qs8-f16-vcvt/qs8-f16-vcvt.inc"
#include "src/qs8-f32-vcvt/qs8-f32-vcvt.inc"
#include "src/qs8-vcvt/qs8-vcvt.inc"
#include "src/qu8-f32-vcvt/qu8-f32-vcvt.inc"
#include "src/qu8-vcvt/qu8-vcvt.inc"
#undef XNN_UKERNEL

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XNNPACK_SRC_XNNPACK_VCVT_H_
