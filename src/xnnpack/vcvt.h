// Copyright 2021 Google LLC
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

#define XNN_CVT_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_tile, vector_tile, type_in, type_out, params_type, init_params) \
  XNN_INTERNAL void ukernel(size_t n, const type_in* input, type_out* output, const params_type* params);
#include "src/f16-f32-vcvt/f16-f32-vcvt.h"
#include "src/f16-qs8-vcvt/f16-qs8-vcvt.h"
#include "src/f32-f16-vcvt/f32-f16-vcvt.h"
#include "src/f32-qs8-vcvt/f32-qs8-vcvt.h"
#include "src/f32-qu8-vcvt/f32-qu8-vcvt.h"
#include "src/qs16-qs8-vcvt/qs16-qs8-vcvt.h"
#include "src/qs8-f16-vcvt/qs8-f16-vcvt.h"
#include "src/qs8-f32-vcvt/qs8-f32-vcvt.h"
#include "src/qu8-f32-vcvt/qu8-f32-vcvt.h"
#include "src/s32-f32-vcvt/s32-f32-vcvt.h"
#include "src/u32-f32-vcvt/u32-f32-vcvt.h"
#include "src/qs8-vcvt/qs8-vcvt.h"
#include "src/qu8-vcvt/qu8-vcvt.h"
#undef XNN_CVT_UKERNEL_WITH_PARAMS

#ifdef __cplusplus
}  // extern "C"
#endif
