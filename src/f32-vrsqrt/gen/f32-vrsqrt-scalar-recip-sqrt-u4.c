// Auto-generated file. Do not edit!
//   Template: src/f32-vrsqrt/scalar-rsqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

// Request extensions specified by ISO/IEC TS 18661-4:2015.
#ifndef __STDC_WANT_IEC_60559_FUNCS_EXT__
#define __STDC_WANT_IEC_60559_FUNCS_EXT__ 1
#endif
#include <math.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>
#include <xnnpack/vunary.h>

// The `rsqrtf` function is defined in ISO/IEC/IEEE 60559:2011 (the current
// revision of IEEE-754), and is likely not available everywhere (should be part
// of C23), so check the corresponding feature flag.
#if defined(__STDC_IEC_60559_FUNCS__) && __STDC_IEC_60559_FUNCS__ >= 201506L
#define HAVE_SYSTEM_RSQRTF 1
#else
#define rsqrtf(x) (1.0f / sqrtf(x))
#endif

void xnn_f32_vrsqrt_ukernel__scalar_recip_sqrt_u4(
    size_t batch, const float* input, float* output,
    const union xnn_f32_rsqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) {
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    const float vx2 = input[2];
    const float vx3 = input[3];
    input += 4;

    #ifdef HAVE_SYSTEM_RSQRTF
    const float vy0 = rsqrtf(vx0);
    const float vy1 = rsqrtf(vx1);
    const float vy2 = rsqrtf(vx2);
    const float vy3 = rsqrtf(vx3);
    #else
    const float vt0 = sqrtf(vx0);
    const float vt1 = sqrtf(vx1);
    const float vt2 = sqrtf(vx2);
    const float vt3 = sqrtf(vx3);
    const float vy0 = 1.0f / vt0;
    const float vy1 = 1.0f / vt1;
    const float vy2 = 1.0f / vt2;
    const float vy3 = 1.0f / vt3;
    #endif  // HAVE_SYSTEM_RSQRTF

    output[0] = vy0;
    output[1] = vy1;
    output[2] = vy2;
    output[3] = vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;
      const float vy = rsqrtf(vx);
      *output++ = vy;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}
