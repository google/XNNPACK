// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/gavgpool.h"
#include "xnnpack/math.h"


void xnn_f32_gavgpool_minmax_ukernel_7x__wasm_c1(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  if (rows < 2) {
    i1 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  if (rows <= 2) {
    i2 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
  if (rows < 4) {
    i3 = zero;
  }
  const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
  if (rows <= 4) {
    i4 = zero;
  }
  const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
  if (rows < 6) {
    i5 = zero;
  }
  const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
  if (rows <= 6) {
    i6 = zero;
  }

  const float vscale = params->scalar.scale;
  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float vi0 = *i0++;
    const float vi1 = *i1++;
    const float vi2 = *i2++;
    const float vi3 = *i3++;
    const float vi4 = *i4++;
    const float vi5 = *i5++;
    const float vi6 = *i6++;

    const float vsum01 = vi0 + vi1;
    const float vsum23 = vi2 + vi3;
    const float vsum45 = vi4 + vi5;

    const float vsum016 = vsum01 + vi6;
    const float vsum2345 = vsum23 + vsum45;

    const float vsum = vsum016 + vsum2345;

    float vout = vsum * vscale;
    vout = __builtin_wasm_max_f32(vout, vmin);
    vout = __builtin_wasm_min_f32(vout, vmax);

    *output++ = vout;
  } while (--channels != 0);
}
