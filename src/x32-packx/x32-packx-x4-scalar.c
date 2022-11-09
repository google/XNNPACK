// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/packx.h>


void xnn_x32_packx_ukernel_4x__scalar(
    size_t m,
    size_t k,
    const uint32_t* restrict x,
    size_t x_stride,
    uint32_t* restrict y)
{
  assert(m != 0);
  assert(k != 0);

  const float* x0 = (const float*) x;
  const float* x1 = (const float*) ((uintptr_t) x0 + x_stride);
  if (m < 2) {
    x1 = x0;
  }
  const float* x2 = (const float*) ((uintptr_t) x1 + x_stride);
  if (m <= 2) {
    x2 = x1;
  }
  const float* x3 = (const float*) ((uintptr_t) x2 + x_stride);
  if (m != 4) {
    x3 = x2;
  }

  float*restrict y_f32 = (float*) y;

  do {
    const float vx0 = *x0++;
    const float vx1 = *x1++;
    const float vx2 = *x2++;
    const float vx3 = *x3++;

    y_f32[0] = vx0;
    y_f32[1] = vx1;
    y_f32[2] = vx2;
    y_f32[3] = vx3;
    y_f32 += 4;
  } while (--k != 0);
}
