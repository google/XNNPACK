// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/pad.h>


void xnn_x32_pad_x2__scalar(
    size_t m,
    size_t n,
    size_t l,
    size_t r,
    uint32_t c,
    const void* x,
    size_t x_stride,
    void* y,
    size_t y_stride)
{
  assert(m <= 2);
  assert(l % 4 == 0);
  assert(n % 4 == 0);
  assert(r % 4 == 0);

  const uint32_t* x0 = x;
  uint32_t* y0 = y;

  const uint32_t* x1 = (const uint32_t*) ((uintptr_t) x0 + x_stride);
  uint32_t* y1 = (uint32_t*) ((uintptr_t) y0 + y_stride);
  if (m != 2) {
    x1 = x0;
    y1 = y0;
  }

  // Pre-pad input channels.
  for (; l != 0; l -= 4) {
    *y0++ = c;
    *y1++ = c;
  }

  // Copy input channels.
  for (; n != 0; n -= 4) {
    *y0++ = *x0++;
    *y1++ = *x1++;
  }

  // Post-pad input channels.
  for (; r != 0; r -= 4) {
    *y0++ = c;
    *y1++ = c;
  }
}
