// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/fill.h>


void xnn_x32_fill_ukernel__scalar_float(
    size_t rows,
    size_t channels,
    uint32_t* output,
    size_t output_stride,
    const uint32_t* fill_value)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);
  assert(fill_value != NULL);

  const size_t output_increment = output_stride - channels;

  const float vfill = *((const float*) fill_value);
  float* o = (float*) output;
  do {
    size_t c = channels;
    for (; c >= 4 * sizeof(uint32_t); c -= 4 * sizeof(uint32_t)) {
      o[0] = vfill;
      o[1] = vfill;
      o[2] = vfill;
      o[3] = vfill;
      o += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      do {
        *o++ = vfill;
        c -= sizeof(uint32_t);
      } while (c != 0);
    }
    o = (void*) ((uintptr_t) o + output_increment);
  } while (--rows != 0);
}
