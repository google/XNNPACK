// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/fill.h>


void xnn_x32_fill_ukernel__scalar_int(
    size_t rows,
    size_t channels,
    uint32_t* output,
    size_t output_stride,
    const uint32_t* fill_value)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(uint32_t) == 0);
  assert(fill_value != NULL);

  const size_t output_increment = output_stride - channels;

  const uint32_t vfill = *fill_value;
  do {
    size_t c = channels;
    for (; c >= 4 * sizeof(uint32_t); c -= 4 * sizeof(uint32_t)) {
      output[0] = vfill;
      output[1] = vfill;
      output[2] = vfill;
      output[3] = vfill;
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      do {
        *output++ = vfill;
        c -= sizeof(uint32_t);
      } while (c != 0);
    }
    output = (void*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}
