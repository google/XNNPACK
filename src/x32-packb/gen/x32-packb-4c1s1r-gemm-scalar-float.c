// Auto-generated file. Do not edit!
//   Template: src/x32-packb/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include "xnnpack/math.h"
#include "xnnpack/packb.h"
#include "xnnpack/unaligned.h"

void xnn_x32_packb_gemm_ukernel_4c1s1r__scalar_float(
  size_t groups,
  size_t channels,
  const uint32_t* bias,
  uint32_t* packed_weights,
  size_t channel_tile_stride,
  size_t channel_subtile_stride,
  const union xnn_x32_packb_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(groups != 0);
  assert(channels != 0);
  assert(packed_weights != NULL);

  float* w = (float*) packed_weights;
  const float* b = (const float*) bias;
  do {
    // channel tile loop multiple of 4
    size_t c = channels;
    for (; c >= 4; c -= 4) {
      const float b0 = b[0];
      const float b1 = b[1];
      const float b2 = b[2];
      const float b3 = b[3];
      unaligned_indexed_store_f32(w, 0, b0);
      unaligned_indexed_store_f32(w, 1, b1);
      unaligned_indexed_store_f32(w, 2, b2);
      unaligned_indexed_store_f32(w, 3, b3);
      b += 4;

      w = (float*) ((uintptr_t) w + channel_tile_stride);
    }

    // channel subtile loop multiple of 1
    for (; c >= 1; c -= 1) {
      const float b0 = b[0];
      unaligned_indexed_store_f32(w, 0, b0);
      b += 1;

      w = (float*) ((uintptr_t) w + channel_subtile_stride);
    }

  } while (--groups != 0);
}
