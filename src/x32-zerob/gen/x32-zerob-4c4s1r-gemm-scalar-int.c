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

#include <xnnpack/math.h>
#include <xnnpack/packb.h>
#include <xnnpack/unaligned.h>

void xnn_x32_zerob_gemm_ukernel_4c4s1r__scalar_int(
  size_t groups,
  size_t channels,
  uint32_t* packed_weights,
  size_t channel_tile_stride,
  size_t channel_subtile_stride,
  const union xnn_x32_packb_params* params)
{
  assert(groups != 0);
  assert(channels != 0);
  assert(packed_weights != NULL);

  uint32_t* w = (uint32_t*) packed_weights;
  const uint32_t vzero = 0;
  do {
    // channel tile loop multiple of 4
    size_t c = channels;
    for (; c >= 4; c -= 4) {
      unaligned_indexed_store_u32(w, 0, vzero);
      unaligned_indexed_store_u32(w, 1, vzero);
      unaligned_indexed_store_u32(w, 2, vzero);
      unaligned_indexed_store_u32(w, 3, vzero);

      w = (uint32_t*) ((uintptr_t) w + channel_tile_stride);
    }


    if XNN_UNLIKELY(c != 0) {
      // channels remainder (1..3)
      uint32_t* prev_w = w;
      if (c & 2) {
        unaligned_indexed_store_u32(w, 0, vzero);
        unaligned_indexed_store_u32(w, 1, vzero);
        w += 2;
      }
      if (c & 1) {
        unaligned_indexed_store_u32(w, 0, vzero);
        w += 1;
      }

      w = (uint32_t*) ((uintptr_t) prev_w + channel_subtile_stride);
    }
  } while (--groups != 0);
}
