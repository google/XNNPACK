// clang-format off
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
#include <stdint.h>

#include "src/xnnpack/microparams.h"
#include "src/xnnpack/packb.h"
#include "src/xnnpack/unaligned.h"

void xnn_x32_packb_gemm_ukernel_2c1s1r__scalar_int(
  size_t groups,
  size_t channels,
  const uint32_t* bias,
  uint32_t* packed_weights,
  size_t channel_tile_stride,
  size_t channel_subtile_stride,
  const struct xnn_x32_packb_params* restrict params)
{
  assert(groups != 0);
  assert(channels != 0);
  assert(packed_weights != NULL);

  uint32_t* w = (uint32_t*) packed_weights;
  const uint32_t* b = (const uint32_t*) bias;
  do {
    // channel tile loop multiple of 2
    size_t c = channels;
    for (; c >= 2; c -= 2) {
      const uint32_t b0 = b[0];
      const uint32_t b1 = b[1];
      unaligned_indexed_store_u32(w, 0, b0);
      unaligned_indexed_store_u32(w, 1, b1);
      b += 2;

      w = (uint32_t*) ((uintptr_t) w + channel_tile_stride);
    }

    // channel subtile loop multiple of 1
    if (c != 0) {
      const uint32_t b0 = b[0];
      unaligned_indexed_store_u32(w, 0, b0);
      b += 1;

      w = (uint32_t*) ((uintptr_t) w + channel_subtile_stride);
    }

  } while (--groups != 0);
}
