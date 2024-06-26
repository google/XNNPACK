// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/gavgpool.h"
#include "xnnpack/math.h"


void xnn_f32_gavgpool_minmax_ukernel_7p7x__wasmsimd_x86_c4(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* buffer,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows > 7);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
  const size_t packed_channels = round_up_po2(channels, 4);
  const size_t input_increment = 7 * input_stride - packed_channels * sizeof(float);

  float* b = buffer;
  for (size_t c = 0; c < channels; c += 4) {
    const v128_t vi0 = wasm_v128_load(i0);
    i0 += 4;
    const v128_t vi1 = wasm_v128_load(i1);
    i1 += 4;
    const v128_t vi2 = wasm_v128_load(i2);
    i2 += 4;
    const v128_t vi3 = wasm_v128_load(i3);
    i3 += 4;
    const v128_t vi4 = wasm_v128_load(i4);
    i4 += 4;
    const v128_t vi5 = wasm_v128_load(i5);
    i5 += 4;
    const v128_t vi6 = wasm_v128_load(i6);
    i6 += 4;

    const v128_t vsum01 = wasm_f32x4_add(vi0, vi1);
    const v128_t vsum23 = wasm_f32x4_add(vi2, vi3);
    const v128_t vsum45 = wasm_f32x4_add(vi4, vi5);

    const v128_t vsum016 = wasm_f32x4_add(vsum01, vi6);
    const v128_t vsum2345 = wasm_f32x4_add(vsum23, vsum45);

    const v128_t vsum = wasm_f32x4_add(vsum016, vsum2345);

    wasm_v128_store(b, vsum);
    b += 4;
  }
  for (rows -= 7; rows > 7; rows -= 7) {
    b = buffer;

    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    i2 = (const float*) ((uintptr_t) i2 + input_increment);
    i3 = (const float*) ((uintptr_t) i3 + input_increment);
    i4 = (const float*) ((uintptr_t) i4 + input_increment);
    i5 = (const float*) ((uintptr_t) i5 + input_increment);
    i6 = (const float*) ((uintptr_t) i6 + input_increment);

    for (size_t c = 0; c < channels; c += 4) {
      const v128_t vi0 = wasm_v128_load(i0);
      i0 += 4;
      const v128_t vi1 = wasm_v128_load(i1);
      i1 += 4;
      const v128_t vi2 = wasm_v128_load(i2);
      i2 += 4;
      const v128_t vi3 = wasm_v128_load(i3);
      i3 += 4;
      const v128_t vi4 = wasm_v128_load(i4);
      i4 += 4;
      const v128_t vi5 = wasm_v128_load(i5);
      i5 += 4;
      const v128_t vi6 = wasm_v128_load(i6);
      i6 += 4;
      const v128_t vacc = wasm_v128_load(b);

      const v128_t vsum01 = wasm_f32x4_add(vi0, vi1);
      const v128_t vsum23 = wasm_f32x4_add(vi2, vi3);
      const v128_t vsum45 = wasm_f32x4_add(vi4, vi5);
      const v128_t vsum6a = wasm_f32x4_add(vi6, vacc);

      const v128_t vsum0123 = wasm_f32x4_add(vsum01, vsum23);
      const v128_t vsum456a = wasm_f32x4_add(vsum45, vsum6a);

      const v128_t vsum = wasm_f32x4_add(vsum0123, vsum456a);

      wasm_v128_store(b, vsum); b += 4;
    }
  }

  i0 = (const float*) ((uintptr_t) i0 + input_increment);
  i1 = (const float*) ((uintptr_t) i1 + input_increment);
  if (rows < 2) {
    i1 = zero;
  }
  i2 = (const float*) ((uintptr_t) i2 + input_increment);
  if (rows <= 2) {
    i2 = zero;
  }
  i3 = (const float*) ((uintptr_t) i3 + input_increment);
  if (rows < 4) {
    i3 = zero;
  }
  i4 = (const float*) ((uintptr_t) i4 + input_increment);
  if (rows <= 4) {
    i4 = zero;
  }
  i5 = (const float*) ((uintptr_t) i5 + input_increment);
  if (rows < 6) {
    i5 = zero;
  }
  i6 = (const float*) ((uintptr_t) i6 + input_increment);
  if (rows <= 6) {
    i6 = zero;
  }
  const v128_t vscale = wasm_v128_load32_splat(&params->scalar.scale);
  const v128_t vmin = wasm_v128_load32_splat(&params->scalar.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->scalar.max);

  b = buffer;
  while (channels >= 4) {
    const v128_t vi0 = wasm_v128_load(i0);
    i0 += 4;
    const v128_t vi1 = wasm_v128_load(i1);
    i1 += 4;
    const v128_t vi2 = wasm_v128_load(i2);
    i2 += 4;
    const v128_t vi3 = wasm_v128_load(i3);
    i3 += 4;
    const v128_t vi4 = wasm_v128_load(i4);
    i4 += 4;
    const v128_t vi5 = wasm_v128_load(i5);
    i5 += 4;
    const v128_t vi6 = wasm_v128_load(i6);
    i6 += 4;
    const v128_t vacc = wasm_v128_load(b);
    b += 4;

    const v128_t vsum01 = wasm_f32x4_add(vi0, vi1);
    const v128_t vsum23 = wasm_f32x4_add(vi2, vi3);
    const v128_t vsum45 = wasm_f32x4_add(vi4, vi5);
    const v128_t vsum6a = wasm_f32x4_add(vi6, vacc);

    const v128_t vsum0123 = wasm_f32x4_add(vsum01, vsum23);
    const v128_t vsum456a = wasm_f32x4_add(vsum45, vsum6a);

    const v128_t vsum = wasm_f32x4_add(vsum0123, vsum456a);

    v128_t vout = wasm_f32x4_mul(vsum, vscale);
    vout = wasm_f32x4_pmax(vmin, vout);
    vout = wasm_f32x4_pmin(vmax, vout);

    wasm_v128_store(output, vout);
    output += 4;

    channels -= 4;
  }
  if (channels != 0) {
    const v128_t vi0 = wasm_v128_load(i0);
    const v128_t vi1 = wasm_v128_load(i1);
    const v128_t vi2 = wasm_v128_load(i2);
    const v128_t vi3 = wasm_v128_load(i3);
    const v128_t vi4 = wasm_v128_load(i4);
    const v128_t vi5 = wasm_v128_load(i5);
    const v128_t vi6 = wasm_v128_load(i6);
    const v128_t vacc = wasm_v128_load(b);

    const v128_t vsum01 = wasm_f32x4_add(vi0, vi1);
    const v128_t vsum23 = wasm_f32x4_add(vi2, vi3);
    const v128_t vsum45 = wasm_f32x4_add(vi4, vi5);
    const v128_t vsum6a = wasm_f32x4_add(vi6, vacc);

    const v128_t vsum0123 = wasm_f32x4_add(vsum01, vsum23);
    const v128_t vsum456a = wasm_f32x4_add(vsum45, vsum6a);

    const v128_t vsum = wasm_f32x4_add(vsum0123, vsum456a);

    v128_t vout = wasm_f32x4_mul(vsum, vscale);
    vout = wasm_f32x4_pmax(vmin, vout);
    vout = wasm_f32x4_pmin(vmax, vout);

    if (channels & 2) {
      wasm_v128_store64_lane(output, vout, 0);
      vout = wasm_v64x2_shuffle(vout, vout, 1, 1);
      output += 2;
    }
    if (channels & 1) {
      wasm_v128_store32_lane(output, vout, 0);
      output += 1;
    }
  }
}
