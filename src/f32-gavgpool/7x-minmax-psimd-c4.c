// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/gavgpool.h>


void xnn_f32_gavgpool_minmax_ukernel_7x__psimd_c4(
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
  const psimd_f32 vscale = psimd_load_splat_f32(&params->scalar.scale);
  const psimd_f32 vmin = psimd_load_splat_f32(&params->scalar.min);
  const psimd_f32 vmax = psimd_load_splat_f32(&params->scalar.max);

  while (channels >= 4) {
    const psimd_f32 vi0 = psimd_load_f32(i0);
    i0 += 4;
    const psimd_f32 vi1 = psimd_load_f32(i1);
    i1 += 4;
    const psimd_f32 vi2 = psimd_load_f32(i2);
    i2 += 4;
    const psimd_f32 vi3 = psimd_load_f32(i3);
    i3 += 4;
    const psimd_f32 vi4 = psimd_load_f32(i4);
    i4 += 4;
    const psimd_f32 vi5 = psimd_load_f32(i5);
    i5 += 4;
    const psimd_f32 vi6 = psimd_load_f32(i6);
    i6 += 4;

    const psimd_f32 vsum01 = psimd_add_f32(vi0, vi1);
    const psimd_f32 vsum23 = psimd_add_f32(vi2, vi3);
    const psimd_f32 vsum45 = psimd_add_f32(vi4, vi5);

    const psimd_f32 vsum016 = psimd_add_f32(vsum01, vi6);
    const psimd_f32 vsum2345 = psimd_add_f32(vsum23, vsum45);

    const psimd_f32 vsum = psimd_add_f32(vsum016, vsum2345);

    psimd_f32 vout = psimd_mul_f32(vsum, vscale);
    vout = psimd_max_f32(vout, vmin);
    vout = psimd_min_f32(vout, vmax);

    psimd_store_f32(output, vout);
    output += 4;

    channels -= 4;
  }
  if (channels != 0) {
    const psimd_f32 vi0 = psimd_load_f32(i0);
    const psimd_f32 vi1 = psimd_load_f32(i1);
    const psimd_f32 vi2 = psimd_load_f32(i2);
    const psimd_f32 vi3 = psimd_load_f32(i3);
    const psimd_f32 vi4 = psimd_load_f32(i4);
    const psimd_f32 vi5 = psimd_load_f32(i5);
    const psimd_f32 vi6 = psimd_load_f32(i6);

    const psimd_f32 vsum01 = psimd_add_f32(vi0, vi1);
    const psimd_f32 vsum23 = psimd_add_f32(vi2, vi3);
    const psimd_f32 vsum45 = psimd_add_f32(vi4, vi5);

    const psimd_f32 vsum016 = psimd_add_f32(vsum01, vi6);
    const psimd_f32 vsum2345 = psimd_add_f32(vsum23, vsum45);

    const psimd_f32 vsum = psimd_add_f32(vsum016, vsum2345);

    psimd_f32 vout = psimd_mul_f32(vsum, vscale);
    vout = psimd_max_f32(vout, vmin);
    vout = psimd_min_f32(vout, vmax);

    if (channels & 2) {
      psimd_store2_f32(output, vout);
      output += 2;
      vout = psimd_concat_hi_f32(vout, vout);
    }
    if (channels & 1) {
      psimd_store1_f32(output, vout);
    }
  }
}
