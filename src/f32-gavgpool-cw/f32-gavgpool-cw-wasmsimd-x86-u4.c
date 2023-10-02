// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_f32_gavgpool_cw_ukernel__wasmsimd_x86_u4(
    size_t elements,
    size_t channels,
    const float* input,
    float* output,
    const union xnn_f32_gavgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(elements != 0);
  assert(elements % sizeof(float) == 0);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + elements);
  const float* i2 = (const float*) ((uintptr_t) i1 + elements);
  const float* i3 = (const float*) ((uintptr_t) i2 + elements);

  const v128_t vmask = wasm_v128_load(params->scalar.mask);
  const v128_t vmultiplier = wasm_v128_load32_splat(&params->scalar.multiplier);
  const v128_t vmin = wasm_v128_load32_splat(&params->scalar.output_min);
  const v128_t vmax = wasm_v128_load32_splat(&params->scalar.output_max);

  while (channels >= 4) {
    v128_t vsum0 = wasm_f32x4_const_splat(0.0f);
    v128_t vsum1 = vsum0;
    v128_t vsum2 = vsum0;
    v128_t vsum3 = vsum0;
    size_t n = elements;
    while (n >= 4 * sizeof(float)) {
      const v128_t vi0 = wasm_v128_load(i0);
      i0 += 4;
      const v128_t vi1 = wasm_v128_load(i1);
      i1 += 4;
      const v128_t vi2 = wasm_v128_load(i2);
      i2 += 4;
      const v128_t vi3 = wasm_v128_load(i3);
      i3 += 4;

      vsum0 = wasm_f32x4_add(vsum0, vi0);
      vsum1 = wasm_f32x4_add(vsum1, vi1);
      vsum2 = wasm_f32x4_add(vsum2, vi2);
      vsum3 = wasm_f32x4_add(vsum3, vi3);
      n -= 4 * sizeof(float);
    }

    if XNN_UNLIKELY(n != 0) {
      const v128_t vi0 = wasm_v128_and(wasm_v128_load(i0), vmask);
      i0 = (const float*) ((uintptr_t) i0 + n);
      const v128_t vi1 = wasm_v128_and(wasm_v128_load(i1), vmask);
      i1 = (const float*) ((uintptr_t) i1 + n);
      const v128_t vi2 = wasm_v128_and(wasm_v128_load(i2), vmask);
      i2 = (const float*) ((uintptr_t) i2 + n);
      const v128_t vi3 = wasm_v128_and(wasm_v128_load(i3), vmask);
      i3 = (const float*) ((uintptr_t) i3 + n);

      vsum0 = wasm_f32x4_add(vsum0, vi0);
      vsum1 = wasm_f32x4_add(vsum1, vi1);
      vsum2 = wasm_f32x4_add(vsum2, vi2);
      vsum3 = wasm_f32x4_add(vsum3, vi3);
    }

    // Having exactly 4 rows makes this work out nicely as we end up with
    // the 4 totals in 4 different lanes of the same vector.
    const v128_t vsum01 = wasm_f32x4_add(wasm_v32x4_shuffle(vsum0, vsum1, 0, 2, 4, 6), wasm_v32x4_shuffle(vsum0, vsum1, 1, 3, 5, 7));
    const v128_t vsum23 = wasm_f32x4_add(wasm_v32x4_shuffle(vsum2, vsum3, 0, 2, 4, 6), wasm_v32x4_shuffle(vsum2, vsum3, 1, 3, 5, 7));
    const v128_t vsum = wasm_f32x4_add(wasm_v32x4_shuffle(vsum01, vsum23, 0, 2, 4, 6), wasm_v32x4_shuffle(vsum01, vsum23, 1, 3, 5, 7));
    v128_t vout = wasm_f32x4_mul(vsum, vmultiplier);

    vout = wasm_f32x4_pmin(vmax, vout);
    vout = wasm_f32x4_pmax(vmin, vout);

    wasm_v128_store(output, vout);
    output += 4;
    i0 = i3;
    i1 = (const float*) ((uintptr_t) i0 + elements);
    i2 = (const float*) ((uintptr_t) i1 + elements);
    i3 = (const float*) ((uintptr_t) i2 + elements);
    channels -= 4;
  }

  while (channels != 0) {
    v128_t vsum = wasm_f32x4_const_splat(0.0f);
    size_t n = elements;
    while (n >= 4 * sizeof(float)) {
      const v128_t vi0 = wasm_v128_load(i0);
      i0 += 4;
      vsum = wasm_f32x4_add(vsum, vi0);
      n -= 4 * sizeof(float);
    }

    if XNN_UNLIKELY(n != 0) {
      v128_t vi0 = wasm_v128_and(vmask, wasm_v128_load(i0));
      i0 = (const float*) ((uintptr_t) i0 + n);
      vsum = wasm_f32x4_add(vsum, vi0);
    }

    vsum = wasm_f32x4_add(wasm_v32x4_shuffle(vsum, vsum, 0, 2, 4, 6), wasm_v32x4_shuffle(vsum, vsum, 1, 3, 5, 7));
    vsum = wasm_f32x4_add(wasm_v32x4_shuffle(vsum, vsum, 0, 2, 4, 6), wasm_v32x4_shuffle(vsum, vsum, 1, 3, 5, 7));

    v128_t vout = wasm_f32x4_mul(vsum, vmultiplier);

    vout = wasm_f32x4_pmin(vmax, vout);
    vout = wasm_f32x4_pmax(vmin, vout);

    *output++ = wasm_f32x4_extract_lane(vout, 0);
    channels -= 1;
  }
}
