// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_f32_gavgpool_cw_ukernel__psimd_x4(
    size_t elements,
    size_t channels,
    const float* input,
    float* output,
    const union xnn_f32_gavgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(elements != 0);
  assert(elements % sizeof(float) == 0);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + elements);
  const float* i2 = (const float*) ((uintptr_t) i1 + elements);
  const float* i3 = (const float*) ((uintptr_t) i2 + elements);

  const psimd_s32 vmask = psimd_load_s32(params->scalar.mask);
  const psimd_f32 vmultiplier = psimd_load_splat_f32(&params->scalar.multiplier);
  const psimd_f32 voutput_min = psimd_load_splat_f32(&params->scalar.output_min);
  const psimd_f32 voutput_max = psimd_load_splat_f32(&params->scalar.output_max);

  while (channels >= 4) {
    psimd_f32 vsum0 = psimd_zero_f32();
    psimd_f32 vsum1 = psimd_zero_f32();
    psimd_f32 vsum2 = psimd_zero_f32();
    psimd_f32 vsum3 = psimd_zero_f32();
    size_t n = elements;
    while (n >= 4 * sizeof(float)) {
      const psimd_f32 vi0 = psimd_load_f32(i0);
      i0 += 4;
      const psimd_f32 vi1 = psimd_load_f32(i1);
      i1 += 4;
      const psimd_f32 vi2 = psimd_load_f32(i2);
      i2 += 4;
      const psimd_f32 vi3 = psimd_load_f32(i3);
      i3 += 4;

      vsum0 = psimd_add_f32(vsum0, vi0);
      vsum1 = psimd_add_f32(vsum1, vi1);
      vsum2 = psimd_add_f32(vsum2, vi2);
      vsum3 = psimd_add_f32(vsum3, vi3);
      n -= 4 * sizeof(float);
    }

    if XNN_UNLIKELY(n != 0) {
      const psimd_f32 vi0 = psimd_andmask_f32(vmask, psimd_load_f32(i0));
      i0 = (const float*) ((uintptr_t) i0 + n);
      const psimd_f32 vi1 = psimd_andmask_f32(vmask, psimd_load_f32(i1));
      i1 = (const float*) ((uintptr_t) i1 + n);
      const psimd_f32 vi2 = psimd_andmask_f32(vmask, psimd_load_f32(i2));
      i2 = (const float*) ((uintptr_t) i2 + n);
      const psimd_f32 vi3 = psimd_andmask_f32(vmask, psimd_load_f32(i3));
      i3 = (const float*) ((uintptr_t) i3 + n);

      vsum0 = psimd_add_f32(vsum0, vi0);
      vsum1 = psimd_add_f32(vsum1, vi1);
      vsum2 = psimd_add_f32(vsum2, vi2);
      vsum3 = psimd_add_f32(vsum3, vi3);
    }

    // Having exaclty 4 rows makes this work out nicely as we end up with
    // the 4 totals in 4 different lanes of the same vector.
    const psimd_f32 vsum01 = psimd_add_f32(psimd_concat_even_f32(vsum0, vsum1), psimd_concat_odd_f32(vsum0, vsum1));
    const psimd_f32 vsum23 = psimd_add_f32(psimd_concat_even_f32(vsum2, vsum3), psimd_concat_odd_f32(vsum2, vsum3));
    const psimd_f32 vsum = psimd_add_f32(psimd_concat_even_f32(vsum01, vsum23), psimd_concat_odd_f32(vsum01, vsum23));
    psimd_f32 vout = psimd_mul_f32(vsum, vmultiplier);

    vout = psimd_max_f32(vout, voutput_min);
    vout = psimd_min_f32(vout, voutput_max);

    psimd_store_f32(output, vout);
    output += 4;
    i0 = i3;
    i1 = (const float*) ((uintptr_t) i0 + elements);
    i2 = (const float*) ((uintptr_t) i1 + elements);
    i3 = (const float*) ((uintptr_t) i2 + elements);
    channels -= 4;
  }

  while (channels != 0) {
    psimd_f32 vsum = psimd_zero_f32();
    size_t n = elements;
    while (n >= 4 * sizeof(float)) {
      const psimd_f32 vi0 = psimd_load_f32(i0);
      i0 += 4;
      vsum = psimd_add_f32(vsum, vi0);
      n -= 4 * sizeof(float);
    }

    if XNN_UNLIKELY(n != 0) {
      psimd_f32 vi0 = psimd_andmask_f32(vmask, psimd_load_f32(i0));
      i0 = (const float*) ((uintptr_t) i0 + n);
      vsum = psimd_add_f32(vsum, vi0);
    }

    vsum = psimd_add_f32(psimd_concat_even_f32(vsum, vsum), psimd_concat_odd_f32(vsum, vsum));
    vsum = psimd_add_f32(psimd_concat_even_f32(vsum, vsum), psimd_concat_odd_f32(vsum, vsum));

    psimd_f32 vout = psimd_mul_f32(vsum, vmultiplier);

    vout = psimd_max_f32(vout, voutput_min);
    vout = psimd_min_f32(vout, voutput_max);

    psimd_store1_f32(output, vout);
    output += 1;
    channels -= 1;
  }
}
