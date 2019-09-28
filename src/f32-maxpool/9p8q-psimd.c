/*
 * Copyright 2019 Google LLC
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <psimd.h>

#include <xnnpack/maxpool.h>


void xnn_f32_maxpool_ukernel_9p8q__psimd(
    size_t n,
    size_t ks,
    size_t kc,
    const float** input,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(n != 0);
  assert(ks != 0);
  assert(kc != 0);

  const psimd_f32 voutput_max = psimd_load_splat_f32(&params->scalar.max);
  const psimd_f32 voutput_min = psimd_load_splat_f32(&params->scalar.min);
  do {
    float* o = output;
    {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      const float* i8 = *input++;
      if (ks < 2) {
        i1 = i0;
      }
      if (ks <= 2) {
        i2 = i0;
      }
      if (ks < 4) {
        i3 = i0;
      }
      if (ks <= 4) {
        i4 = i0;
      }
      if (ks < 6) {
        i5 = i0;
      }
      if (ks <= 6) {
        i6 = i0;
      }
      if (ks < 8) {
        i7 = i0;
      }
      if (ks <= 8) {
        i8 = i0;
      }

      size_t k = kc;
      for (; k >= 4; k -= 4) {
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
        const psimd_f32 vi7 = psimd_load_f32(i7);
        i7 += 4;
        const psimd_f32 vi8 = psimd_load_f32(i8);
        i8 += 4;

        const psimd_f32 vmax018 = psimd_max_f32(psimd_max_f32(vi0, vi1), vi8);
        const psimd_f32 vmax23 = psimd_max_f32(vi2, vi3);
        const psimd_f32 vmax45 = psimd_max_f32(vi4, vi5);
        const psimd_f32 vmax67 = psimd_max_f32(vi6, vi7);

        const psimd_f32 vmax2345 = psimd_max_f32(vmax23, vmax45);
        const psimd_f32 vmax01678 = psimd_max_f32(vmax018, vmax67);
        const psimd_f32 vmax = psimd_max_f32(vmax2345, vmax01678);
        const psimd_f32 vout = psimd_max_f32(psimd_min_f32(vmax, voutput_max), voutput_min);

        psimd_store_f32(o, vout);
        o += 4;
      }
      if (k != 0) {
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
        const psimd_f32 vi7 = psimd_load_f32(i7);
        i7 += 4;
        const psimd_f32 vi8 = psimd_load_f32(i8);
        i8 += 4;

        const psimd_f32 vmax018 = psimd_max_f32(psimd_max_f32(vi0, vi1), vi8);
        const psimd_f32 vmax23 = psimd_max_f32(vi2, vi3);
        const psimd_f32 vmax45 = psimd_max_f32(vi4, vi5);
        const psimd_f32 vmax67 = psimd_max_f32(vi6, vi7);

        const psimd_f32 vmax2345 = psimd_max_f32(vmax23, vmax45);
        const psimd_f32 vmax01678 = psimd_max_f32(vmax018, vmax67);
        const psimd_f32 vmax = psimd_max_f32(vmax2345, vmax01678);
        psimd_f32 vout = psimd_max_f32(psimd_min_f32(vmax, voutput_max), voutput_min);

        if (k & 2) {
          psimd_store2_f32(o, vout);
          vout = psimd_concat_hi_f32(vout, vout);
          o += 2;
        }
        if (k & 1) {
          psimd_store1_f32(o, vout);
          o += 1;
        }
      }
    }

    for (ptrdiff_t m = (ptrdiff_t) ks - 9; m > 0; m -= 8) {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      if (m < 2) {
        i1 = i0;
      }
      if (m <= 2) {
        i2 = i0;
      }
      if (m < 4) {
        i3 = i0;
      }
      if (m <= 4) {
        i4 = i0;
      }
      if (m < 6) {
        i5 = i0;
      }
      if (m <= 6) {
        i6 = i0;
      }
      if (m < 8) {
        i7 = i0;
      }

      o = output;
      size_t k = kc;
      for (; k >= 4; k -= 4) {
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
        const psimd_f32 vi7 = psimd_load_f32(i7);
        i7 += 4;
        const psimd_f32 vo = psimd_load_f32(o);

        const psimd_f32 vmax01 = psimd_max_f32(psimd_max_f32(vi0, vi1), vo);
        const psimd_f32 vmax23 = psimd_max_f32(vi2, vi3);
        const psimd_f32 vmax45 = psimd_max_f32(vi4, vi5);
        const psimd_f32 vmax67 = psimd_max_f32(vi6, vi7);

        const psimd_f32 vmax2345 = psimd_max_f32(vmax23, vmax45);
        const psimd_f32 vmax0167 = psimd_max_f32(vmax01, vmax67);
        const psimd_f32 vmax = psimd_max_f32(vmax2345, vmax0167);
        const psimd_f32 vout = psimd_max_f32(psimd_min_f32(vmax, voutput_max), voutput_min);

        psimd_store_f32(o, vout);
        o += 4;
      }
      if (k != 0) {
        const psimd_f32 vi0 = psimd_load_f32(i0);
        const psimd_f32 vi1 = psimd_load_f32(i1);
        const psimd_f32 vi2 = psimd_load_f32(i2);
        const psimd_f32 vi3 = psimd_load_f32(i3);
        const psimd_f32 vi4 = psimd_load_f32(i4);
        const psimd_f32 vi5 = psimd_load_f32(i5);
        const psimd_f32 vi6 = psimd_load_f32(i6);
        const psimd_f32 vi7 = psimd_load_f32(i7);
        const psimd_f32 vo = psimd_load_f32(o);

        const psimd_f32 vmax01 = psimd_max_f32(psimd_max_f32(vi0, vi1), vo);
        const psimd_f32 vmax23 = psimd_max_f32(vi2, vi3);
        const psimd_f32 vmax45 = psimd_max_f32(vi4, vi5);
        const psimd_f32 vmax67 = psimd_max_f32(vi6, vi7);

        const psimd_f32 vmax2345 = psimd_max_f32(vmax23, vmax45);
        const psimd_f32 vmax0167 = psimd_max_f32(vmax01, vmax67);
        const psimd_f32 vmax = psimd_max_f32(vmax2345, vmax0167);
        psimd_f32 vout = psimd_max_f32(psimd_min_f32(vmax, voutput_max), voutput_min);

        if (k & 2) {
          psimd_store2_f32(o, vout);
          vout = psimd_concat_hi_f32(vout, vout);
          o += 2;
        }
        if (k & 1) {
          psimd_store1_f32(o, vout);
          o += 1;
        }
      }
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) o + output_increment);
  } while (--n != 0);
}
