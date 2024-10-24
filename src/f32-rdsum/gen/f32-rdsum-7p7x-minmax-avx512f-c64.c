// Auto-generated file. Do not edit!
//   Template: src/f32-rdsum/avx512.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include "xnnpack/math.h"


void xnn_f32_rdsum_ukernel_7p7x__avx512f_c64(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const struct xnn_f32_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vscale = _mm512_set1_ps(params->scalar.scale);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 64; channels -= 64) {
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);
    const float* i2 = (const float*) ((uintptr_t) input + 2 * input_stride);
    const float* i3 = (const float*) ((uintptr_t) input + 3 * input_stride);
    const float* i4 = (const float*) ((uintptr_t) input + 4 * input_stride);
    const float* i5 = (const float*) ((uintptr_t) input + 5 * input_stride);
    const float* i6 = (const float*) ((uintptr_t) input + 6 * input_stride);

    __m512 vacc0 = _mm512_setzero_ps();
    __m512 vacc1 = _mm512_setzero_ps();
    __m512 vacc2 = _mm512_setzero_ps();
    __m512 vacc3 = _mm512_setzero_ps();

    for (int r = rows; r > 0; r -= 7) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 2) {
        i2 = zero;
      }
      if XNN_UNPREDICTABLE(r < 4) {
        i3 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 4) {
        i4 = zero;
      }
      if XNN_UNPREDICTABLE(r < 6) {
        i5 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 6) {
        i6 = zero;
      }
      __m512 vin0;
      __m512 vin1;
      __m512 vin2;
      __m512 vin3;
      vin0 = _mm512_loadu_ps(&i0[0]);
      vin1 = _mm512_loadu_ps(&i0[16]);
      vin2 = _mm512_loadu_ps(&i0[32]);
      vin3 = _mm512_loadu_ps(&i0[48]);
      vacc0 = _mm512_add_ps(vin0, vacc0);
      vacc1 = _mm512_add_ps(vin1, vacc1);
      vacc2 = _mm512_add_ps(vin2, vacc2);
      vacc3 = _mm512_add_ps(vin3, vacc3);
      vin0 = _mm512_loadu_ps(&i1[0]);
      vin1 = _mm512_loadu_ps(&i1[16]);
      vin2 = _mm512_loadu_ps(&i1[32]);
      vin3 = _mm512_loadu_ps(&i1[48]);
      vacc0 = _mm512_add_ps(vin0, vacc0);
      vacc1 = _mm512_add_ps(vin1, vacc1);
      vacc2 = _mm512_add_ps(vin2, vacc2);
      vacc3 = _mm512_add_ps(vin3, vacc3);
      vin0 = _mm512_loadu_ps(&i2[0]);
      vin1 = _mm512_loadu_ps(&i2[16]);
      vin2 = _mm512_loadu_ps(&i2[32]);
      vin3 = _mm512_loadu_ps(&i2[48]);
      vacc0 = _mm512_add_ps(vin0, vacc0);
      vacc1 = _mm512_add_ps(vin1, vacc1);
      vacc2 = _mm512_add_ps(vin2, vacc2);
      vacc3 = _mm512_add_ps(vin3, vacc3);
      vin0 = _mm512_loadu_ps(&i3[0]);
      vin1 = _mm512_loadu_ps(&i3[16]);
      vin2 = _mm512_loadu_ps(&i3[32]);
      vin3 = _mm512_loadu_ps(&i3[48]);
      vacc0 = _mm512_add_ps(vin0, vacc0);
      vacc1 = _mm512_add_ps(vin1, vacc1);
      vacc2 = _mm512_add_ps(vin2, vacc2);
      vacc3 = _mm512_add_ps(vin3, vacc3);
      vin0 = _mm512_loadu_ps(&i4[0]);
      vin1 = _mm512_loadu_ps(&i4[16]);
      vin2 = _mm512_loadu_ps(&i4[32]);
      vin3 = _mm512_loadu_ps(&i4[48]);
      vacc0 = _mm512_add_ps(vin0, vacc0);
      vacc1 = _mm512_add_ps(vin1, vacc1);
      vacc2 = _mm512_add_ps(vin2, vacc2);
      vacc3 = _mm512_add_ps(vin3, vacc3);
      vin0 = _mm512_loadu_ps(&i5[0]);
      vin1 = _mm512_loadu_ps(&i5[16]);
      vin2 = _mm512_loadu_ps(&i5[32]);
      vin3 = _mm512_loadu_ps(&i5[48]);
      vacc0 = _mm512_add_ps(vin0, vacc0);
      vacc1 = _mm512_add_ps(vin1, vacc1);
      vacc2 = _mm512_add_ps(vin2, vacc2);
      vacc3 = _mm512_add_ps(vin3, vacc3);
      vin0 = _mm512_loadu_ps(&i6[0]);
      vin1 = _mm512_loadu_ps(&i6[16]);
      vin2 = _mm512_loadu_ps(&i6[32]);
      vin3 = _mm512_loadu_ps(&i6[48]);
      vacc0 = _mm512_add_ps(vin0, vacc0);
      vacc1 = _mm512_add_ps(vin1, vacc1);
      vacc2 = _mm512_add_ps(vin2, vacc2);
      vacc3 = _mm512_add_ps(vin3, vacc3);
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    vacc0 = _mm512_mul_ps(vacc0, vscale);
    vacc1 = _mm512_mul_ps(vacc1, vscale);
    vacc2 = _mm512_mul_ps(vacc2, vscale);
    vacc3 = _mm512_mul_ps(vacc3, vscale);

    const float* o = output;
    const __m512 vo0 = _mm512_loadu_ps(o); o += 16;
    const __m512 vo1 = _mm512_loadu_ps(o); o += 16;
    const __m512 vo2 = _mm512_loadu_ps(o); o += 16;
    const __m512 vo3 = _mm512_loadu_ps(o); o += 16;
    vacc0 = _mm512_add_ps(vo0, vacc0);
    vacc1 = _mm512_add_ps(vo1, vacc1);
    vacc2 = _mm512_add_ps(vo2, vacc2);
    vacc3 = _mm512_add_ps(vo3, vacc3);
    _mm512_storeu_ps(output, vacc0); output += 16;
    _mm512_storeu_ps(output, vacc1); output += 16;
    _mm512_storeu_ps(output, vacc2); output += 16;
    _mm512_storeu_ps(output, vacc3); output += 16;

    input = (const float*) ((uintptr_t) input + 64 * sizeof(float));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    const float* i0 = input;
    const float* i1 = (const float*) ((uintptr_t) input + 1 * input_stride);
    const float* i2 = (const float*) ((uintptr_t) input + 2 * input_stride);
    const float* i3 = (const float*) ((uintptr_t) input + 3 * input_stride);
    const float* i4 = (const float*) ((uintptr_t) input + 4 * input_stride);
    const float* i5 = (const float*) ((uintptr_t) input + 5 * input_stride);
    const float* i6 = (const float*) ((uintptr_t) input + 6 * input_stride);
    __m512 vacc[4];
    vacc[0] = _mm512_setzero_ps();
    vacc[1] = _mm512_setzero_ps();
    vacc[2] = _mm512_setzero_ps();
    vacc[3] = _mm512_setzero_ps();

    const size_t num_full_chunks = channels >> 4;
    const size_t num_chunks = round_up_po2(channels, 16) >> 4;
    const size_t remainder = channels & 0xF;
    const size_t batch = channels & 0xF;
    __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));
    if (remainder) {
      assert(batch >= 1);
      assert(batch <= 15);
      vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));
    }
    for (int r = rows; r > 0; r -= 7) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 2) {
        i2 = zero;
      }
      if XNN_UNPREDICTABLE(r < 4) {
        i3 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 4) {
        i4 = zero;
      }
      if XNN_UNPREDICTABLE(r < 6) {
        i5 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 6) {
        i6 = zero;
      }
      for (int i = 0; i < num_full_chunks; ++i) {
        vacc[i] = _mm512_add_ps(_mm512_loadu_ps(&i0[i*16]), vacc[i]);
        vacc[i] = _mm512_add_ps(_mm512_loadu_ps(&i1[i*16]), vacc[i]);
        vacc[i] = _mm512_add_ps(_mm512_loadu_ps(&i2[i*16]), vacc[i]);
        vacc[i] = _mm512_add_ps(_mm512_loadu_ps(&i3[i*16]), vacc[i]);
        vacc[i] = _mm512_add_ps(_mm512_loadu_ps(&i4[i*16]), vacc[i]);
        vacc[i] = _mm512_add_ps(_mm512_loadu_ps(&i5[i*16]), vacc[i]);
        vacc[i] = _mm512_add_ps(_mm512_loadu_ps(&i6[i*16]), vacc[i]);
      }

      if (remainder) {
        vacc[num_full_chunks] = _mm512_maskz_add_ps(vmask, vacc[num_full_chunks],  _mm512_maskz_loadu_ps(vmask, &i0[num_full_chunks*16]));
        vacc[num_full_chunks] = _mm512_maskz_add_ps(vmask, vacc[num_full_chunks],  _mm512_maskz_loadu_ps(vmask, &i1[num_full_chunks*16]));
        vacc[num_full_chunks] = _mm512_maskz_add_ps(vmask, vacc[num_full_chunks],  _mm512_maskz_loadu_ps(vmask, &i2[num_full_chunks*16]));
        vacc[num_full_chunks] = _mm512_maskz_add_ps(vmask, vacc[num_full_chunks],  _mm512_maskz_loadu_ps(vmask, &i3[num_full_chunks*16]));
        vacc[num_full_chunks] = _mm512_maskz_add_ps(vmask, vacc[num_full_chunks],  _mm512_maskz_loadu_ps(vmask, &i4[num_full_chunks*16]));
        vacc[num_full_chunks] = _mm512_maskz_add_ps(vmask, vacc[num_full_chunks],  _mm512_maskz_loadu_ps(vmask, &i5[num_full_chunks*16]));
        vacc[num_full_chunks] = _mm512_maskz_add_ps(vmask, vacc[num_full_chunks],  _mm512_maskz_loadu_ps(vmask, &i6[num_full_chunks*16]));
      }
      i0 = (const float*) ((uintptr_t) i0 + input_increment);
      i1 = (const float*) ((uintptr_t) i1 + input_increment);
      i2 = (const float*) ((uintptr_t) i2 + input_increment);
      i3 = (const float*) ((uintptr_t) i3 + input_increment);
      i4 = (const float*) ((uintptr_t) i4 + input_increment);
      i5 = (const float*) ((uintptr_t) i5 + input_increment);
      i6 = (const float*) ((uintptr_t) i6 + input_increment);
    }
    for (size_t i = 0; i < num_chunks; ++i) {
      vacc[i] = _mm512_mul_ps(vacc[i], vscale);
    }

    __m512 vo[4];
    const float* o = output;
    for (int i = 0; i < channels >> 4; ++i) {
      vo[i] = _mm512_loadu_ps(o); o += 16;
    }
    for (int i = 0; i < channels >> 4; ++i) {
      vacc[i] = _mm512_add_ps(vo[i], vacc[i]);
    }
    for (int i = 0; i < channels >> 4; ++i) {
      _mm512_storeu_ps(output, vacc[i]); output += 16;
    }
    if (remainder) {
      const size_t pos = num_full_chunks;
      __m512 vout = vacc[pos];
      vout = _mm512_maskz_add_ps(vmask, vout,  _mm512_maskz_loadu_ps(vmask, output));
      _mm512_mask_storeu_ps(output, vmask, vout);
    }
  }
}
