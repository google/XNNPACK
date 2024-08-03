// Auto-generated file. Do not edit!
//   Template: src/f16-f32acc-rdsum/avx512skx.c.in
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


void xnn_f16_f32acc_rdsum_ukernel_7p7x__avx512skx_c64(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    float* output,
    const union xnn_f16_f32acc_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vscale = _mm512_set1_ps(params->scale);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 64; channels -= 64) {
    const uint16_t* i0 = input;
    const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input + 1 * input_stride);
    const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input + 2 * input_stride);
    const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input + 3 * input_stride);
    const uint16_t* i4 = (const uint16_t*) ((uintptr_t) input + 4 * input_stride);
    const uint16_t* i5 = (const uint16_t*) ((uintptr_t) input + 5 * input_stride);
    const uint16_t* i6 = (const uint16_t*) ((uintptr_t) input + 6 * input_stride);

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
      vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[0])));
      vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[16])));
      vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[32])));
      vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[48])));
      vacc0 = _mm512_add_ps(vin0, vacc0);
      vacc1 = _mm512_add_ps(vin1, vacc1);
      vacc2 = _mm512_add_ps(vin2, vacc2);
      vacc3 = _mm512_add_ps(vin3, vacc3);
      vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[0])));
      vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[16])));
      vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[32])));
      vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[48])));
      vacc0 = _mm512_add_ps(vin0, vacc0);
      vacc1 = _mm512_add_ps(vin1, vacc1);
      vacc2 = _mm512_add_ps(vin2, vacc2);
      vacc3 = _mm512_add_ps(vin3, vacc3);
      vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[0])));
      vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[16])));
      vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[32])));
      vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[48])));
      vacc0 = _mm512_add_ps(vin0, vacc0);
      vacc1 = _mm512_add_ps(vin1, vacc1);
      vacc2 = _mm512_add_ps(vin2, vacc2);
      vacc3 = _mm512_add_ps(vin3, vacc3);
      vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[0])));
      vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[16])));
      vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[32])));
      vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[48])));
      vacc0 = _mm512_add_ps(vin0, vacc0);
      vacc1 = _mm512_add_ps(vin1, vacc1);
      vacc2 = _mm512_add_ps(vin2, vacc2);
      vacc3 = _mm512_add_ps(vin3, vacc3);
      vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[0])));
      vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[16])));
      vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[32])));
      vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[48])));
      vacc0 = _mm512_add_ps(vin0, vacc0);
      vacc1 = _mm512_add_ps(vin1, vacc1);
      vacc2 = _mm512_add_ps(vin2, vacc2);
      vacc3 = _mm512_add_ps(vin3, vacc3);
      vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[0])));
      vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[16])));
      vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[32])));
      vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[48])));
      vacc0 = _mm512_add_ps(vin0, vacc0);
      vacc1 = _mm512_add_ps(vin1, vacc1);
      vacc2 = _mm512_add_ps(vin2, vacc2);
      vacc3 = _mm512_add_ps(vin3, vacc3);
      vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[0])));
      vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[16])));
      vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[32])));
      vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[48])));
      vacc0 = _mm512_add_ps(vin0, vacc0);
      vacc1 = _mm512_add_ps(vin1, vacc1);
      vacc2 = _mm512_add_ps(vin2, vacc2);
      vacc3 = _mm512_add_ps(vin3, vacc3);
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_increment);
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_increment);
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_increment);
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_increment);
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_increment);
    }
    vacc0 = _mm512_mul_ps(vacc0, vscale);
    vacc1 = _mm512_mul_ps(vacc1, vscale);
    vacc2 = _mm512_mul_ps(vacc2, vscale);
    vacc3 = _mm512_mul_ps(vacc3, vscale);

    __m512 vo0 = _mm512_loadu_ps(output + 0 * 16);
    __m512 vo1 = _mm512_loadu_ps(output + 1 * 16);
    __m512 vo2 = _mm512_loadu_ps(output + 2 * 16);
    __m512 vo3 = _mm512_loadu_ps(output + 3 * 16);
    vacc0 = _mm512_add_ps(vo0, vacc0);
    vacc1 = _mm512_add_ps(vo1, vacc1);
    vacc2 = _mm512_add_ps(vo2, vacc2);
    vacc3 = _mm512_add_ps(vo3, vacc3);
    _mm512_storeu_ps(output, vacc0); output = (void*) ((uintptr_t) output + 16 * sizeof(float));
    _mm512_storeu_ps(output, vacc1); output = (void*) ((uintptr_t) output + 16 * sizeof(float));
    _mm512_storeu_ps(output, vacc2); output = (void*) ((uintptr_t) output + 16 * sizeof(float));
    _mm512_storeu_ps(output, vacc3); output = (void*) ((uintptr_t) output + 16 * sizeof(float));

    input = (const uint16_t*) ((uintptr_t) input + 64 * sizeof(uint16_t));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    const uint16_t* i0 = input;
    const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input + 1 * input_stride);
    const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input + 2 * input_stride);
    const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input + 3 * input_stride);
    const uint16_t* i4 = (const uint16_t*) ((uintptr_t) input + 4 * input_stride);
    const uint16_t* i5 = (const uint16_t*) ((uintptr_t) input + 5 * input_stride);
    const uint16_t* i6 = (const uint16_t*) ((uintptr_t) input + 6 * input_stride);
    __m512 vacc[4];
    vacc[0] = _mm512_setzero_ps();
    vacc[1] = _mm512_setzero_ps();
    vacc[2] = _mm512_setzero_ps();
    vacc[3] = _mm512_setzero_ps();

    const size_t num_full_chunks = channels >> 4;
    // AVX512 has 16 float lanes.
    const size_t num_chunks = round_up_po2(channels, 16) >> 4;
    // 0xF masks the remainder.
    const size_t remainder = channels & 0xF;
    const size_t batch = channels & 0xF;
    __mmask16 vmask;
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
        vacc[i] = _mm512_add_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i0[i*16])), vacc[i]);
        vacc[i] = _mm512_add_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i1[i*16])), vacc[i]);
        vacc[i] = _mm512_add_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i2[i*16])), vacc[i]);
        vacc[i] = _mm512_add_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i3[i*16])), vacc[i]);
        vacc[i] = _mm512_add_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i4[i*16])), vacc[i]);
        vacc[i] = _mm512_add_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i5[i*16])), vacc[i]);
        vacc[i] = _mm512_add_ps(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i6[i*16])), vacc[i]);
      }

      if (remainder) {
        vacc[num_full_chunks] = _mm512_maskz_add_ps(vmask, vacc[num_full_chunks],  _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i0[num_full_chunks*16])));
        vacc[num_full_chunks] = _mm512_maskz_add_ps(vmask, vacc[num_full_chunks],  _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i1[num_full_chunks*16])));
        vacc[num_full_chunks] = _mm512_maskz_add_ps(vmask, vacc[num_full_chunks],  _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i2[num_full_chunks*16])));
        vacc[num_full_chunks] = _mm512_maskz_add_ps(vmask, vacc[num_full_chunks],  _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i3[num_full_chunks*16])));
        vacc[num_full_chunks] = _mm512_maskz_add_ps(vmask, vacc[num_full_chunks],  _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i4[num_full_chunks*16])));
        vacc[num_full_chunks] = _mm512_maskz_add_ps(vmask, vacc[num_full_chunks],  _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i5[num_full_chunks*16])));
        vacc[num_full_chunks] = _mm512_maskz_add_ps(vmask, vacc[num_full_chunks],  _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i6[num_full_chunks*16])));
      }
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_increment);
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_increment);
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_increment);
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_increment);
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_increment);
    }
    for (size_t i = 0; i < num_chunks; ++i) {
      vacc[i] = _mm512_mul_ps(vacc[i], vscale);
    }

    __m512 vo[4];
    for (int i = 0; i < num_full_chunks; ++i) {
      vo[i] = _mm512_loadu_ps(output + i * 16);
    }
    for (int i = 0; i < num_full_chunks; ++i) {
      vacc[i] = _mm512_add_ps(vo[i], vacc[i]);
    }
    for (int i = 0; i < num_full_chunks; ++i) {
      _mm512_storeu_ps(output, vacc[i]); output = (void*) ((uintptr_t) output + 16 * sizeof(float));
    }
    if (remainder) {
      __m512 vout = vacc[num_full_chunks];
      vout = _mm512_maskz_add_ps(vmask, vout,  _mm512_maskz_loadu_ps(vmask, output));
      _mm512_mask_storeu_ps(output, vmask, vout);
    }
  }
}
