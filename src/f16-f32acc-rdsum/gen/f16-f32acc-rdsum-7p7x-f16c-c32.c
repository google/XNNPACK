// Auto-generated file. Do not edit!
//   Template: src/f16-f32acc-rdsum/avx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/unaligned.h"
#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include "xnnpack/math.h"


void xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32(
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

  const __m256 vscale = _mm256_set1_ps(params->scale);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 32; channels -= 32) {
    const uint16_t* i0 = input;
    const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input + 1 * input_stride);
    const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input + 2 * input_stride);
    const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input + 3 * input_stride);
    const uint16_t* i4 = (const uint16_t*) ((uintptr_t) input + 4 * input_stride);
    const uint16_t* i5 = (const uint16_t*) ((uintptr_t) input + 5 * input_stride);
    const uint16_t* i6 = (const uint16_t*) ((uintptr_t) input + 6 * input_stride);

    __m256 vacc0 = _mm256_setzero_ps();
    __m256 vacc1 = _mm256_setzero_ps();
    __m256 vacc2 = _mm256_setzero_ps();
    __m256 vacc3 = _mm256_setzero_ps();

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
      __m256 vin0;
      __m256 vin1;
      __m256 vin2;
      __m256 vin3;
      vin0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i0[0])));
      vin1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i0[8])));
      vin2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i0[16])));
      vin3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i0[24])));
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i1[0])));
      vin1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i1[8])));
      vin2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i1[16])));
      vin3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i1[24])));
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i2[0])));
      vin1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i2[8])));
      vin2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i2[16])));
      vin3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i2[24])));
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i3[0])));
      vin1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i3[8])));
      vin2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i3[16])));
      vin3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i3[24])));
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i4[0])));
      vin1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i4[8])));
      vin2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i4[16])));
      vin3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i4[24])));
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i5[0])));
      vin1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i5[8])));
      vin2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i5[16])));
      vin3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i5[24])));
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i6[0])));
      vin1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i6[8])));
      vin2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i6[16])));
      vin3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i6[24])));
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_increment);
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_increment);
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_increment);
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_increment);
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_increment);
    }
    vacc0 = _mm256_mul_ps(vacc0, vscale);
    vacc1 = _mm256_mul_ps(vacc1, vscale);
    vacc2 = _mm256_mul_ps(vacc2, vscale);
    vacc3 = _mm256_mul_ps(vacc3, vscale);

    const float* o = output;
    __m256 vo0 = _mm256_loadu_ps(o); o = (const void*) ((uintptr_t) o + 8 * sizeof(float));
    __m256 vo1 = _mm256_loadu_ps(o); o = (const void*) ((uintptr_t) o + 8 * sizeof(float));
    __m256 vo2 = _mm256_loadu_ps(o); o = (const void*) ((uintptr_t) o + 8 * sizeof(float));
    __m256 vo3 = _mm256_loadu_ps(o); o = (const void*) ((uintptr_t) o + 8 * sizeof(float));
    vacc0 = _mm256_add_ps(vo0, vacc0);
    vacc1 = _mm256_add_ps(vo1, vacc1);
    vacc2 = _mm256_add_ps(vo2, vacc2);
    vacc3 = _mm256_add_ps(vo3, vacc3);
    _mm256_storeu_ps(output, vacc0); output = (void*) ((uintptr_t) output + 8 * sizeof(float));
    _mm256_storeu_ps(output, vacc1); output = (void*) ((uintptr_t) output + 8 * sizeof(float));
    _mm256_storeu_ps(output, vacc2); output = (void*) ((uintptr_t) output + 8 * sizeof(float));
    _mm256_storeu_ps(output, vacc3); output = (void*) ((uintptr_t) output + 8 * sizeof(float));

    input = (const uint16_t*) ((uintptr_t) input + 32 * sizeof(uint16_t));
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
    __m256 vacc[4];
    vacc[0] = _mm256_setzero_ps();
    vacc[1] = _mm256_setzero_ps();
    vacc[2] = _mm256_setzero_ps();
    vacc[3] = _mm256_setzero_ps();

    const size_t num_full_chunks = channels >> 3;
    const size_t num_chunks = round_up_po2(channels, 8) >> 3;
    const size_t remainder = channels & 0x7;
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
        vacc[i] = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) &i0[i*8])), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) &i1[i*8])), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) &i2[i*8])), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) &i3[i*8])), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) &i4[i*8])), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) &i5[i*8])), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) &i6[i*8])), vacc[i]);
      }

      if (remainder) {
        vacc[num_full_chunks] = _mm256_add_ps(vacc[num_full_chunks],  _mm256_cvtph_ps(_mm_loadu_si128((const  __m128i*) &i0[num_full_chunks*8])));
        vacc[num_full_chunks] = _mm256_add_ps(vacc[num_full_chunks],  _mm256_cvtph_ps(_mm_loadu_si128((const  __m128i*) &i1[num_full_chunks*8])));
        vacc[num_full_chunks] = _mm256_add_ps(vacc[num_full_chunks],  _mm256_cvtph_ps(_mm_loadu_si128((const  __m128i*) &i2[num_full_chunks*8])));
        vacc[num_full_chunks] = _mm256_add_ps(vacc[num_full_chunks],  _mm256_cvtph_ps(_mm_loadu_si128((const  __m128i*) &i3[num_full_chunks*8])));
        vacc[num_full_chunks] = _mm256_add_ps(vacc[num_full_chunks],  _mm256_cvtph_ps(_mm_loadu_si128((const  __m128i*) &i4[num_full_chunks*8])));
        vacc[num_full_chunks] = _mm256_add_ps(vacc[num_full_chunks],  _mm256_cvtph_ps(_mm_loadu_si128((const  __m128i*) &i5[num_full_chunks*8])));
        vacc[num_full_chunks] = _mm256_add_ps(vacc[num_full_chunks],  _mm256_cvtph_ps(_mm_loadu_si128((const  __m128i*) &i6[num_full_chunks*8])));
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
      vacc[i] = _mm256_mul_ps(vacc[i], vscale);
    }

    __m256 vo[4];
    const float* o = output;
    for (int i = 0; i < num_full_chunks; ++i) {
      vo[i] = _mm256_loadu_ps(o); o = (const void*) ((uintptr_t) o + 8 * sizeof(float));
    }
    for (int i = 0; i < num_full_chunks; ++i) {
      vacc[i] = _mm256_add_ps(vo[i], vacc[i]);
    }
    for (int i = 0; i < num_full_chunks; ++i) {
      _mm256_storeu_ps(output, vacc[i]); output = (void*) ((uintptr_t) output + 8 * sizeof(float));
    }
    if (remainder) {
      __m256 vout = vacc[num_full_chunks];
      __m128 vout_low = _mm256_castps256_ps128(vout);
      if (channels & 4) {
        __m128 vo =  _mm_loadu_ps(output);
        vo = _mm_add_ps(vout_low, vo);
        _mm_storeu_ps(output, vo);
        vout_low  = _mm256_castps256_ps128(_mm256_permute2f128_ps(vout, vout, 1));
        output = (void*) ((uintptr_t) output + 4 * sizeof(float));
      }
      if (channels & 2) {
        __m128 vo =  _mm_castsi128_ps(_mm_loadl_epi64((__m128i*) output));
        vo = _mm_add_ps(vout_low, vo);
        _mm_storel_pi((__m64*) output, vo);
        vout_low = _mm_movehl_ps(vout_low, vout_low);
        output = (void*) ((uintptr_t) output + 2 * sizeof(float));
      }
      if (channels & 1) {
        __m128 vo = _mm_castsi128_ps(_mm_cvtsi32_si128(unaligned_load_s32(output)));
        vo = _mm_add_ps(vout_low, vo);
        _mm_store_ss(output, vo);
      }
    }
  }
}
