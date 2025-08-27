// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-f32acc-rdsum2/avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"

    
void xnn_f16_f32acc_rdsum2_ukernel_7p7x__avx512skx_u16(
    size_t channels,
    size_t k1,
    size_t k2,
    size_t k3,
    const xnn_float16* input,
    size_t input_stride1,
    size_t input_stride2,
    size_t input_stride3,
    const xnn_float16* zero,
    float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params)
{
  assert(k1 != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vscale = _mm512_set1_ps(params->scalar.scale);
  float* original_output = output;
  size_t original_channels = channels;

  for (size_t k = 0; k < k3; ++k) {
    for (size_t j = 0; j < k2; ++j) {
      const xnn_float16* input_row = (const xnn_float16*)((uintptr_t)input + j * input_stride2 + k * input_stride3);
      output = original_output;
      channels = original_channels;

      assert(input_row != NULL);

      size_t input_increment = 7 * input_stride1;
      for (; channels >= 16; channels -= 16) {
        const uint16_t* i0 = (const uint16_t*) input_row;
        const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input_row + 1 * input_stride1);
        const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input_row + 2 * input_stride1);
        const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input_row + 3 * input_stride1);
        const uint16_t* i4 = (const uint16_t*) ((uintptr_t) input_row + 4 * input_stride1);
        const uint16_t* i5 = (const uint16_t*) ((uintptr_t) input_row + 5 * input_stride1);
        const uint16_t* i6 = (const uint16_t*) ((uintptr_t) input_row + 6 * input_stride1);

        __m512 vacc0 = _mm512_setzero_ps();

        for (int r = k1; r > 0; r -= 7) {
          if XNN_UNPREDICTABLE(r < 2) {
            i1 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 2) {
            i2 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 4) {
            i3 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 4) {
            i4 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 6) {
            i5 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 6) {
            i6 = (const uint16_t*) zero;
          }
          __m512 vin0;
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[0])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[0])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[0])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[0])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[0])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[0])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[0])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
          i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
          i2 = (const uint16_t*) ((uintptr_t) i2 + input_increment);
          i3 = (const uint16_t*) ((uintptr_t) i3 + input_increment);
          i4 = (const uint16_t*) ((uintptr_t) i4 + input_increment);
          i5 = (const uint16_t*) ((uintptr_t) i5 + input_increment);
          i6 = (const uint16_t*) ((uintptr_t) i6 + input_increment);
        }
        vacc0 = _mm512_mul_ps(vacc0, vscale);

        __m512 vo0 = _mm512_loadu_ps(output + 0 * 16);
        vacc0 = _mm512_add_ps(vo0, vacc0);
        _mm512_storeu_ps(output, vacc0); output = (void*) ((uintptr_t) output + 16 * sizeof(float));

        input_row = (const xnn_float16*) ((uintptr_t) input_row + 16 * sizeof(uint16_t));
      }
      if (channels != 0) {
        input_increment = 7 * input_stride1;
        const uint16_t* i0 = (const uint16_t*) input_row;
        const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input_row + 1 * input_stride1);
        const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input_row + 2 * input_stride1);
        const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input_row + 3 * input_stride1);
        const uint16_t* i4 = (const uint16_t*) ((uintptr_t) input_row + 4 * input_stride1);
        const uint16_t* i5 = (const uint16_t*) ((uintptr_t) input_row + 5 * input_stride1);
        const uint16_t* i6 = (const uint16_t*) ((uintptr_t) input_row + 6 * input_stride1);
        __m512 vacc[1];
        vacc[0] = _mm512_setzero_ps();

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
        for (int r = k1; r > 0; r -= 7) {
          if XNN_UNPREDICTABLE(r < 2) {
            i1 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 2) {
            i2 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 4) {
            i3 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 4) {
            i4 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 6) {
            i5 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 6) {
            i6 = (const uint16_t*) zero;
          }
          for (int i = 0; i < num_full_chunks; ++i) {
            __m512 vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i0[i*16]));
            __m512 vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i1[i*16]));
            __m512 vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i2[i*16]));
            __m512 vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i3[i*16]));
            __m512 vin4 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i4[i*16]));
            __m512 vin5 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i5[i*16]));
            __m512 vin6 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i6[i*16]));
            vacc[i] = _mm512_fmadd_ps(vin0, vin0, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin1, vin1, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin2, vin2, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin3, vin3, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin4, vin4, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin5, vin5, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin6, vin6, vacc[i]);
          }

          if (remainder) {
            __m512 vin0 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i0[num_full_chunks*16]));
            __m512 vin1 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i1[num_full_chunks*16]));
            __m512 vin2 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i2[num_full_chunks*16]));
            __m512 vin3 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i3[num_full_chunks*16]));
            __m512 vin4 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i4[num_full_chunks*16]));
            __m512 vin5 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i5[num_full_chunks*16]));
            __m512 vin6 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i6[num_full_chunks*16]));
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin0, vin0, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin1, vin1, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin2, vin2, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin3, vin3, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin4, vin4, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin5, vin5, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin6, vin6, vacc[num_full_chunks]);
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

        __m512 vo[1];
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
  }
}
    
void xnn_f16_f32acc_rdsum2_ukernel_7p7x__avx512skx_u32(
    size_t channels,
    size_t k1,
    size_t k2,
    size_t k3,
    const xnn_float16* input,
    size_t input_stride1,
    size_t input_stride2,
    size_t input_stride3,
    const xnn_float16* zero,
    float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params)
{
  assert(k1 != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vscale = _mm512_set1_ps(params->scalar.scale);
  float* original_output = output;
  size_t original_channels = channels;

  for (size_t k = 0; k < k3; ++k) {
    for (size_t j = 0; j < k2; ++j) {
      const xnn_float16* input_row = (const xnn_float16*)((uintptr_t)input + j * input_stride2 + k * input_stride3);
      output = original_output;
      channels = original_channels;

      assert(input_row != NULL);

      size_t input_increment = 7 * input_stride1;
      for (; channels >= 32; channels -= 32) {
        const uint16_t* i0 = (const uint16_t*) input_row;
        const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input_row + 1 * input_stride1);
        const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input_row + 2 * input_stride1);
        const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input_row + 3 * input_stride1);
        const uint16_t* i4 = (const uint16_t*) ((uintptr_t) input_row + 4 * input_stride1);
        const uint16_t* i5 = (const uint16_t*) ((uintptr_t) input_row + 5 * input_stride1);
        const uint16_t* i6 = (const uint16_t*) ((uintptr_t) input_row + 6 * input_stride1);

        __m512 vacc0 = _mm512_setzero_ps();
        __m512 vacc1 = _mm512_setzero_ps();

        for (int r = k1; r > 0; r -= 7) {
          if XNN_UNPREDICTABLE(r < 2) {
            i1 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 2) {
            i2 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 4) {
            i3 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 4) {
            i4 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 6) {
            i5 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 6) {
            i6 = (const uint16_t*) zero;
          }
          __m512 vin0;
          __m512 vin1;
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[16])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[16])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[16])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[16])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[16])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[16])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[16])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
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

        __m512 vo0 = _mm512_loadu_ps(output + 0 * 16);
        __m512 vo1 = _mm512_loadu_ps(output + 1 * 16);
        vacc0 = _mm512_add_ps(vo0, vacc0);
        vacc1 = _mm512_add_ps(vo1, vacc1);
        _mm512_storeu_ps(output, vacc0); output = (void*) ((uintptr_t) output + 16 * sizeof(float));
        _mm512_storeu_ps(output, vacc1); output = (void*) ((uintptr_t) output + 16 * sizeof(float));

        input_row = (const xnn_float16*) ((uintptr_t) input_row + 32 * sizeof(uint16_t));
      }
      if (channels != 0) {
        input_increment = 7 * input_stride1;
        const uint16_t* i0 = (const uint16_t*) input_row;
        const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input_row + 1 * input_stride1);
        const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input_row + 2 * input_stride1);
        const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input_row + 3 * input_stride1);
        const uint16_t* i4 = (const uint16_t*) ((uintptr_t) input_row + 4 * input_stride1);
        const uint16_t* i5 = (const uint16_t*) ((uintptr_t) input_row + 5 * input_stride1);
        const uint16_t* i6 = (const uint16_t*) ((uintptr_t) input_row + 6 * input_stride1);
        __m512 vacc[2];
        vacc[0] = _mm512_setzero_ps();
        vacc[1] = _mm512_setzero_ps();

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
        for (int r = k1; r > 0; r -= 7) {
          if XNN_UNPREDICTABLE(r < 2) {
            i1 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 2) {
            i2 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 4) {
            i3 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 4) {
            i4 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 6) {
            i5 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 6) {
            i6 = (const uint16_t*) zero;
          }
          for (int i = 0; i < num_full_chunks; ++i) {
            __m512 vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i0[i*16]));
            __m512 vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i1[i*16]));
            __m512 vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i2[i*16]));
            __m512 vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i3[i*16]));
            __m512 vin4 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i4[i*16]));
            __m512 vin5 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i5[i*16]));
            __m512 vin6 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i6[i*16]));
            vacc[i] = _mm512_fmadd_ps(vin0, vin0, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin1, vin1, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin2, vin2, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin3, vin3, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin4, vin4, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin5, vin5, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin6, vin6, vacc[i]);
          }

          if (remainder) {
            __m512 vin0 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i0[num_full_chunks*16]));
            __m512 vin1 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i1[num_full_chunks*16]));
            __m512 vin2 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i2[num_full_chunks*16]));
            __m512 vin3 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i3[num_full_chunks*16]));
            __m512 vin4 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i4[num_full_chunks*16]));
            __m512 vin5 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i5[num_full_chunks*16]));
            __m512 vin6 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i6[num_full_chunks*16]));
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin0, vin0, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin1, vin1, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin2, vin2, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin3, vin3, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin4, vin4, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin5, vin5, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin6, vin6, vacc[num_full_chunks]);
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

        __m512 vo[2];
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
  }
}
    
void xnn_f16_f32acc_rdsum2_ukernel_7p7x__avx512skx_u64(
    size_t channels,
    size_t k1,
    size_t k2,
    size_t k3,
    const xnn_float16* input,
    size_t input_stride1,
    size_t input_stride2,
    size_t input_stride3,
    const xnn_float16* zero,
    float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params)
{
  assert(k1 != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vscale = _mm512_set1_ps(params->scalar.scale);
  float* original_output = output;
  size_t original_channels = channels;

  for (size_t k = 0; k < k3; ++k) {
    for (size_t j = 0; j < k2; ++j) {
      const xnn_float16* input_row = (const xnn_float16*)((uintptr_t)input + j * input_stride2 + k * input_stride3);
      output = original_output;
      channels = original_channels;

      assert(input_row != NULL);

      size_t input_increment = 7 * input_stride1;
      for (; channels >= 64; channels -= 64) {
        const uint16_t* i0 = (const uint16_t*) input_row;
        const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input_row + 1 * input_stride1);
        const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input_row + 2 * input_stride1);
        const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input_row + 3 * input_stride1);
        const uint16_t* i4 = (const uint16_t*) ((uintptr_t) input_row + 4 * input_stride1);
        const uint16_t* i5 = (const uint16_t*) ((uintptr_t) input_row + 5 * input_stride1);
        const uint16_t* i6 = (const uint16_t*) ((uintptr_t) input_row + 6 * input_stride1);

        __m512 vacc0 = _mm512_setzero_ps();
        __m512 vacc1 = _mm512_setzero_ps();
        __m512 vacc2 = _mm512_setzero_ps();
        __m512 vacc3 = _mm512_setzero_ps();

        for (int r = k1; r > 0; r -= 7) {
          if XNN_UNPREDICTABLE(r < 2) {
            i1 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 2) {
            i2 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 4) {
            i3 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 4) {
            i4 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 6) {
            i5 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 6) {
            i6 = (const uint16_t*) zero;
          }
          __m512 vin0;
          __m512 vin1;
          __m512 vin2;
          __m512 vin3;
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[16])));
          vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[32])));
          vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[48])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vacc2 = _mm512_fmadd_ps(vin2, vin2, vacc2);
          vacc3 = _mm512_fmadd_ps(vin3, vin3, vacc3);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[16])));
          vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[32])));
          vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[48])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vacc2 = _mm512_fmadd_ps(vin2, vin2, vacc2);
          vacc3 = _mm512_fmadd_ps(vin3, vin3, vacc3);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[16])));
          vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[32])));
          vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[48])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vacc2 = _mm512_fmadd_ps(vin2, vin2, vacc2);
          vacc3 = _mm512_fmadd_ps(vin3, vin3, vacc3);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[16])));
          vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[32])));
          vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[48])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vacc2 = _mm512_fmadd_ps(vin2, vin2, vacc2);
          vacc3 = _mm512_fmadd_ps(vin3, vin3, vacc3);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[16])));
          vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[32])));
          vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[48])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vacc2 = _mm512_fmadd_ps(vin2, vin2, vacc2);
          vacc3 = _mm512_fmadd_ps(vin3, vin3, vacc3);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[16])));
          vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[32])));
          vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[48])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vacc2 = _mm512_fmadd_ps(vin2, vin2, vacc2);
          vacc3 = _mm512_fmadd_ps(vin3, vin3, vacc3);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[16])));
          vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[32])));
          vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[48])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vacc2 = _mm512_fmadd_ps(vin2, vin2, vacc2);
          vacc3 = _mm512_fmadd_ps(vin3, vin3, vacc3);
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

        input_row = (const xnn_float16*) ((uintptr_t) input_row + 64 * sizeof(uint16_t));
      }
      if (channels != 0) {
        input_increment = 7 * input_stride1;
        const uint16_t* i0 = (const uint16_t*) input_row;
        const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input_row + 1 * input_stride1);
        const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input_row + 2 * input_stride1);
        const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input_row + 3 * input_stride1);
        const uint16_t* i4 = (const uint16_t*) ((uintptr_t) input_row + 4 * input_stride1);
        const uint16_t* i5 = (const uint16_t*) ((uintptr_t) input_row + 5 * input_stride1);
        const uint16_t* i6 = (const uint16_t*) ((uintptr_t) input_row + 6 * input_stride1);
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
        for (int r = k1; r > 0; r -= 7) {
          if XNN_UNPREDICTABLE(r < 2) {
            i1 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 2) {
            i2 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 4) {
            i3 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 4) {
            i4 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 6) {
            i5 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 6) {
            i6 = (const uint16_t*) zero;
          }
          for (int i = 0; i < num_full_chunks; ++i) {
            __m512 vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i0[i*16]));
            __m512 vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i1[i*16]));
            __m512 vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i2[i*16]));
            __m512 vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i3[i*16]));
            __m512 vin4 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i4[i*16]));
            __m512 vin5 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i5[i*16]));
            __m512 vin6 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i6[i*16]));
            vacc[i] = _mm512_fmadd_ps(vin0, vin0, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin1, vin1, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin2, vin2, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin3, vin3, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin4, vin4, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin5, vin5, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin6, vin6, vacc[i]);
          }

          if (remainder) {
            __m512 vin0 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i0[num_full_chunks*16]));
            __m512 vin1 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i1[num_full_chunks*16]));
            __m512 vin2 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i2[num_full_chunks*16]));
            __m512 vin3 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i3[num_full_chunks*16]));
            __m512 vin4 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i4[num_full_chunks*16]));
            __m512 vin5 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i5[num_full_chunks*16]));
            __m512 vin6 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i6[num_full_chunks*16]));
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin0, vin0, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin1, vin1, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin2, vin2, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin3, vin3, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin4, vin4, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin5, vin5, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin6, vin6, vacc[num_full_chunks]);
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
  }
}
    
void xnn_f16_f32acc_rdsum2_ukernel_7p7x__avx512skx_u128(
    size_t channels,
    size_t k1,
    size_t k2,
    size_t k3,
    const xnn_float16* input,
    size_t input_stride1,
    size_t input_stride2,
    size_t input_stride3,
    const xnn_float16* zero,
    float* output,
    const struct xnn_f16_f32acc_scale_params* restrict params)
{
  assert(k1 != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vscale = _mm512_set1_ps(params->scalar.scale);
  float* original_output = output;
  size_t original_channels = channels;

  for (size_t k = 0; k < k3; ++k) {
    for (size_t j = 0; j < k2; ++j) {
      const xnn_float16* input_row = (const xnn_float16*)((uintptr_t)input + j * input_stride2 + k * input_stride3);
      output = original_output;
      channels = original_channels;

      assert(input_row != NULL);

      size_t input_increment = 7 * input_stride1;
      for (; channels >= 128; channels -= 128) {
        const uint16_t* i0 = (const uint16_t*) input_row;
        const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input_row + 1 * input_stride1);
        const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input_row + 2 * input_stride1);
        const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input_row + 3 * input_stride1);
        const uint16_t* i4 = (const uint16_t*) ((uintptr_t) input_row + 4 * input_stride1);
        const uint16_t* i5 = (const uint16_t*) ((uintptr_t) input_row + 5 * input_stride1);
        const uint16_t* i6 = (const uint16_t*) ((uintptr_t) input_row + 6 * input_stride1);

        __m512 vacc0 = _mm512_setzero_ps();
        __m512 vacc1 = _mm512_setzero_ps();
        __m512 vacc2 = _mm512_setzero_ps();
        __m512 vacc3 = _mm512_setzero_ps();
        __m512 vacc4 = _mm512_setzero_ps();
        __m512 vacc5 = _mm512_setzero_ps();
        __m512 vacc6 = _mm512_setzero_ps();
        __m512 vacc7 = _mm512_setzero_ps();

        for (int r = k1; r > 0; r -= 7) {
          if XNN_UNPREDICTABLE(r < 2) {
            i1 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 2) {
            i2 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 4) {
            i3 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 4) {
            i4 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 6) {
            i5 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 6) {
            i6 = (const uint16_t*) zero;
          }
          __m512 vin0;
          __m512 vin1;
          __m512 vin2;
          __m512 vin3;
          __m512 vin4;
          __m512 vin5;
          __m512 vin6;
          __m512 vin7;
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[16])));
          vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[32])));
          vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[48])));
          vin4 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[64])));
          vin5 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[80])));
          vin6 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[96])));
          vin7 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i0[112])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vacc2 = _mm512_fmadd_ps(vin2, vin2, vacc2);
          vacc3 = _mm512_fmadd_ps(vin3, vin3, vacc3);
          vacc4 = _mm512_fmadd_ps(vin4, vin4, vacc4);
          vacc5 = _mm512_fmadd_ps(vin5, vin5, vacc5);
          vacc6 = _mm512_fmadd_ps(vin6, vin6, vacc6);
          vacc7 = _mm512_fmadd_ps(vin7, vin7, vacc7);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[16])));
          vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[32])));
          vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[48])));
          vin4 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[64])));
          vin5 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[80])));
          vin6 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[96])));
          vin7 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i1[112])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vacc2 = _mm512_fmadd_ps(vin2, vin2, vacc2);
          vacc3 = _mm512_fmadd_ps(vin3, vin3, vacc3);
          vacc4 = _mm512_fmadd_ps(vin4, vin4, vacc4);
          vacc5 = _mm512_fmadd_ps(vin5, vin5, vacc5);
          vacc6 = _mm512_fmadd_ps(vin6, vin6, vacc6);
          vacc7 = _mm512_fmadd_ps(vin7, vin7, vacc7);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[16])));
          vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[32])));
          vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[48])));
          vin4 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[64])));
          vin5 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[80])));
          vin6 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[96])));
          vin7 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i2[112])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vacc2 = _mm512_fmadd_ps(vin2, vin2, vacc2);
          vacc3 = _mm512_fmadd_ps(vin3, vin3, vacc3);
          vacc4 = _mm512_fmadd_ps(vin4, vin4, vacc4);
          vacc5 = _mm512_fmadd_ps(vin5, vin5, vacc5);
          vacc6 = _mm512_fmadd_ps(vin6, vin6, vacc6);
          vacc7 = _mm512_fmadd_ps(vin7, vin7, vacc7);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[16])));
          vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[32])));
          vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[48])));
          vin4 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[64])));
          vin5 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[80])));
          vin6 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[96])));
          vin7 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i3[112])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vacc2 = _mm512_fmadd_ps(vin2, vin2, vacc2);
          vacc3 = _mm512_fmadd_ps(vin3, vin3, vacc3);
          vacc4 = _mm512_fmadd_ps(vin4, vin4, vacc4);
          vacc5 = _mm512_fmadd_ps(vin5, vin5, vacc5);
          vacc6 = _mm512_fmadd_ps(vin6, vin6, vacc6);
          vacc7 = _mm512_fmadd_ps(vin7, vin7, vacc7);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[16])));
          vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[32])));
          vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[48])));
          vin4 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[64])));
          vin5 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[80])));
          vin6 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[96])));
          vin7 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i4[112])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vacc2 = _mm512_fmadd_ps(vin2, vin2, vacc2);
          vacc3 = _mm512_fmadd_ps(vin3, vin3, vacc3);
          vacc4 = _mm512_fmadd_ps(vin4, vin4, vacc4);
          vacc5 = _mm512_fmadd_ps(vin5, vin5, vacc5);
          vacc6 = _mm512_fmadd_ps(vin6, vin6, vacc6);
          vacc7 = _mm512_fmadd_ps(vin7, vin7, vacc7);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[16])));
          vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[32])));
          vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[48])));
          vin4 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[64])));
          vin5 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[80])));
          vin6 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[96])));
          vin7 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i5[112])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vacc2 = _mm512_fmadd_ps(vin2, vin2, vacc2);
          vacc3 = _mm512_fmadd_ps(vin3, vin3, vacc3);
          vacc4 = _mm512_fmadd_ps(vin4, vin4, vacc4);
          vacc5 = _mm512_fmadd_ps(vin5, vin5, vacc5);
          vacc6 = _mm512_fmadd_ps(vin6, vin6, vacc6);
          vacc7 = _mm512_fmadd_ps(vin7, vin7, vacc7);
          vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[0])));
          vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[16])));
          vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[32])));
          vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[48])));
          vin4 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[64])));
          vin5 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[80])));
          vin6 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[96])));
          vin7 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) (&i6[112])));
          vacc0 = _mm512_fmadd_ps(vin0, vin0, vacc0);
          vacc1 = _mm512_fmadd_ps(vin1, vin1, vacc1);
          vacc2 = _mm512_fmadd_ps(vin2, vin2, vacc2);
          vacc3 = _mm512_fmadd_ps(vin3, vin3, vacc3);
          vacc4 = _mm512_fmadd_ps(vin4, vin4, vacc4);
          vacc5 = _mm512_fmadd_ps(vin5, vin5, vacc5);
          vacc6 = _mm512_fmadd_ps(vin6, vin6, vacc6);
          vacc7 = _mm512_fmadd_ps(vin7, vin7, vacc7);
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
        vacc4 = _mm512_mul_ps(vacc4, vscale);
        vacc5 = _mm512_mul_ps(vacc5, vscale);
        vacc6 = _mm512_mul_ps(vacc6, vscale);
        vacc7 = _mm512_mul_ps(vacc7, vscale);

        __m512 vo0 = _mm512_loadu_ps(output + 0 * 16);
        __m512 vo1 = _mm512_loadu_ps(output + 1 * 16);
        __m512 vo2 = _mm512_loadu_ps(output + 2 * 16);
        __m512 vo3 = _mm512_loadu_ps(output + 3 * 16);
        __m512 vo4 = _mm512_loadu_ps(output + 4 * 16);
        __m512 vo5 = _mm512_loadu_ps(output + 5 * 16);
        __m512 vo6 = _mm512_loadu_ps(output + 6 * 16);
        __m512 vo7 = _mm512_loadu_ps(output + 7 * 16);
        vacc0 = _mm512_add_ps(vo0, vacc0);
        vacc1 = _mm512_add_ps(vo1, vacc1);
        vacc2 = _mm512_add_ps(vo2, vacc2);
        vacc3 = _mm512_add_ps(vo3, vacc3);
        vacc4 = _mm512_add_ps(vo4, vacc4);
        vacc5 = _mm512_add_ps(vo5, vacc5);
        vacc6 = _mm512_add_ps(vo6, vacc6);
        vacc7 = _mm512_add_ps(vo7, vacc7);
        _mm512_storeu_ps(output, vacc0); output = (void*) ((uintptr_t) output + 16 * sizeof(float));
        _mm512_storeu_ps(output, vacc1); output = (void*) ((uintptr_t) output + 16 * sizeof(float));
        _mm512_storeu_ps(output, vacc2); output = (void*) ((uintptr_t) output + 16 * sizeof(float));
        _mm512_storeu_ps(output, vacc3); output = (void*) ((uintptr_t) output + 16 * sizeof(float));
        _mm512_storeu_ps(output, vacc4); output = (void*) ((uintptr_t) output + 16 * sizeof(float));
        _mm512_storeu_ps(output, vacc5); output = (void*) ((uintptr_t) output + 16 * sizeof(float));
        _mm512_storeu_ps(output, vacc6); output = (void*) ((uintptr_t) output + 16 * sizeof(float));
        _mm512_storeu_ps(output, vacc7); output = (void*) ((uintptr_t) output + 16 * sizeof(float));

        input_row = (const xnn_float16*) ((uintptr_t) input_row + 128 * sizeof(uint16_t));
      }
      if (channels != 0) {
        input_increment = 7 * input_stride1;
        const uint16_t* i0 = (const uint16_t*) input_row;
        const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input_row + 1 * input_stride1);
        const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input_row + 2 * input_stride1);
        const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input_row + 3 * input_stride1);
        const uint16_t* i4 = (const uint16_t*) ((uintptr_t) input_row + 4 * input_stride1);
        const uint16_t* i5 = (const uint16_t*) ((uintptr_t) input_row + 5 * input_stride1);
        const uint16_t* i6 = (const uint16_t*) ((uintptr_t) input_row + 6 * input_stride1);
        __m512 vacc[8];
        vacc[0] = _mm512_setzero_ps();
        vacc[1] = _mm512_setzero_ps();
        vacc[2] = _mm512_setzero_ps();
        vacc[3] = _mm512_setzero_ps();
        vacc[4] = _mm512_setzero_ps();
        vacc[5] = _mm512_setzero_ps();
        vacc[6] = _mm512_setzero_ps();
        vacc[7] = _mm512_setzero_ps();

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
        for (int r = k1; r > 0; r -= 7) {
          if XNN_UNPREDICTABLE(r < 2) {
            i1 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 2) {
            i2 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 4) {
            i3 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 4) {
            i4 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r < 6) {
            i5 = (const uint16_t*) zero;
          }
          if XNN_UNPREDICTABLE(r <= 6) {
            i6 = (const uint16_t*) zero;
          }
          for (int i = 0; i < num_full_chunks; ++i) {
            __m512 vin0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i0[i*16]));
            __m512 vin1 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i1[i*16]));
            __m512 vin2 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i2[i*16]));
            __m512 vin3 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i3[i*16]));
            __m512 vin4 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i4[i*16]));
            __m512 vin5 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i5[i*16]));
            __m512 vin6 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*) &i6[i*16]));
            vacc[i] = _mm512_fmadd_ps(vin0, vin0, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin1, vin1, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin2, vin2, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin3, vin3, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin4, vin4, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin5, vin5, vacc[i]);
            vacc[i] = _mm512_fmadd_ps(vin6, vin6, vacc[i]);
          }

          if (remainder) {
            __m512 vin0 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i0[num_full_chunks*16]));
            __m512 vin1 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i1[num_full_chunks*16]));
            __m512 vin2 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i2[num_full_chunks*16]));
            __m512 vin3 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i3[num_full_chunks*16]));
            __m512 vin4 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i4[num_full_chunks*16]));
            __m512 vin5 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i5[num_full_chunks*16]));
            __m512 vin6 = _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(vmask, &i6[num_full_chunks*16]));
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin0, vin0, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin1, vin1, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin2, vin2, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin3, vin3, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin4, vin4, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin5, vin5, vacc[num_full_chunks]);
            vacc[num_full_chunks] = _mm512_maskz_fmadd_ps(vmask, vin6, vin6, vacc[num_full_chunks]);
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

        __m512 vo[8];
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
  }
}
