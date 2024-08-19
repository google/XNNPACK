// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/transpose.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/vcvt.h"
#include "xnnpack/vlrelu.h"


void xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_2x4_acc2(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 1);

  const __m128 vmask = _mm_load_ps((const float*) params->sse_stride1.mask);
  const __m128 vmax = _mm_set1_ps(params->sse_stride1.max);
  const __m128 vmin = _mm_set1_ps(params->sse_stride1.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const __m128 vbias = _mm_load1_ps(weights);
  const __m128 vk00 = _mm_load1_ps(weights + 1);
  const __m128 vk01 = _mm_load1_ps(weights + 2);
  const __m128 vk02 = _mm_load1_ps(weights + 3);
  const __m128 vk10 = _mm_load1_ps(weights + 4);
  const __m128 vk11 = _mm_load1_ps(weights + 5);
  const __m128 vk12 = _mm_load1_ps(weights + 6);
  const __m128 vk20 = _mm_load1_ps(weights + 7);
  const __m128 vk21 = _mm_load1_ps(weights + 8);
  const __m128 vk22 = _mm_load1_ps(weights + 9);

  const size_t input_decrement = round_up_po2(input_width, 4 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = input;
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i2 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i3 = zero;
    }

    __m128 vi0x0123 = _mm_setzero_ps();
    __m128 vi1x0123 = _mm_setzero_ps();
    __m128 vi2x0123 = _mm_setzero_ps();
    __m128 vi3x0123 = _mm_setzero_ps();

    __m128 vi0x4567 = _mm_loadu_ps(i0);
    i0 += 4;
    __m128 vi1x4567 = _mm_loadu_ps(i1);
    i1 += 4;
    __m128 vi2x4567 = _mm_loadu_ps(i2);
    i2 += 4;
    __m128 vi3x4567 = _mm_loadu_ps(i3);
    i3 += 4;

    size_t w = input_width;
    for (; w > 4 * sizeof(float); w -= 4 * sizeof(float)) {
      const __m128 vi0x89AB = _mm_loadu_ps(i0);
      i0 += 4;
      const __m128 vi1x89AB = _mm_loadu_ps(i1);
      i1 += 4;
      const __m128 vi2x89AB = _mm_loadu_ps(i2);
      i2 += 4;
      const __m128 vi3x89AB = _mm_loadu_ps(i3);
      i3 += 4;

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk01));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi1x4567, vk01));
      __m128 vo0p1 = _mm_mul_ps(vi1x4567, vk11);
      __m128 vo1p1 = _mm_mul_ps(vi2x4567, vk11);
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x4567, vk21));

      const __m128 vi0x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi0x4567), _mm_castps_si128(vi0x0123), 12));
      const __m128 vi1x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi1x4567), _mm_castps_si128(vi1x0123), 12));
      const __m128 vi2x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi2x4567), _mm_castps_si128(vi2x0123), 12));
      const __m128 vi3x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi3x4567), _mm_castps_si128(vi3x0123), 12));

      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi0x3456, vk00));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi1x3456, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x3456, vk10));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi2x3456, vk20));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi3x3456, vk20));

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;

      const __m128 vi0x5678 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi0x89AB), _mm_castps_si128(vi0x4567), 4));
      const __m128 vi1x5678 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi1x89AB), _mm_castps_si128(vi1x4567), 4));
      const __m128 vi2x5678 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi2x89AB), _mm_castps_si128(vi2x4567), 4));
      const __m128 vi3x5678 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi3x89AB), _mm_castps_si128(vi3x4567), 4));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk02));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x5678, vk02));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi1x5678, vk12));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi2x5678, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x5678, vk22));

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;

      vo0p0 = _mm_add_ps(vo0p0, vo0p1);
      vo1p0 = _mm_add_ps(vo1p0, vo1p1);

      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);

      _mm_storeu_ps(o1, vo1);
      o1 += 4;
      _mm_storeu_ps(o0, vo0);
      o0 += 4;
    }
    // Always process the last block of 1..4 pixels.
    assert(w >= 1 * sizeof(float));
    assert(w <= 4 * sizeof(float));
    {
      vi0x4567 = _mm_and_ps(vmask, vi0x4567);
      vi1x4567 = _mm_and_ps(vmask, vi1x4567);
      vi2x4567 = _mm_and_ps(vmask, vi2x4567);
      vi3x4567 = _mm_and_ps(vmask, vi3x4567);

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk01));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi1x4567, vk01));
      __m128 vo0p1 = _mm_mul_ps(vi1x4567, vk11);
      __m128 vo1p1 = _mm_mul_ps(vi2x4567, vk11);
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x4567, vk21));

      const __m128 vi0x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi0x4567), _mm_castps_si128(vi0x0123), 12));
      const __m128 vi1x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi1x4567), _mm_castps_si128(vi1x0123), 12));
      const __m128 vi2x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi2x4567), _mm_castps_si128(vi2x0123), 12));
      const __m128 vi3x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi3x4567), _mm_castps_si128(vi3x0123), 12));

      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi0x3456, vk00));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi1x3456, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x3456, vk10));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi2x3456, vk20));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi3x3456, vk20));

      const __m128i vzero = _mm_setzero_si128();
      const __m128 vi0x5678 = _mm_castsi128_ps(_mm_alignr_epi8(vzero, _mm_castps_si128(vi0x4567), 4));
      const __m128 vi1x5678 = _mm_castsi128_ps(_mm_alignr_epi8(vzero, _mm_castps_si128(vi1x4567), 4));
      const __m128 vi2x5678 = _mm_castsi128_ps(_mm_alignr_epi8(vzero, _mm_castps_si128(vi2x4567), 4));
      const __m128 vi3x5678 = _mm_castsi128_ps(_mm_alignr_epi8(vzero, _mm_castps_si128(vi3x4567), 4));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk02));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x5678, vk02));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi1x5678, vk12));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi2x5678, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x5678, vk22));

      vo0p0 = _mm_add_ps(vo0p0, vo0p1);
      vo1p0 = _mm_add_ps(vo1p0, vo1p1);

      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);

      if XNN_LIKELY(w == 4 * sizeof(float)) {
        _mm_storeu_ps(o1, vo1);
        o1 += 4;
        _mm_storeu_ps(o0, vo0);
        o0 += 4;
      } else {
        if (w & (2 * sizeof(float))) {
          _mm_storel_pi((__m64*) o1, vo1);
          o1 += 2;
          _mm_storel_pi((__m64*) o0, vo0);
          o0 += 2;

          vo0 = _mm_movehl_ps(vo0, vo0);
          vo1 = _mm_movehl_ps(vo1, vo1);
        }
        if (w & (1 * sizeof(float))) {
          _mm_store_ss(o1, vo1);
          o1 += 1;
          _mm_store_ss(o0, vo0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i2 - input_decrement);
    i1 = (const float*) ((uintptr_t) i3 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);

    o0 = o1;
    o1 = (float*) ((uintptr_t) o0 + input_width);

    output_height = doz(output_height, 2);
  } while (output_height != 0);
}

void xnn_qs16_qs8_vcvt_ukernel__ssse3_u16(
    size_t batch,
    const int16_t* input,
    int8_t* output,
    const union xnn_qs16_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);
  XNN_ALIGN(16) static const uint8_t shuffle01[16] = {
    0x80, 0x80, 0, 1, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 2, 3, 0x80, 0x80, 0x80, 0x80,
  };
  XNN_ALIGN(16) static const uint8_t shuffle23[16] = {
    0x80, 0x80, 4, 5, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 6, 7, 0x80, 0x80, 0x80, 0x80,
  };
  XNN_ALIGN(16) static const uint8_t shuffle45[16] = {
    0x80, 0x80, 8, 9, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 10, 11, 0x80, 0x80, 0x80, 0x80,
  };
  XNN_ALIGN(16) static const uint8_t shuffle67[16] = {
    0x80, 0x80, 12, 13, 0x80, 0x80, 0x80, 0x80,
    0x80, 0x80, 14, 15, 0x80, 0x80, 0x80, 0x80,
  };

  const __m128i vinput_bias = _mm_set1_epi16(UINT16_C(0x8000));
  const __m128i vmultiplier = _mm_set1_epi32(params->scalar.multiplier);
  const __m128i vbias = _mm_set1_epi64x(
      (int64_t) ((uint64_t) params->scalar.output_zero_point << 32) + 
      INT64_C(0x80000000) -
      (INT64_C(0x80000000) * (int64_t) params->scalar.multiplier));
  XNN_FORCE_REALIZATION(vinput_bias);
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(vbias);
  const __m128i vshuffle01 = _mm_load_si128((const __m128i*) shuffle01);
  const __m128i vshuffle23 = _mm_load_si128((const __m128i*) shuffle23);

  const __m128i vshuffle45 = _mm_load_si128((const __m128i*) shuffle45);
  const __m128i vshuffle67 = _mm_load_si128((const __m128i*) shuffle67);
  for (; batch >= 16 * sizeof(int16_t); batch -= 16 * sizeof(int16_t)) {
    __m128i vx0 = _mm_loadu_si128((const __m128i*) input); input += 8;
    __m128i vx2 = _mm_loadu_si128((const __m128i*) input); input += 8;

    // Add 0x8000 to convert signed inputs to unsigned.
    vx0 = _mm_xor_si128(vx0, vinput_bias);
    vx2 = _mm_xor_si128(vx2, vinput_bias);

    // Move int16 to upper part of int32
    __m128i vacc0lo   = _mm_shuffle_epi8(vx0, vshuffle01);
    __m128i vacc0hi   = _mm_shuffle_epi8(vx0, vshuffle23);
    __m128i vacc1lo = _mm_shuffle_epi8(vx0, vshuffle45);
    __m128i vacc1hi = _mm_shuffle_epi8(vx0, vshuffle67);
    __m128i vacc2lo   = _mm_shuffle_epi8(vx2, vshuffle01);
    __m128i vacc2hi   = _mm_shuffle_epi8(vx2, vshuffle23);
    __m128i vacc3lo = _mm_shuffle_epi8(vx2, vshuffle45);
    __m128i vacc3hi = _mm_shuffle_epi8(vx2, vshuffle67);

    vacc0lo = _mm_mul_epu32(vacc0lo, vmultiplier);
    vacc0hi = _mm_mul_epu32(vacc0hi, vmultiplier);
    vacc1lo = _mm_mul_epu32(vacc1lo, vmultiplier);
    vacc1hi = _mm_mul_epu32(vacc1hi, vmultiplier);
    vacc2lo = _mm_mul_epu32(vacc2lo, vmultiplier);
    vacc2hi = _mm_mul_epu32(vacc2hi, vmultiplier);
    vacc3lo = _mm_mul_epu32(vacc3lo, vmultiplier);
    vacc3hi = _mm_mul_epu32(vacc3hi, vmultiplier);

    vacc0lo = _mm_add_epi64(vacc0lo, vbias);
    vacc0hi = _mm_add_epi64(vacc0hi, vbias);
    vacc1lo = _mm_add_epi64(vacc1lo, vbias);
    vacc1hi = _mm_add_epi64(vacc1hi, vbias);
    vacc2lo = _mm_add_epi64(vacc2lo, vbias);
    vacc2hi = _mm_add_epi64(vacc2hi, vbias);
    vacc3lo = _mm_add_epi64(vacc3lo, vbias);
    vacc3hi = _mm_add_epi64(vacc3hi, vbias);

    __m128i vacc0 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacc0lo), _mm_castsi128_ps(vacc0hi), _MM_SHUFFLE(3, 1, 3, 1)));
    __m128i vacc1 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacc1lo), _mm_castsi128_ps(vacc1hi), _MM_SHUFFLE(3, 1, 3, 1)));
    __m128i vacc2 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacc2lo), _mm_castsi128_ps(vacc2hi), _MM_SHUFFLE(3, 1, 3, 1)));
    __m128i vacc3 = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacc3lo), _mm_castsi128_ps(vacc3hi), _MM_SHUFFLE(3, 1, 3, 1)));

    // Pack 8 ints into 8 shorts
    vacc0 = _mm_packs_epi32(vacc0, vacc1);
    vacc2 = _mm_packs_epi32(vacc2, vacc3);

    // Pack 16 shorts into 16 bytes
    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc2);

    _mm_storeu_si128((__m128i*) output, vy0); output += 16;
  }

  for (; batch >= 4 * sizeof(int16_t); batch -= 4 * sizeof(int16_t)) {
    __m128i vx = _mm_loadu_si128((const __m128i*) input); input += 4;
    vx = _mm_xor_si128(vx, vinput_bias);
    __m128i vacclo = _mm_shuffle_epi8(vx, vshuffle01);
    __m128i vacchi = _mm_shuffle_epi8(vx, vshuffle23);
    vacclo = _mm_mul_epu32(vacclo, vmultiplier);
    vacchi = _mm_mul_epu32(vacchi, vmultiplier);
    vacclo = _mm_add_epi64(vacclo, vbias);
    vacchi = _mm_add_epi64(vacchi, vbias);
    __m128i vacc = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacclo), _mm_castsi128_ps(vacchi), _MM_SHUFFLE(3, 1, 3, 1)));
    vacc = _mm_packs_epi32(vacc, vacc);
    const __m128i vy = _mm_packs_epi16(vacc, vacc);

    _mm_storeu_si32(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int16_t));
    assert(batch <= 3 * sizeof(int16_t));

    __m128i vx = _mm_loadu_si128((const __m128i*) input);
    vx = _mm_xor_si128(vx, vinput_bias);
    __m128i vacclo = _mm_shuffle_epi8(vx, vshuffle01);
    __m128i vacchi = _mm_shuffle_epi8(vx, vshuffle23);
    vacclo = _mm_mul_epu32(vacclo, vmultiplier);
    vacchi = _mm_mul_epu32(vacchi, vmultiplier);
    vacclo = _mm_add_epi64(vacclo, vbias);
    vacchi = _mm_add_epi64(vacchi, vbias);
    __m128i vacc = _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(vacclo), _mm_castsi128_ps(vacchi), _MM_SHUFFLE(3, 1, 3, 1)));
    vacc = _mm_packs_epi32(vacc, vacc);
    const __m128i vy = _mm_packs_epi16(vacc, vacc);

    uint32_t vy_lo = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(int16_t))) {
      unaligned_store_u16(output, (uint16_t) vy_lo);
      vy_lo >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(int16_t))) {
      *output = (int8_t) vy_lo;
    }
  }
}

void xnn_qs8_vcvt_ukernel__ssse3_u32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_zero_point = _mm_set1_epi16(params->scalar.input_zero_point);
  const __m128i vmultiplier = _mm_set1_epi16(-params->scalar.multiplier);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) input);
    const __m128i vx1 = _mm_loadu_si128((const __m128i*) (input + 16));
    input += 32;

    const __m128i vm0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vx0);
    __m128i vacc0 = _mm_unpacklo_epi8(vx0, vm0);
    __m128i vacc1 = _mm_unpackhi_epi8(vx0, vm0);
    const __m128i vm1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vx1);
    __m128i vacc2 = _mm_unpacklo_epi8(vx1, vm1);
    __m128i vacc3 = _mm_unpackhi_epi8(vx1, vm1);

    vacc0 = _mm_sub_epi16(vinput_zero_point, vacc0);
    vacc1 = _mm_sub_epi16(vinput_zero_point, vacc1);
    vacc2 = _mm_sub_epi16(vinput_zero_point, vacc2);
    vacc3 = _mm_sub_epi16(vinput_zero_point, vacc3);

    vacc0 = _mm_slli_epi16(vacc0, 7);
    vacc1 = _mm_slli_epi16(vacc1, 7);
    vacc2 = _mm_slli_epi16(vacc2, 7);
    vacc3 = _mm_slli_epi16(vacc3, 7);

    vacc0 = _mm_mulhrs_epi16(vacc0, vmultiplier);
    vacc1 = _mm_mulhrs_epi16(vacc1, vmultiplier);
    vacc2 = _mm_mulhrs_epi16(vacc2, vmultiplier);
    vacc3 = _mm_mulhrs_epi16(vacc3, vmultiplier);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);
    vacc2 = _mm_adds_epi16(vacc2, voutput_zero_point);
    vacc3 = _mm_adds_epi16(vacc3, voutput_zero_point);

    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packs_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    const __m128i vm = _mm_cmpgt_epi8(_mm_setzero_si128(), vx);
    __m128i vacc_lo = _mm_unpacklo_epi8(vx, vm);
    __m128i vacc_hi = _mm_unpackhi_epi8(vx, vm);
    vacc_lo = _mm_sub_epi16(vinput_zero_point, vacc_lo);
    vacc_hi = _mm_sub_epi16(vinput_zero_point, vacc_hi);
    vacc_lo = _mm_slli_epi16(vacc_lo, 7);
    vacc_hi = _mm_slli_epi16(vacc_hi, 7);
    vacc_lo = _mm_mulhrs_epi16(vacc_lo, vmultiplier);
    vacc_hi = _mm_mulhrs_epi16(vacc_hi, vmultiplier);
    vacc_lo = _mm_adds_epi16(vacc_lo, voutput_zero_point);
    vacc_hi = _mm_adds_epi16(vacc_hi, voutput_zero_point);

    const __m128i vy = _mm_packs_epi16(vacc_lo, vacc_hi);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 15 * sizeof(int8_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);

    const __m128i vm = _mm_cmpgt_epi8(_mm_setzero_si128(), vx);
    __m128i vacc_lo = _mm_unpacklo_epi8(vx, vm);
    __m128i vacc_hi = _mm_unpackhi_epi8(vx, vm);
    vacc_lo = _mm_sub_epi16(vinput_zero_point, vacc_lo);
    vacc_hi = _mm_sub_epi16(vinput_zero_point, vacc_hi);
    vacc_lo = _mm_slli_epi16(vacc_lo, 7);
    vacc_hi = _mm_slli_epi16(vacc_hi, 7);
    vacc_lo = _mm_mulhrs_epi16(vacc_lo, vmultiplier);
    vacc_hi = _mm_mulhrs_epi16(vacc_hi, vmultiplier);
    vacc_lo = _mm_adds_epi16(vacc_lo, voutput_zero_point);
    vacc_hi = _mm_adds_epi16(vacc_hi, voutput_zero_point);

    __m128i vy = _mm_packs_epi16(vacc_lo, vacc_hi);
    if (batch & (8 * sizeof(int8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(int8_t))) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    uint32_t vy_lo = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(int8_t))) {
      unaligned_store_u16(output, (uint16_t) vy_lo);
      vy_lo >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      *output = (int8_t) vy_lo;
    }
  }
}

void xnn_qs8_vlrelu_ukernel__ssse3_u32(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_zero_point = _mm_set1_epi16(params->scalar.input_zero_point);
  const __m128i vmultiplier_diff = _mm_set1_epi16(-params->scalar.negative_multiplier ^ -params->scalar.positive_multiplier);
  const __m128i vmultiplier_base = _mm_set1_epi16(-params->scalar.negative_multiplier);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier_diff);
  XNN_FORCE_REALIZATION(vmultiplier_base);
  XNN_FORCE_REALIZATION(voutput_zero_point);

  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) input);
    const __m128i vx1 = _mm_loadu_si128((const __m128i*) (input + 16));
    input += 32;

    const __m128i vm0 = _mm_cmpgt_epi8(_mm_setzero_si128(), vx0);
    __m128i vacc0 = _mm_unpacklo_epi8(vx0, vm0);
    __m128i vacc1 = _mm_unpackhi_epi8(vx0, vm0);
    const __m128i vm1 = _mm_cmpgt_epi8(_mm_setzero_si128(), vx1);
    __m128i vacc2 = _mm_unpacklo_epi8(vx1, vm1);
    __m128i vacc3 = _mm_unpackhi_epi8(vx1, vm1);

    __m128i vmultiplier0 = _mm_cmpgt_epi16(vacc0, vinput_zero_point);
    vacc0 = _mm_sub_epi16(vinput_zero_point, vacc0);
    __m128i vmultiplier1 = _mm_cmpgt_epi16(vacc1, vinput_zero_point);
    vacc1 = _mm_sub_epi16(vinput_zero_point, vacc1);
    __m128i vmultiplier2 = _mm_cmpgt_epi16(vacc2, vinput_zero_point);
    vacc2 = _mm_sub_epi16(vinput_zero_point, vacc2);
    __m128i vmultiplier3 = _mm_cmpgt_epi16(vacc3, vinput_zero_point);
    vacc3 = _mm_sub_epi16(vinput_zero_point, vacc3);

    vmultiplier0 = _mm_and_si128(vmultiplier0, vmultiplier_diff);
    vacc0 = _mm_slli_epi16(vacc0, 7);
    vmultiplier0 = _mm_xor_si128(vmultiplier0, vmultiplier_base);
    vmultiplier1 = _mm_and_si128(vmultiplier1, vmultiplier_diff);
    vacc1 = _mm_slli_epi16(vacc1, 7);
    vmultiplier1 = _mm_xor_si128(vmultiplier1, vmultiplier_base);
    vmultiplier2 = _mm_and_si128(vmultiplier2, vmultiplier_diff);
    vacc2 = _mm_slli_epi16(vacc2, 7);
    vmultiplier2 = _mm_xor_si128(vmultiplier2, vmultiplier_base);
    vmultiplier3 = _mm_and_si128(vmultiplier3, vmultiplier_diff);
    vacc3 = _mm_slli_epi16(vacc3, 7);
    vmultiplier3 = _mm_xor_si128(vmultiplier3, vmultiplier_base);

    vacc0 = _mm_mulhrs_epi16(vacc0, vmultiplier0);
    vacc1 = _mm_mulhrs_epi16(vacc1, vmultiplier1);
    vacc2 = _mm_mulhrs_epi16(vacc2, vmultiplier2);
    vacc3 = _mm_mulhrs_epi16(vacc3, vmultiplier3);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);
    vacc2 = _mm_adds_epi16(vacc2, voutput_zero_point);
    vacc3 = _mm_adds_epi16(vacc3, voutput_zero_point);

    const __m128i vy0 = _mm_packs_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packs_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    const __m128i vm = _mm_cmpgt_epi8(_mm_setzero_si128(), vx);
    __m128i vacc_lo = _mm_unpacklo_epi8(vx, vm);
    __m128i vacc_hi = _mm_unpackhi_epi8(vx, vm);
    __m128i vmultiplier_lo = _mm_cmpgt_epi16(vacc_lo, vinput_zero_point);
    __m128i vmultiplier_hi = _mm_cmpgt_epi16(vacc_hi, vinput_zero_point);
    vacc_lo = _mm_sub_epi16(vinput_zero_point, vacc_lo);
    vacc_hi = _mm_sub_epi16(vinput_zero_point, vacc_hi);
    vmultiplier_lo = _mm_and_si128(vmultiplier_lo, vmultiplier_diff);
    vmultiplier_hi = _mm_and_si128(vmultiplier_hi, vmultiplier_diff);
    vacc_lo = _mm_slli_epi16(vacc_lo, 7);
    vacc_hi = _mm_slli_epi16(vacc_hi, 7);
    vmultiplier_lo = _mm_xor_si128(vmultiplier_lo, vmultiplier_base);
    vmultiplier_hi = _mm_xor_si128(vmultiplier_hi, vmultiplier_base);
    vacc_lo = _mm_mulhrs_epi16(vacc_lo, vmultiplier_lo);
    vacc_hi = _mm_mulhrs_epi16(vacc_hi, vmultiplier_hi);
    vacc_lo = _mm_adds_epi16(vacc_lo, voutput_zero_point);
    vacc_hi = _mm_adds_epi16(vacc_hi, voutput_zero_point);

    const __m128i vy = _mm_packs_epi16(vacc_lo, vacc_hi);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 15 * sizeof(int8_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);

    const __m128i vm = _mm_cmpgt_epi8(_mm_setzero_si128(), vx);
    __m128i vacc_lo = _mm_unpacklo_epi8(vx, vm);
    __m128i vacc_hi = _mm_unpackhi_epi8(vx, vm);
    __m128i vmultiplier_lo = _mm_cmpgt_epi16(vacc_lo, vinput_zero_point);
    __m128i vmultiplier_hi = _mm_cmpgt_epi16(vacc_hi, vinput_zero_point);
    vacc_lo = _mm_sub_epi16(vinput_zero_point, vacc_lo);
    vacc_hi = _mm_sub_epi16(vinput_zero_point, vacc_hi);
    vmultiplier_lo = _mm_and_si128(vmultiplier_lo, vmultiplier_diff);
    vmultiplier_hi = _mm_and_si128(vmultiplier_hi, vmultiplier_diff);
    vacc_lo = _mm_slli_epi16(vacc_lo, 7);
    vacc_hi = _mm_slli_epi16(vacc_hi, 7);
    vmultiplier_lo = _mm_xor_si128(vmultiplier_lo, vmultiplier_base);
    vmultiplier_hi = _mm_xor_si128(vmultiplier_hi, vmultiplier_base);
    vacc_lo = _mm_mulhrs_epi16(vacc_lo, vmultiplier_lo);
    vacc_hi = _mm_mulhrs_epi16(vacc_hi, vmultiplier_hi);
    vacc_lo = _mm_adds_epi16(vacc_lo, voutput_zero_point);
    vacc_hi = _mm_adds_epi16(vacc_hi, voutput_zero_point);

    __m128i vy = _mm_packs_epi16(vacc_lo, vacc_hi);
    if (batch & (8 * sizeof(int8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(int8_t))) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    uint32_t vy_lo = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(int8_t))) {
      unaligned_store_u16(output, (uint16_t) vy_lo);
      vy_lo >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(int8_t))) {
      *output = (int8_t) vy_lo;
    }
  }
}

void xnn_qu8_vcvt_ukernel__ssse3_u32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_zero_point = _mm_set1_epi16(params->scalar.input_zero_point);
  const __m128i vmultiplier = _mm_set1_epi16(-params->scalar.multiplier);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  const __m128i vzero = _mm_setzero_si128();
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) input);
    const __m128i vx1 = _mm_loadu_si128((const __m128i*) (input + 16));
    input += 32;

    __m128i vacc0 = _mm_unpacklo_epi8(vx0, vzero);
    __m128i vacc1 = _mm_unpackhi_epi8(vx0, vzero);
    __m128i vacc2 = _mm_unpacklo_epi8(vx1, vzero);
    __m128i vacc3 = _mm_unpackhi_epi8(vx1, vzero);

    vacc0 = _mm_sub_epi16(vinput_zero_point, vacc0);
    vacc1 = _mm_sub_epi16(vinput_zero_point, vacc1);
    vacc2 = _mm_sub_epi16(vinput_zero_point, vacc2);
    vacc3 = _mm_sub_epi16(vinput_zero_point, vacc3);

    vacc0 = _mm_slli_epi16(vacc0, 7);
    vacc1 = _mm_slli_epi16(vacc1, 7);
    vacc2 = _mm_slli_epi16(vacc2, 7);
    vacc3 = _mm_slli_epi16(vacc3, 7);

    vacc0 = _mm_mulhrs_epi16(vacc0, vmultiplier);
    vacc1 = _mm_mulhrs_epi16(vacc1, vmultiplier);
    vacc2 = _mm_mulhrs_epi16(vacc2, vmultiplier);
    vacc3 = _mm_mulhrs_epi16(vacc3, vmultiplier);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);
    vacc2 = _mm_adds_epi16(vacc2, voutput_zero_point);
    vacc3 = _mm_adds_epi16(vacc3, voutput_zero_point);

    const __m128i vy0 = _mm_packus_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packus_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    __m128i vacc_lo = _mm_unpacklo_epi8(vx, vzero);
    __m128i vacc_hi = _mm_unpackhi_epi8(vx, vzero);
    vacc_lo = _mm_sub_epi16(vinput_zero_point, vacc_lo);
    vacc_hi = _mm_sub_epi16(vinput_zero_point, vacc_hi);
    vacc_lo = _mm_slli_epi16(vacc_lo, 7);
    vacc_hi = _mm_slli_epi16(vacc_hi, 7);
    vacc_lo = _mm_mulhrs_epi16(vacc_lo, vmultiplier);
    vacc_hi = _mm_mulhrs_epi16(vacc_hi, vmultiplier);
    vacc_lo = _mm_adds_epi16(vacc_lo, voutput_zero_point);
    vacc_hi = _mm_adds_epi16(vacc_hi, voutput_zero_point);

    const __m128i vy = _mm_packus_epi16(vacc_lo, vacc_hi);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 15 * sizeof(uint8_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);

    __m128i vacc_lo = _mm_unpacklo_epi8(vx, vzero);
    __m128i vacc_hi = _mm_unpackhi_epi8(vx, vzero);
    vacc_lo = _mm_sub_epi16(vinput_zero_point, vacc_lo);
    vacc_hi = _mm_sub_epi16(vinput_zero_point, vacc_hi);
    vacc_lo = _mm_slli_epi16(vacc_lo, 7);
    vacc_hi = _mm_slli_epi16(vacc_hi, 7);
    vacc_lo = _mm_mulhrs_epi16(vacc_lo, vmultiplier);
    vacc_hi = _mm_mulhrs_epi16(vacc_hi, vmultiplier);
    vacc_lo = _mm_adds_epi16(vacc_lo, voutput_zero_point);
    vacc_hi = _mm_adds_epi16(vacc_hi, voutput_zero_point);

    __m128i vy = _mm_packus_epi16(vacc_lo, vacc_hi);
    if (batch & (8 * sizeof(uint8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(uint8_t))) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    uint32_t vy_lo = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(uint8_t))) {
      unaligned_store_u16(output, (uint16_t) vy_lo);
      vy_lo >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) vy_lo;
    }
  }
}

void xnn_qu8_vlrelu_ukernel__ssse3_u32(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vinput_zero_point = _mm_set1_epi16(params->scalar.input_zero_point);
  const __m128i vmultiplier_diff = _mm_set1_epi16(-params->scalar.negative_multiplier ^ -params->scalar.positive_multiplier);
  const __m128i vmultiplier_base = _mm_set1_epi16(-params->scalar.negative_multiplier);
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  XNN_FORCE_REALIZATION(vinput_zero_point);
  XNN_FORCE_REALIZATION(vmultiplier_diff);
  XNN_FORCE_REALIZATION(vmultiplier_base);
  XNN_FORCE_REALIZATION(voutput_zero_point);

  const __m128i vzero = _mm_setzero_si128();
  for (; batch >= 32 * sizeof(uint8_t); batch -= 32 * sizeof(uint8_t)) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) input);
    const __m128i vx1 = _mm_loadu_si128((const __m128i*) (input + 16));
    input += 32;

    __m128i vacc0 = _mm_unpacklo_epi8(vx0, vzero);
    __m128i vacc1 = _mm_unpackhi_epi8(vx0, vzero);
    __m128i vacc2 = _mm_unpacklo_epi8(vx1, vzero);
    __m128i vacc3 = _mm_unpackhi_epi8(vx1, vzero);

    __m128i vmultiplier0 = _mm_cmpgt_epi16(vacc0, vinput_zero_point);
    vacc0 = _mm_sub_epi16(vinput_zero_point, vacc0);
    __m128i vmultiplier1 = _mm_cmpgt_epi16(vacc1, vinput_zero_point);
    vacc1 = _mm_sub_epi16(vinput_zero_point, vacc1);
    __m128i vmultiplier2 = _mm_cmpgt_epi16(vacc2, vinput_zero_point);
    vacc2 = _mm_sub_epi16(vinput_zero_point, vacc2);
    __m128i vmultiplier3 = _mm_cmpgt_epi16(vacc3, vinput_zero_point);
    vacc3 = _mm_sub_epi16(vinput_zero_point, vacc3);

    vmultiplier0 = _mm_and_si128(vmultiplier0, vmultiplier_diff);
    vacc0 = _mm_slli_epi16(vacc0, 7);
    vmultiplier0 = _mm_xor_si128(vmultiplier0, vmultiplier_base);
    vmultiplier1 = _mm_and_si128(vmultiplier1, vmultiplier_diff);
    vacc1 = _mm_slli_epi16(vacc1, 7);
    vmultiplier1 = _mm_xor_si128(vmultiplier1, vmultiplier_base);
    vmultiplier2 = _mm_and_si128(vmultiplier2, vmultiplier_diff);
    vacc2 = _mm_slli_epi16(vacc2, 7);
    vmultiplier2 = _mm_xor_si128(vmultiplier2, vmultiplier_base);
    vmultiplier3 = _mm_and_si128(vmultiplier3, vmultiplier_diff);
    vacc3 = _mm_slli_epi16(vacc3, 7);
    vmultiplier3 = _mm_xor_si128(vmultiplier3, vmultiplier_base);

    vacc0 = _mm_mulhrs_epi16(vacc0, vmultiplier0);
    vacc1 = _mm_mulhrs_epi16(vacc1, vmultiplier1);
    vacc2 = _mm_mulhrs_epi16(vacc2, vmultiplier2);
    vacc3 = _mm_mulhrs_epi16(vacc3, vmultiplier3);

    vacc0 = _mm_adds_epi16(vacc0, voutput_zero_point);
    vacc1 = _mm_adds_epi16(vacc1, voutput_zero_point);
    vacc2 = _mm_adds_epi16(vacc2, voutput_zero_point);
    vacc3 = _mm_adds_epi16(vacc3, voutput_zero_point);

    const __m128i vy0 = _mm_packus_epi16(vacc0, vacc1);
    const __m128i vy1 = _mm_packus_epi16(vacc2, vacc3);

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    output += 32;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    __m128i vacc_lo = _mm_unpacklo_epi8(vx, vzero);
    __m128i vacc_hi = _mm_unpackhi_epi8(vx, vzero);
    __m128i vmultiplier_lo = _mm_cmpgt_epi16(vacc_lo, vinput_zero_point);
    __m128i vmultiplier_hi = _mm_cmpgt_epi16(vacc_hi, vinput_zero_point);
    vacc_lo = _mm_sub_epi16(vinput_zero_point, vacc_lo);
    vacc_hi = _mm_sub_epi16(vinput_zero_point, vacc_hi);
    vmultiplier_lo = _mm_and_si128(vmultiplier_lo, vmultiplier_diff);
    vmultiplier_hi = _mm_and_si128(vmultiplier_hi, vmultiplier_diff);
    vacc_lo = _mm_slli_epi16(vacc_lo, 7);
    vacc_hi = _mm_slli_epi16(vacc_hi, 7);
    vmultiplier_lo = _mm_xor_si128(vmultiplier_lo, vmultiplier_base);
    vmultiplier_hi = _mm_xor_si128(vmultiplier_hi, vmultiplier_base);
    vacc_lo = _mm_mulhrs_epi16(vacc_lo, vmultiplier_lo);
    vacc_hi = _mm_mulhrs_epi16(vacc_hi, vmultiplier_hi);
    vacc_lo = _mm_adds_epi16(vacc_lo, voutput_zero_point);
    vacc_hi = _mm_adds_epi16(vacc_hi, voutput_zero_point);

    const __m128i vy = _mm_packus_epi16(vacc_lo, vacc_hi);
    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint8_t));
    assert(batch <= 15 * sizeof(uint8_t));

    const __m128i vx = _mm_loadu_si128((const __m128i*) input);

    __m128i vacc_lo = _mm_unpacklo_epi8(vx, vzero);
    __m128i vacc_hi = _mm_unpackhi_epi8(vx, vzero);
    __m128i vmultiplier_lo = _mm_cmpgt_epi16(vacc_lo, vinput_zero_point);
    __m128i vmultiplier_hi = _mm_cmpgt_epi16(vacc_hi, vinput_zero_point);
    vacc_lo = _mm_sub_epi16(vinput_zero_point, vacc_lo);
    vacc_hi = _mm_sub_epi16(vinput_zero_point, vacc_hi);
    vmultiplier_lo = _mm_and_si128(vmultiplier_lo, vmultiplier_diff);
    vmultiplier_hi = _mm_and_si128(vmultiplier_hi, vmultiplier_diff);
    vacc_lo = _mm_slli_epi16(vacc_lo, 7);
    vacc_hi = _mm_slli_epi16(vacc_hi, 7);
    vmultiplier_lo = _mm_xor_si128(vmultiplier_lo, vmultiplier_base);
    vmultiplier_hi = _mm_xor_si128(vmultiplier_hi, vmultiplier_base);
    vacc_lo = _mm_mulhrs_epi16(vacc_lo, vmultiplier_lo);
    vacc_hi = _mm_mulhrs_epi16(vacc_hi, vmultiplier_hi);
    vacc_lo = _mm_adds_epi16(vacc_lo, voutput_zero_point);
    vacc_hi = _mm_adds_epi16(vacc_hi, voutput_zero_point);

    __m128i vy = _mm_packus_epi16(vacc_lo, vacc_hi);
    if (batch & (8 * sizeof(uint8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(uint8_t))) {
      unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vy));
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    uint32_t vy_lo = (uint32_t) _mm_cvtsi128_si32(vy);
    if (batch & (2 * sizeof(uint8_t))) {
      unaligned_store_u16(output, (uint16_t) vy_lo);
      vy_lo >>= 16;
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) vy_lo;
    }
  }
}

void xnn_x24_transposec_ukernel__4x4_ssse3(
    const void *input,
    void * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{ 
  static const uint8_t pos0[16] = {0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11, -1, -1, -1, -1};
  static const uint8_t pos1[16] = {4, 8, 12, 6, 10, 14, 5, 9, 13, 7, 11, 15, -1, -1, -1, -1};
  static const uint8_t pos2[16] = {12, -1, -1, 14, -1, -1, 13, -1, -1, 15, -1, -1, -1, -1, -1, -1};
  static const uint8_t pos3[16] = {-1, 0, 4, -1, 2, 6, -1, 1, 5, -1, 3, 7, -1, -1, -1, -1};
  static const uint8_t pos4[16] = {8, 12, -1, 10, 14, -1, 9, 13, -1, 11, 15, -1, -1, -1, -1, -1};
  static const uint8_t pos5[16] = {-1, -1, 0, -1, -1, 2, -1, -1, 1, -1, -1, 3, -1, -1, -1, -1};

  assert(output_stride >= block_height * 3);
  assert(input_stride >= block_width * 3);

  assert(output_stride >= block_height * 3);
  assert(input_stride >= block_width * 3);

  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_wbytes = tile_width * 3;
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - block_height * 3;
  const size_t tile_stride = tile_height * input_stride;

  const uint8_t* i0 = (const uint8_t*) input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);

  uint8_t* o0 = (uint8_t*) output;
  uint8_t* o1 = (uint8_t*) ((uintptr_t) o0 + output_stride);
  uint8_t* o2 = (uint8_t*) ((uintptr_t) o1 + output_stride);
  uint8_t* o3 = (uint8_t*) ((uintptr_t) o2 + output_stride);

  const __m128i vperm0 = _mm_load_si128((const __m128i*) pos0);
  const __m128i vperm1 = _mm_load_si128((const __m128i*) pos1);
  const __m128i vperm2 = _mm_load_si128((const __m128i*) pos2);
  const __m128i vperm3 = _mm_load_si128((const __m128i*) pos3);
  const __m128i vperm4 = _mm_load_si128((const __m128i*) pos4);
  const __m128i vperm5 = _mm_load_si128((const __m128i*) pos5);
  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 2) {
      o2 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 4) {
      o3 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 4; bh -= 4) {
      const __m128i v0 = _mm_loadu_si128((const __m128i*) i0);
      const __m128i v1 = _mm_loadu_si128((const __m128i*) i1);
      const __m128i v2 = _mm_loadu_si128((const __m128i*) i2);
      const __m128i v3 = _mm_loadu_si128((const __m128i*) i3);
      i0 = (const uint8_t*) ((uintptr_t) i0 + tile_stride);
      i1 = (const uint8_t*) ((uintptr_t) i1 + tile_stride);
      i2 = (const uint8_t*) ((uintptr_t) i2 + tile_stride);
      i3 = (const uint8_t*) ((uintptr_t) i3 + tile_stride);

      const __m128i v1_0 = _mm_unpacklo_epi8(v0, v1);
      const __m128i v1_1 = _mm_unpackhi_epi8(v0, v1);
      const __m128i v1_2 = _mm_unpacklo_epi8(v2, v3);
      const __m128i v1_3 = _mm_unpackhi_epi8(v2, v3);

      const __m128i v3_0 = _mm_unpacklo_epi8(v1_0, v1_2);
      const __m128i v3_1 = _mm_unpackhi_epi8(v1_0, v1_2);
      const __m128i v3_2 = _mm_unpacklo_epi8(v1_1, v1_3);

      __m128i v4_0 = _mm_shuffle_epi8(v3_0, vperm0);
      __m128i v4_1 = _mm_or_si128(_mm_shuffle_epi8(v3_0, vperm2), _mm_shuffle_epi8(v3_1, vperm3));
      __m128i v4_2 = _mm_or_si128(_mm_shuffle_epi8(v3_1, vperm4), _mm_shuffle_epi8(v3_2, vperm5));
      __m128i v4_3 = _mm_shuffle_epi8(v3_2, vperm1);

      _mm_storel_epi64((__m128i*) o3, v4_3);
      _mm_storel_epi64((__m128i*) o2, v4_2);
      _mm_storel_epi64((__m128i*) o1, v4_1);
      _mm_storel_epi64((__m128i*) o0, v4_0);
      o3 += 8;
      o2 += 8;
      o1 += 8;
      o0 += 8;

      v4_3 = _mm_unpackhi_epi64(v4_3, v4_3);
      unaligned_store_u32(o3, (uint32_t) _mm_cvtsi128_si32(v4_3));
      v4_2 = _mm_unpackhi_epi64(v4_2, v4_2);
      unaligned_store_u32(o2, (uint32_t) _mm_cvtsi128_si32(v4_2));
      v4_1 = _mm_unpackhi_epi64(v4_1, v4_1);
      unaligned_store_u32(o1, (uint32_t) _mm_cvtsi128_si32(v4_1));
      v4_0 = _mm_unpackhi_epi64(v4_0, v4_0);
      unaligned_store_u32(o0, (uint32_t) _mm_cvtsi128_si32(v4_0));
      o3 += 4;
      o2 += 4;
      o1 += 4;
      o0 += 4;
    }

    if (bh != 0) {
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      const __m128i v0 = _mm_loadu_si128((const __m128i*) i0);
      const __m128i v1 = _mm_loadu_si128((const __m128i*) i1);
      const __m128i v2 = _mm_loadu_si128((const __m128i*) i2);

      const __m128i v1_0 = _mm_unpacklo_epi8(v0, v1);
      const __m128i v1_1 = _mm_unpackhi_epi8(v0, v1);
      const __m128i v1_2 = _mm_unpacklo_epi8(v2, v2);
      const __m128i v1_3 = _mm_unpackhi_epi8(v2, v2);

      const __m128i v3_0 = _mm_unpacklo_epi8(v1_0, v1_2);
      const __m128i v3_1 = _mm_unpackhi_epi8(v1_0, v1_2);
      const __m128i v3_2 = _mm_unpacklo_epi8(v1_1, v1_3);

      __m128i v4_0 = _mm_shuffle_epi8(v3_0, vperm0);
      __m128i v4_1 = _mm_or_si128(_mm_shuffle_epi8(v3_0, vperm2), _mm_shuffle_epi8(v3_1, vperm3));
      __m128i v4_2 = _mm_or_si128(_mm_shuffle_epi8(v3_1, vperm4), _mm_shuffle_epi8(v3_2, vperm5));
      __m128i v4_3 = _mm_shuffle_epi8(v3_2, vperm1);

      if (bh & 2) {
        unaligned_store_u32(o3, (uint32_t) _mm_cvtsi128_si32(v4_3));
        unaligned_store_u32(o2, (uint32_t) _mm_cvtsi128_si32(v4_2));
        unaligned_store_u32(o1, (uint32_t) _mm_cvtsi128_si32(v4_1));
        unaligned_store_u32(o0, (uint32_t) _mm_cvtsi128_si32(v4_0));
        o3 += 4;
        o2 += 4;
        o1 += 4;
        o0 += 4;
        unaligned_store_u16(o3, (uint16_t) _mm_extract_epi16(v4_3, 2));
        unaligned_store_u16(o2, (uint16_t) _mm_extract_epi16(v4_2, 2));
        unaligned_store_u16(o1, (uint16_t) _mm_extract_epi16(v4_1, 2));
        unaligned_store_u16(o0, (uint16_t) _mm_extract_epi16(v4_0, 2));
        o3 += 2;
        o2 += 2;
        o1 += 2;
        o0 += 2;
        v4_3 = _mm_bsrli_si128(v4_3, 6);
        v4_2 = _mm_bsrli_si128(v4_2, 6);
        v4_1 = _mm_bsrli_si128(v4_1, 6);
        v4_0 = _mm_bsrli_si128(v4_0, 6);
      }
      if (bh & 1) {
        unaligned_store_u16(o3, (uint16_t) _mm_cvtsi128_si32(v4_3));
        unaligned_store_u16(o2, (uint16_t) _mm_cvtsi128_si32(v4_2));
        unaligned_store_u16(o1, (uint16_t) _mm_cvtsi128_si32(v4_1));
        unaligned_store_u16(o0, (uint16_t) _mm_cvtsi128_si32(v4_0));
        o3 += 2;
        o2 += 2;
        o1 += 2;
        o0 += 2;
        *((uint8_t*) o3) = (uint8_t) _mm_cvtsi128_si32(_mm_bsrli_si128(v4_3, 2));
        *((uint8_t*) o2) = (uint8_t) _mm_cvtsi128_si32(_mm_bsrli_si128(v4_2, 2));
        *((uint8_t*) o1) = (uint8_t) _mm_cvtsi128_si32(_mm_bsrli_si128(v4_1, 2));
        *((uint8_t*) o0) = (uint8_t) _mm_cvtsi128_si32(_mm_bsrli_si128(v4_0, 2));
        o3 += 1;
        o2 += 1;
        o1 += 1;
        o0 += 1;
      }
    }
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
    i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
    i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
    o0 = (uint8_t*) ((uintptr_t) o0 + output_reset);
    o1 = (uint8_t*) ((uintptr_t) o1 + output_reset);
    o2 = (uint8_t*) ((uintptr_t) o2 + output_reset);
    o3 = (uint8_t*) ((uintptr_t) o3 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
