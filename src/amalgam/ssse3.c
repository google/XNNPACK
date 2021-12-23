// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


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

  const __m128 vmask = _mm_load_ps((const float*) params->sse.mask);
  const __m128 vmax = _mm_load_ps(params->sse.max);
  const __m128 vmin = _mm_load_ps(params->sse.min);

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

void xnn_qs8_gavgpool_minmax_ukernel_7p7x__ssse3_c8_acc2(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* buffer,
    int8_t* output,
    const union xnn_qs8_avgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows > 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 8);

  const __m128i vbias = _mm_load_si128((const __m128i*) params->sse2.bias);
  int32_t* b = buffer;
  size_t c = channels;
  for (; c != 0; c = doz(c, 8)) {
    const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
    i0 += 8;
    const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
    i1 += 8;
    const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
    i2 += 8;
    const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
    i3 += 8;
    const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
    i4 += 8;
    const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
    i5 += 8;
    const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
    i6 += 8;

    const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x01234567));
    const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x01234567));
    const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x01234567));
    const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x01234567));
    const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x01234567));
    const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x01234567));
    const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x01234567));

    __m128i vacc0x01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    __m128i vacc1x01234567 = _mm_add_epi16(vxi2x01234567, vxi3x01234567);

    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi4x01234567);
    vacc1x01234567 = _mm_add_epi16(vacc1x01234567, vxi5x01234567);
    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi6x01234567);

    // Add up all accumulators to vacc0x01234567
    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vacc1x01234567);

    const __m128i vsgnacc0x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x01234567);
    const __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vacc0x01234567, vsgnacc0x01234567));
    const __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vacc0x01234567, vsgnacc0x01234567));

    _mm_store_si128((__m128i*) b, vacc0123);
    _mm_store_si128((__m128i*) (b + 4), vacc4567);
    b += 8;
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);

    int32_t* b = buffer;
    size_t c = channels;
    for (; c != 0; c = doz(c, 8)) {
      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      i0 += 8;
      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      i1 += 8;
      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      i2 += 8;
      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      i3 += 8;
      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      i4 += 8;
      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      i5 += 8;
      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      i6 += 8;

      const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x01234567));
      const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x01234567));
      const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x01234567));
      const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x01234567));
      const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x01234567));
      const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x01234567));
      const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x01234567));

      __m128i vacc0x01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      __m128i vacc1x01234567 = _mm_add_epi16(vxi2x01234567, vxi3x01234567);

      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi4x01234567);
      vacc1x01234567 = _mm_add_epi16(vacc1x01234567, vxi5x01234567);
      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi6x01234567);

      // Add up all accumulators to vacc0x01234567
      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vacc1x01234567);

      const __m128i vsgnacc0x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x01234567);
      const __m128i vacc0123 = _mm_add_epi32(_mm_unpacklo_epi16(vacc0x01234567, vsgnacc0x01234567), _mm_load_si128((const __m128i*) (b + 0)));
      const __m128i vacc4567 = _mm_add_epi32(_mm_unpackhi_epi16(vacc0x01234567, vsgnacc0x01234567), _mm_load_si128((const __m128i*) (b + 4)));

      _mm_store_si128((__m128i*) b, vacc0123);
      _mm_store_si128((__m128i*) (b + 4), vacc4567);
      b += 8;
    }
  }

  i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->sse2.multiplier);
  const __m128i vrounding = _mm_load_si128((const __m128i*) params->sse2.rounding);
  const __m128i vshift = _mm_loadl_epi64((const __m128i*) params->sse2.shift);
  while (channels >= 8) {
    const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
    i0 += 8;
    const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
    i1 += 8;
    const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
    i2 += 8;
    const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
    i3 += 8;
    const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
    i4 += 8;
    const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
    i5 += 8;
    const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
    i6 += 8;

    const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x01234567));
    const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x01234567));
    const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x01234567));
    const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x01234567));
    const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x01234567));
    const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x01234567));
    const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x01234567));

    __m128i vacc0x01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    __m128i vacc1x01234567 = _mm_add_epi16(vxi2x01234567, vxi3x01234567);

    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi4x01234567);
    vacc1x01234567 = _mm_add_epi16(vacc1x01234567, vxi5x01234567);
    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi6x01234567);

    // Add up all accumulators to vacc0x01234567
    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vacc1x01234567);

    const __m128i vsgnacc0x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x01234567);
    const __m128i vacc0123 = _mm_add_epi32(_mm_unpacklo_epi16(vacc0x01234567, vsgnacc0x01234567), _mm_load_si128((const __m128i*) (buffer + 0)));
    const __m128i vacc4567 = _mm_add_epi32(_mm_unpackhi_epi16(vacc0x01234567, vsgnacc0x01234567), _mm_load_si128((const __m128i*) (buffer + 4)));
    buffer += 8;

    const __m128i vabsacc0123 = _mm_abs_epi32(vacc0123);
    const __m128i vabsacc4567 = _mm_abs_epi32(vacc4567);

    const __m128i vabsacc13 = _mm_shuffle_epi32(vabsacc0123, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128i vabsacc57 = _mm_shuffle_epi32(vabsacc4567, _MM_SHUFFLE(3, 3, 1, 1));

    const __m128i vabsprod02 = _mm_mul_epu32(vabsacc0123, vmultiplier);
    const __m128i vabsprod13 = _mm_mul_epu32(vabsacc13, vmultiplier);
    const __m128i vabsprod46 = _mm_mul_epu32(vabsacc4567, vmultiplier);
    const __m128i vabsprod57 = _mm_mul_epu32(vabsacc57, vmultiplier);

    const __m128i vabsout02 = _mm_srl_epi64(_mm_add_epi64(vabsprod02, vrounding), vshift);
    const __m128i vabsout13 = _mm_srl_epi64(_mm_add_epi64(vabsprod13, vrounding), vshift);
    const __m128i vabsout46 = _mm_srl_epi64(_mm_add_epi64(vabsprod46, vrounding), vshift);
    const __m128i vabsout57 = _mm_srl_epi64(_mm_add_epi64(vabsprod57, vrounding), vshift);

    const __m128i vabsout0213 = _mm_castps_si128(
        _mm_shuffle_ps(_mm_castsi128_ps(vabsout02), _mm_castsi128_ps(vabsout13), _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i vabsout4657 = _mm_castps_si128(
        _mm_shuffle_ps(_mm_castsi128_ps(vabsout46), _mm_castsi128_ps(vabsout57), _MM_SHUFFLE(2, 0, 2, 0)));

    const __m128i vabsout0123 = _mm_shuffle_epi32(vabsout0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i vabsout4567 = _mm_shuffle_epi32(vabsout4657, _MM_SHUFFLE(3, 1, 2, 0));

    const __m128i vout0123 = _mm_sign_epi32(vabsout0123, vacc0123);
    const __m128i vout4567 = _mm_sign_epi32(vabsout4567, vacc4567);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vout0123, vout4567), voutput_zero_point);

    const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse2.output_min);
    const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse2.output_max);
    vout01234567 = _mm_min_epi16(_mm_max_epi16(vout01234567, voutput_min), voutput_max);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;

    channels -= 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      i0 += 8;
      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      i1 += 8;
      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      i2 += 8;
      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      i3 += 8;
      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      i4 += 8;
      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      i5 += 8;
      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      i6 += 8;

      const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x01234567));
      const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x01234567));
      const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x01234567));
      const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x01234567));
      const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x01234567));
      const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x01234567));
      const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x01234567));

      __m128i vacc0x01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      __m128i vacc1x01234567 = _mm_add_epi16(vxi2x01234567, vxi3x01234567);

      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi4x01234567);
      vacc1x01234567 = _mm_add_epi16(vacc1x01234567, vxi5x01234567);
      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi6x01234567);

      // Add up all accumulators to vacc0x01234567
      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vacc1x01234567);

      const __m128i vsgnacc0x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x01234567);
      const __m128i vacc0123 = _mm_add_epi32(_mm_unpacklo_epi16(vacc0x01234567, vsgnacc0x01234567), _mm_load_si128((const __m128i*) buffer));
      const __m128i vacc4567 = _mm_add_epi32(_mm_unpackhi_epi16(vacc0x01234567, vsgnacc0x01234567), _mm_load_si128((const __m128i*) (buffer + 4)));
      buffer += 8;

      const __m128i vabsacc0123 = _mm_abs_epi32(vacc0123);
      const __m128i vabsacc4567 = _mm_abs_epi32(vacc4567);

      const __m128i vabsacc13 = _mm_shuffle_epi32(vabsacc0123, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vabsacc57 = _mm_shuffle_epi32(vabsacc4567, _MM_SHUFFLE(3, 3, 1, 1));

      const __m128i vabsprod02 = _mm_mul_epu32(vabsacc0123, vmultiplier);
      const __m128i vabsprod13 = _mm_mul_epu32(vabsacc13, vmultiplier);
      const __m128i vabsprod46 = _mm_mul_epu32(vabsacc4567, vmultiplier);
      const __m128i vabsprod57 = _mm_mul_epu32(vabsacc57, vmultiplier);

      const __m128i vabsout02 = _mm_srl_epi64(_mm_add_epi64(vabsprod02, vrounding), vshift);
      const __m128i vabsout13 = _mm_srl_epi64(_mm_add_epi64(vabsprod13, vrounding), vshift);
      const __m128i vabsout46 = _mm_srl_epi64(_mm_add_epi64(vabsprod46, vrounding), vshift);
      const __m128i vabsout57 = _mm_srl_epi64(_mm_add_epi64(vabsprod57, vrounding), vshift);

      const __m128i vabsout0213 = _mm_castps_si128(
          _mm_shuffle_ps(_mm_castsi128_ps(vabsout02), _mm_castsi128_ps(vabsout13), _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128i vabsout4657 = _mm_castps_si128(
          _mm_shuffle_ps(_mm_castsi128_ps(vabsout46), _mm_castsi128_ps(vabsout57), _MM_SHUFFLE(2, 0, 2, 0)));

      const __m128i vabsout0123 = _mm_shuffle_epi32(vabsout0213, _MM_SHUFFLE(3, 1, 2, 0));
      const __m128i vabsout4567 = _mm_shuffle_epi32(vabsout4657, _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i vout0123 = _mm_sign_epi32(vabsout0123, vacc0123);
      const __m128i vout4567 = _mm_sign_epi32(vabsout4567, vacc4567);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vout0123, vout4567), voutput_zero_point);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse2.output_min);
      const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse2.output_max);
      vout01234567 = _mm_min_epi16(_mm_max_epi16(vout01234567, voutput_min), voutput_max);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      if (channels & 4) {
        *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (channels & 2) {
        *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (channels & 1) {
        *output = (int32_t) _mm_cvtsi128_si32(vout0123456701234567);
      }
    }
  }
}

void xnn_qs8_gavgpool_minmax_ukernel_7x__ssse3_c8_acc2(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const union xnn_qs8_avgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const __m128i vbias = _mm_load_si128((const __m128i*) params->sse2.bias);
  const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->sse2.multiplier);
  const __m128i vrounding = _mm_load_si128((const __m128i*) params->sse2.rounding);
  const __m128i vshift = _mm_loadl_epi64((const __m128i*) params->sse2.shift);
  while (channels >= 8) {
    const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
    i0 += 8;
    const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
    i1 += 8;
    const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
    i2 += 8;
    const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
    i3 += 8;
    const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
    i4 += 8;
    const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
    i5 += 8;
    const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
    i6 += 8;

    const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x01234567));
    const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x01234567));
    const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x01234567));
    const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x01234567));
    const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x01234567));
    const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x01234567));
    const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x01234567));

    __m128i vacc0x01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    __m128i vacc1x01234567 = _mm_add_epi16(vxi2x01234567, vxi3x01234567);

    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi4x01234567);
    vacc1x01234567 = _mm_add_epi16(vacc1x01234567, vxi5x01234567);
    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi6x01234567);

    // Add up all accumulators to vacc0x01234567
    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vacc1x01234567);

    const __m128i vsgnacc0x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x01234567);
    const __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vacc0x01234567, vsgnacc0x01234567));
    const __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vacc0x01234567, vsgnacc0x01234567));

    const __m128i vabsacc0123 = _mm_abs_epi32(vacc0123);
    const __m128i vabsacc4567 = _mm_abs_epi32(vacc4567);

    const __m128i vabsacc13 = _mm_shuffle_epi32(vabsacc0123, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128i vabsacc57 = _mm_shuffle_epi32(vabsacc4567, _MM_SHUFFLE(3, 3, 1, 1));

    const __m128i vabsprod02 = _mm_mul_epu32(vabsacc0123, vmultiplier);
    const __m128i vabsprod13 = _mm_mul_epu32(vabsacc13, vmultiplier);
    const __m128i vabsprod46 = _mm_mul_epu32(vabsacc4567, vmultiplier);
    const __m128i vabsprod57 = _mm_mul_epu32(vabsacc57, vmultiplier);

    const __m128i vabsout02 = _mm_srl_epi64(_mm_add_epi64(vabsprod02, vrounding), vshift);
    const __m128i vabsout13 = _mm_srl_epi64(_mm_add_epi64(vabsprod13, vrounding), vshift);
    const __m128i vabsout46 = _mm_srl_epi64(_mm_add_epi64(vabsprod46, vrounding), vshift);
    const __m128i vabsout57 = _mm_srl_epi64(_mm_add_epi64(vabsprod57, vrounding), vshift);

    const __m128i vabsout0213 = _mm_castps_si128(
        _mm_shuffle_ps(_mm_castsi128_ps(vabsout02), _mm_castsi128_ps(vabsout13), _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i vabsout4657 = _mm_castps_si128(
        _mm_shuffle_ps(_mm_castsi128_ps(vabsout46), _mm_castsi128_ps(vabsout57), _MM_SHUFFLE(2, 0, 2, 0)));

    const __m128i vabsout0123 = _mm_shuffle_epi32(vabsout0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i vabsout4567 = _mm_shuffle_epi32(vabsout4657, _MM_SHUFFLE(3, 1, 2, 0));

    const __m128i vout0123 = _mm_sign_epi32(vabsout0123, vacc0123);
    const __m128i vout4567 = _mm_sign_epi32(vabsout4567, vacc4567);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vout0123, vout4567), voutput_zero_point);

    const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse2.output_min);
    const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse2.output_max);
    vout01234567 = _mm_min_epi16(_mm_max_epi16(vout01234567, voutput_min), voutput_max);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;

    channels -= 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      i0 += 8;
      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      i1 += 8;
      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      i2 += 8;
      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      i3 += 8;
      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      i4 += 8;
      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      i5 += 8;
      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      i6 += 8;

      const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x01234567));
      const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x01234567));
      const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x01234567));
      const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x01234567));
      const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x01234567));
      const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x01234567));
      const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x01234567));

      __m128i vacc0x01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      __m128i vacc1x01234567 = _mm_add_epi16(vxi2x01234567, vxi3x01234567);

      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi4x01234567);
      vacc1x01234567 = _mm_add_epi16(vacc1x01234567, vxi5x01234567);
      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi6x01234567);

      // Add up all accumulators to vacc0x01234567
      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vacc1x01234567);

      const __m128i vsgnacc0x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x01234567);
      const __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vacc0x01234567, vsgnacc0x01234567));
      const __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vacc0x01234567, vsgnacc0x01234567));

      const __m128i vabsacc0123 = _mm_abs_epi32(vacc0123);
      const __m128i vabsacc4567 = _mm_abs_epi32(vacc4567);

      const __m128i vabsacc13 = _mm_shuffle_epi32(vabsacc0123, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vabsacc57 = _mm_shuffle_epi32(vabsacc4567, _MM_SHUFFLE(3, 3, 1, 1));

      const __m128i vabsprod02 = _mm_mul_epu32(vabsacc0123, vmultiplier);
      const __m128i vabsprod13 = _mm_mul_epu32(vabsacc13, vmultiplier);
      const __m128i vabsprod46 = _mm_mul_epu32(vabsacc4567, vmultiplier);
      const __m128i vabsprod57 = _mm_mul_epu32(vabsacc57, vmultiplier);

      const __m128i vabsout02 = _mm_srl_epi64(_mm_add_epi64(vabsprod02, vrounding), vshift);
      const __m128i vabsout13 = _mm_srl_epi64(_mm_add_epi64(vabsprod13, vrounding), vshift);
      const __m128i vabsout46 = _mm_srl_epi64(_mm_add_epi64(vabsprod46, vrounding), vshift);
      const __m128i vabsout57 = _mm_srl_epi64(_mm_add_epi64(vabsprod57, vrounding), vshift);

      const __m128i vabsout0213 = _mm_castps_si128(
          _mm_shuffle_ps(_mm_castsi128_ps(vabsout02), _mm_castsi128_ps(vabsout13), _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128i vabsout4657 = _mm_castps_si128(
          _mm_shuffle_ps(_mm_castsi128_ps(vabsout46), _mm_castsi128_ps(vabsout57), _MM_SHUFFLE(2, 0, 2, 0)));

      const __m128i vabsout0123 = _mm_shuffle_epi32(vabsout0213, _MM_SHUFFLE(3, 1, 2, 0));
      const __m128i vabsout4567 = _mm_shuffle_epi32(vabsout4657, _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i vout0123 = _mm_sign_epi32(vabsout0123, vacc0123);
      const __m128i vout4567 = _mm_sign_epi32(vabsout4567, vacc4567);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vout0123, vout4567), voutput_zero_point);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse2.output_min);
      const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse2.output_max);
      vout01234567 = _mm_min_epi16(_mm_max_epi16(vout01234567, voutput_min), voutput_max);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      if (channels & 4) {
        *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (channels & 2) {
        *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (channels & 1) {
        *output = (int32_t) _mm_cvtsi128_si32(vout0123456701234567);
      }
    }
  }
}
