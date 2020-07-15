// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <immintrin.h>

#include <xnnpack/dwconv.h>


void xnn_qu8_dwconv_minmax_ukernel_up8x9__sse2(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_gemm_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  const __m128i vkernel_zero_point = _mm_load_si128((const __m128i*) params->sse2.kernel_zero_point);
  const __m128i vzero = _mm_setzero_si128();

  do {
    const uint8_t* i0 = input[0];
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }

    input = (const uint8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 8; c -= 8) {
      __m128i vacc_lo = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc_hi = _mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16));

      const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0); i0 += 8;
      const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
      const __m128i vk0 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32));
      const __m128i vxk0 = _mm_sub_epi16(_mm_unpacklo_epi8(vk0, vzero), vkernel_zero_point);
      const __m128i vprod0_odd  = _mm_mullo_epi16(vxi0, vxk0);
      const __m128i vprod0_even = _mm_mulhi_epi16(vxi0, vxk0);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod0_odd, vprod0_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod0_odd, vprod0_even));

      const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1); i1 += 8;
      const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
      const __m128i vk1 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 40));
      const __m128i vxk1 = _mm_sub_epi16(_mm_unpacklo_epi8(vk1, vzero), vkernel_zero_point);
      const __m128i vprod1_odd  = _mm_mullo_epi16(vxi1, vxk1);
      const __m128i vprod1_even = _mm_mulhi_epi16(vxi1, vxk1);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod1_odd, vprod1_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod1_odd, vprod1_even));

      const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2); i2 += 8;
      const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
      const __m128i vk2 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 48));
      const __m128i vxk2 = _mm_sub_epi16(_mm_unpacklo_epi8(vk2, vzero), vkernel_zero_point);
      const __m128i vprod2_odd  = _mm_mullo_epi16(vxi2, vxk2);
      const __m128i vprod2_even = _mm_mulhi_epi16(vxi2, vxk2);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod2_odd, vprod2_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod2_odd, vprod2_even));

      const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3); i3 += 8;
      const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
      const __m128i vk3 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 56));
      const __m128i vxk3 = _mm_sub_epi16(_mm_unpacklo_epi8(vk3, vzero), vkernel_zero_point);
      const __m128i vprod3_odd  = _mm_mullo_epi16(vxi3, vxk3);
      const __m128i vprod3_even = _mm_mulhi_epi16(vxi3, vxk3);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod3_odd, vprod3_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod3_odd, vprod3_even));

      const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4); i4 += 8;
      const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
      const __m128i vk4 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 64));
      const __m128i vxk4 = _mm_sub_epi16(_mm_unpacklo_epi8(vk4, vzero), vkernel_zero_point);
      const __m128i vprod4_odd  = _mm_mullo_epi16(vxi4, vxk4);
      const __m128i vprod4_even = _mm_mulhi_epi16(vxi4, vxk4);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod4_odd, vprod4_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod4_odd, vprod4_even));

      const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5); i5 += 8;
      const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
      const __m128i vk5 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 72));
      const __m128i vxk5 = _mm_sub_epi16(_mm_unpacklo_epi8(vk5, vzero), vkernel_zero_point);
      const __m128i vprod5_odd  = _mm_mullo_epi16(vxi5, vxk5);
      const __m128i vprod5_even = _mm_mulhi_epi16(vxi5, vxk5);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod5_odd, vprod5_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod5_odd, vprod5_even));

      const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6); i6 += 8;
      const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);
      const __m128i vk6 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 80));
      const __m128i vxk6 = _mm_sub_epi16(_mm_unpacklo_epi8(vk6, vzero), vkernel_zero_point);
      const __m128i vprod6_odd  = _mm_mullo_epi16(vxi6, vxk6);
      const __m128i vprod6_even = _mm_mulhi_epi16(vxi6, vxk6);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod6_odd, vprod6_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod6_odd, vprod6_even));

      const __m128i vi7 = _mm_loadl_epi64((const __m128i*) i7); i7 += 8;
      const __m128i vxi7 = _mm_unpacklo_epi8(vi7, vzero);
      const __m128i vk7 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 88));
      const __m128i vxk7 = _mm_sub_epi16(_mm_unpacklo_epi8(vk7, vzero), vkernel_zero_point);
      const __m128i vprod7_odd  = _mm_mullo_epi16(vxi7, vxk7);
      const __m128i vprod7_even = _mm_mulhi_epi16(vxi7, vxk7);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod7_odd, vprod7_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod7_odd, vprod7_even));

      const __m128i vi8 = _mm_loadl_epi64((const __m128i*) i8); i8 += 8;
      const __m128i vxi8 = _mm_unpacklo_epi8(vi8, vzero);
      const __m128i vk8 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 96));
      const __m128i vxk8 = _mm_sub_epi16(_mm_unpacklo_epi8(vk8, vzero), vkernel_zero_point);
      const __m128i vprod8_odd  = _mm_mullo_epi16(vxi8, vxk8);
      const __m128i vprod8_even = _mm_mulhi_epi16(vxi8, vxk8);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod8_odd, vprod8_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod8_odd, vprod8_even));

      w = (void*) ((uintptr_t) w + 104);

      const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->sse2.multiplier);
      const __m128i vrounding = _mm_load_si128((const __m128i*) params->sse2.rounding);

      const __m128i vnmask_lo0123 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_lo);
      const __m128i vnmask_hi0123 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_hi);

      const __m128i vabsacc_lo0123 = _mm_sub_epi32(_mm_xor_si128(vacc_lo, vnmask_lo0123), vnmask_lo0123);
      const __m128i vabsacc_hi0123 = _mm_sub_epi32(_mm_xor_si128(vacc_hi, vnmask_hi0123), vnmask_hi0123);

      const __m128i vabsacc_lo1032 = _mm_shuffle_epi32(vabsacc_lo0123, _MM_SHUFFLE(2, 3, 0, 1));
      const __m128i vabsacc_hi1032 = _mm_shuffle_epi32(vabsacc_hi0123, _MM_SHUFFLE(2, 3, 0, 1));

      const __m128i vabsprod_lo02 = _mm_mul_epu32(vabsacc_lo0123, vmultiplier);
      const __m128i vabsprod_hi02 = _mm_mul_epu32(vabsacc_hi0123, vmultiplier);

      const __m128i vnmask_lo02 = _mm_shuffle_epi32(vnmask_lo0123, _MM_SHUFFLE(2, 2, 0, 0));
      const __m128i vnmask_hi02 = _mm_shuffle_epi32(vnmask_hi0123, _MM_SHUFFLE(2, 2, 0, 0));

      const __m128i vprod_lo02 = _mm_sub_epi64(_mm_xor_si128(vabsprod_lo02, vnmask_lo02), vnmask_lo02);
      const __m128i vprod_hi02 = _mm_sub_epi64(_mm_xor_si128(vabsprod_hi02, vnmask_hi02), vnmask_hi02);

      const __m128i vq31prod_lo02 = _mm_srli_epi64(_mm_add_epi64(vprod_lo02, vrounding), 31);
      const __m128i vq31prod_hi02 = _mm_srli_epi64(_mm_add_epi64(vprod_hi02, vrounding), 31);

      const __m128i vabsprod_lo13 = _mm_mul_epu32(vabsacc_lo1032, vmultiplier);
      const __m128i vabsprod_hi13 = _mm_mul_epu32(vabsacc_hi1032, vmultiplier);

      const __m128i vnmask_lo13 = _mm_shuffle_epi32(vnmask_lo0123, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vnmask_hi13 = _mm_shuffle_epi32(vnmask_hi0123, _MM_SHUFFLE(3, 3, 1, 1));

      const __m128i vprod_lo13 = _mm_sub_epi64(_mm_xor_si128(vabsprod_lo13, vnmask_lo13), vnmask_lo13);
      const __m128i vprod_hi13 = _mm_sub_epi64(_mm_xor_si128(vabsprod_hi13, vnmask_hi13), vnmask_hi13);

      const __m128i vq31prod_lo13 = _mm_srli_epi64(_mm_add_epi64(vprod_lo13, vrounding), 31);
      const __m128i vq31prod_hi13 = _mm_srli_epi64(_mm_add_epi64(vprod_hi13, vrounding), 31);

      const __m128i vq31prod_lo0213 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prod_lo02), _mm_castsi128_ps(vq31prod_lo13), _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128i vq31prod_hi0213 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prod_hi02), _mm_castsi128_ps(vq31prod_hi13), _MM_SHUFFLE(2, 0, 2, 0)));

      const __m128i vq31prod_lo0123 = _mm_shuffle_epi32(vq31prod_lo0213, _MM_SHUFFLE(3, 1, 2, 0));
      const __m128i vq31prod_hi0123 = _mm_shuffle_epi32(vq31prod_hi0213, _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->sse2.remainder_mask);

      const __m128i vrem_lo0123 =
        _mm_add_epi32(_mm_and_si128(vq31prod_lo0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod_lo0123));
      const __m128i vrem_hi0123 =
        _mm_add_epi32(_mm_and_si128(vq31prod_hi0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod_hi0123));

      const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->sse2.remainder_threshold);
      const __m128i vshift = _mm_load_si128((const __m128i*) params->sse2.shift);

      const __m128i vout_lo = _mm_sub_epi32(_mm_sra_epi32(vq31prod_lo0123, vshift), _mm_cmpgt_epi32(vrem_lo0123, vremainder_threshold));
      const __m128i vout_hi = _mm_sub_epi32(_mm_sra_epi32(vq31prod_hi0123, vshift), _mm_cmpgt_epi32(vrem_hi0123, vremainder_threshold));

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
      __m128i vout = _mm_adds_epi16(_mm_packs_epi32(vout_lo, vout_hi), voutput_zero_point);
      vout = _mm_packus_epi16(vout, vout);
      vout = _mm_min_epu8(vout, _mm_load_si128((const __m128i*) params->sse2.output_max));
      vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->sse2.output_min));

      _mm_storel_epi64((__m128i*) output, vout); output += 8;
    }
    if (c != 0) {
      __m128i vacc_lo = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc_hi = _mm_loadu_si128((const __m128i*) ((uintptr_t) w + 16));

      const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0); i0 += 8;
      const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
      const __m128i vk0 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 32));
      const __m128i vxk0 = _mm_sub_epi16(_mm_unpacklo_epi8(vk0, vzero), vkernel_zero_point);
      const __m128i vprod0_odd  = _mm_mullo_epi16(vxi0, vxk0);
      const __m128i vprod0_even = _mm_mulhi_epi16(vxi0, vxk0);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod0_odd, vprod0_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod0_odd, vprod0_even));

      const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1); i1 += 8;
      const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
      const __m128i vk1 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 40));
      const __m128i vxk1 = _mm_sub_epi16(_mm_unpacklo_epi8(vk1, vzero), vkernel_zero_point);
      const __m128i vprod1_odd  = _mm_mullo_epi16(vxi1, vxk1);
      const __m128i vprod1_even = _mm_mulhi_epi16(vxi1, vxk1);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod1_odd, vprod1_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod1_odd, vprod1_even));

      const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2); i2 += 8;
      const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
      const __m128i vk2 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 48));
      const __m128i vxk2 = _mm_sub_epi16(_mm_unpacklo_epi8(vk2, vzero), vkernel_zero_point);
      const __m128i vprod2_odd  = _mm_mullo_epi16(vxi2, vxk2);
      const __m128i vprod2_even = _mm_mulhi_epi16(vxi2, vxk2);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod2_odd, vprod2_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod2_odd, vprod2_even));

      const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3); i3 += 8;
      const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
      const __m128i vk3 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 56));
      const __m128i vxk3 = _mm_sub_epi16(_mm_unpacklo_epi8(vk3, vzero), vkernel_zero_point);
      const __m128i vprod3_odd  = _mm_mullo_epi16(vxi3, vxk3);
      const __m128i vprod3_even = _mm_mulhi_epi16(vxi3, vxk3);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod3_odd, vprod3_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod3_odd, vprod3_even));

      const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4); i4 += 8;
      const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
      const __m128i vk4 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 64));
      const __m128i vxk4 = _mm_sub_epi16(_mm_unpacklo_epi8(vk4, vzero), vkernel_zero_point);
      const __m128i vprod4_odd  = _mm_mullo_epi16(vxi4, vxk4);
      const __m128i vprod4_even = _mm_mulhi_epi16(vxi4, vxk4);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod4_odd, vprod4_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod4_odd, vprod4_even));

      const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5); i5 += 8;
      const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
      const __m128i vk5 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 72));
      const __m128i vxk5 = _mm_sub_epi16(_mm_unpacklo_epi8(vk5, vzero), vkernel_zero_point);
      const __m128i vprod5_odd  = _mm_mullo_epi16(vxi5, vxk5);
      const __m128i vprod5_even = _mm_mulhi_epi16(vxi5, vxk5);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod5_odd, vprod5_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod5_odd, vprod5_even));

      const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6); i6 += 8;
      const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);
      const __m128i vk6 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 80));
      const __m128i vxk6 = _mm_sub_epi16(_mm_unpacklo_epi8(vk6, vzero), vkernel_zero_point);
      const __m128i vprod6_odd  = _mm_mullo_epi16(vxi6, vxk6);
      const __m128i vprod6_even = _mm_mulhi_epi16(vxi6, vxk6);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod6_odd, vprod6_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod6_odd, vprod6_even));

      const __m128i vi7 = _mm_loadl_epi64((const __m128i*) i7); i7 += 8;
      const __m128i vxi7 = _mm_unpacklo_epi8(vi7, vzero);
      const __m128i vk7 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 88));
      const __m128i vxk7 = _mm_sub_epi16(_mm_unpacklo_epi8(vk7, vzero), vkernel_zero_point);
      const __m128i vprod7_odd  = _mm_mullo_epi16(vxi7, vxk7);
      const __m128i vprod7_even = _mm_mulhi_epi16(vxi7, vxk7);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod7_odd, vprod7_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod7_odd, vprod7_even));

      const __m128i vi8 = _mm_loadl_epi64((const __m128i*) i8); i8 += 8;
      const __m128i vxi8 = _mm_unpacklo_epi8(vi8, vzero);
      const __m128i vk8 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 96));
      const __m128i vxk8 = _mm_sub_epi16(_mm_unpacklo_epi8(vk8, vzero), vkernel_zero_point);
      const __m128i vprod8_odd  = _mm_mullo_epi16(vxi8, vxk8);
      const __m128i vprod8_even = _mm_mulhi_epi16(vxi8, vxk8);
      vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vprod8_odd, vprod8_even));
      vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vprod8_odd, vprod8_even));

      const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->sse2.multiplier);
      const __m128i vrounding = _mm_load_si128((const __m128i*) params->sse2.rounding);

      const __m128i vnmask_lo0123 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_lo);
      const __m128i vnmask_hi0123 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_hi);

      const __m128i vabsacc_lo0123 = _mm_sub_epi32(_mm_xor_si128(vacc_lo, vnmask_lo0123), vnmask_lo0123);
      const __m128i vabsacc_hi0123 = _mm_sub_epi32(_mm_xor_si128(vacc_hi, vnmask_hi0123), vnmask_hi0123);

      const __m128i vabsacc_lo1032 = _mm_shuffle_epi32(vabsacc_lo0123, _MM_SHUFFLE(2, 3, 0, 1));
      const __m128i vabsacc_hi1032 = _mm_shuffle_epi32(vabsacc_hi0123, _MM_SHUFFLE(2, 3, 0, 1));

      const __m128i vabsprod_lo02 = _mm_mul_epu32(vabsacc_lo0123, vmultiplier);
      const __m128i vabsprod_hi02 = _mm_mul_epu32(vabsacc_hi0123, vmultiplier);

      const __m128i vnmask_lo02 = _mm_shuffle_epi32(vnmask_lo0123, _MM_SHUFFLE(2, 2, 0, 0));
      const __m128i vnmask_hi02 = _mm_shuffle_epi32(vnmask_hi0123, _MM_SHUFFLE(2, 2, 0, 0));

      const __m128i vprod_lo02 = _mm_sub_epi64(_mm_xor_si128(vabsprod_lo02, vnmask_lo02), vnmask_lo02);
      const __m128i vprod_hi02 = _mm_sub_epi64(_mm_xor_si128(vabsprod_hi02, vnmask_hi02), vnmask_hi02);

      const __m128i vq31prod_lo02 = _mm_srli_epi64(_mm_add_epi64(vprod_lo02, vrounding), 31);
      const __m128i vq31prod_hi02 = _mm_srli_epi64(_mm_add_epi64(vprod_hi02, vrounding), 31);

      const __m128i vabsprod_lo13 = _mm_mul_epu32(vabsacc_lo1032, vmultiplier);
      const __m128i vabsprod_hi13 = _mm_mul_epu32(vabsacc_hi1032, vmultiplier);

      const __m128i vnmask_lo13 = _mm_shuffle_epi32(vnmask_lo0123, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vnmask_hi13 = _mm_shuffle_epi32(vnmask_hi0123, _MM_SHUFFLE(3, 3, 1, 1));

      const __m128i vprod_lo13 = _mm_sub_epi64(_mm_xor_si128(vabsprod_lo13, vnmask_lo13), vnmask_lo13);
      const __m128i vprod_hi13 = _mm_sub_epi64(_mm_xor_si128(vabsprod_hi13, vnmask_hi13), vnmask_hi13);

      const __m128i vq31prod_lo13 = _mm_srli_epi64(_mm_add_epi64(vprod_lo13, vrounding), 31);
      const __m128i vq31prod_hi13 = _mm_srli_epi64(_mm_add_epi64(vprod_hi13, vrounding), 31);

      const __m128i vq31prod_lo0213 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prod_lo02), _mm_castsi128_ps(vq31prod_lo13), _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128i vq31prod_hi0213 = _mm_castps_si128(_mm_shuffle_ps(
          _mm_castsi128_ps(vq31prod_hi02), _mm_castsi128_ps(vq31prod_hi13), _MM_SHUFFLE(2, 0, 2, 0)));

      const __m128i vq31prod_lo0123 = _mm_shuffle_epi32(vq31prod_lo0213, _MM_SHUFFLE(3, 1, 2, 0));
      const __m128i vq31prod_hi0123 = _mm_shuffle_epi32(vq31prod_hi0213, _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->sse2.remainder_mask);

      const __m128i vrem_lo0123 =
        _mm_add_epi32(_mm_and_si128(vq31prod_lo0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod_lo0123));
      const __m128i vrem_hi0123 =
        _mm_add_epi32(_mm_and_si128(vq31prod_hi0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod_hi0123));

      const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->sse2.remainder_threshold);
      const __m128i vshift = _mm_load_si128((const __m128i*) params->sse2.shift);

      const __m128i vout_lo = _mm_sub_epi32(_mm_sra_epi32(vq31prod_lo0123, vshift), _mm_cmpgt_epi32(vrem_lo0123, vremainder_threshold));
      const __m128i vout_hi = _mm_sub_epi32(_mm_sra_epi32(vq31prod_hi0123, vshift), _mm_cmpgt_epi32(vrem_hi0123, vremainder_threshold));

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
      __m128i vout = _mm_adds_epi16(_mm_packs_epi32(vout_lo, vout_hi), voutput_zero_point);
      vout = _mm_packus_epi16(vout, vout);
      vout = _mm_min_epu8(vout, _mm_load_si128((const __m128i*) params->sse2.output_max));
      vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->sse2.output_min));

      if (c & 4) {
        *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout);
        output += 4;
        vout = _mm_srli_epi64(vout, 32);
      }
      if (c & 2) {
        *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout, 0);
        output += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (c & 1) {
        *((uint8_t*) output) = (uint8_t) _mm_cvtsi128_si32(vout);
        output += 1;
      }
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
