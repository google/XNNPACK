// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/gemm.h>
#include <xnnpack/math.h>


static inline __m128i sse_reduce4_i32(__m128i x, __m128i y, __m128i z, __m128i w) {
#if defined(__SSSE3__) && !defined(__ANDROID__)
  // xxyy = ( y2 + y3, y0 + y1, x2 + x3, x0 + x1 )
  const __m128i xxyy = _mm_hadd_epi32(x, y);
  // zzww = ( w2 + w3, w0 + w1, z2 + z3, z0 + z1 )
  const __m128i zzww = _mm_hadd_epi32(z, w);
  // xyzw = ( w0 + w1 + w2 + w3, y0 + y1 + y2 + y3, z0 + z1 + z2 + z3, x0 + x1 + x2 + x3 )
  return _mm_hadd_epi32(xxyy, zzww);
#else
  // xzxz = ( z1 + z3, x1 + x3, z0 + z2, x0 + x2 )
  const __m128i xzxz = _mm_add_epi32(_mm_unpacklo_epi32(x, z), _mm_unpackhi_epi32(x, z));
  // ywyw = ( w1 + w3, y1 + y3, w0 + w2, y0 + y2 )
  const __m128i ywyw = _mm_add_epi32(_mm_unpacklo_epi32(y, w), _mm_unpackhi_epi32(y, w));
  // xyzw = ( w0 + w2 + w1 + w3, y0 + y2 + y1 + y3, z0 + z2 + z1 + z3, x0 + x2 + x1 + x3 )
  return _mm_add_epi32(_mm_unpacklo_epi32(xzxz, ywyw), _mm_unpackhi_epi32(xzxz, ywyw));
#endif
}

void xnn_qu8_gemm_minmax_ukernel_2x4c8__sse2(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_gemm_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);

  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if (mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  const size_t kc_stride = round_up_po2(kc, 8);
  const __m128i vb_zero_point = _mm_load_si128((const __m128i*) params->sse2.kernel_zero_point);

  do {
    __m128i vacc00 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[0]);
    __m128i vacc01 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[1]);
    __m128i vacc02 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[2]);
    __m128i vacc03 = _mm_cvtsi32_si128((int) ((const int32_t*) w)[3]);
    __m128i vacc10 = vacc00;
    __m128i vacc11 = vacc01;
    __m128i vacc12 = vacc02;
    __m128i vacc13 = vacc03;
    w = (const void*) ((uintptr_t) w + 16);

    const __m128i vzero = _mm_setzero_si128();
    for (size_t k = 0; k < kc; k += 8 * sizeof(uint8_t)) {
      const __m128i va0 = _mm_loadl_epi64((const __m128i*) a0);
      const __m128i vxa0 = _mm_unpacklo_epi8(va0, vzero);
      a0 += 8;
      const __m128i va1 = _mm_loadl_epi64((const __m128i*) a1);
      const __m128i vxa1 = _mm_unpacklo_epi8(va1, vzero);
      a1 += 8;

      const __m128i vb0 = _mm_loadl_epi64((const __m128i*) w);
      const __m128i vxb0 = _mm_sub_epi16(_mm_unpacklo_epi8(vb0, vzero), vb_zero_point);
      const __m128i vb1 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 8));
      const __m128i vxb1 = _mm_sub_epi16(_mm_unpacklo_epi8(vb1, vzero), vb_zero_point);
      const __m128i vb2 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 16));
      const __m128i vxb2 = _mm_sub_epi16(_mm_unpacklo_epi8(vb2, vzero), vb_zero_point);
      const __m128i vb3 = _mm_loadl_epi64((const __m128i*) ((uintptr_t) w + 24));
      const __m128i vxb3 = _mm_sub_epi16(_mm_unpacklo_epi8(vb3, vzero), vb_zero_point);
      w = (const void*) ((uintptr_t) w + 32);

      vacc00 = _mm_add_epi32(vacc00, _mm_madd_epi16(vxa0, vxb0));
      vacc01 = _mm_add_epi32(vacc01, _mm_madd_epi16(vxa0, vxb1));
      vacc02 = _mm_add_epi32(vacc02, _mm_madd_epi16(vxa0, vxb2));
      vacc03 = _mm_add_epi32(vacc03, _mm_madd_epi16(vxa0, vxb3));
      vacc10 = _mm_add_epi32(vacc10, _mm_madd_epi16(vxa1, vxb0));
      vacc11 = _mm_add_epi32(vacc11, _mm_madd_epi16(vxa1, vxb1));
      vacc12 = _mm_add_epi32(vacc12, _mm_madd_epi16(vxa1, vxb2));
      vacc13 = _mm_add_epi32(vacc13, _mm_madd_epi16(vxa1, vxb3));
    }

    __m128i vacc0x0123 = sse_reduce4_i32(vacc00, vacc01, vacc02, vacc03);
    __m128i vacc1x0123 = sse_reduce4_i32(vacc10, vacc11, vacc12, vacc13);

    const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->sse2.multiplier);
    const __m128i vrounding = _mm_load_si128((const __m128i*) params->sse2.rounding);

    const __m128i vnmask0x0123 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc0x0123);
    const __m128i vnmask1x0123 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc1x0123);

    const __m128i vabsacc0x0123 = _mm_sub_epi32(_mm_xor_si128(vacc0x0123, vnmask0x0123), vnmask0x0123);
    const __m128i vabsacc1x0123 = _mm_sub_epi32(_mm_xor_si128(vacc1x0123, vnmask1x0123), vnmask1x0123);

    const __m128i vabsacc0x1032 = _mm_shuffle_epi32(vabsacc0x0123, _MM_SHUFFLE(2, 3, 0, 1));
    const __m128i vabsacc1x1032 = _mm_shuffle_epi32(vabsacc1x0123, _MM_SHUFFLE(2, 3, 0, 1));

    const __m128i vabsprod0x02 = _mm_mul_epu32(vabsacc0x0123, vmultiplier);
    const __m128i vabsprod1x02 = _mm_mul_epu32(vabsacc1x0123, vmultiplier);

    const __m128i vnmask0x02 = _mm_shuffle_epi32(vnmask0x0123, _MM_SHUFFLE(2, 2, 0, 0));
    const __m128i vnmask1x02 = _mm_shuffle_epi32(vnmask1x0123, _MM_SHUFFLE(2, 2, 0, 0));

    const __m128i vprod0x02 = _mm_sub_epi64(_mm_xor_si128(vabsprod0x02, vnmask0x02), vnmask0x02);
    const __m128i vprod1x02 = _mm_sub_epi64(_mm_xor_si128(vabsprod1x02, vnmask1x02), vnmask1x02);

    const __m128i vq31prod0x02 = _mm_srli_epi64(_mm_add_epi64(vprod0x02, vrounding), 31);
    const __m128i vq31prod1x02 = _mm_srli_epi64(_mm_add_epi64(vprod1x02, vrounding), 31);

    const __m128i vabsprod0x13 = _mm_mul_epu32(vabsacc0x1032, vmultiplier);
    const __m128i vabsprod1x13 = _mm_mul_epu32(vabsacc1x1032, vmultiplier);

    const __m128i vnmask0x13 = _mm_shuffle_epi32(vnmask0x0123, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128i vnmask1x13 = _mm_shuffle_epi32(vnmask1x0123, _MM_SHUFFLE(3, 3, 1, 1));

    const __m128i vprod0x13 = _mm_sub_epi64(_mm_xor_si128(vabsprod0x13, vnmask0x13), vnmask0x13);
    const __m128i vprod1x13 = _mm_sub_epi64(_mm_xor_si128(vabsprod1x13, vnmask1x13), vnmask1x13);

    const __m128i vq31prod0x13 = _mm_srli_epi64(_mm_add_epi64(vprod0x13, vrounding), 31);
    const __m128i vq31prod1x13 = _mm_srli_epi64(_mm_add_epi64(vprod1x13, vrounding), 31);

    const __m128i vq31prod0x0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(vq31prod0x02), _mm_castsi128_ps(vq31prod0x13), _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i vq31prod1x0213 = _mm_castps_si128(_mm_shuffle_ps(
        _mm_castsi128_ps(vq31prod1x02), _mm_castsi128_ps(vq31prod1x13), _MM_SHUFFLE(2, 0, 2, 0)));

    const __m128i vq31prod0x0123 = _mm_shuffle_epi32(vq31prod0x0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i vq31prod1x0123 = _mm_shuffle_epi32(vq31prod1x0213, _MM_SHUFFLE(3, 1, 2, 0));

    const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->sse2.remainder_mask);

    const __m128i vrem0x0123 =
      _mm_add_epi32(_mm_and_si128(vq31prod0x0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod0x0123));
    const __m128i vrem1x0123 =
      _mm_add_epi32(_mm_and_si128(vq31prod1x0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod1x0123));

    const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->sse2.remainder_threshold);
    const __m128i vshift = _mm_load_si128((const __m128i*) params->sse2.shift);

    vacc0x0123 =
      _mm_sub_epi32(_mm_sra_epi32(vq31prod0x0123, vshift), _mm_cmpgt_epi32(vrem0x0123, vremainder_threshold));
    vacc1x0123 =
      _mm_sub_epi32(_mm_sra_epi32(vq31prod1x0123, vshift), _mm_cmpgt_epi32(vrem1x0123, vremainder_threshold));

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
    const __m128i vacc01x0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0x0123, vacc1x0123), voutput_zero_point);
    __m128i vout = _mm_packus_epi16(vacc01x0123, vacc01x0123);
    vout = _mm_min_epu8(vout, _mm_load_si128((const __m128i*) params->sse2.output_max));
    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->sse2.output_min));

    if (nc >= 4) {
      *((uint32_t*) c0) = (uint32_t) _mm_cvtsi128_si32(vout);
      *((uint32_t*) c1) = (uint32_t) _mm_cvtsi128_si32(_mm_srli_epi64(vout, 32));

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc_stride);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc_stride);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        *((uint16_t*) c0) = (uint16_t) _mm_extract_epi16(vout, 0);
        c0 += 2;
        *((uint16_t*) c1) = (uint16_t) _mm_extract_epi16(vout, 2);
        c1 += 2;
        vout = _mm_srli_epi32(vout, 16);
      }
      if (nc & 1) {
        *((uint8_t*) c0) = (uint8_t) _mm_cvtsi128_si32(vout);
        *((uint8_t*) c1) = (uint8_t) _mm_extract_epi16(vout, 2);
      }

      nc = 0;
    }
  } while (nc != 0);
}
