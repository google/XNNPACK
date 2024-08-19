// Auto-generated file. Do not edit!
//   Template: src/f32-velu/sse-rr2-lut16-p3.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/vunary.h"
#include "xnnpack/common.h"


extern XNN_INTERNAL const float xnn_table_exp2minus_k_over_16[16];

void xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_u12(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_elu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128 vsat_cutoff = _mm_set1_ps(-0x1.154246p+4f);
  const __m128 vmagic_bias = _mm_set1_ps(0x1.800000p19f);
  const __m128 vlog2e = _mm_set1_ps(0x1.715476p+0f);
  const __m128i vindex_mask = _mm_set1_epi32(UINT32_C(0xF));
  const __m128 vminus_ln2_hi = _mm_set1_ps(-0x1.62E400p-1f);
  const __m128 vminus_ln2_lo = _mm_set1_ps(-0x1.7F7D1Cp-20f);
  const __m128 vc3 = _mm_set1_ps(0x1.55561Cp-3f);
  const __m128 vc2 = _mm_set1_ps(0x1.0001ECp-1f);
  const __m128 vone = _mm_set1_ps(1.0f);

  XNN_FORCE_REALIZATION(vsat_cutoff);
  XNN_FORCE_REALIZATION(vmagic_bias);
  XNN_FORCE_REALIZATION(vlog2e);
  XNN_FORCE_REALIZATION(vindex_mask);
  XNN_FORCE_REALIZATION(vminus_ln2_hi);
  XNN_FORCE_REALIZATION(vminus_ln2_lo);
  XNN_FORCE_REALIZATION(vc3);
  XNN_FORCE_REALIZATION(vc2);
  XNN_FORCE_REALIZATION(vone);

  const __m128 vprescale = _mm_set1_ps(params->scalar.prescale);
  const __m128 valpha = _mm_set1_ps(params->scalar.alpha);
  const __m128 vbeta = _mm_set1_ps(params->scalar.beta);

  for (; batch >= 12 * sizeof(float); batch -= 12 * sizeof(float)) {
    __m128 vx0123 = _mm_loadu_ps(input);
    __m128 vx4567 = _mm_loadu_ps(input + 4);
    __m128 vx89AB = _mm_loadu_ps(input + 8);
    input += 12;

    const __m128 vz0123 = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx0123, vprescale));
    const __m128 vz4567 = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx4567, vprescale));
    const __m128 vz89AB = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx89AB, vprescale));

    __m128 vn0123 = _mm_add_ps(_mm_mul_ps(vz0123, vlog2e), vmagic_bias);
    __m128 vn4567 = _mm_add_ps(_mm_mul_ps(vz4567, vlog2e), vmagic_bias);
    __m128 vn89AB = _mm_add_ps(_mm_mul_ps(vz89AB, vlog2e), vmagic_bias);

    const __m128i vidx0123 = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn0123), vindex_mask), 2);
    const __m128i ven0123 = _mm_slli_epi32(_mm_castps_si128(vn0123), 19);
    const __m128i vidx4567 = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn4567), vindex_mask), 2);
    const __m128i ven4567 = _mm_slli_epi32(_mm_castps_si128(vn4567), 19);
    const __m128i vidx89AB = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn89AB), vindex_mask), 2);
    const __m128i ven89AB = _mm_slli_epi32(_mm_castps_si128(vn89AB), 19);

    #if XNN_ARCH_X86_64
      const uint64_t vidx01 = (uint64_t) _mm_cvtsi128_si64(vidx0123);
      const uint64_t vidx23 = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx0123, vidx0123));
      const __m128i vl0   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx01)));
      const __m128i vl2 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx23)));
      const __m128i vl1 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx01 >> 32))));
      const __m128i vl01 = _mm_unpacklo_epi32(vl0, vl1);
      const __m128i vl3 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx23 >> 32))));
      const __m128i vl23 = _mm_unpacklo_epi32(vl2, vl3);
      const __m128i vl0123 = _mm_unpacklo_epi64(vl01, vl23);
      const uint64_t vidx45 = (uint64_t) _mm_cvtsi128_si64(vidx4567);
      const uint64_t vidx67 = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx4567, vidx4567));
      const __m128i vl4   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx45)));
      const __m128i vl6 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx67)));
      const __m128i vl5 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx45 >> 32))));
      const __m128i vl45 = _mm_unpacklo_epi32(vl4, vl5);
      const __m128i vl7 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx67 >> 32))));
      const __m128i vl67 = _mm_unpacklo_epi32(vl6, vl7);
      const __m128i vl4567 = _mm_unpacklo_epi64(vl45, vl67);
      const uint64_t vidx89 = (uint64_t) _mm_cvtsi128_si64(vidx89AB);
      const uint64_t vidxAB = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx89AB, vidx89AB));
      const __m128i vl8   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx89)));
      const __m128i vlA = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidxAB)));
      const __m128i vl9 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx89 >> 32))));
      const __m128i vl89 = _mm_unpacklo_epi32(vl8, vl9);
      const __m128i vlB = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidxAB >> 32))));
      const __m128i vlAB = _mm_unpacklo_epi32(vlA, vlB);
      const __m128i vl89AB = _mm_unpacklo_epi64(vl89, vlAB);
    #else  // !XNN_ARCH_X86_64
      const uint32_t vidx0 = (uint32_t) _mm_cvtsi128_si32(vidx0123);
      const uint32_t vidx1 = (uint32_t) _mm_extract_epi16(vidx0123, 2);
      const uint32_t vidx2 = (uint32_t) _mm_extract_epi16(vidx0123, 4);
      const uint32_t vidx3 = (uint32_t) _mm_extract_epi16(vidx0123, 6);
      const __m128i vl0   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx0)));
      const __m128i vl2 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx2)));
      const __m128i vl1 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx1)));
      const __m128i vl01 = _mm_unpacklo_epi32(vl0, vl1);
      const __m128i vl3 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx3)));
      const __m128i vl23 = _mm_unpacklo_epi32(vl2, vl3);
      const __m128i vl0123 = _mm_unpacklo_epi64(vl01, vl23);
      const uint32_t vidx4 = (uint32_t) _mm_cvtsi128_si32(vidx4567);
      const uint32_t vidx5 = (uint32_t) _mm_extract_epi16(vidx4567, 2);
      const uint32_t vidx6 = (uint32_t) _mm_extract_epi16(vidx4567, 4);
      const uint32_t vidx7 = (uint32_t) _mm_extract_epi16(vidx4567, 6);
      const __m128i vl4   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx4)));
      const __m128i vl6 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx6)));
      const __m128i vl5 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx5)));
      const __m128i vl45 = _mm_unpacklo_epi32(vl4, vl5);
      const __m128i vl7 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx7)));
      const __m128i vl67 = _mm_unpacklo_epi32(vl6, vl7);
      const __m128i vl4567 = _mm_unpacklo_epi64(vl45, vl67);
      const uint32_t vidx8 = (uint32_t) _mm_cvtsi128_si32(vidx89AB);
      const uint32_t vidx9 = (uint32_t) _mm_extract_epi16(vidx89AB, 2);
      const uint32_t vidxA = (uint32_t) _mm_extract_epi16(vidx89AB, 4);
      const uint32_t vidxB = (uint32_t) _mm_extract_epi16(vidx89AB, 6);
      const __m128i vl8   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx8)));
      const __m128i vlA = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidxA)));
      const __m128i vl9 = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidx9)));
      const __m128i vl89 = _mm_unpacklo_epi32(vl8, vl9);
      const __m128i vlB = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + vidxB)));
      const __m128i vlAB = _mm_unpacklo_epi32(vlA, vlB);
      const __m128i vl89AB = _mm_unpacklo_epi64(vl89, vlAB);
    #endif  // XNN_ARCH_X86_64

    vn0123 = _mm_sub_ps(vn0123, vmagic_bias);
    __m128 vs0123 = _mm_castsi128_ps(_mm_add_epi32(vl0123, ven0123));
    vn4567 = _mm_sub_ps(vn4567, vmagic_bias);
    __m128 vs4567 = _mm_castsi128_ps(_mm_add_epi32(vl4567, ven4567));
    vn89AB = _mm_sub_ps(vn89AB, vmagic_bias);
    __m128 vs89AB = _mm_castsi128_ps(_mm_add_epi32(vl89AB, ven89AB));

    __m128 vt0123 = _mm_add_ps(_mm_mul_ps(vn0123, vminus_ln2_hi), vz0123);
    __m128 vt4567 = _mm_add_ps(_mm_mul_ps(vn4567, vminus_ln2_hi), vz4567);
    __m128 vt89AB = _mm_add_ps(_mm_mul_ps(vn89AB, vminus_ln2_hi), vz89AB);

    vt0123 = _mm_add_ps(_mm_mul_ps(vn0123, vminus_ln2_lo), vt0123);
    vt4567 = _mm_add_ps(_mm_mul_ps(vn4567, vminus_ln2_lo), vt4567);
    vt89AB = _mm_add_ps(_mm_mul_ps(vn89AB, vminus_ln2_lo), vt89AB);

    __m128 vp0123 = _mm_add_ps(_mm_mul_ps(vc3, vt0123), vc2);
    __m128 vp4567 = _mm_add_ps(_mm_mul_ps(vc3, vt4567), vc2);
    __m128 vp89AB = _mm_add_ps(_mm_mul_ps(vc3, vt89AB), vc2);

    vp0123 = _mm_mul_ps(vp0123, vt0123);
    vp4567 = _mm_mul_ps(vp4567, vt4567);
    vp89AB = _mm_mul_ps(vp89AB, vt89AB);

    vt0123 = _mm_mul_ps(vt0123, vs0123);
    vs0123 = _mm_sub_ps(vs0123, vone);
    vt4567 = _mm_mul_ps(vt4567, vs4567);
    vs4567 = _mm_sub_ps(vs4567, vone);
    vt89AB = _mm_mul_ps(vt89AB, vs89AB);
    vs89AB = _mm_sub_ps(vs89AB, vone);

    vp0123 = _mm_add_ps(_mm_mul_ps(vp0123, vt0123), vt0123);
    vp4567 = _mm_add_ps(_mm_mul_ps(vp4567, vt4567), vt4567);
    vp89AB = _mm_add_ps(_mm_mul_ps(vp89AB, vt89AB), vt89AB);

    const __m128 ve0123 = _mm_mul_ps(_mm_add_ps(vp0123, vs0123), valpha);
    const __m128 ve4567 = _mm_mul_ps(_mm_add_ps(vp4567, vs4567), valpha);
    const __m128 ve89AB = _mm_mul_ps(_mm_add_ps(vp89AB, vs89AB), valpha);

    const __m128 vm0123 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx0123)));
    vx0123 = _mm_mul_ps(vx0123, vbeta);
    const __m128 vm4567 = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx4567)));
    vx4567 = _mm_mul_ps(vx4567, vbeta);
    const __m128 vm89AB = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx89AB)));
    vx89AB = _mm_mul_ps(vx89AB, vbeta);

    const __m128 vy0123 = _mm_or_ps(_mm_and_ps(ve0123, vm0123), _mm_andnot_ps(vm0123, vx0123));
    const __m128 vy4567 = _mm_or_ps(_mm_and_ps(ve4567, vm4567), _mm_andnot_ps(vm4567, vx4567));
    const __m128 vy89AB = _mm_or_ps(_mm_and_ps(ve89AB, vm89AB), _mm_andnot_ps(vm89AB, vx89AB));

    _mm_storeu_ps(output, vy0123);
    _mm_storeu_ps(output + 4, vy4567);
    _mm_storeu_ps(output + 8, vy89AB);
    output += 12;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    __m128 vx = _mm_loadu_ps(input);
    input += 4;

    const __m128 vz = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx, vprescale));

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128i ven = _mm_slli_epi32(_mm_castps_si128(vn), 19);
    const __m128i vidx = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn), vindex_mask), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx, vidx));
      const __m128i vl_ll   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lo)));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hi)));
      const __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lo >> 32))));
      const __m128i vl_lo = _mm_unpacklo_epi32(vl_ll, vl_lh);
      const __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hi >> 32))));
      const __m128i vl_hi = _mm_unpacklo_epi32(vl_hl, vl_hh);
    #else  // !XNN_ARCH_X86_64
      const __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx))));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 4))));
      const __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 2))));
      const __m128i vl_lo = _mm_unpacklo_epi32(vl_ll, vl_lh);
      const __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 6))));
      const __m128i vl_hi = _mm_unpacklo_epi32(vl_hl, vl_hh);
    #endif  // XNN_ARCH_X86_64
    const __m128i vl = _mm_unpacklo_epi64(vl_lo, vl_hi);
    __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ven));
    vn = _mm_sub_ps(vn, vmagic_bias);

    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc3, vt), vc2);
    vp = _mm_mul_ps(vp, vt);

    vt = _mm_mul_ps(vt, vs);
    vs = _mm_sub_ps(vs, vone);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vt);
    const __m128 ve = _mm_mul_ps(_mm_add_ps(vp, vs), valpha);

    const __m128 vm = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx)));
    vx = _mm_mul_ps(vx, vbeta);
    const __m128 vy = _mm_or_ps(_mm_and_ps(ve, vm), _mm_andnot_ps(vm, vx));

    _mm_storeu_ps(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128 vx = _mm_loadu_ps(input);

    const __m128 vz = _mm_max_ps(vsat_cutoff, _mm_mul_ps(vx, vprescale));

    __m128 vn = _mm_add_ps(_mm_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128i ven = _mm_slli_epi32(_mm_castps_si128(vn), 19);
    const __m128i vidx = _mm_slli_epi32(_mm_and_si128(_mm_castps_si128(vn), vindex_mask), 2);
    #if XNN_ARCH_X86_64
      const uint64_t vidx_lo = (uint64_t) _mm_cvtsi128_si64(vidx);
      const uint64_t vidx_hi = (uint64_t) _mm_cvtsi128_si64(_mm_unpackhi_epi64(vidx, vidx));
      const __m128i vl_ll   = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_lo)));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) vidx_hi)));
      const __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_lo >> 32))));
      const __m128i vl_lo = _mm_unpacklo_epi32(vl_ll, vl_lh);
      const __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) (vidx_hi >> 32))));
      const __m128i vl_hi = _mm_unpacklo_epi32(vl_hl, vl_hh);
    #else  // !XNN_ARCH_X86_64
      const __m128i vl_ll = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_cvtsi128_si32(vidx))));
      const __m128i vl_hl = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 4))));
      const __m128i vl_lh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 2))));
      const __m128i vl_lo = _mm_unpacklo_epi32(vl_ll, vl_lh);
      const __m128i vl_hh = _mm_cvtsi32_si128(*((const int*) ((uintptr_t) xnn_table_exp2minus_k_over_16 + (uint32_t) _mm_extract_epi16(vidx, 6))));
      const __m128i vl_hi = _mm_unpacklo_epi32(vl_hl, vl_hh);
    #endif  // XNN_ARCH_X86_64
    const __m128i vl = _mm_unpacklo_epi64(vl_lo, vl_hi);
    __m128 vs = _mm_castsi128_ps(_mm_add_epi32(vl, ven));
    vn = _mm_sub_ps(vn, vmagic_bias);

    __m128 vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_hi), vz);
    vt = _mm_add_ps(_mm_mul_ps(vn, vminus_ln2_lo), vt);

    __m128 vp = _mm_add_ps(_mm_mul_ps(vc3, vt), vc2);
    vp = _mm_mul_ps(vp, vt);

    vt = _mm_mul_ps(vt, vs);
    vs = _mm_sub_ps(vs, vone);
    vp = _mm_add_ps(_mm_mul_ps(vp, vt), vt);
    const __m128 ve = _mm_mul_ps(_mm_add_ps(vp, vs), valpha);

    const __m128 vm = _mm_castsi128_ps(_mm_cmpgt_epi32(_mm_setzero_si128(), _mm_castps_si128(vx)));
    vx = _mm_mul_ps(vx, vbeta);
    __m128 vy = _mm_or_ps(_mm_and_ps(ve, vm), _mm_andnot_ps(vm, vx));

    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy);
      vy = _mm_movehl_ps(vy, vy);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy);
    }
  }
}
