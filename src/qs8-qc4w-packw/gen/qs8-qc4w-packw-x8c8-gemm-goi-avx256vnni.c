// Auto-generated file. Do not edit!
//   Template: src/x8-packw/kr-avxvnni.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "xnnpack/packw.h"
#include "xnnpack/unaligned.h"

XNN_INLINE static uint64_t safe_load_u64(const void* address, size_t n) {
  uint64_t value = 0;
  assert(n <= sizeof(uint64_t));
  const uint8_t* bytes = (const uint8_t*) address;
  for (size_t i = 0; i < n; ++i) {
    value |= (uint64_t) bytes[i] << (i * 8);
  }
  return value;
}


void xnn_qs8_qc4w_packw_gemm_goi_ukernel_x8c8__avx256vnni(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint8_t* weights,
  const int32_t* bias,
  const float* scale,
  void* packed_weights,
  size_t extra_bytes,
  const struct xnn_qs8_qc4w_packing_params* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 8);
  assert(kr == 8);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  assert(params != NULL);

  // Use scalar pack if not an even block size
  // TODO: Support NC remainder
  if ((kc & 1) || (nc & 7)) {
    xnn_qs8_qc4w_packw_gemm_goi_ukernel_x8c8__scalar(
      g, nc, kc, nr, kr, sr,
      weights, bias, scale, packed_weights, extra_bytes, params);
    return;
  }
  kc = (kc + 1) / 2;

  int8_t* out = (int8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;

  const __m256i vone = _mm256_set1_epi8(1);
  const __m256i vmask = _mm256_set1_epi8(0xF0);
  const __m256i vzeropoint = _mm256_set1_epi32((int32_t) params->input_zero_point + 0);
  const __m256i vkernel_zero_point = _mm256_set1_epi32((uint32_t) params->kernel_zero_point * 0x11111111);
  assert(params->kernel_zero_point == 8 || params->kernel_zero_point == 0);
  static const uint8_t perm0[32] = {
    0, -1,  1, -1,  2, -1,  3, -1,
    8, -1,  9, -1, 10, -1, 11, -1,
    0, -1,  1, -1,  2, -1,  3, -1,
    8, -1,  9, -1, 10, -1, 11, -1 };

  static const uint8_t perm1[32] = {
    -1,  0, -1,  1, -1,  2, -1,  3,
    -1,  8, -1,  9, -1, 10, -1, 11,
    -1,  0, -1,  1, -1,  2, -1,  3,
    -1,  8, -1,  9, -1, 10, -1, 11 };

  static const uint8_t perm2[32] = {
     4, -1,  5, -1,  6, -1,  7, -1,
    12, -1, 13, -1, 14, -1, 15, -1,
     4, -1,  5, -1,  6, -1,  7, -1,
    12, -1, 13, -1, 14, -1, 15, -1 };

  static const uint8_t perm3[32] = {
    -1,  4, -1,  5, -1,  6, -1,  7,
    -1, 12, -1, 13, -1, 14, -1, 15,
    -1,  4, -1,  5, -1,  6, -1,  7,
    -1, 12, -1, 13, -1, 14, -1, 15 };
  const __m256i vperm0 = _mm256_load_si256((const __m256i*) perm0);
  const __m256i vperm1 = _mm256_load_si256((const __m256i*) perm1);
  const __m256i vperm2 = _mm256_load_si256((const __m256i*) perm2);
  const __m256i vperm3 = _mm256_load_si256((const __m256i*) perm3);

  do {
    // NC main loop multiple of 8
    const int8_t* w0 = (const int8_t*) weights;
    size_t n = nc;
    for (;n >= 8; n -= 8) {
      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        const __m256i vb0 = _mm256_loadu_si256((const __m256i*) (b + 0));
        _mm256_storeu_si256((__m256i*) (out + 0), vb0);
        b += 8;
      } else {
        _mm256_storeu_si256((__m256i*) (out + 0), _mm256_setzero_si256());
      }
      out += 8 * sizeof(int32_t);

      const int8_t* w1 = w0 + kc;
      const int8_t* w2 = w1 + kc;
      const int8_t* w3 = w2 + kc;
      const int8_t* w4 = w3 + kc;
      const int8_t* w5 = w4 + kc;
      const int8_t* w6 = w5 + kc;
      const int8_t* w7 = w6 + kc;

      __m256i vacc0 = _mm256_setzero_si256();
      __m256i vacc4 = _mm256_setzero_si256();

      size_t k = kc;
      // KC main loop multiple of 8x32
      for (; k >= 32; k -= 32) {
        const __m256i v0_0123 = _mm256_loadu_si256((const __m256i*) w0);
        const __m256i v1_0123 = _mm256_loadu_si256((const __m256i*) w1);
        const __m256i v2_0123 = _mm256_loadu_si256((const __m256i*) w2);
        const __m256i v3_0123 = _mm256_loadu_si256((const __m256i*) w3);
        const __m256i v4_0123 = _mm256_loadu_si256((const __m256i*) w4);
        const __m256i v5_0123 = _mm256_loadu_si256((const __m256i*) w5);
        const __m256i v6_0123 = _mm256_loadu_si256((const __m256i*) w6);
        const __m256i v7_0123 = _mm256_loadu_si256((const __m256i*) w7);

        const __m256i v01_02 = _mm256_unpacklo_epi64(v0_0123, v1_0123);
        const __m256i v01_13 = _mm256_unpackhi_epi64(v0_0123, v1_0123);
        const __m256i v23_02 = _mm256_unpacklo_epi64(v2_0123, v3_0123);
        const __m256i v23_13 = _mm256_unpackhi_epi64(v2_0123, v3_0123);
        const __m256i v45_02 = _mm256_unpacklo_epi64(v4_0123, v5_0123);
        const __m256i v45_13 = _mm256_unpackhi_epi64(v4_0123, v5_0123);
        const __m256i v67_02 = _mm256_unpacklo_epi64(v6_0123, v7_0123);
        const __m256i v67_13 = _mm256_unpackhi_epi64(v6_0123, v7_0123);


        __m256i v0_0 = _mm256_permute2f128_si256(v01_02, v23_02, _MM_SHUFFLE(0, 2, 0, 0));
        __m256i v0_1 = _mm256_permute2f128_si256(v01_13, v23_13, _MM_SHUFFLE(0, 2, 0, 0));
        __m256i v0_2 = _mm256_permute2f128_si256(v01_02, v23_02, _MM_SHUFFLE(0, 3, 0, 1));
        __m256i v0_3 = _mm256_permute2f128_si256(v01_13, v23_13, _MM_SHUFFLE(0, 3, 0, 1));
        __m256i v4_0 = _mm256_permute2f128_si256(v45_02, v67_02, _MM_SHUFFLE(0, 2, 0, 0));
        __m256i v4_1 = _mm256_permute2f128_si256(v45_13, v67_13, _MM_SHUFFLE(0, 2, 0, 0));
        __m256i v4_2 = _mm256_permute2f128_si256(v45_02, v67_02, _MM_SHUFFLE(0, 3, 0, 1));
        __m256i v4_3 = _mm256_permute2f128_si256(v45_13, v67_13, _MM_SHUFFLE(0, 3, 0, 1));

        v0_0 = _mm256_xor_si256(v0_0, vkernel_zero_point);    // uint4 -> int4
        const __m256i vt0_0 = _mm256_slli_epi32(v0_0, 4);     // isolate lower int4
        const __m256i vh0_0 = _mm256_and_si256(v0_0, vmask);  // isolate upper int4
        const __m256i vl0_0 = _mm256_and_si256(vt0_0, vmask);
        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, vh0_0);
        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, vl0_0);
        const __m256i v0x0_0 = _mm256_shuffle_epi8(vl0_0, vperm0);  // 0,2,4,6
        const __m256i v1x0_0 = _mm256_shuffle_epi8(vh0_0, vperm1);  // 1,3,5,7
        const __m256i v01x0_0 = _mm256_or_si256(v0x0_0, v1x0_0);
        const __m256i v2x0_0 = _mm256_shuffle_epi8(vl0_0, vperm2);  // 8,A,C,E
        const __m256i v3x0_0 = _mm256_shuffle_epi8(vh0_0, vperm3);  // 9,B,D,F
        const __m256i v23x0_0 = _mm256_or_si256(v2x0_0, v3x0_0);
        const __m256i vt010_0 = _mm256_srli_epi32(v01x0_0, 4);  // first plane 0-7
        v0_0 = _mm256_or_si256(vt010_0, v23x0_0);            // + second plane 8-F
        v0_1 = _mm256_xor_si256(v0_1, vkernel_zero_point);    // uint4 -> int4
        const __m256i vt0_1 = _mm256_slli_epi32(v0_1, 4);     // isolate lower int4
        const __m256i vh0_1 = _mm256_and_si256(v0_1, vmask);  // isolate upper int4
        const __m256i vl0_1 = _mm256_and_si256(vt0_1, vmask);
        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, vh0_1);
        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, vl0_1);
        const __m256i v0x0_1 = _mm256_shuffle_epi8(vl0_1, vperm0);  // 0,2,4,6
        const __m256i v1x0_1 = _mm256_shuffle_epi8(vh0_1, vperm1);  // 1,3,5,7
        const __m256i v01x0_1 = _mm256_or_si256(v0x0_1, v1x0_1);
        const __m256i v2x0_1 = _mm256_shuffle_epi8(vl0_1, vperm2);  // 8,A,C,E
        const __m256i v3x0_1 = _mm256_shuffle_epi8(vh0_1, vperm3);  // 9,B,D,F
        const __m256i v23x0_1 = _mm256_or_si256(v2x0_1, v3x0_1);
        const __m256i vt010_1 = _mm256_srli_epi32(v01x0_1, 4);  // first plane 0-7
        v0_1 = _mm256_or_si256(vt010_1, v23x0_1);            // + second plane 8-F
        v0_2 = _mm256_xor_si256(v0_2, vkernel_zero_point);    // uint4 -> int4
        const __m256i vt0_2 = _mm256_slli_epi32(v0_2, 4);     // isolate lower int4
        const __m256i vh0_2 = _mm256_and_si256(v0_2, vmask);  // isolate upper int4
        const __m256i vl0_2 = _mm256_and_si256(vt0_2, vmask);
        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, vh0_2);
        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, vl0_2);
        const __m256i v0x0_2 = _mm256_shuffle_epi8(vl0_2, vperm0);  // 0,2,4,6
        const __m256i v1x0_2 = _mm256_shuffle_epi8(vh0_2, vperm1);  // 1,3,5,7
        const __m256i v01x0_2 = _mm256_or_si256(v0x0_2, v1x0_2);
        const __m256i v2x0_2 = _mm256_shuffle_epi8(vl0_2, vperm2);  // 8,A,C,E
        const __m256i v3x0_2 = _mm256_shuffle_epi8(vh0_2, vperm3);  // 9,B,D,F
        const __m256i v23x0_2 = _mm256_or_si256(v2x0_2, v3x0_2);
        const __m256i vt010_2 = _mm256_srli_epi32(v01x0_2, 4);  // first plane 0-7
        v0_2 = _mm256_or_si256(vt010_2, v23x0_2);            // + second plane 8-F
        v0_3 = _mm256_xor_si256(v0_3, vkernel_zero_point);    // uint4 -> int4
        const __m256i vt0_3 = _mm256_slli_epi32(v0_3, 4);     // isolate lower int4
        const __m256i vh0_3 = _mm256_and_si256(v0_3, vmask);  // isolate upper int4
        const __m256i vl0_3 = _mm256_and_si256(vt0_3, vmask);
        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, vh0_3);
        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, vl0_3);
        const __m256i v0x0_3 = _mm256_shuffle_epi8(vl0_3, vperm0);  // 0,2,4,6
        const __m256i v1x0_3 = _mm256_shuffle_epi8(vh0_3, vperm1);  // 1,3,5,7
        const __m256i v01x0_3 = _mm256_or_si256(v0x0_3, v1x0_3);
        const __m256i v2x0_3 = _mm256_shuffle_epi8(vl0_3, vperm2);  // 8,A,C,E
        const __m256i v3x0_3 = _mm256_shuffle_epi8(vh0_3, vperm3);  // 9,B,D,F
        const __m256i v23x0_3 = _mm256_or_si256(v2x0_3, v3x0_3);
        const __m256i vt010_3 = _mm256_srli_epi32(v01x0_3, 4);  // first plane 0-7
        v0_3 = _mm256_or_si256(vt010_3, v23x0_3);            // + second plane 8-F
        v4_0 = _mm256_xor_si256(v4_0, vkernel_zero_point);    // uint4 -> int4
        const __m256i vt4_0 = _mm256_slli_epi32(v4_0, 4);     // isolate lower int4
        const __m256i vh4_0 = _mm256_and_si256(v4_0, vmask);  // isolate upper int4
        const __m256i vl4_0 = _mm256_and_si256(vt4_0, vmask);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, vh4_0);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, vl4_0);
        const __m256i v0x4_0 = _mm256_shuffle_epi8(vl4_0, vperm0);  // 0,2,4,6
        const __m256i v1x4_0 = _mm256_shuffle_epi8(vh4_0, vperm1);  // 1,3,5,7
        const __m256i v01x4_0 = _mm256_or_si256(v0x4_0, v1x4_0);
        const __m256i v2x4_0 = _mm256_shuffle_epi8(vl4_0, vperm2);  // 8,A,C,E
        const __m256i v3x4_0 = _mm256_shuffle_epi8(vh4_0, vperm3);  // 9,B,D,F
        const __m256i v23x4_0 = _mm256_or_si256(v2x4_0, v3x4_0);
        const __m256i vt014_0 = _mm256_srli_epi32(v01x4_0, 4);  // first plane 0-7
        v4_0 = _mm256_or_si256(vt014_0, v23x4_0);            // + second plane 8-F
        v4_1 = _mm256_xor_si256(v4_1, vkernel_zero_point);    // uint4 -> int4
        const __m256i vt4_1 = _mm256_slli_epi32(v4_1, 4);     // isolate lower int4
        const __m256i vh4_1 = _mm256_and_si256(v4_1, vmask);  // isolate upper int4
        const __m256i vl4_1 = _mm256_and_si256(vt4_1, vmask);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, vh4_1);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, vl4_1);
        const __m256i v0x4_1 = _mm256_shuffle_epi8(vl4_1, vperm0);  // 0,2,4,6
        const __m256i v1x4_1 = _mm256_shuffle_epi8(vh4_1, vperm1);  // 1,3,5,7
        const __m256i v01x4_1 = _mm256_or_si256(v0x4_1, v1x4_1);
        const __m256i v2x4_1 = _mm256_shuffle_epi8(vl4_1, vperm2);  // 8,A,C,E
        const __m256i v3x4_1 = _mm256_shuffle_epi8(vh4_1, vperm3);  // 9,B,D,F
        const __m256i v23x4_1 = _mm256_or_si256(v2x4_1, v3x4_1);
        const __m256i vt014_1 = _mm256_srli_epi32(v01x4_1, 4);  // first plane 0-7
        v4_1 = _mm256_or_si256(vt014_1, v23x4_1);            // + second plane 8-F
        v4_2 = _mm256_xor_si256(v4_2, vkernel_zero_point);    // uint4 -> int4
        const __m256i vt4_2 = _mm256_slli_epi32(v4_2, 4);     // isolate lower int4
        const __m256i vh4_2 = _mm256_and_si256(v4_2, vmask);  // isolate upper int4
        const __m256i vl4_2 = _mm256_and_si256(vt4_2, vmask);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, vh4_2);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, vl4_2);
        const __m256i v0x4_2 = _mm256_shuffle_epi8(vl4_2, vperm0);  // 0,2,4,6
        const __m256i v1x4_2 = _mm256_shuffle_epi8(vh4_2, vperm1);  // 1,3,5,7
        const __m256i v01x4_2 = _mm256_or_si256(v0x4_2, v1x4_2);
        const __m256i v2x4_2 = _mm256_shuffle_epi8(vl4_2, vperm2);  // 8,A,C,E
        const __m256i v3x4_2 = _mm256_shuffle_epi8(vh4_2, vperm3);  // 9,B,D,F
        const __m256i v23x4_2 = _mm256_or_si256(v2x4_2, v3x4_2);
        const __m256i vt014_2 = _mm256_srli_epi32(v01x4_2, 4);  // first plane 0-7
        v4_2 = _mm256_or_si256(vt014_2, v23x4_2);            // + second plane 8-F
        v4_3 = _mm256_xor_si256(v4_3, vkernel_zero_point);    // uint4 -> int4
        const __m256i vt4_3 = _mm256_slli_epi32(v4_3, 4);     // isolate lower int4
        const __m256i vh4_3 = _mm256_and_si256(v4_3, vmask);  // isolate upper int4
        const __m256i vl4_3 = _mm256_and_si256(vt4_3, vmask);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, vh4_3);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, vl4_3);
        const __m256i v0x4_3 = _mm256_shuffle_epi8(vl4_3, vperm0);  // 0,2,4,6
        const __m256i v1x4_3 = _mm256_shuffle_epi8(vh4_3, vperm1);  // 1,3,5,7
        const __m256i v01x4_3 = _mm256_or_si256(v0x4_3, v1x4_3);
        const __m256i v2x4_3 = _mm256_shuffle_epi8(vl4_3, vperm2);  // 8,A,C,E
        const __m256i v3x4_3 = _mm256_shuffle_epi8(vh4_3, vperm3);  // 9,B,D,F
        const __m256i v23x4_3 = _mm256_or_si256(v2x4_3, v3x4_3);
        const __m256i vt014_3 = _mm256_srli_epi32(v01x4_3, 4);  // first plane 0-7
        v4_3 = _mm256_or_si256(vt014_3, v23x4_3);            // + second plane 8-F

        _mm256_storeu_si256((__m256i *)&out[0],  v0_0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4_0);
        _mm256_storeu_si256((__m256i *)&out[64],  v0_1);
        _mm256_storeu_si256((__m256i *)&out[96],  v4_1);
        _mm256_storeu_si256((__m256i *)&out[128],  v0_2);
        _mm256_storeu_si256((__m256i *)&out[160],  v4_2);
        _mm256_storeu_si256((__m256i *)&out[192],  v0_3);
        _mm256_storeu_si256((__m256i *)&out[224],  v4_3);

        w0 += 32;
        w1 += 32;
        w2 += 32;
        w3 += 32;
        w4 += 32;
        w5 += 32;
        w6 += 32;
        w7 += 32;
        out += 256;
      }

      // KC main loop multiple of 8x8
      for (; k >= 8; k -= 8) {
        __m256i v0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w0));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w1)), 0x0C);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w2)), 0x30);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w3)), 0xC0);
        __m256i v4 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w4));
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w5)), 0x0C);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w6)), 0x30);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w7)), 0xC0);

        v0 = _mm256_xor_si256(v0, vkernel_zero_point);    // uint4 -> int4
        const __m256i vt0 = _mm256_slli_epi32(v0, 4);     // isolate lower int4
        const __m256i vh0 = _mm256_and_si256(v0, vmask);  // isolate upper int4
        const __m256i vl0 = _mm256_and_si256(vt0, vmask);
        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, vh0);
        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, vl0);
        const __m256i v0x0 = _mm256_shuffle_epi8(vl0, vperm0);  // 0,2,4,6
        const __m256i v1x0 = _mm256_shuffle_epi8(vh0, vperm1);  // 1,3,5,7
        const __m256i v01x0 = _mm256_or_si256(v0x0, v1x0);
        const __m256i v2x0 = _mm256_shuffle_epi8(vl0, vperm2);  // 8,A,C,E
        const __m256i v3x0 = _mm256_shuffle_epi8(vh0, vperm3);  // 9,B,D,F
        const __m256i v23x0 = _mm256_or_si256(v2x0, v3x0);
        const __m256i vt010 = _mm256_srli_epi32(v01x0, 4);  // first plane 0-7
        v0 = _mm256_or_si256(vt010, v23x0);            // + second plane 8-F
        v4 = _mm256_xor_si256(v4, vkernel_zero_point);    // uint4 -> int4
        const __m256i vt4 = _mm256_slli_epi32(v4, 4);     // isolate lower int4
        const __m256i vh4 = _mm256_and_si256(v4, vmask);  // isolate upper int4
        const __m256i vl4 = _mm256_and_si256(vt4, vmask);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, vh4);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, vl4);
        const __m256i v0x4 = _mm256_shuffle_epi8(vl4, vperm0);  // 0,2,4,6
        const __m256i v1x4 = _mm256_shuffle_epi8(vh4, vperm1);  // 1,3,5,7
        const __m256i v01x4 = _mm256_or_si256(v0x4, v1x4);
        const __m256i v2x4 = _mm256_shuffle_epi8(vl4, vperm2);  // 8,A,C,E
        const __m256i v3x4 = _mm256_shuffle_epi8(vh4, vperm3);  // 9,B,D,F
        const __m256i v23x4 = _mm256_or_si256(v2x4, v3x4);
        const __m256i vt014 = _mm256_srli_epi32(v01x4, 4);  // first plane 0-7
        v4 = _mm256_or_si256(vt014, v23x4);            // + second plane 8-F

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        w4 += 8;
        w5 += 8;
        w6 += 8;
        w7 += 8;
        out += 64;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);

        __m256i v0 = _mm256_set1_epi64x((int64_t) safe_load_u64(w0, k));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) safe_load_u64(w1, k)), 0x0C);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) safe_load_u64(w2, k)), 0x30);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) safe_load_u64(w3, k)), 0xC0);
        __m256i v4 = _mm256_set1_epi64x((int64_t) safe_load_u64(w4, k));
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) safe_load_u64(w5, k)), 0x0C);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) safe_load_u64(w6, k)), 0x30);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) safe_load_u64(w7, k)), 0xC0);

        w0 += k;
        w1 += k;
        w2 += k;
        w3 += k;
        w4 += k;
        w5 += k;
        w6 += k;
        w7 += k;

        v0 = _mm256_xor_si256(v0, vkernel_zero_point);    // uint4 -> int4
        const __m256i vt0 = _mm256_slli_epi32(v0, 4);     // isolate lower int4
        const __m256i vh0 = _mm256_and_si256(v0, vmask);  // isolate upper int4
        const __m256i vl0 = _mm256_and_si256(vt0, vmask);
        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, vh0);
        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, vl0);
        const __m256i v0x0 = _mm256_shuffle_epi8(vl0, vperm0);  // 0,2,4,6
        const __m256i v1x0 = _mm256_shuffle_epi8(vh0, vperm1);  // 1,3,5,7
        const __m256i v01x0 = _mm256_or_si256(v0x0, v1x0);
        const __m256i v2x0 = _mm256_shuffle_epi8(vl0, vperm2);  // 8,A,C,E
        const __m256i v3x0 = _mm256_shuffle_epi8(vh0, vperm3);  // 9,B,D,F
        const __m256i v23x0 = _mm256_or_si256(v2x0, v3x0);
        const __m256i vt010 = _mm256_srli_epi32(v01x0, 4);  // first plane 0-7
        v0 = _mm256_or_si256(vt010, v23x0);            // + second plane 8-F
        v4 = _mm256_xor_si256(v4, vkernel_zero_point);    // uint4 -> int4
        const __m256i vt4 = _mm256_slli_epi32(v4, 4);     // isolate lower int4
        const __m256i vh4 = _mm256_and_si256(v4, vmask);  // isolate upper int4
        const __m256i vl4 = _mm256_and_si256(vt4, vmask);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, vh4);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, vl4);
        const __m256i v0x4 = _mm256_shuffle_epi8(vl4, vperm0);  // 0,2,4,6
        const __m256i v1x4 = _mm256_shuffle_epi8(vh4, vperm1);  // 1,3,5,7
        const __m256i v01x4 = _mm256_or_si256(v0x4, v1x4);
        const __m256i v2x4 = _mm256_shuffle_epi8(vl4, vperm2);  // 8,A,C,E
        const __m256i v3x4 = _mm256_shuffle_epi8(vh4, vperm3);  // 9,B,D,F
        const __m256i v23x4 = _mm256_or_si256(v2x4, v3x4);
        const __m256i vt014 = _mm256_srli_epi32(v01x4, 4);  // first plane 0-7
        v4 = _mm256_or_si256(vt014, v23x4);            // + second plane 8-F

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);

        out += 64;
      }

      __m256i vksum0 = _mm256_hadd_epi32(vacc0, vacc4);
      vksum0 = _mm256_permute4x64_epi64(vksum0, _MM_SHUFFLE(3, 1, 2, 0));
      vksum0 = _mm256_mullo_epi32(vksum0, vzeropoint);
      __m256i vpack0 =  _mm256_loadu_si256((const __m256i*) (packed_b + 0));
      vpack0 = _mm256_sub_epi32(vpack0, vksum0);
      _mm256_storeu_si256((__m256i *) (packed_b + 0), vpack0);
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
      w0 = w7;
    }

    // NC remainder (1..7)
    if XNN_UNLIKELY(n != 0) {
      assert(n >= 1 && n <= 7);

      int32_t* packed_b = (int32_t*) out;
      if XNN_LIKELY(b != NULL) {
        size_t nb = n;
        do {
          *((int32_t*) out) = *b++;
          out += sizeof(int32_t);
        } while (--nb != 0);
      } else {
        size_t nb = n;
        do {
          *((int32_t*) out) = 0;
          out += sizeof(int32_t);
        } while (--nb != 0);
      }
      out += (8 - n) * sizeof(int32_t);

      const int8_t* w1 = w0 + kc;
      if XNN_UNPREDICTABLE(n < 2) {
        w1 = w0;
      }
      const int8_t* w2 = w1 + kc;
      if XNN_UNPREDICTABLE(n <= 2) {
        w2 = w1;
      }
      const int8_t* w3 = w2 + kc;
      if XNN_UNPREDICTABLE(n < 4) {
        w3 = w2;
      }
      const int8_t* w4 = w3 + kc;
      if XNN_UNPREDICTABLE(n <= 4) {
        w4 = w3;
      }
      const int8_t* w5 = w4 + kc;
      if XNN_UNPREDICTABLE(n < 6) {
        w5 = w4;
      }
      const int8_t* w6 = w5 + kc;
      if XNN_UNPREDICTABLE(n <= 6) {
        w6 = w5;
      }
      const int8_t* w7 = w6 + kc;
      if XNN_UNPREDICTABLE(n < 8) {
        w7 = w6;
      }

      __m256i vacc0 = _mm256_setzero_si256();
      __m256i vacc4 = _mm256_setzero_si256();

      // KC main loop multiple of 8x8
      size_t k = kc;
      for (; k >= 8; k -= 8) {
        __m256i v0 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w0));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w1)), 0x0C);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w2)), 0x30);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w3)), 0xC0);
        __m256i v4 = _mm256_set1_epi64x((int64_t) unaligned_load_u64(w4));
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w5)), 0x0C);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w6)), 0x30);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) unaligned_load_u64(w7)), 0xC0);

        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, v0);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, v4);

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);

        w0 += 8;
        w1 += 8;
        w2 += 8;
        w3 += 8;
        w4 += 8;
        w5 += 8;
        w6 += 8;
        w7 += 8;
        out += 64;
      }

      // KC remainder of 1..7
      if (k != 0) {
        assert(k >= 1 && k <= 7);

        __m256i v0 = _mm256_set1_epi64x((int64_t) safe_load_u64(w0, k));
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) safe_load_u64(w1, k)), 0x0C);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) safe_load_u64(w2, k)), 0x30);
        v0 = _mm256_blend_epi32(v0, _mm256_set1_epi64x((int64_t) safe_load_u64(w3, k)), 0xC0);
        __m256i v4 = _mm256_set1_epi64x((int64_t) safe_load_u64(w4, k));
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) safe_load_u64(w5, k)), 0x0C);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) safe_load_u64(w6, k)), 0x30);
        v4 = _mm256_blend_epi32(v4, _mm256_set1_epi64x((int64_t) safe_load_u64(w7, k)), 0xC0);

        w0 += k;
        w1 += k;
        w2 += k;
        w3 += k;
        w4 += k;
        w5 += k;
        w6 += k;
        w7 += k;

        vacc0 = _mm256_dpbusd_epi32(vacc0, vone, v0);
        vacc4 = _mm256_dpbusd_epi32(vacc4, vone, v4);

        _mm256_storeu_si256((__m256i *)&out[0],  v0);
        _mm256_storeu_si256((__m256i *)&out[32],  v4);

        out += 64;
      }

      __m256i vksum0 = _mm256_hadd_epi32(vacc0, vacc4);
      vksum0 = _mm256_permute4x64_epi64(vksum0, _MM_SHUFFLE(3, 1, 2, 0));
      vksum0 = _mm256_mullo_epi32(vksum0, vzeropoint);
      __m256i vpack0 =  _mm256_loadu_si256((const __m256i*) (packed_b + 0));
      vpack0 = _mm256_sub_epi32(vpack0, vksum0);
      _mm256_storeu_si256((__m256i *) (packed_b + 0), vpack0);
      out = (int8_t*) ((uintptr_t) out + extra_bytes);
    }

    weights = (const uint8_t*)((intptr_t) weights + nc * kc);
  } while (--g != 0);
}
