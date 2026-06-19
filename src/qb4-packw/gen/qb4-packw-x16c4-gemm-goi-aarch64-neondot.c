// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qb4-packw/c4-aarch64-neondot.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/packw.h"

// convert a vector from packed nibbles to planar, and accumulate sum
static XNN_INTRINSIC
int8x16_t xnn_packed2planar(
  int32x4_t *vacc,
  const int8x16_t v,
  const int8x16_t vmask,
  const int8x16_t vones)
{
  const int8x16_t vl = vandq_s8(v, vmask);  // isolate lower int 4
  const int8x16_t vh = vshlq_n_s8(v, 4);// isolate upper int 4
  *vacc = vdotq_s32(*vacc, vh, vones);
  *vacc = vdotq_s32(*vacc, vl, vones);
  const int8x16_t v0123 = vzip1q_s8(vh, vl);
  const int8x16_t v4567 = vzip2q_s8(vh, vl);
  const uint8x16_t v0246 = vreinterpretq_u8_u32(vuzp1q_u32(vreinterpretq_u32_s8(v0123), vreinterpretq_u32_s8(v4567)));
  const uint8x16_t v1357 = vreinterpretq_u8_u32(vuzp2q_u32(vreinterpretq_u32_s8(v0123), vreinterpretq_u32_s8(v4567)));
  const uint8x16_t vl0246 = vshrq_n_u8(v0246, 4);
  const uint8x16_t v01234567 = vorrq_u8(vl0246, v1357);
  return vreinterpretq_s8_u8(v01234567);
}

void xnn_qb4_packw_gemm_goi_ukernel_x16c4__aarch64_neondot(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  size_t bl,
  const uint8_t* weights,
  const int32_t* bias,
  const void* scale,
  int8_t* packed_weights,
  size_t extra_bytes_bl,
  size_t extra_bytes_n,
  const void* params) XNN_OOB_READS
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 16);
  assert(kr == 4);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);
  assert(extra_bytes_bl == nr * sizeof(uint16_t));
  assert(extra_bytes_n == nr * sizeof(float));
  assert(params != NULL);
  assert(kc % bl == 0);
  size_t num_blocks = kc / bl;
  size_t weight_stride = (kc >> 1);
  const int8x16_t vmask = vmovq_n_s8(INT8_C(0xF0));
  const int8x16_t vones = vmovq_n_s8(UINT8_C(0x01));

  uint8_t* out = (uint8_t*) packed_weights;
  const int32_t* b = (const int32_t*) bias;
  const float32x4_t vzeropoint = vmovq_n_f32((float) (((const struct xnn_qs8_qc4w_packing_params*) params)->input_zero_point + 0));
  const float32x4_t vrecip_sixteen = vmovq_n_f32(1.0f/ 16.0f);
  const int8_t kzp = (int8_t) (((const struct xnn_qs8_qc4w_packing_params*) params)->kernel_zero_point);
  assert(kzp == 0 || kzp == 8);
  const int8x16_t vkernel_zero_point = vmovq_n_s8(kzp * 0x11);

  do {
    // NC main loop multiple of 16
    const int8_t* w0 = (const int8_t*) weights;
    const uint16_t* s0 = (const uint16_t*) scale;
    int n = nc;
    for (;n > 0; n -= 16) {
      float* packed_k_scaled_sum = (float*) out;
      float32x4_t packed_k_scaled_sums0 = vdupq_n_f32(0.0f);
      float32x4_t packed_k_scaled_sums1 = vdupq_n_f32(0.0f);
      float32x4_t packed_k_scaled_sums2 = vdupq_n_f32(0.0f);
      float32x4_t packed_k_scaled_sums3 = vdupq_n_f32(0.0f);
      out += 16 * sizeof(float);

      // KC/2 bytes is KC Nibbles
      const int8_t* w1 = w0 + weight_stride;
      const int8_t* w2 = w1 + weight_stride;
      const int8_t* w3 = w2 + weight_stride;
      const int8_t* w4 = w3 + weight_stride;
      const int8_t* w5 = w4 + weight_stride;
      const int8_t* w6 = w5 + weight_stride;
      const int8_t* w7 = w6 + weight_stride;
      const int8_t* w8 = w7 + weight_stride;
      const int8_t* w9 = w8 + weight_stride;
      const int8_t* w10 = w9 + weight_stride;
      const int8_t* w11 = w10 + weight_stride;
      const int8_t* w12 = w11 + weight_stride;
      const int8_t* w13 = w12 + weight_stride;
      const int8_t* w14 = w13 + weight_stride;
      const int8_t* w15 = w14 + weight_stride;

      // scales
      const uint16_t* s1 = s0 + num_blocks;
      const uint16_t* s2 = s1 + num_blocks;
      const uint16_t* s3 = s2 + num_blocks;
      const uint16_t* s4 = s3 + num_blocks;
      const uint16_t* s5 = s4 + num_blocks;
      const uint16_t* s6 = s5 + num_blocks;
      const uint16_t* s7 = s6 + num_blocks;
      const uint16_t* s8 = s7 + num_blocks;
      const uint16_t* s9 = s8 + num_blocks;
      const uint16_t* s10 = s9 + num_blocks;
      const uint16_t* s11 = s10 + num_blocks;
      const uint16_t* s12 = s11 + num_blocks;
      const uint16_t* s13 = s12 + num_blocks;
      const uint16_t* s14 = s13 + num_blocks;
      const uint16_t* s15 = s14 + num_blocks;

      if XNN_UNLIKELY(n < 16){
        if XNN_UNPREDICTABLE(n < 2) {
          w1 = w0;
          s1 = s0;
        }
        if XNN_UNPREDICTABLE(n <= 2) {
          w2 = w1;
          s2 = s1;
        }
        if XNN_UNPREDICTABLE(n < 4) {
          w3 = w2;
          s3 = s2;
        }
        if XNN_UNPREDICTABLE(n <= 4) {
          w4 = w3;
          s4 = s3;
        }
        if XNN_UNPREDICTABLE(n < 6) {
          w5 = w4;
          s5 = s4;
        }
        if XNN_UNPREDICTABLE(n <= 6) {
          w6 = w5;
          s6 = s5;
        }
        if XNN_UNPREDICTABLE(n < 8) {
          w7 = w6;
          s7 = s6;
        }
        if XNN_UNPREDICTABLE(n <= 8) {
          w8 = w7;
          s8 = s7;
        }
        if XNN_UNPREDICTABLE(n < 10) {
          w9 = w8;
          s9 = s8;
        }
        if XNN_UNPREDICTABLE(n <= 10) {
          w10 = w9;
          s10 = s9;
        }
        if XNN_UNPREDICTABLE(n < 12) {
          w11 = w10;
          s11 = s10;
        }
        if XNN_UNPREDICTABLE(n <= 12) {
          w12 = w11;
          s12 = s11;
        }
        if XNN_UNPREDICTABLE(n < 14) {
          w13 = w12;
          s13 = s12;
        }
        if XNN_UNPREDICTABLE(n <= 14) {
          w14 = w13;
          s14 = s13;
        }
        if XNN_UNPREDICTABLE(n < 16) {
          w15 = w14;
          s15 = s14;
        }
      }

      size_t kb = kc;
      // Process k by blocks (bl)
      for (; kb >= bl; kb-=bl) {
        // Initialize KSum as subtracting bl zero points (8)
        int32x4_t ksum0 = vdupq_n_s32(0);
        int32x4_t ksum1 = vdupq_n_s32(0);
        int32x4_t ksum2 = vdupq_n_s32(0);
        int32x4_t ksum3 = vdupq_n_s32(0);
        size_t k = bl;

        // KC Main loop multiple of 16x32
        for(; k >= 32; k-=32) {
          const uint32x4_t w0x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w0), vkernel_zero_point)); w0 += 16;
          const uint32x4_t w1x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w1), vkernel_zero_point)); w1 += 16;
          const uint32x4_t w2x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w2), vkernel_zero_point)); w2 += 16;
          const uint32x4_t w3x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w3), vkernel_zero_point)); w3 += 16;
          const uint32x4_t w4x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w4), vkernel_zero_point)); w4 += 16;
          const uint32x4_t w5x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w5), vkernel_zero_point)); w5 += 16;
          const uint32x4_t w6x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w6), vkernel_zero_point)); w6 += 16;
          const uint32x4_t w7x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w7), vkernel_zero_point)); w7 += 16;
          const uint32x4_t w8x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w8), vkernel_zero_point)); w8 += 16;
          const uint32x4_t w9x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w9), vkernel_zero_point)); w9 += 16;
          const uint32x4_t w10x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w10), vkernel_zero_point)); w10 += 16;
          const uint32x4_t w11x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w11), vkernel_zero_point)); w11 += 16;
          const uint32x4_t w12x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w12), vkernel_zero_point)); w12 += 16;
          const uint32x4_t w13x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w13), vkernel_zero_point)); w13 += 16;
          const uint32x4_t w14x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w14), vkernel_zero_point)); w14 += 16;
          const uint32x4_t w15x0123 = vreinterpretq_u32_s8(veorq_s8(vld1q_s8(w15), vkernel_zero_point)); w15 += 16;

          const uint32x4_t v0_02 = vtrn1q_u32(w0x0123, w1x0123);
          const uint32x4_t v0_13 = vtrn2q_u32(w0x0123, w1x0123);
          const uint32x4_t v2_02 = vtrn1q_u32(w2x0123, w3x0123);
          const uint32x4_t v2_13 = vtrn2q_u32(w2x0123, w3x0123);
          const uint32x4_t v4_02 = vtrn1q_u32(w4x0123, w5x0123);
          const uint32x4_t v4_13 = vtrn2q_u32(w4x0123, w5x0123);
          const uint32x4_t v6_02 = vtrn1q_u32(w6x0123, w7x0123);
          const uint32x4_t v6_13 = vtrn2q_u32(w6x0123, w7x0123);
          const uint32x4_t v8_02 = vtrn1q_u32(w8x0123, w9x0123);
          const uint32x4_t v8_13 = vtrn2q_u32(w8x0123, w9x0123);
          const uint32x4_t v10_02 = vtrn1q_u32(w10x0123, w11x0123);
          const uint32x4_t v10_13 = vtrn2q_u32(w10x0123, w11x0123);
          const uint32x4_t v12_02 = vtrn1q_u32(w12x0123, w13x0123);
          const uint32x4_t v12_13 = vtrn2q_u32(w12x0123, w13x0123);
          const uint32x4_t v14_02 = vtrn1q_u32(w14x0123, w15x0123);
          const uint32x4_t v14_13 = vtrn2q_u32(w14x0123, w15x0123);

          int8x16_t v0_0 = vreinterpretq_s8_u32(vcombine_u32(vget_low_u32(v0_02), vget_low_u32(v2_02)));
          int8x16_t v0_2 = vreinterpretq_s8_u32(vcombine_u32(vget_high_u32(v0_02), vget_high_u32(v2_02)));
          int8x16_t v0_1 = vreinterpretq_s8_u32(vcombine_u32(vget_low_u32(v0_13), vget_low_u32(v2_13)));
          int8x16_t v0_3 = vreinterpretq_s8_u32(vcombine_u32(vget_high_u32(v0_13), vget_high_u32(v2_13)));
          int8x16_t v1_0 = vreinterpretq_s8_u32(vcombine_u32(vget_low_u32(v4_02), vget_low_u32(v6_02)));
          int8x16_t v1_2 = vreinterpretq_s8_u32(vcombine_u32(vget_high_u32(v4_02), vget_high_u32(v6_02)));
          int8x16_t v1_1 = vreinterpretq_s8_u32(vcombine_u32(vget_low_u32(v4_13), vget_low_u32(v6_13)));
          int8x16_t v1_3 = vreinterpretq_s8_u32(vcombine_u32(vget_high_u32(v4_13), vget_high_u32(v6_13)));
          int8x16_t v2_0 = vreinterpretq_s8_u32(vcombine_u32(vget_low_u32(v8_02), vget_low_u32(v10_02)));
          int8x16_t v2_2 = vreinterpretq_s8_u32(vcombine_u32(vget_high_u32(v8_02), vget_high_u32(v10_02)));
          int8x16_t v2_1 = vreinterpretq_s8_u32(vcombine_u32(vget_low_u32(v8_13), vget_low_u32(v10_13)));
          int8x16_t v2_3 = vreinterpretq_s8_u32(vcombine_u32(vget_high_u32(v8_13), vget_high_u32(v10_13)));
          int8x16_t v3_0 = vreinterpretq_s8_u32(vcombine_u32(vget_low_u32(v12_02), vget_low_u32(v14_02)));
          int8x16_t v3_2 = vreinterpretq_s8_u32(vcombine_u32(vget_high_u32(v12_02), vget_high_u32(v14_02)));
          int8x16_t v3_1 = vreinterpretq_s8_u32(vcombine_u32(vget_low_u32(v12_13), vget_low_u32(v14_13)));
          int8x16_t v3_3 = vreinterpretq_s8_u32(vcombine_u32(vget_high_u32(v12_13), vget_high_u32(v14_13)));

          v0_0 = xnn_packed2planar(&ksum0, v0_0, vmask, vones);
          v0_1 = xnn_packed2planar(&ksum0, v0_1, vmask, vones);
          v0_2 = xnn_packed2planar(&ksum0, v0_2, vmask, vones);
          v0_3 = xnn_packed2planar(&ksum0, v0_3, vmask, vones);
          v1_0 = xnn_packed2planar(&ksum1, v1_0, vmask, vones);
          v1_1 = xnn_packed2planar(&ksum1, v1_1, vmask, vones);
          v1_2 = xnn_packed2planar(&ksum1, v1_2, vmask, vones);
          v1_3 = xnn_packed2planar(&ksum1, v1_3, vmask, vones);
          v2_0 = xnn_packed2planar(&ksum2, v2_0, vmask, vones);
          v2_1 = xnn_packed2planar(&ksum2, v2_1, vmask, vones);
          v2_2 = xnn_packed2planar(&ksum2, v2_2, vmask, vones);
          v2_3 = xnn_packed2planar(&ksum2, v2_3, vmask, vones);
          v3_0 = xnn_packed2planar(&ksum3, v3_0, vmask, vones);
          v3_1 = xnn_packed2planar(&ksum3, v3_1, vmask, vones);
          v3_2 = xnn_packed2planar(&ksum3, v3_2, vmask, vones);
          v3_3 = xnn_packed2planar(&ksum3, v3_3, vmask, vones);

          vst1q_s8((int8_t*)&out[0], v0_0);
          vst1q_s8((int8_t*)&out[16], v1_0);
          vst1q_s8((int8_t*)&out[32], v2_0);
          vst1q_s8((int8_t*)&out[48], v3_0);
          vst1q_s8((int8_t*)&out[64], v0_1);
          vst1q_s8((int8_t*)&out[80], v1_1);
          vst1q_s8((int8_t*)&out[96], v2_1);
          vst1q_s8((int8_t*)&out[112], v3_1);
          vst1q_s8((int8_t*)&out[128], v0_2);
          vst1q_s8((int8_t*)&out[144], v1_2);
          vst1q_s8((int8_t*)&out[160], v2_2);
          vst1q_s8((int8_t*)&out[176], v3_2);
          vst1q_s8((int8_t*)&out[192], v0_3);
          vst1q_s8((int8_t*)&out[208], v1_3);
          vst1q_s8((int8_t*)&out[224], v2_3);
          vst1q_s8((int8_t*)&out[240], v3_3);

          out += 256;
        }

        uint16x8_t bf_scales0 = vdupq_n_u16(0);
        bf_scales0 = vsetq_lane_u16(s0[0], bf_scales0, 0);
        bf_scales0 = vsetq_lane_u16(s1[0], bf_scales0, 1);
        bf_scales0 = vsetq_lane_u16(s2[0], bf_scales0, 2);
        bf_scales0 = vsetq_lane_u16(s3[0], bf_scales0, 3);
        bf_scales0 = vsetq_lane_u16(s4[0], bf_scales0, 4);
        bf_scales0 = vsetq_lane_u16(s5[0], bf_scales0, 5);
        bf_scales0 = vsetq_lane_u16(s6[0], bf_scales0, 6);
        bf_scales0 = vsetq_lane_u16(s7[0], bf_scales0, 7);
        uint16x8_t bf_scales1 = vdupq_n_u16(0);
        bf_scales1 = vsetq_lane_u16(s8[0], bf_scales1, 0);
        bf_scales1 = vsetq_lane_u16(s9[0], bf_scales1, 1);
        bf_scales1 = vsetq_lane_u16(s10[0], bf_scales1, 2);
        bf_scales1 = vsetq_lane_u16(s11[0], bf_scales1, 3);
        bf_scales1 = vsetq_lane_u16(s12[0], bf_scales1, 4);
        bf_scales1 = vsetq_lane_u16(s13[0], bf_scales1, 5);
        bf_scales1 = vsetq_lane_u16(s14[0], bf_scales1, 6);
        bf_scales1 = vsetq_lane_u16(s15[0], bf_scales1, 7);

        float32x4_t f_scales0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf_scales0), 16));
        float32x4_t f_scales1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf_scales0), 16));
        float32x4_t f_scales2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(bf_scales1), 16));
        float32x4_t f_scales3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(bf_scales1), 16));

        s0 += 1;
        s1 += 1;
        s2 += 1;
        s3 += 1;
        s4 += 1;
        s5 += 1;
        s6 += 1;
        s7 += 1;
        s8 += 1;
        s9 += 1;
        s10 += 1;
        s11 += 1;
        s12 += 1;
        s13 += 1;
        s14 += 1;
        s15 += 1;

        f_scales0 = vmulq_f32(f_scales0, vrecip_sixteen);
        f_scales1 = vmulq_f32(f_scales1, vrecip_sixteen);
        f_scales2 = vmulq_f32(f_scales2, vrecip_sixteen);
        f_scales3 = vmulq_f32(f_scales3, vrecip_sixteen);

        float32x4_t f_ksum0 = vcvtq_f32_s32(ksum0);
        f_ksum0 = vmulq_f32(f_ksum0, vzeropoint);
        packed_k_scaled_sums0 = vfmsq_f32(packed_k_scaled_sums0, f_ksum0, f_scales0);
        float32x4_t f_ksum1 = vcvtq_f32_s32(ksum1);
        f_ksum1 = vmulq_f32(f_ksum1, vzeropoint);
        packed_k_scaled_sums1 = vfmsq_f32(packed_k_scaled_sums1, f_ksum1, f_scales1);
        float32x4_t f_ksum2 = vcvtq_f32_s32(ksum2);
        f_ksum2 = vmulq_f32(f_ksum2, vzeropoint);
        packed_k_scaled_sums2 = vfmsq_f32(packed_k_scaled_sums2, f_ksum2, f_scales2);
        float32x4_t f_ksum3 = vcvtq_f32_s32(ksum3);
        f_ksum3 = vmulq_f32(f_ksum3, vzeropoint);
        packed_k_scaled_sums3 = vfmsq_f32(packed_k_scaled_sums3, f_ksum3, f_scales3);

        vst1q_f32(&packed_k_scaled_sum[0], packed_k_scaled_sums0);
        vst1q_f32(&packed_k_scaled_sum[4], packed_k_scaled_sums1);
        vst1q_f32(&packed_k_scaled_sum[8], packed_k_scaled_sums2);
        vst1q_f32(&packed_k_scaled_sum[12], packed_k_scaled_sums3);


        vst1_u16((uint16_t*)out+0, vreinterpret_u16_s16(vshrn_n_s32(vreinterpretq_s32_f32(f_scales0), 16)));
        vst1_u16((uint16_t*)out+4, vreinterpret_u16_s16(vshrn_n_s32(vreinterpretq_s32_f32(f_scales1), 16)));
        vst1_u16((uint16_t*)out+8, vreinterpret_u16_s16(vshrn_n_s32(vreinterpretq_s32_f32(f_scales2), 16)));
        vst1_u16((uint16_t*)out+12, vreinterpret_u16_s16(vshrn_n_s32(vreinterpretq_s32_f32(f_scales3), 16)));

        out += 16 * sizeof(uint16_t);
      }


      if XNN_LIKELY(b != NULL){
        const int32x4_t b0 = vld1q_s32(&b[0]);
        vst1q_s32((int32_t*)out + 0, b0);
        const int32x4_t b1 = vld1q_s32(&b[4]);
        vst1q_s32((int32_t*)out + 4, b1);
        const int32x4_t b2 = vld1q_s32(&b[8]);
        vst1q_s32((int32_t*)out + 8, b2);
        const int32x4_t b3 = vld1q_s32(&b[12]);
        vst1q_s32((int32_t*)out + 12, b3);
        b += 16;
      } else {
        vst1q_s32((int32_t*)out + 0, vdupq_n_s32(0));
        vst1q_s32((int32_t*)out + 4, vdupq_n_s32(0));
        vst1q_s32((int32_t*)out + 8, vdupq_n_s32(0));
        vst1q_s32((int32_t*)out + 12, vdupq_n_s32(0));
      }
      out += 16 * sizeof(uint32_t);
      w0 = w15;
      s0 = s15;
    }
  } while (--g != 0);
}
