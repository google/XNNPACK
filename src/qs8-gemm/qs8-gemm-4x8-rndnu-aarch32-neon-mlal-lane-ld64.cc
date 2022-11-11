// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <cassert>

#include <xnnpack.h>
#include <xnnpack/aarch32-assembler.h>
#include <xnnpack/gemm.h>
#include <xnnpack/memory.h>


namespace xnnpack {
namespace aarch32 {
namespace {
class Generator : public MacroAssembler {
  using MacroAssembler::MacroAssembler;
 public:
  void generate(bool prefetch, size_t max_mr, size_t nc_mod_nr, size_t kc, const void* params);
};


// void xnn_qs8_gemm_minmax_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64(
//     size_t mr,                            r0
//     size_t nc,                            r1
//     size_t kc,                            r2 -> r5
//     const uint8_t*restrict a,             r3
//     size_t a_stride,           sp + 64 -> (r7)
//     const void*restrict w,     sp + 68 -> r9
//     uint8_t*restrict c,        sp + 72 -> r11
//     size_t cm_stride,          sp + 76 -> (r6)
//     size_t cn_stride,          sp + 80 -> r7
//     xnn_qs8_conv_minmax_params params)  sp + 84 -> (r5)

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// A0   r3  d0-d1 q0
// A1  r12  d2-d3 q1
// A2  r10  d4-d5 q2
// A3   r0  d6-d7 q3
// B    r9  d8-d9 q4
// C0  r11 d16-d17  q8  d18-d19  q9
// C1   r4 d20-d21 q10  d22-d23 q11
// C2   r8 d24-d25 q12  d26-d27 q13
// C3   r6 d28-d29 q14  d30-d31 q15
// unused q6 q7

// params structure is 16 bytes
//  struct {
//    int32_t right_pre_shift;    d10[0]
//    int32_t multiplier;         d10[1]
//    int32_t right_post_shift;   d11[0]
//    int16_t output_zero_point;  d11[2]
//    int8_t output_min;          d11[6]
//    int8_t output_max;          d11[7]
//  } rndnu_neon;

// Converted from: src/qs8-gemm/gen/4x8-minmax-rndnu-aarch32-neon-mlal-lane-prfm-ld64.S
void Generator::generate(bool prefetch, size_t max_mr, size_t nc_mod_nr, size_t kc, const void* params)
{
  assert(nc_mod_nr < 8);
  assert(kc != 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7;

  // Push 64 bytes
  push({r4, r5, r6, r7, r8, r9, r10, r11}); // 32
  vpush({d8-d11}); // +32 = 64

  ldr(r7, mem[sp, 64]); // a_stride
  ldr(r11, mem[sp, 72]); // c
  ldr(r6, mem[sp, 76]); // cm_stride
  ldr(r9, mem[sp, 68]); // w
  ldr(r5, mem[sp, 84]); // params

  // Clamp A and C pointers
  cmp(r0, 2); // if mr >= 2
  add(r12, r3, r7); //   a1 = a0 + a_stride
  add(r4, r11, r6); //   c1 = c0 + cm_stride
  movlo(r12, r3); // a1
  movlo(r4, r11); // c1
  // if mr > 2
  add(r10, r12, r7); //   a2 = a1 + a_stride
  add(r8, r4, r6); //   c2 = c1 + cm_stride
  movls(r10, r12); // a2
  movls(r8, r4); // c2

  cmp(r0, 4); // if mr >=4
  add(r0, r10, r7); //   a3 = a2 + a_stride
  add(r6, r8, r6); //   c3 = c2 + cm_stride
  movlo(r0, r10); // a3
  movlo(r6, r8); // c3

  // Load params values
  vldm(mem[r5], {d10-d11}); // RNDNU params
  ldr(r7, mem[sp, 80]); // cn_stride

  if (prefetch) {
    pld(mem[r9, 64]); // Prefetch B
    pld(mem[r9, 128]);
    pld(mem[r9, 192]);
    pld(mem[r9, 256]);
    pld(mem[r9, 320]);
    pld(mem[r9, 384]);
  }

  align(8);
  bind(l0);
  // Load initial bias from w into accumulators
  vldm(mem[r9]++, {d16-d19}); // Bias
  subs(r5, r2, 8); // k = kc - 8

  vmov(q10, q8);
  if (prefetch) {
    pld(mem[r3, 64]); // Prefetch A
  }
  vmov(q11, q9);
  if (prefetch) {
    pld(mem[r12, 64]);
  }
  vmov(q12, q8);
  if (prefetch) {
    pld(mem[r10, 64]);
  }
  vmov(q13, q9);
  if (prefetch) {
    pld(mem[r0, 64]);
  }
  vmov(q14, q8);
  vmov(q15, q9);
  blo(l3); // less than 8 channels?

  // Main loop - 8 bytes
  // 64 bytes for weights.
  align(8);
  bind(l1);
  vld1_8({d0}, mem[r3]++); // A0
  vld1_8({d8}, mem[r9]++); // B
  vld1_8({d2}, mem[r12]++); // A1
  vld1_8({d4}, mem[r10]++); // A2
  vld1_8({d6}, mem[r0]++); // A3
  subs(r5, r5, 8);
  if (prefetch) {
    pld(mem[r3, 128]);
  }
  vmovl_s8(q0, d0);
  if (prefetch) {
    pld(mem[r12, 128]);
  }
  vmovl_s8(q4, d8);
  if (prefetch) {
    pld(mem[r10, 128]);
  }
  vmovl_s8(q1, d2);
  if (prefetch) {
    pld(mem[r0, 128]);
  }
  vmovl_s8(q2, d4);
  if (prefetch) {
    pld(mem[r9, 448]);
  }
  vmovl_s8(q3, d6);
  vmlal_s16(q8, d8, d0[0]);
  vmlal_s16(q9, d9, d0[0]);
  vmlal_s16(q10, d8, d2[0]);
  vmlal_s16(q11, d9, d2[0]);
  vmlal_s16(q12, d8, d4[0]);
  vmlal_s16(q13, d9, d4[0]);
  vmlal_s16(q14, d8, d6[0]);
  vmlal_s16(q15, d9, d6[0]);

  vld1_8({d8}, mem[r9]++);
  vmovl_s8(q4, d8);
  vmlal_s16(q8, d8, d0[1]);
  vmlal_s16(q9, d9, d0[1]);
  vmlal_s16(q10, d8, d2[1]);
  vmlal_s16(q11, d9, d2[1]);
  vmlal_s16(q12, d8, d4[1]);
  vmlal_s16(q13, d9, d4[1]);
  vmlal_s16(q14, d8, d6[1]);
  vmlal_s16(q15, d9, d6[1]);

  vld1_8({d8}, mem[r9]++);
  vmovl_s8(q4, d8);
  vmlal_s16(q8, d8, d0[2]);
  vmlal_s16(q9, d9, d0[2]);
  vmlal_s16(q10, d8, d2[2]);
  vmlal_s16(q11, d9, d2[2]);
  vmlal_s16(q12, d8, d4[2]);
  vmlal_s16(q13, d9, d4[2]);
  vmlal_s16(q14, d8, d6[2]);
  vmlal_s16(q15, d9, d6[2]);

  vld1_8({d8}, mem[r9]++);
  vmovl_s8(q4, d8);
  vmlal_s16(q8, d8, d0[3]);
  vmlal_s16(q9, d9, d0[3]);
  vmlal_s16(q10, d8, d2[3]);
  vmlal_s16(q11, d9, d2[3]);
  vmlal_s16(q12, d8, d4[3]);
  vmlal_s16(q13, d9, d4[3]);
  vmlal_s16(q14, d8, d6[3]);
  vmlal_s16(q15, d9, d6[3]);

  vld1_8({d8}, mem[r9]++);
  vmovl_s8(q4, d8);
  vmlal_s16(q8, d8, d1[0]);
  vmlal_s16(q9, d9, d1[0]);
  vmlal_s16(q10, d8, d3[0]);
  vmlal_s16(q11, d9, d3[0]);
  vmlal_s16(q12, d8, d5[0]);
  vmlal_s16(q13, d9, d5[0]);
  vmlal_s16(q14, d8, d7[0]);
  vmlal_s16(q15, d9, d7[0]);

  vld1_8({d8}, mem[r9]++);
  vmovl_s8(q4, d8);
  vmlal_s16(q8, d8, d1[1]);
  vmlal_s16(q9, d9, d1[1]);
  vmlal_s16(q10, d8, d3[1]);
  vmlal_s16(q11, d9, d3[1]);
  vmlal_s16(q12, d8, d5[1]);
  vmlal_s16(q13, d9, d5[1]);
  vmlal_s16(q14, d8, d7[1]);
  vmlal_s16(q15, d9, d7[1]);

  vld1_8({d8}, mem[r9]++);
  vmovl_s8(q4, d8);
  vmlal_s16(q8, d8, d1[2]);
  vmlal_s16(q9, d9, d1[2]);
  vmlal_s16(q10, d8, d3[2]);
  vmlal_s16(q11, d9, d3[2]);
  vmlal_s16(q12, d8, d5[2]);
  vmlal_s16(q13, d9, d5[2]);
  vmlal_s16(q14, d8, d7[2]);
  vmlal_s16(q15, d9, d7[2]);

  vld1_8({d8}, mem[r9]++);
  vmovl_s8(q4, d8);
  vmlal_s16(q8, d8, d1[3]);
  vmlal_s16(q9, d9, d1[3]);
  vmlal_s16(q10, d8, d3[3]);
  vmlal_s16(q11, d9, d3[3]);
  vmlal_s16(q12, d8, d5[3]);
  vmlal_s16(q13, d9, d5[3]);
  vmlal_s16(q14, d8, d7[3]);
  vmlal_s16(q15, d9, d7[3]);
  bhs(l1);

  // Is there a remainder?- 1-7 bytes of A
  adds(r5, r5, 8);
  bne(l3);

  bind(l2);
  // RNDNU quantization
  vdup_32(q0, d10[0]); // right_pre_shift

  vqshl_s32(q8, q8, q0);
  vqshl_s32(q9, q9, q0);
  vqshl_s32(q10, q10, q0);
  vqshl_s32(q11, q11, q0);
  vqshl_s32(q12, q12, q0);
  vqshl_s32(q13, q13, q0);
  vqshl_s32(q14, q14, q0);
  vqshl_s32(q15, q15, q0);

  vdup_32(q2, d11[0]); // right_post_shift

  vqdmulh_s32(q8, q8, d10[1]); // multiplier
  vqdmulh_s32(q9, q9, d10[1]);
  vqdmulh_s32(q10, q10, d10[1]);
  vqdmulh_s32(q11, q11, d10[1]);
  vqdmulh_s32(q12, q12, d10[1]);
  vqdmulh_s32(q13, q13, d10[1]);
  vqdmulh_s32(q14, q14, d10[1]);
  vqdmulh_s32(q15, q15, d10[1]);

  vrshl_s32(q8, q8, q2);
  vrshl_s32(q9, q9, q2);
  vrshl_s32(q10, q10, q2);
  vrshl_s32(q11, q11, q2);
  vrshl_s32(q12, q12, q2);
  vrshl_s32(q13, q13, q2);
  vrshl_s32(q14, q14, q2);
  vrshl_s32(q15, q15, q2);

  vdup_16(q0, d11[2]); // output_zero_point

  vqmovn_s32(d16, q8);
  vqmovn_s32(d17, q9);
  vqmovn_s32(d18, q10);
  vqmovn_s32(d19, q11);
  vqmovn_s32(d20, q12);
  vqmovn_s32(d21, q13);
  vqmovn_s32(d22, q14);
  vqmovn_s32(d23, q15);

  vqadd_s16(q8, q8, q0);
  vqadd_s16(q9, q9, q0);
  vqadd_s16(q10, q10, q0);
  vqadd_s16(q11, q11, q0);

  vdup_8(q12, d11[6]); // output_min

  vqmovn_s16(d0, q8);
  vqmovn_s16(d1, q9);
  vqmovn_s16(d2, q10);
  vqmovn_s16(d3, q11);

  vdup_8(q13, d11[7]); // output_max

  vmax_s8(q0, q0, q12);
  vmax_s8(q1, q1, q12);

  subs(r1, r1, 8);

  vmin_s8(q0, q0, q13);
  vmin_s8(q1, q1, q13);

  // Store full 4 x 8
  blo(l4);
  vst1_8({d0}, mem[r11], r7);
  sub(r3, r3, r2);
  vst1_8({d1}, mem[r4], r7);
  sub(r12, r12, r2);
  vst1_8({d2}, mem[r8], r7);
  sub(r10, r10, r2);
  vst1_8({d3}, mem[r6], r7);
  sub(r0, r0, r2);
  bhi(l0);

  vpop({d8-d11});
  pop({r4, r5, r6, r7, r8, r9, r10, r11});
  bx(lr);

  // Remainder- 1 to 7 bytes of A
  align(8);
  bind(l3);
  and_(r5, r5, 7); // kc remainder 1 to 7

  vld1_8({d0}, mem[r3], r5);
  vld1_8({d8}, mem[r9]++);
  vld1_8({d2}, mem[r12], r5);
  vld1_8({d4}, mem[r10], r5);
  vld1_8({d6}, mem[r0], r5);

  vmovl_s8(q0, d0);
  vmovl_s8(q4, d8);
  vmovl_s8(q1, d2);
  vmovl_s8(q2, d4);
  vmovl_s8(q3, d6);
  vmlal_s16(q8, d8, d0[0]);
  vmlal_s16(q9, d9, d0[0]);
  vmlal_s16(q10, d8, d2[0]);
  vmlal_s16(q11, d9, d2[0]);
  vmlal_s16(q12, d8, d4[0]);
  vmlal_s16(q13, d9, d4[0]);
  vmlal_s16(q14, d8, d6[0]);
  vmlal_s16(q15, d9, d6[0]);
  cmp(r5, 2);
  blo(l2);

  vld1_8({d8}, mem[r9]++);
  vmovl_s8(q4, d8);
  vmlal_s16(q8, d8, d0[1]);
  vmlal_s16(q9, d9, d0[1]);
  vmlal_s16(q10, d8, d2[1]);
  vmlal_s16(q11, d9, d2[1]);
  vmlal_s16(q12, d8, d4[1]);
  vmlal_s16(q13, d9, d4[1]);
  vmlal_s16(q14, d8, d6[1]);
  vmlal_s16(q15, d9, d6[1]);
  beq(l2);

  vld1_8({d8}, mem[r9]++);
  vmovl_s8(q4, d8);
  vmlal_s16(q8, d8, d0[2]);
  vmlal_s16(q9, d9, d0[2]);
  vmlal_s16(q10, d8, d2[2]);
  vmlal_s16(q11, d9, d2[2]);
  vmlal_s16(q12, d8, d4[2]);
  vmlal_s16(q13, d9, d4[2]);
  vmlal_s16(q14, d8, d6[2]);
  vmlal_s16(q15, d9, d6[2]);
  cmp(r5, 4);
  blo(l2);

  vld1_8({d8}, mem[r9]++);
  vmovl_s8(q4, d8);
  vmlal_s16(q8, d8, d0[3]);
  vmlal_s16(q9, d9, d0[3]);
  vmlal_s16(q10, d8, d2[3]);
  vmlal_s16(q11, d9, d2[3]);
  vmlal_s16(q12, d8, d4[3]);
  vmlal_s16(q13, d9, d4[3]);
  vmlal_s16(q14, d8, d6[3]);
  vmlal_s16(q15, d9, d6[3]);
  beq(l2);

  vld1_8({d8}, mem[r9]++);
  vmovl_s8(q4, d8);
  vmlal_s16(q8, d8, d1[0]);
  vmlal_s16(q9, d9, d1[0]);
  vmlal_s16(q10, d8, d3[0]);
  vmlal_s16(q11, d9, d3[0]);
  vmlal_s16(q12, d8, d5[0]);
  vmlal_s16(q13, d9, d5[0]);
  vmlal_s16(q14, d8, d7[0]);
  vmlal_s16(q15, d9, d7[0]);
  cmp(r5, 6);
  blo(l2);

  vld1_8({d8}, mem[r9]++);
  vmovl_s8(q4, d8);
  vmlal_s16(q8, d8, d1[1]);
  vmlal_s16(q9, d9, d1[1]);
  vmlal_s16(q10, d8, d3[1]);
  vmlal_s16(q11, d9, d3[1]);
  vmlal_s16(q12, d8, d5[1]);
  vmlal_s16(q13, d9, d5[1]);
  vmlal_s16(q14, d8, d7[1]);
  vmlal_s16(q15, d9, d7[1]);
  beq(l2);

  vld1_8({d8}, mem[r9]++);
  vmovl_s8(q4, d8);
  vmlal_s16(q8, d8, d1[2]);
  vmlal_s16(q9, d9, d1[2]);
  vmlal_s16(q10, d8, d3[2]);
  vmlal_s16(q11, d9, d3[2]);
  vmlal_s16(q12, d8, d5[2]);
  vmlal_s16(q13, d9, d5[2]);
  vmlal_s16(q14, d8, d7[2]);
  vmlal_s16(q15, d9, d7[2]);
  b(l2);

  // Store odd width
  align(8);
  bind(l4);
  tst(r1, 4);
  beq(l5);
  vst1_32({d0[0]}, mem[r11]++);
  vst1_32({d1[0]}, mem[r4]++);
  vst1_32({d2[0]}, mem[r8]++);
  vst1_32({d3[0]}, mem[r6]++);
  vext_8(q0, q0, q0, 4);
  vext_8(q1, q1, q1, 4);
  bind(l5);
  tst(r1, 2);
  beq(l6);
  vst1_16({d0[0]}, mem[r11]++);
  vst1_16({d1[0]}, mem[r4]++);
  vst1_16({d2[0]}, mem[r8]++);
  vst1_16({d3[0]}, mem[r6]++);
  vext_8(q0, q0, q0, 2);
  vext_8(q1, q1, q1, 2);

  bind(l6);
  tst(r1, 1);
  beq(l7);
  vst1_8({d0[0]}, mem[r11]);
  vst1_8({d1[0]}, mem[r4]);
  vst1_8({d2[0]}, mem[r8]);
  vst1_8({d3[0]}, mem[r6]);

  bind(l7);
  vpop({d8-d11});
  pop({r4, r5, r6, r7, r8, r9, r10, r11});
  bx(lr);
}
}  // namespace
}  // aarch32
}  // xnnpack

xnn_status_t xnn_generate_qs8_gemm_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, const void* params) {
  using namespace xnnpack::aarch32;
  Generator g(code);
  g.generate(false, max_mr, nc_mod_nr, kc, nullptr);
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}

xnn_status_t xnn_generate_qs8_gemm_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_prfm_ld64(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, const void* params) {
  using namespace xnnpack::aarch32;
  Generator g(code);
  g.generate(true, max_mr, nc_mod_nr, kc, nullptr);
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
