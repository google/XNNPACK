// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <cassert>

#include <xnnpack/aarch32-assembler.h>
#include <xnnpack/allocator.h>
#include <xnnpack/igemm.h>

namespace xnnpack {
namespace aarch32 {
namespace {
class Generator : public Assembler {
  using Assembler::Assembler;
 public:
  void generate(size_t nc_mod_nr, size_t kc, size_t ks, const void* params);
};


// void xnn_qc8_igemm_minmax_fp32_ukernel_4x8c4__aarch32_neondot_ld64(
//     size_t mr,                                      r0
//     size_t nc,                                      r1
//     size_t kc,                                      r2 -> r5 -> sp + 52
//     size_t ks,                                      r3 -> sp + 56 -> r14
//     const int8_t**restrict a,           sp + 96  -> r2
//     const void*restrict w,              sp + 100 -> r9
//     int8_t*restrict c,                  sp + 104 -> r11
//     size_t cm_stride,                   sp + 108 -> (r6)
//     size_t cn_stride,                   sp + 112 -> (r7)
//     size_t a_offset,                    sp + 116 -> (r5)
//     const int8_t* zero,                 sp + 120 -> (r7)
//     xnn_qs8_minmax_params*params); sp + 124 -> (r5)

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage

// A0   r3  d0
// A1  r12  d1
// A2  r10  d2
// A3   r0  d3

// B    r9  q2 q3 q4 q5

// C0  r11 d16-d17  q8  d18-d19  q9
// C1   r4 d20-d21 q10  d22-d23 q11
// C2   r8 d24-d25 q12  d26-d27 q13
// C3   r6 d28-d29 q14  d30-d31 q15

// unused q7

// params structure is 4 bytes
//  struct {
//    int16_t output_zero_point;  d13[2]
//    int8_t output_min;          d13[6]
//    int8_t output_max;          d13[7]
//  } xnn_qs8_minmax_params.neonv8;

void Generator::generate(size_t nc_mod_nr, size_t kc, size_t ks, const void* params)
{
  assert(ks != 0);
  assert(kc != 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8;

  // Auto-generated file. Do not edit!
  add(r2, r2, 3); // kc = (kc + 3) & ~3
  bic(r2, r2, 3);

  // Push 96 bytes
  // r2 will be reloaded in outer loop.  r3 is ks
  push({r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, lr}); // +44
  sub(sp, sp, 4); // 4
  vpush({d8-d13}); // +48 = 96

  ldr(r11, mem[sp, 104]); // c
  ldr(r6, mem[sp, 108]); // cm_stride
  ldr(r2, mem[sp, 96]); // a
  ldr(r9, mem[sp, 100]); // w
  ldr(r5, mem[sp, 124]); // params
  mov(r14, r3); // p = ks

  // Clamp C pointers
  cmp(r0, 2); // if mr >= 2
  add(r4, r11, r6); //   c1 = c0 + cm_stride
  movlo(r4, r11); // c1
  // if mr > 2
  add(r8, r4, r6); //   c2 = c1 + cm_stride
  movls(r8, r4); // c2
  cmp(r0, 4); // if mr >=4
  add(r6, r8, r6); //   c3 = c2 + cm_stride
  movlo(r6, r8); // c3

  // Load params values
  vld1r_32({d13}, mem[r5]); // QC8 params

  bind(l0);
  // Load initial bias from w into accumulators
  vldm(mem[r9]++, {d16-d19}); // Bias
  vmov(q10, q8);
  vmov(q11, q9);
  vmov(q12, q8);
  vmov(q13, q9);
  vmov(q14, q8);
  vmov(q15, q9);

  bind(l1);
  // Load next 4 A pointers
  ldr(r3, mem[r2, 0]);
  ldr(r12, mem[r2, 4]);
  ldr(r10, mem[r2, 8]);
  ldr(r0, mem[r2, 12]);
  add(r2, r2, 16);

  // Add a_offset
  ldr(r5, mem[sp, 116]); // a_offset
  ldr(r7, mem[sp, 120]); // zero
  cmp(r3, r7); // if a0 == zero
  add(r3, r3, r5); // a0 += a_offset
  moveq(r3, r7); //   a0 = zero, else += a0 + a_offset
  cmp(r12, r7); // if a1 == zero
  add(r12, r12, r5); // a1 += a_offset
  moveq(r12, r7); //   a1 = zero, else += a1 + a_offset
  cmp(r10, r7); // if a2 == zero
  add(r10, r10, r5); // a2 += a_offset
  moveq(r10, r7); //   a2 = zero, else += a2 + a_offset
  cmp(r0, r7); // if a3 == zero
  add(r0, r0, r5); // a3 += a_offset
  ldr(r5, mem[sp, 52]); // kc
  moveq(r0, r7); //   a3 = zero, else += a3 + a_offset

  subs(r5, r5, 8); // kc - 8
  blo(l4); // less than 8 channels?

  // Main loop - 8 bytes of A.
  // 16 SDOT, 4 LD64 A, 4 LD128 B
  align(8);
  bind(l2);
  vld1_8({d0}, mem[r3]++); // A0
  vld1_8({q2}, mem[r9]++); // B0
  vld1_8({d1}, mem[r12]++); // A1
  vld1_8({q3}, mem[r9]++); // B1
  vld1_8({d2}, mem[r10]++); // A2
  vld1_8({q4}, mem[r9]++); // B2
  vld1_8({d3}, mem[r0]++); // A3
  vld1_8({q5}, mem[r9]++); // B3
  subs(r5, r5, 8);

  vsdot_s8(q8, q2, d0[0]);
  vsdot_s8(q9, q3, d0[0]);
  vsdot_s8(q10, q2, d1[0]);
  vsdot_s8(q11, q3, d1[0]);
  vsdot_s8(q12, q2, d2[0]);
  vsdot_s8(q13, q3, d2[0]);
  vsdot_s8(q14, q2, d3[0]);
  vsdot_s8(q15, q3, d3[0]);

  vsdot_s8(q8, q4, d0[1]);
  vsdot_s8(q9, q5, d0[1]);
  vsdot_s8(q10, q4, d1[1]);
  vsdot_s8(q11, q5, d1[1]);
  vsdot_s8(q12, q4, d2[1]);
  vsdot_s8(q13, q5, d2[1]);
  vsdot_s8(q14, q4, d3[1]);
  vsdot_s8(q15, q5, d3[1]);
  bhs(l2);

  // Is there a remainder?- 4 bytes of A
  tst(r5, 4);
  bne(l4);

  bind(l3);
  // ks loop
  subs(r14, r14, 16); // ks -= MR * sizeof(void*)
  bhi(l1);

  ldr(r7, mem[sp, 112]); // cn_stride
  ldr(r14, mem[sp, 56]); // p = ks

  // QC8 FP32 quantization
  vld1_8({q0-q1}, mem[r9]++);

  vcvt_f32_s32(q8, q8);
  vcvt_f32_s32(q9, q9);
  vcvt_f32_s32(q10, q10);
  vcvt_f32_s32(q11, q11);
  vcvt_f32_s32(q12, q12);
  vcvt_f32_s32(q13, q13);
  vcvt_f32_s32(q14, q14);
  vcvt_f32_s32(q15, q15);

  vmul_f32(q8, q8, q0); // multiplier
  vmul_f32(q9, q9, q1);
  vmul_f32(q10, q10, q0);
  vmul_f32(q11, q11, q1);
  vmul_f32(q12, q12, q0);
  vmul_f32(q13, q13, q1);
  vmul_f32(q14, q14, q0);
  vmul_f32(q15, q15, q1);

  vcvtn_s32_f32(q8, q8);
  vcvtn_s32_f32(q9, q9);
  vcvtn_s32_f32(q10, q10);
  vcvtn_s32_f32(q11, q11);
  vcvtn_s32_f32(q12, q12);
  vcvtn_s32_f32(q13, q13);
  vcvtn_s32_f32(q14, q14);
  vcvtn_s32_f32(q15, q15);
  vdup_16(q0, d13[2]); // output_zero_point

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

  vdup_8(q12, d13[6]); // output_min

  vqmovn_s16(d0, q8);
  vqmovn_s16(d1, q9);
  vqmovn_s16(d2, q10);
  vqmovn_s16(d3, q11);

  vdup_8(q13, d13[7]); // output_min

  vmax_s8(q0, q0, q12);
  vmax_s8(q1, q1, q12);

  subs(r1, r1, 8); // nc -= 8

  vmin_s8(q0, q0, q13);
  vmin_s8(q1, q1, q13);

  // Store full 4 x 8
  blo(l5);
  vst1_8({d3}, mem[r6], r7);
  vst1_8({d2}, mem[r8], r7);
  vst1_8({d1}, mem[r4], r7);
  vst1_8({d0}, mem[r11], r7);
  sub(r2, r2, r14); // a -= ks
  bhi(l0);

  vpop({d8-d13});
  add(sp, sp, 12); // skip pad, r2, r3
  pop({r4, r5, r6, r7, r8, r9, r10, r11, pc});

  bind(l4);
  // Remainder- 4 bytes of A
  vld1_32({d0[0]}, mem[r3]++); // A0
  vld1_32({q2}, mem[r9]++); // B0
  vld1_32({d1[0]}, mem[r12]++); // A1
  vld1_32({q3}, mem[r9]++); // B1
  vld1_32({d2[0]}, mem[r10]++); // A2
  vld1_32({d3[0]}, mem[r0]++); // A3

  vsdot_s8(q8, q2, d0[0]);
  vsdot_s8(q9, q3, d0[0]);
  vsdot_s8(q10, q2, d1[0]);
  vsdot_s8(q11, q3, d1[0]);
  vsdot_s8(q12, q2, d2[0]);
  vsdot_s8(q13, q3, d2[0]);
  vsdot_s8(q14, q2, d3[0]);
  vsdot_s8(q15, q3, d3[0]);
  b(l3);

  // Store odd width
  align(8);
  bind(l5);
  tst(r1, 4);
  beq(l6);
  vst1_32({d3[0]}, mem[r6]++);
  vst1_32({d2[0]}, mem[r8]++);
  vst1_32({d1[0]}, mem[r4]++);
  vst1_32({d0[0]}, mem[r11]++);
  vext_8(q0, q0, q0, 4);
  vext_8(q1, q1, q1, 4);
  bind(l6);
  tst(r1, 2);
  beq(l7);
  vst1_16({d3[0]}, mem[r6]++);
  vst1_16({d2[0]}, mem[r8]++);
  vst1_16({d1[0]}, mem[r4]++);
  vst1_16({d0[0]}, mem[r11]++);
  vext_8(q0, q0, q0, 2);
  vext_8(q1, q1, q1, 2);

  bind(l7);
  tst(r1, 1);
  beq(l8);
  vst1_8({d3[0]}, mem[r6]);
  vst1_8({d2[0]}, mem[r8]);
  vst1_8({d1[0]}, mem[r4]);
  vst1_8({d0[0]}, mem[r11]);

  bind(l8);
  vpop({d8-d13});
  add(sp, sp, 12); // skip pad, r2, r3
  pop({r4, r5, r6, r7, r8, r9, r10, r11, pc});
}
}  // namespace
}  // aarch32
}  // xnnpack

xnn_status xnn_generate_qc8_igemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64(xnn_code_buffer* code, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
  using namespace xnnpack::aarch32;
  Generator g(code);
  g.generate(nc_mod_nr, kc, ks, nullptr);
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
