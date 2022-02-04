// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/aarch32-assembler.h>
#include <xnnpack/allocator.h>
#include <xnnpack/igemm.h>

namespace xnnpack {
namespace aarch32 {
namespace {
class Generator : public Assembler {
  using Assembler::Assembler;
 public:
  void generate(size_t nc_mod_nr, size_t kc, const void* params);
};


// void xnn_f32_igemm_minmax_ukernel_4x8__aarch32_neon_cortex_a55(
//     size_t mr,                            r0
//     size_t nc,                            r1
//     size_t kc,                            r2 -> r5
//     size_t ks,                            r3 -> sp + 64 -> r14
//     const float**restrict a,  sp + 104 -> (r5)
//     const void*restrict w,    sp + 108 -> r9
//     uint8_t*restrict c,       sp + 112 -> r11
//     size_t cm_stride,         sp + 116 -> (r6)
//     size_t cn_stride,         sp + 120 -> (r0)
//     size_t a_offset,          sp + 124 -> (r5)
//     const float* zero,        sp + 128 -> (r0)
//     minmax_params*params,     sp + 132 -> (r5)

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage

// A0   r3  d0
// A1  r12  d1
// A2  r10  d2
// A3   r7  d3

// B    r9  d8,  d9, d10, d11
// B       d12, d13, d14, d15

// C0  r11 d16-d17  q8  d18-d19  q9
// C1   r4 d20-d21 q10  d22-d23 q11
// C2   r8 d24-d25 q12  d26-d27 q13
// C3   r6 d28-d29 q14  d30-d31 q15

// Clamp (r5) d4 d5 d6 d7

// Converted from: src/f32-igemm/4x8-minmax-aarch32-neon-cortex-a55.S
void Generator::generate(size_t nc_mod_nr, size_t kc, const void* params) {
  Label l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;

  // Push 104 bytes
  push({r3, r4, r5, r6, r7, r8, r9, r10, r11, lr}); // +40
  vpush({d8-d15}); // +64 = 104

  ldr(r11, mem[sp, 112]); // c
  ldr(r6, mem[sp, 116]); // cm_stride
  ldr(r5, mem[sp, 104]); // a
  ldr(r9, mem[sp, 108]); // w
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


  align(8);
  bind(l0);
  // Load initial bias from w into accumulators
  vldm(mem[r9]++, {d16-d19}); // Bias

  vmov(q10, q8);
  vmov(q11, q9);
  vmov(q12, q8);
  vmov(q13, q9);
  pld(mem[r9, 0]); // Prefetch B
  pld(mem[r9, 64]);
  vmov(q14, q8);
  pld(mem[r9, 128]);
  pld(mem[r9, 192]);
  vmov(q15, q9);
  pld(mem[r9, 256]);
  pld(mem[r9, 320]);

  bind(l1);
  // Load next 4 A pointers
  ldr(r3, mem[r5, 0]);
  ldr(r12, mem[r5, 4]);
  ldr(r10, mem[r5, 8]);
  ldr(r7, mem[r5, 12]);
  add(r5, r5, 16);
  pld(mem[r3, 0]); // Prefetch A
  str(r5, mem[sp, 104]); // a
  pld(mem[r3, 64]);
  ldr(r0, mem[sp, 128]); // zero
  pld(mem[r12, 0]);
  ldr(r5, mem[sp, 124]); // a_offset
  pld(mem[r12, 64]);
  pld(mem[r10, 0]);
  pld(mem[r10, 64]);
  pld(mem[r7, 0]);
  pld(mem[r7, 64]);

  // Add a_offset
  cmp(r3, r0); // if a0 == zero
  add(r3, r3, r5); // a0 += a_offset
  moveq(r3, r0); //   a0 = zero, else += a0 + a_offset
  cmp(r12, r0); // if a1 == zero
  add(r12, r12, r5); // a1 += a_offset
  moveq(r12, r0); //   a1 = zero, else += a1 + a_offset
  cmp(r10, r0); // if a2 == zero
  add(r10, r10, r5); // a2 += a_offset
  moveq(r10, r0); //   a2 = zero, else += a2 + a_offset
  cmp(r7, r0); // if a3 == zero
  add(r7, r7, r5); // a3 += a_offset
  moveq(r7, r0); //   a3 = zero, else += a3 + a_offset

  subs(r5, r2, 16); // kc - 16
  blo(l5); // less than 4 channels?

  // Prologue
  vld1_32({d0}, mem[r3]++); // A0
  vld1_32({d1}, mem[r12]++); // A1
  vld1_32({d2}, mem[r10]++); // A2
  vld1_32({d3}, mem[r7]++); // A3
  subs(r5, r5, 16);
  vldm(mem[r9], {d8-d11}); // B0
  vldr(d15, mem[r9, 56]); // B1CK 0
  vldr(d13, mem[r9, 40]); // B1

  blo(l3); // less than 4 channels?  skip main loop

  // Main loop - 4 floats of A (16 bytes)
  // 32 FMA + 8 LD64 A + 8 LDR B
  align(8);
  bind(l2);
  // First group of 16 FMA, Second group loads
  // BLOCK 0
  vmla_f32(q8, q4, d0[0]);
  vld1_32({d4}, mem[r3]++); // A0
  vmla_f32(q10, q4, d1[0]);
  vld1_32({d5}, mem[r12]++); // A1
  vmla_f32(q12, q4, d2[0]);

  // BLOCK 1
  vmla_f32(q14, q4, d3[0]);
  vldr(d12, mem[r9, 32]); // B1
  vmla_f32(q9, q5, d0[0]);
  vldr(d9, mem[r9, 72]); // B0
  vmla_f32(q11, q5, d1[0]);

  // BLOCK 2
  vmla_f32(q13, q5, d2[0]);
  vld1_32({d6}, mem[r10]++); // A2
  vmla_f32(q15, q5, d3[0]);
  vld1_32({d7}, mem[r7]++); // A3
  vmla_f32(q8, q6, d0[1]);

  // BLOCK 3
  vmla_f32(q10, q6, d1[1]);
  vldr(d14, mem[r9, 48]); // B1
  vmla_f32(q12, q6, d2[1]);
  vldr(d11, mem[r9, 88]); // B0
  vmla_f32(q14, q6, d3[1]);

  // BLOCK 4
  vmla_f32(q9, q7, d0[1]);
  vldr(d8, mem[r9, 64]); // B0
  vmla_f32(q11, q7, d1[1]);
  vldr(d13, mem[r9, 104]); // B1
  vmla_f32(q13, q7, d2[1]);
  vldr(d10, mem[r9, 80]); // B0

  // BLOCK 5
  vmla_f32(q15, q7, d3[1]);
  vldr(d15, mem[r9, 120]); // B1

  // Second group of 16 FMA, First group of loads
  // BLOCK 0
  vmla_f32(q8, q4, d4[0]);
  vld1_32({d0}, mem[r3]++); // A0
  vmla_f32(q10, q4, d5[0]);
  vld1_32({d1}, mem[r12]++); // A1
  vmla_f32(q12, q4, d6[0]);

  // BLOCK 1
  vmla_f32(q14, q4, d7[0]);
  vldr(d12, mem[r9, 96]); // B1
  vmla_f32(q9, q5, d4[0]);
  vldr(d9, mem[r9, 136]); // B0
  vmla_f32(q11, q5, d5[0]);

  // BLOCK 2
  vmla_f32(q13, q5, d6[0]);
  vld1_32({d2}, mem[r10]++); // A2
  vmla_f32(q15, q5, d7[0]);
  vld1_32({d3}, mem[r7]++); // A3
  vmla_f32(q8, q6, d4[1]);
  subs(r5, r5, 16);

  // BLOCK 3
  vmla_f32(q10, q6, d5[1]);
  vldr(d14, mem[r9, 112]); // B1
  vmla_f32(q12, q6, d6[1]);
  vldr(d11, mem[r9, 152]); // B0
  vmla_f32(q14, q6, d7[1]);

  // BLOCK 4
  vmla_f32(q9, q7, d4[1]);
  vldr(d8, mem[r9, 128]); // B0
  vmla_f32(q11, q7, d5[1]);
  vldr(d13, mem[r9, 168]); // B1
  vmla_f32(q13, q7, d6[1]);
  vldr(d10, mem[r9, 144]); // B0

  // BLOCK 5
  vmla_f32(q15, q7, d7[1]);
  vldr(d15, mem[r9, 184]); // B1
  add(r9, r9, 128); // B++
  bhs(l2);

  // Epilogue - 4 floats of A (16 bytes)
  bind(l3);
  // First group of 16 FMA, Second group loads
  // BLOCK 0
  vmla_f32(q8, q4, d0[0]);
  vld1_32({d4}, mem[r3]++); // A0
  vmla_f32(q10, q4, d1[0]);
  vld1_32({d5}, mem[r12]++); // A1
  vmla_f32(q12, q4, d2[0]);

  // BLOCK 1
  vmla_f32(q14, q4, d3[0]);
  vldr(d12, mem[r9, 32]); // B1
  vmla_f32(q9, q5, d0[0]);
  vldr(d9, mem[r9, 72]); // B0
  vmla_f32(q11, q5, d1[0]);

  // BLOCK 2
  vmla_f32(q13, q5, d2[0]);
  vld1_32({d6}, mem[r10]++); // A2
  vmla_f32(q15, q5, d3[0]);
  vld1_32({d7}, mem[r7]++); // A3
  vmla_f32(q8, q6, d0[1]);

  // BLOCK 3
  vmla_f32(q10, q6, d1[1]);
  vldr(d14, mem[r9, 48]); // B1
  vmla_f32(q12, q6, d2[1]);
  vldr(d11, mem[r9, 88]); // B0
  vmla_f32(q14, q6, d3[1]);

  // BLOCK 4
  vmla_f32(q9, q7, d0[1]);
  vldr(d8, mem[r9, 64]); // B0
  vmla_f32(q11, q7, d1[1]);
  vldr(d13, mem[r9, 104]); // B1
  vmla_f32(q13, q7, d2[1]);
  vldr(d10, mem[r9, 80]); // B0

  // BLOCK 5
  vmla_f32(q15, q7, d3[1]);
  vldr(d15, mem[r9, 120]); // B1

  // Second group of 16 FMA, First group of loads
  // BLOCK 0
  vmla_f32(q8, q4, d4[0]);
  vldr(d12, mem[r9, 96]); // B1
  vmla_f32(q10, q4, d5[0]);
  vmla_f32(q12, q4, d6[0]);

  // BLOCK 1
  vmla_f32(q14, q4, d7[0]);
  vldr(d14, mem[r9, 112]); // B1
  vmla_f32(q9, q5, d4[0]);
  vmla_f32(q11, q5, d5[0]);

  // BLOCK 2
  vmla_f32(q13, q5, d6[0]);
  vmla_f32(q15, q5, d7[0]);
  vmla_f32(q8, q6, d4[1]);
  add(r9, r9, 128); // B++

  // BLOCK 3
  vmla_f32(q10, q6, d5[1]);
  vmla_f32(q12, q6, d6[1]);
  vmla_f32(q14, q6, d7[1]);
  tst(r5, 15);

  // BLOCK 4
  vmla_f32(q9, q7, d4[1]);
  vmla_f32(q11, q7, d5[1]);
  vmla_f32(q13, q7, d6[1]);

  // BLOCK 5
  vmla_f32(q15, q7, d7[1]);

  // Is there a remainder?- 1 to 3 floats of A (4, 8 or 12 bytes)
  bne(l5);

  align(8);
  bind(l4);
  ldr(r5, mem[sp, 104]); // a
  subs(r14, r14, 16); // ks -= MR * sizeof(void*)

  // ks loop
  bhi(l1);

  // Load params pointer
  ldr(r0, mem[sp, 132]); // params
  ldr(r14, mem[sp, 64]); // p = ks
  // Load min/max values
  vld1r_32({d4,d5}, mem[r0]++);
  vld1r_32({d6,d7}, mem[r0]);
  subs(r1, r1, 8);
  ldr(r0, mem[sp, 120]); // cn_stride

  // Clamp
  vmax_f32(q8, q8, q2);
  vmax_f32(q9, q9, q2);
  vmax_f32(q10, q10, q2);
  vmax_f32(q11, q11, q2);
  vmax_f32(q12, q12, q2);
  vmax_f32(q13, q13, q2);
  vmax_f32(q14, q14, q2);
  vmax_f32(q15, q15, q2);
  vmin_f32(q8, q8, q3);
  vmin_f32(q9, q9, q3);
  vmin_f32(q10, q10, q3);
  vmin_f32(q11, q11, q3);
  vmin_f32(q12, q12, q3);
  vmin_f32(q13, q13, q3);
  vmin_f32(q14, q14, q3);
  vmin_f32(q15, q15, q3);

  // Store full 4 x 8
  blo(l7);
  vst1_32({d28-d31}, mem[r6], r0);
  vst1_32({d24-d27}, mem[r8], r0);
  vst1_32({d20-d23}, mem[r4], r0);
  vst1_32({d16-d19}, mem[r11], r0);

  sub(r5, r5, r14); // a -= ks

  bhi(l0);

  vpop({d8-d15});
  add(sp, sp, 4); // skip r3
  pop({r4, r5, r6, r7, r8, r9, r10, r11, pc});

  align(8);
  bind(l5);
  // Is there a remainder?- 2 floats of A (8 bytes)
  tst(r5, 8);
  beq(l6);

  // Remainder - 2 floats of A (8 bytes)
  vld1_32({d0}, mem[r3]++); // A0
  vldm(mem[r9]++, {d8-d11}); // B0
  vld1_32({d1}, mem[r12]++); // A1
  vld1_32({d2}, mem[r10]++); // A2
  vld1_32({d3}, mem[r7]++); // A3

  vmla_f32(q8, q4, d0[0]);
  vmla_f32(q9, q5, d0[0]);
  vmla_f32(q10, q4, d1[0]);
  vmla_f32(q11, q5, d1[0]);
  vldm(mem[r9]++, {d12-d15}); // B1
  vmla_f32(q12, q4, d2[0]);
  vmla_f32(q13, q5, d2[0]);
  vmla_f32(q14, q4, d3[0]);
  vmla_f32(q15, q5, d3[0]);
  vmla_f32(q8, q6, d0[1]);
  vmla_f32(q9, q7, d0[1]);
  vmla_f32(q10, q6, d1[1]);
  vmla_f32(q11, q7, d1[1]);
  vmla_f32(q12, q6, d2[1]);
  vmla_f32(q13, q7, d2[1]);
  vmla_f32(q14, q6, d3[1]);
  vmla_f32(q15, q7, d3[1]);

  // Is there a remainder?- 1 floats of A (4 bytes)
  tst(r5, 4);
  beq(l4);

  bind(l6);
  // Remainder- 1 floats of A (4 bytes)
  vldm(mem[r3]++, {s0}); // A0
  vldm(mem[r9]++, {d8-d11}); // B0
  vldm(mem[r12]++, {s2}); // A1
  vldm(mem[r10]++, {s4}); // A2
  vldm(mem[r7]++, {s6}); // A3
  vmla_f32(q8, q4, d0[0]);
  vmla_f32(q9, q5, d0[0]);
  vmla_f32(q10, q4, d1[0]);
  vmla_f32(q11, q5, d1[0]);
  vmla_f32(q12, q4, d2[0]);
  vmla_f32(q13, q5, d2[0]);
  vmla_f32(q14, q4, d3[0]);
  vmla_f32(q15, q5, d3[0]);
  b(l4);

  // Store odd width
  bind(l7);
  tst(r1, 4);
  beq(l8);
  vst1_32({d28-d29}, mem[r6]++);
  vst1_32({d24-d25}, mem[r8]++);
  vmov(q14, q15);
  vmov(q12, q13);
  vst1_32({d20-d21}, mem[r4]++);
  vst1_32({d16-d17}, mem[r11]++);
  vmov(q10, q11);
  vmov(q8, q9);

  bind(l8);
  tst(r1, 2);
  beq(l9);
  vst1_32({d28}, mem[r6]++);
  vst1_32({d24}, mem[r8]++);
  vmov(d28, d29);
  vmov(d24, d25);
  vst1_32({d20}, mem[r4]++);
  vst1_32({d16}, mem[r11]++);
  vmov(d20, d21);
  vmov(d16, d17);

  bind(l9);
  tst(r1, 1);
  beq(l10);
  vst1_32({d28[0]}, mem[r6]++);
  vst1_32({d24[0]}, mem[r8]++);
  vst1_32({d20[0]}, mem[r4]++);
  vst1_32({d16[0]}, mem[r11]++);

  bind(l10);
  vpop({d8-d15});
  add(sp, sp, 4); // skip r3
  pop({r4, r5, r6, r7, r8, r9, r10, r11, pc});
}
}  // namespace
}  // aarch32
}  // xnnpack

xnn_status xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a55(xnn_code_buffer* code, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
  using namespace xnnpack::aarch32;
  Generator g(code);
  g.generate(nc_mod_nr, kc, nullptr);
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
