// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <limits>

#include <xnnpack.h>
#include <xnnpack/aarch32-assembler.h>
#include <xnnpack/igemm.h>
#include <xnnpack/log.h>
#include <xnnpack/memory.h>
#include <xnnpack/microparams.h>
#include <xnnpack/post-operation.h>

namespace xnnpack {
namespace aarch32 {
namespace {
class Generator : public MacroAssembler {
  using MacroAssembler::MacroAssembler;

 public:
  void generate(bool prefetch, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const jit_gemm_params* jit_gemm_params);
  void perform_post_operations(size_t max_mr, size_t num_post_operations, const xnn_post_operation* post_operations);
};


// void xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm(
//     size_t mr,                            r0
//     size_t nc,                            r1
//     size_t kc,                            r2 -> r5 -> sp + 68
//     size_t ks,                            r3 -> sp + 72 -> r14
//     const float** restrict a,  sp + 112 -> r2
//     const void* restrict w,    sp + 116 -> r9
//     uint8_t* restrict c,       sp + 120 -> r11
//     size_t cm_stride,         sp + 124 -> (r6)
//     size_t cn_stride,         sp + 128 -> (r7)
//     size_t a_offset,          sp + 132 -> (r5)
//     const float* zero,        sp + 136 -> (r7)
//     minmax_params*params,     sp + 140 -> (r5)

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// A0   r3  d0 d4
// A1  r12  d1 d5
// A2  r10  d2 d6
// A3   r0  d3 d7
// B    r9  d8,  d9, d10, d11
// B       d12, d13, d14, d15
// C0  r11 d16-d17  q8  d18-d19  q9
// C1   r4 d20-d21 q10  d22-d23 q11
// C2   r8 d24-d25 q12  d26-d27 q13
// C3   r6 d28-d29 q14  d30-d31 q15
// clamp  (r5) d4 d5 d6 d7

// Converted from: src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch32-neon-cortex-a75-prfm.S
void Generator::generate(bool prefetch, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const jit_gemm_params* jit_gemm_params)
{
  assert(max_mr <= 4);
  assert(nc_mod_nr < 8 || nc_mod_nr == SIZE_MAX);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;
  const size_t num_post_operations = jit_gemm_params->num_post_operations;
  const xnn_post_operation* post_operations = jit_gemm_params->post_operations;
  const float min = jit_gemm_params->f32_minmax.min;
  const float max = jit_gemm_params->f32_minmax.max;
  const bool clamp_min = min != -std::numeric_limits<float>::infinity();
  const bool clamp_max = max != +std::numeric_limits<float>::infinity();
  assert(num_post_operations == 0 || (!clamp_min && !clamp_max));
  // Push 112 bytes
  // r2 will be reloaded in outer loop.  r3 is ks
  push({r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, lr}); // +44
  sub(sp, sp, 4); // 4
  vpush({d8-d15}); // +64 = 112

  ldr(r11, mem[sp, 120]); // c
  ldr(r6, mem[sp, 124]); // cm_stride
  ldr(r2, mem[sp, 112]); // a
  ldr(r9, mem[sp, 116]); // w
  mov(r14, r3); // p = ks

  // Clamp C pointers
  if (max_mr > 1) {
    cmp(r0, 2); // if mr >= 2
    add(r4, r11, r6); //   c1 = c0 + cm_stride
    movlo(r4, r11); // c1
  }
  if (max_mr > 2) {
    // if mr > 2
    add(r8, r4, r6); //   c2 = c1 + cm_stride
    movls(r8, r4); // c2
  }
  if (max_mr > 3) {
    cmp(r0, 4); // if mr >=4
    add(r6, r8, r6); //   c3 = c2 + cm_stride
    movlo(r6, r8); // c3
  }

  align(8);
  bind(l0);
  // Load initial bias from w into accumulators
  vldm(mem[r9]++, {d16-d19}); // Bias
  if (max_mr > 1) {
    vmov(q10, q8);
    vmov(q11, q9);
  }
  if (max_mr > 2) {
    vmov(q12, q8);
    vmov(q13, q9);
  }
  if (max_mr > 3) {
    vmov(q14, q8);
    vmov(q15, q9);
  }

  if (prefetch) {
    pld(mem[r9, 0]); // Prefetch B
    pld(mem[r9, 64]);
    pld(mem[r9, 128]);
    pld(mem[r9, 192]);
    pld(mem[r9, 256]);
    pld(mem[r9, 320]);
    pld(mem[r9, 384]);
  }

  bind(l1);
  // Load next 4 A pointers
  ldr(r3, mem[r2, 0]);
  if (max_mr > 1) {
    ldr(r12, mem[r2, 4]);
  }
  if (max_mr > 2) {
    ldr(r10, mem[r2, 8]);
  }
  if (max_mr > 3) {
    ldr(r0, mem[r2, 12]);
  }
  add(r2, r2, max_mr * sizeof(void*)); // a += MR * sizeof(void*)

  // Add a_offset
  ldr(r5, mem[sp, 132]); // a_offset
  ldr(r7, mem[sp, 136]); // zero
  cmp(r3, r7); // if a0 == zero
  add(r3, r3, r5); // a0 += a_offset
  moveq(r3, r7); //   a0 = zero, else += a0 + a_offset
  if (max_mr > 1) {
    cmp(r12, r7); // if a1 == zero
    add(r12, r12, r5); // a1 += a_offset
    moveq(r12, r7); //   a1 = zero, else += a1 + a_offset
  }
  if (max_mr > 2) {
    cmp(r10, r7); // if a2 == zero
    add(r10, r10, r5); // a2 += a_offset
    moveq(r10, r7); //   a2 = zero, else += a2 + a_offset
  }
  if (max_mr > 3) {
    cmp(r0, r7); // if a3 == zero
    add(r0, r0, r5); // a3 += a_offset
  }
  ldr(r5, mem[sp, 68]); // kc
  if (max_mr > 3) {
    moveq(r0, r7); //   a3 = zero, else += a3 + a_offset
  }

  if (prefetch) {
    pld(mem[r3, 0]); // Prefetch A
    pld(mem[r3, 64]);
    pld(mem[r12, 0]);
    pld(mem[r12, 64]);
    pld(mem[r10, 0]);
    pld(mem[r10, 64]);
    pld(mem[r0, 0]);
    pld(mem[r0, 64]);
  }

  subs(r5, r5, 16); // kc - 16
  blo(l5); // less than 4 channels?

  // Prologue
  vld1_32({d0}, mem[r3]++); // A0
  vldm(mem[r9]++, {d8-d11}); // B0
  if (max_mr > 1) {
    vld1_32({d1}, mem[r12]++); // A1
  }
  if (max_mr > 2) {
    vld1_32({d2}, mem[r10]++); // A2
  }
  if (max_mr > 3) {
    vld1_32({d3}, mem[r0]++); // A3
  }

  subs(r5, r5, 16);
  blo(l3); // less than 4 channels?  skip main loop

  align(8);

  // Main loop - 4 floats of A (16 bytes)
  bind(l2);
  vmla_f32(q8, q4, d0[0]);
  vldm(mem[r9]++, {d12-d15}); // B1
  if (max_mr > 1) {
    vmla_f32(q10, q4, d1[0]);
  }
  if (max_mr > 2) {
    vmla_f32(q12, q4, d2[0]);
  }
  vld1_32({d4}, mem[r3]++); // A0
  if (max_mr > 3) {
    vmla_f32(q14, q4, d3[0]);
  }
  vmla_f32(q9, q5, d0[0]);
  if (max_mr > 1) {
    vld1_32({d5}, mem[r12]++); // A1
    vmla_f32(q11, q5, d1[0]);
  }
  if (max_mr > 2) {
    vmla_f32(q13, q5, d2[0]);
  }
  if (max_mr > 3) {
    vmla_f32(q15, q5, d3[0]);
  }
  if (max_mr > 2) {
    vld1_32({d6}, mem[r10]++); // A2
  }
  vmla_f32(q8, q6, d0[1]);
  if (max_mr > 1) {
    vmla_f32(q10, q6, d1[1]);
  }
  if (max_mr > 3) {
    vld1_32({d7}, mem[r0]++); // A3
  }
  if (max_mr > 2) {
    vmla_f32(q12, q6, d2[1]);
  }
  if (max_mr > 3) {
    vmla_f32(q14, q6, d3[1]);
  }
  vldm(mem[r9]++, {d8-d11}); // B0
  vmla_f32(q9, q7, d0[1]);
  if (max_mr > 1) {
    vmla_f32(q11, q7, d1[1]);
  }
  if (max_mr > 2) {
    vmla_f32(q13, q7, d2[1]);
  }
  if (max_mr > 3) {
    vmla_f32(q15, q7, d3[1]);
  }

  vmla_f32(q8, q4, d4[0]);
  vldm(mem[r9]++, {d12-d15}); // B1
  if (max_mr > 1) {
    vmla_f32(q10, q4, d5[0]);
  }
  if (prefetch) {
    pld(mem[r3, 128]); // Prefetch A0
  }
  if (max_mr > 2) {
    vmla_f32(q12, q4, d6[0]);
  }
  vld1_32({d0}, mem[r3]++); // A0
  if (max_mr > 3) {
    vmla_f32(q14, q4, d7[0]);
  }
  if (prefetch) {
    pld(mem[r12, 128]); // Prefetch A1
  }
  vmla_f32(q9, q5, d4[0]);
  if (max_mr > 1) {
    vld1_32({d1}, mem[r12]++); // A1
    vmla_f32(q11, q5, d5[0]);
  }
  if (prefetch) {
    pld(mem[r10, 128]); // Prefetch A2
  }
  if (max_mr > 2) {
    vmla_f32(q13, q5, d6[0]);
    vld1_32({d2}, mem[r10]++); // A2
  }
  if (max_mr > 3) {
    vmla_f32(q15, q5, d7[0]);
  }
  if (prefetch) {
    pld(mem[r0, 128]); // Prefetch A3
  }
  vmla_f32(q8, q6, d4[1]);
  if (max_mr > 3) {
    vld1_32({d3}, mem[r0]++); // A3
  }
  if (max_mr > 1) {
    vmla_f32(q10, q6, d5[1]);
  }
  if (prefetch) {
    pld(mem[r9, 352]); // Prefetch B
  }
  if (max_mr > 2) {
    vmla_f32(q12, q6, d6[1]);
  }
  if (prefetch) {
    pld(mem[r9, 416]); // Prefetch B
  }
  if (max_mr > 3) {
    vmla_f32(q14, q6, d7[1]);
  }
  vldm(mem[r9]++, {d8-d11}); // B0
  vmla_f32(q9, q7, d4[1]);
  if (max_mr > 1) {
    vmla_f32(q11, q7, d5[1]);
  }
  subs(r5, r5, 16);
  if (max_mr > 2) {
    vmla_f32(q13, q7, d6[1]);
  }
  if (max_mr > 3) {
    vmla_f32(q15, q7, d7[1]);
  }
  bhs(l2);

  // Epilogue
  bind(l3);
  vmla_f32(q8, q4, d0[0]);
  vldm(mem[r9]++, {d12-d15}); // B1
  if (max_mr > 1) {
    vmla_f32(q10, q4, d1[0]);
  }
  if (max_mr > 2) {
    vmla_f32(q12, q4, d2[0]);
  }
  vld1_32({d4}, mem[r3]++); // A0
  if (max_mr > 3) {
    vmla_f32(q14, q4, d3[0]);
  }
  vmla_f32(q9, q5, d0[0]);
  if (max_mr > 1) {
    vld1_32({d5}, mem[r12]++); // A1
    vmla_f32(q11, q5, d1[0]);
  }
  if (max_mr > 2) {
    vmla_f32(q13, q5, d2[0]);
  }
  if (max_mr > 3) {
    vmla_f32(q15, q5, d3[0]);
  }
  if (max_mr > 2) {
    vld1_32({d6}, mem[r10]++); // A2
  }
  vmla_f32(q8, q6, d0[1]);
  if (max_mr > 1) {
    vmla_f32(q10, q6, d1[1]);
  }
  if (max_mr > 3) {
    vld1_32({d7}, mem[r0]++); // A3
  }
  if (max_mr > 2) {
    vmla_f32(q12, q6, d2[1]);
  }
  if (max_mr > 3) {
    vmla_f32(q14, q6, d3[1]);
  }
  vldm(mem[r9]++, {d8-d11}); // B0
  vmla_f32(q9, q7, d0[1]);
  if (max_mr > 1) {
    vmla_f32(q11, q7, d1[1]);
  }
  if (max_mr > 2) {
    vmla_f32(q13, q7, d2[1]);
  }
  if (max_mr > 3) {
    vmla_f32(q15, q7, d3[1]);
  }

  vmla_f32(q8, q4, d4[0]);
  vldm(mem[r9]++, {d12-d15}); // B1
  if (max_mr > 1) {
    vmla_f32(q10, q4, d5[0]);
  }
  if (max_mr > 2) {
    vmla_f32(q12, q4, d6[0]);
  }
  if (max_mr > 3) {
    vmla_f32(q14, q4, d7[0]);
  }
  vmla_f32(q9, q5, d4[0]);
  if (max_mr > 1) {
    vmla_f32(q11, q5, d5[0]);
  }
  if (max_mr > 2) {
    vmla_f32(q13, q5, d6[0]);
  }
  if (max_mr > 3) {
    vmla_f32(q15, q5, d7[0]);
  }
  vmla_f32(q8, q6, d4[1]);
  if (max_mr > 1) {
    vmla_f32(q10, q6, d5[1]);
  }
  if (max_mr > 2) {
    vmla_f32(q12, q6, d6[1]);
  }
  if (max_mr > 3) {
    vmla_f32(q14, q6, d7[1]);
  }
  vmla_f32(q9, q7, d4[1]);
  if (max_mr > 1) {
    vmla_f32(q11, q7, d5[1]);
  }
  if (max_mr > 2) {
    vmla_f32(q13, q7, d6[1]);
  }
  if (max_mr > 3) {
    vmla_f32(q15, q7, d7[1]);
  }

  // Is there a remainder?- 1 to 3 floats of A (4, 8 or 12 bytes)
  tst(r5, 12);
  bne(l5);

  align(8);
  bind(l4);
  // ks loop
  subs(r14, r14, max_mr * sizeof(void*)); // ks -= MR * sizeof(void*)
  bhi(l1);

  // Load params pointer
  ldr(r5, mem[sp, 140]); // params
  ldr(r7, mem[sp, 128]); // cn_stride
  ldr(r14, mem[sp, 72]); // p = ks

  // Load min/max values
  if (clamp_min || clamp_max) {
    vld1r_32({d4,d5}, mem[r5]++);
  }
  subs(r1, r1, 8);
  if (clamp_min || clamp_max) {
    vld1r_32({d6,d7}, mem[r5]);
  }

  // Clamp
  if (clamp_min) {
    vmax_f32(q8, q8, q2);
    vmax_f32(q9, q9, q2);
    if (max_mr > 1) {
      vmax_f32(q10, q10, q2);
      vmax_f32(q11, q11, q2);
    }
    if (max_mr > 2) {
      vmax_f32(q12, q12, q2);
      vmax_f32(q13, q13, q2);
    }
    if (max_mr > 3) {
      vmax_f32(q14, q14, q2);
      vmax_f32(q15, q15, q2);
    }
  }
  if (clamp_max) {
    vmin_f32(q8, q8, q3);
    vmin_f32(q9, q9, q3);
    if (max_mr > 1) {
      vmin_f32(q10, q10, q3);
      vmin_f32(q11, q11, q3);
    }
    if (max_mr > 2) {
      vmin_f32(q12, q12, q3);
      vmin_f32(q13, q13, q3);
    }
    if (max_mr > 3) {
      vmin_f32(q14, q14, q3);
      vmin_f32(q15, q15, q3);
    }
  }
  perform_post_operations(max_mr, num_post_operations, post_operations);

  // Store full 4 x 8
  blo(l7);
  if (max_mr > 3) {
    vst1_32({d28-d31}, mem[r6], r7);
  }
  if (max_mr > 2) {
    vst1_32({d24-d27}, mem[r8], r7);
  }
  if (max_mr > 1) {
    vst1_32({d20-d23}, mem[r4], r7);
  }
  vst1_32({d16-d19}, mem[r11], r7);
  sub(r2, r2, r14); // a -= ks
  bhi(l0);

  vpop({d8-d15});
  add(sp, sp, 12); // skip pad, r2, r3
  pop({r4, r5, r6, r7, r8, r9, r10, r11, pc});

  align(8);
  bind(l5);
  // Is there a remainder?- 2 floats of A (8 bytes)
  tst(r5, 8);
  beq(l6);

  // Remainder - 2 floats of A (8 bytes)
  vld1_32({d0}, mem[r3]++); // A0
  vldm(mem[r9]++, {d8-d11}); // B0
  if (max_mr > 1) {
    vld1_32({d1}, mem[r12]++); // A1
  }
  if (max_mr > 2) {
    vld1_32({d2}, mem[r10]++); // A2
  }
  if (max_mr > 3) {
    vld1_32({d3}, mem[r0]++); // A3
  }

  vmla_f32(q8, q4, d0[0]);
  vmla_f32(q9, q5, d0[0]);
  if (max_mr > 1) {
    vmla_f32(q10, q4, d1[0]);
    vmla_f32(q11, q5, d1[0]);
  }
  vldm(mem[r9]++, {d12-d15}); // B1
  if (max_mr > 2) {
    vmla_f32(q12, q4, d2[0]);
    vmla_f32(q13, q5, d2[0]);
  }
  if (max_mr > 3) {
    vmla_f32(q14, q4, d3[0]);
    vmla_f32(q15, q5, d3[0]);
  }
  vmla_f32(q8, q6, d0[1]);
  vmla_f32(q9, q7, d0[1]);
  if (max_mr > 1) {
    vmla_f32(q10, q6, d1[1]);
    vmla_f32(q11, q7, d1[1]);
  }
  if (max_mr > 2) {
    vmla_f32(q12, q6, d2[1]);
    vmla_f32(q13, q7, d2[1]);
  }
  if (max_mr > 3) {
    vmla_f32(q14, q6, d3[1]);
    vmla_f32(q15, q7, d3[1]);
  }

  // Is there a remainder?- 1 float of A (4 bytes)
  tst(r5, 4);
  beq(l4);

  bind(l6);
  // Remainder- 1 float of A (4 bytes)
  vldm(mem[r3]++, {s0}); // A0
  vldm(mem[r9]++, {d8-d11}); // B0
  if (max_mr > 1) {
    vldm(mem[r12]++, {s2}); // A1
  }
  if (max_mr > 2) {
    vldm(mem[r10]++, {s4}); // A2
  }
  if (max_mr > 3) {
    vldm(mem[r0]++, {s6}); // A3
  }
  vmla_f32(q8, q4, d0[0]);
  vmla_f32(q9, q5, d0[0]);
  if (max_mr > 1) {
    vmla_f32(q10, q4, d1[0]);
    vmla_f32(q11, q5, d1[0]);
  }
  if (max_mr > 2) {
    vmla_f32(q12, q4, d2[0]);
    vmla_f32(q13, q5, d2[0]);
  }
  if (max_mr > 3) {
    vmla_f32(q14, q4, d3[0]);
    vmla_f32(q15, q5, d3[0]);
  }
  b(l4);

  // Store odd width
  bind(l7);
  tst(r1, 4);
  beq(l8);
  if (max_mr > 3) {
    vst1_32({d28-d29}, mem[r6]++);
  }
  if (max_mr > 2) {
    vst1_32({d24-d25}, mem[r8]++);
  }
  if (max_mr > 3) {
    vmov(q14, q15);
  }
  if (max_mr > 2) {
    vmov(q12, q13);
  }
  if (max_mr > 1) {
    vst1_32({d20-d21}, mem[r4]++);
  }
  vst1_32({d16-d17}, mem[r11]++);
  if (max_mr > 1) {
    vmov(q10, q11);
  }
  vmov(q8, q9);

  bind(l8);
  tst(r1, 2);
  beq(l9);
  if (max_mr > 3) {
    vst1_32({d28}, mem[r6]++);
  }
  if (max_mr > 2) {
    vst1_32({d24}, mem[r8]++);
  }
  if (max_mr > 3) {
    vmov(d28, d29);
  }
  if (max_mr > 2) {
    vmov(d24, d25);
  }
  if (max_mr > 1) {
    vst1_32({d20}, mem[r4]++);
  }
  vst1_32({d16}, mem[r11]++);
  if (max_mr > 1) {
    vmov(d20, d21);
  }
  vmov(d16, d17);

  bind(l9);
  tst(r1, 1);
  beq(l10);
  if (max_mr > 3) {
    vst1_32({d28[0]}, mem[r6]++);
  }
  if (max_mr > 2) {
    vst1_32({d24[0]}, mem[r8]++);
  }
  if (max_mr > 1) {
    vst1_32({d20[0]}, mem[r4]++);
  }
  vst1_32({d16[0]}, mem[r11]++);

  bind(l10);
  vpop({d8-d15});
  add(sp, sp, 12); // skip pad, r2, r3
  pop({r4, r5, r6, r7, r8, r9, r10, r11, pc});

  align(16);
}

void Generator::perform_post_operations(
  size_t max_mr,
  size_t num_post_operations,
  const xnn_post_operation* post_operations)
{
  if (num_post_operations == 0) {
    return;
  }
  for (size_t i = 0; i < num_post_operations; i++) {
    switch (post_operations[i].op_type) {
      case xnn_post_operation_type_hardswish: {
        const auto sixth = q0;
        const auto three = q1;
        const auto six = q2;
        const auto zero = q3;
        vld3r_32({sixth.low(), three.low(), six.low()}, mem[r5]++);
        vmov(zero, 0);
        vmov(three.high(), three.low());
        vmov(six.high(), six.low());
        const QRegister accs[] = {q8, q9, q10, q11, q12, q13, q14, q15};
        const QRegister tmps[] = {q4, q5, q6, q7};
        f32_hardswish(sixth, three, six, zero, &accs[0], XNN_COUNT_OF(accs), &tmps[0], XNN_COUNT_OF(tmps));
        break;
      }
      default:
        XNN_LOG_UNREACHABLE("unsupported post operation: %u", post_operations[i].op_type);
    }
  }
}

}  // namespace
}  // namespace aarch32
}  // namespace xnnpack

xnn_status_t xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
  using namespace xnnpack::aarch32;
  Generator g(code);
  assert(params != nullptr);
  g.generate(false, max_mr, nc_mod_nr, kc, ks, static_cast<const jit_gemm_params*>(params));
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}

xnn_status_t xnn_generate_f32_igemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
  using namespace xnnpack::aarch32;
  Generator g(code);
  assert(params != nullptr);
  g.generate(true, max_mr, nc_mod_nr, kc, ks, static_cast<const jit_gemm_params*>(params));
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
