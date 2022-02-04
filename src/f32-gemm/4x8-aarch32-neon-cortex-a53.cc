// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/aarch32-assembler.h>
#include <xnnpack/allocator.h>
#include <xnnpack/gemm.h>

#include <cassert>
#include <limits>

namespace xnnpack {
namespace aarch32 {
namespace {
class Generator : public Assembler {
  using Assembler::Assembler;
 public:
  void generate(size_t nc_mod_nr, size_t kc, float min, float max);
};


// void xnn_f32_gemm_minmax_ukernel_4x8__aarch32_neon_cortex_a53(
//     size_t mr,                            r0
//     size_t nc,                            r1
//     size_t kc,                            r2 -> r5 -> sp + 0
//     const uint8_t*restrict a,             r3
//     size_t a_stride,          sp + 100 -> (r7)
//     const void*restrict w,    sp + 104 -> r9
//     uint8_t*restrict c,       sp + 108 -> r11
//     size_t cm_stride,         sp + 112 -> (r6)
//     size_t cn_stride,         sp + 116 -> (r0)
//     const union xnn_f32_minmax_params params)  sp + 120 -> (r5)

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage

// r0, r2   scratch temporaries for loads
// r14 (lr) unused

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

// Converted from: src/f32-gemm/4x8-minmax-aarch32-neon-cortex-a53.S
void Generator::generate(size_t nc_mod_nr, size_t kc, float min, float max) {
  assert(kc % sizeof(float) == 0);

  Label nc_loop, kc_loop, epilogue, clamp, remainder_kc, store_odd_width;
  const bool clamp_min = min != -std::numeric_limits<float>::infinity();
  const bool clamp_max = max != +std::numeric_limits<float>::infinity();

  // Push 100 bytes
  // r2 will be reloaded in outer loop
  vpush({d8-d15}); // 64
  push({r2, r4, r5, r6, r7, r8, r9, r10, r11}); // +36 = 100

  ldr(r7, mem[sp, 100]); // a_stride
  ldr(r11, mem[sp, 108]); // c
  ldr(r6, mem[sp, 112]); // cm_stride
  ldr(r9, mem[sp, 104]); // w

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
  add(r7, r10, r7); //   a3 = a2 + a_stride
  add(r6, r8, r6); //   c3 = c2 + cm_stride
  movlo(r7, r10); // a3
  movlo(r6, r8); // c3

  align(8);
  bind(nc_loop);
  // Load initial bias from w into accumulators
  vldm(mem[r9]++, {d16-d19}); // Bias

  subs(r5, r2, 16); // kc - 16
  pld(mem[r3, 0]); // Prefetch A
  pld(mem[r3, 64]);
  vmov(q10, q8);
  pld(mem[r12, 0]);
  pld(mem[r12, 64]);
  vmov(q11, q9);
  pld(mem[r10, 0]);
  pld(mem[r10, 64]);
  vmov(q12, q8);
  pld(mem[r7, 0]);
  pld(mem[r7, 64]);
  vmov(q13, q9);
  pld(mem[r9, 0]); // Prefetch B
  pld(mem[r9, 64]);
  vmov(q14, q8);
  pld(mem[r9, 128]);
  pld(mem[r9, 192]);
  vmov(q15, q9);
  pld(mem[r9, 256]);
  pld(mem[r9, 320]);
  blo(remainder_kc); // less than 4 channels?

  // Prologue
  vld1_32({d0}, mem[r3]++); // A0
  vld1_32({d1}, mem[r12]++); // A1
  vld1_32({d2}, mem[r10]++); // A2
  vld1_32({d3}, mem[r7]++); // A3
  subs(r5, r5, 16);
  vldm(mem[r9], {d8-d11}); // B0
  ldr(r0, mem[r9, 56]); // B1 low   VMOV is in BLOCK 0
  ldr(r2, mem[r9, 60]); // B1 high
  vldr(d13, mem[r9, 40]); // B1

  blo(epilogue); // less than 4 channels?  skip main loop

  // Main loop - 4 floats of A (16 bytes)
  // 32 FMA + 8 LD64 A + 8 LDR B
  align(8);
  bind(kc_loop);
  // First group of 16 FMA, Second group loads
  // BLOCK 0
  vld1_32({d4}, mem[r3]++); // A0
  vmov(d15, r0, r2); // b1 VMOV b from second group
  vmla_f32(q8, q4, d0[0]);
  ldr(r0, mem[r12]); // A1 low
  vmla_f32(q10, q4, d1[0]);
  ldr(r2, mem[r12, 4]); // A1 high
  vmla_f32(q12, q4, d2[0]);
  pld(mem[r3, 128]); // Prefetch A0

  // BLOCK 1
  vldr(d12, mem[r9, 32]); // B1
  vmov(d5, r0, r2); // a1 VMOV
  vmla_f32(q14, q4, d3[0]);
  ldr(r0, mem[r9, 72]); // B0 low
  vmla_f32(q9, q5, d0[0]);
  ldr(r2, mem[r9, 76]); // B0 high
  vmla_f32(q11, q5, d1[0]);
  pld(mem[r12, 128]); // Prefetch A1

  // BLOCK 2
  vld1_32({d6}, mem[r10]++); // A2
  vmov(d9, r0, r2); // b0 VMOV
  vmla_f32(q13, q5, d2[0]);
  ldr(r0, mem[r7]); // A3 low
  vmla_f32(q15, q5, d3[0]);
  ldr(r2, mem[r7, 4]); // A3 high
  vmla_f32(q8, q6, d0[1]);
  pld(mem[r10, 128]); // Prefetch A2

  // BLOCK 3
  vldr(d14, mem[r9, 48]); // B1
  vmov(d7, r0, r2); // a3 VMOV
  vmla_f32(q10, q6, d1[1]);
  ldr(r0, mem[r9, 88]); // B0 low
  vmla_f32(q12, q6, d2[1]);
  ldr(r2, mem[r9, 92]); // B0 high
  vmla_f32(q14, q6, d3[1]);
  pld(mem[r7, 128]); // Prefetch A3

  // BLOCK 4
  vldr(d8, mem[r9, 64]); // B0
  vmov(d11, r0, r2); // B0 VMOV
  vmla_f32(q9, q7, d0[1]);
  ldr(r0, mem[r9, 104]); // B1 low   VMOV is in BLOCK 0
  vmla_f32(q11, q7, d1[1]);
  ldr(r2, mem[r9, 108]); // B1 high
  vmla_f32(q13, q7, d2[1]);
  pld(mem[r9, 384]); // Prefetch B

  // BLOCK 5
  vldr(d10, mem[r9, 80]); // B0
  vmov(d13, r0, r2); // b1 VMOV b from second group
  vmla_f32(q15, q7, d3[1]);
  ldr(r0, mem[r9, 120]); // B1 low   VMOV is in BLOCK 0
  nop();
  ldr(r2, mem[r9, 124]); // B1 high
  nop();
  pld(mem[r9, 448]); // Prefetch B

  // Second group of 16 FMA, First group of loads
  // BLOCK 0
  vld1_32({d0}, mem[r3]++); // A0
  vmov(d15, r0, r2); // b1 VMOV b from second group
  vmla_f32(q8, q4, d4[0]);
  ldr(r0, mem[r12, 8]); // A1 low
  vmla_f32(q10, q4, d5[0]);
  ldr(r2, mem[r12, 12]); // A1 high
  vmla_f32(q12, q4, d6[0]);
  // NOP

  // BLOCK 1
  vldr(d12, mem[r9, 96]); // B1
  vmov(d1, r0, r2); // a1 VMOV
  vmla_f32(q14, q4, d7[0]);
  ldr(r0, mem[r9, 136]); // B0 low
  vmla_f32(q9, q5, d4[0]);
  ldr(r2, mem[r9, 140]); // B0 high
  vmla_f32(q11, q5, d5[0]);
  // NOP

  // BLOCK 2
  vld1_32({d2}, mem[r10]++); // A2
  vmov(d9, r0, r2); // b0 VMOV
  vmla_f32(q13, q5, d6[0]);
  ldr(r0, mem[r7, 8]); // A3 low
  vmla_f32(q15, q5, d7[0]);
  ldr(r2, mem[r7, 12]); // A3 high
  vmla_f32(q8, q6, d4[1]);
  // NOP

  // BLOCK 3
  vldr(d14, mem[r9, 112]); // B1
  vmov(d3, r0, r2); // a3 VMOV
  vmla_f32(q10, q6, d5[1]);
  ldr(r0, mem[r9, 152]); // B0 low
  vmla_f32(q12, q6, d6[1]);
  ldr(r2, mem[r9, 156]); // B0 high
  vmla_f32(q14, q6, d7[1]);
  add(r12, r12, 16); // A1++

  // BLOCK 4
  vldr(d8, mem[r9, 128]); // B0
  vmov(d11, r0, r2); // B0 VMOV
  vmla_f32(q9, q7, d4[1]);
  ldr(r0, mem[r9, 168]); // B1 low
  vmla_f32(q11, q7, d5[1]);
  ldr(r2, mem[r9, 172]); // B1 high
  vmla_f32(q13, q7, d6[1]);
  add(r7, r7, 16); // A3++

  // BLOCK 5
  vldr(d10, mem[r9, 144]); // B0
  vmov(d13, r0, r2); // b1 VMOV b
  vmla_f32(q15, q7, d7[1]);
  ldr(r0, mem[r9, 184]); // B1 low   VMOV is in BLOCK 0
  subs(r5, r5, 16);
  ldr(r2, mem[r9, 188]); // B1 high
  add(r9, r9, 128); // B++
  bhs(kc_loop);

  // Epilogue - 4 floats of A (16 bytes)
  bind(epilogue);
  // First group of 16 FMA, Second group loads
  // BLOCK 0
  vld1_32({d4}, mem[r3]++); // A0
  vmov(d15, r0, r2); // b1 VMOV b from second group
  vmla_f32(q8, q4, d0[0]);
  ldr(r0, mem[r12]); // A1 low
  vmla_f32(q10, q4, d1[0]);
  ldr(r2, mem[r12, 4]); // A1 high
  vmla_f32(q12, q4, d2[0]);
  // NOP

  // BLOCK 1
  vldr(d12, mem[r9, 32]); // B1
  vmov(d5, r0, r2); // a1 VMOV
  vmla_f32(q14, q4, d3[0]);
  ldr(r0, mem[r9, 72]); // B0 low
  vmla_f32(q9, q5, d0[0]);
  ldr(r2, mem[r9, 76]); // B0 high
  vmla_f32(q11, q5, d1[0]);
  // NOP

  // BLOCK 2
  vld1_32({d6}, mem[r10]++); // A2
  vmov(d9, r0, r2); // b0 VMOV
  vmla_f32(q13, q5, d2[0]);
  ldr(r0, mem[r7]); // A3 low
  vmla_f32(q15, q5, d3[0]);
  ldr(r2, mem[r7, 4]); // A3 high
  vmla_f32(q8, q6, d0[1]);
  // NOP

  // BLOCK 3
  vldr(d14, mem[r9, 48]); // B1
  vmov(d7, r0, r2); // a3 VMOV
  vmla_f32(q10, q6, d1[1]);
  ldr(r0, mem[r9, 88]); // B0 low
  vmla_f32(q12, q6, d2[1]);
  ldr(r2, mem[r9, 92]); // B0 high
  vmla_f32(q14, q6, d3[1]);
  // NOP

  // BLOCK 4
  vldr(d8, mem[r9, 64]); // B0
  vmov(d11, r0, r2); // B0 VMOV
  vmla_f32(q9, q7, d0[1]);
  ldr(r0, mem[r9, 104]); // B1 low
  vmla_f32(q11, q7, d1[1]);
  ldr(r2, mem[r9, 108]); // B1 high
  vmla_f32(q13, q7, d2[1]);
  // NOP

  // BLOCK 5
  vldr(d10, mem[r9, 80]); // B0
  vmov(d13, r0, r2); // b1 VMOV b
  vmla_f32(q15, q7, d3[1]);
  ldr(r0, mem[r9, 120]); // B1 low   VMOV is in BLOCK 0
  nop();
  ldr(r2, mem[r9, 124]); // B1 high
  nop();
  nop();

  // Second group of 16 FMA, First group of loads
  // BLOCK 0
  vldr(d12, mem[r9, 96]); // B1
  vmov(d15, r0, r2); // b1 VMOV b from second group
  vmla_f32(q8, q4, d4[0]);
  vmla_f32(q10, q4, d5[0]);
  vmla_f32(q12, q4, d6[0]);

  // BLOCK 1
  vldr(d14, mem[r9, 112]); // B1
  vmla_f32(q14, q4, d7[0]);
  vmla_f32(q9, q5, d4[0]);
  vmla_f32(q11, q5, d5[0]);
  add(r12, r12, 8); // A1++

  // BLOCK 2
  add(r7, r7, 8); // A3++ VLDR B1 land_s here
  add(r9, r9, 128); // B++
  vmla_f32(q13, q5, d6[0]);
  vmla_f32(q15, q5, d7[0]);
  vmla_f32(q8, q6, d4[1]);

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
  if (kc % 16 != 0) {
    bne(remainder_kc);
  }

  align(8);
  bind(clamp);

  ldr(r0, mem[sp, 116]); // cn_stride
  ldr(r2, mem[sp]); // kc
  subs(r1, r1, 8);

  if (clamp_min || clamp_max) {
    // Load params pointer
    ldr(r5, mem[sp, 120]); // params

    if (clamp_min) {
      vld1r_32({d4, d5}, mem[r5]++);
      vmax_f32(q8, q8, q2);
      vmax_f32(q9, q9, q2);
      vmax_f32(q10, q10, q2);
      vmax_f32(q11, q11, q2);
      vmax_f32(q12, q12, q2);
      vmax_f32(q13, q13, q2);
      vmax_f32(q14, q14, q2);
      vmax_f32(q15, q15, q2);
    } else {
      add(r5, r5, 4);
    }

    if (clamp_max) {
      vld1r_32({d6, d7}, mem[r5]);
      vmin_f32(q8, q8, q3);
      vmin_f32(q9, q9, q3);
      vmin_f32(q10, q10, q3);
      vmin_f32(q11, q11, q3);
      vmin_f32(q12, q12, q3);
      vmin_f32(q13, q13, q3);
      vmin_f32(q14, q14, q3);
      vmin_f32(q15, q15, q3);
    }
  }

  if (nc_mod_nr != 0) {
    blo(store_odd_width);
  }

  // Store full 4 x 8
  vst1_32({d16-d19}, mem[r11], r0);
  sub(r7, r7, r2);
  vst1_32({d20-d23}, mem[r4], r0);
  sub(r10, r10, r2);
  vst1_32({d24-d27}, mem[r8], r0);
  sub(r12, r12, r2);
  vst1_32({d28-d31}, mem[r6], r0);
  sub(r3, r3, r2);
  bhi(nc_loop);

  add(sp, sp, 4);
  pop({r4, r5, r6, r7, r8, r9, r10, r11});
  vpop({d8-d15});
  bx(lr);

  align(8);
  bind(remainder_kc);

  if (kc & 8) {
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
  }
  if (kc & 4) {
    // Remainder - 1 float of A (4 bytes)
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
  }
  b(clamp);

  // Store odd width
  bind(store_odd_width);

  switch (nc_mod_nr) {
    case 0:
      // Do nothing.
      break;
    case 1:
      vst1_32({d16[0]}, mem[r11]);
      vst1_32({d20[0]}, mem[r4]);
      vst1_32({d24[0]}, mem[r8]);
      vst1_32({d28[0]}, mem[r6]);
      break;
    case 2:
      vst1_32({d16}, mem[r11]);
      vst1_32({d20}, mem[r4]);
      vst1_32({d24}, mem[r8]);
      vst1_32({d28}, mem[r6]);
      break;
    case 3:
      vst1_32({d16}, mem[r11]++);
      vst1_32({d20}, mem[r4]++);
      vst1_32({d24}, mem[r8]++);
      vst1_32({d28}, mem[r6]++);
      vst1_32({d17[0]}, mem[r11]);
      vst1_32({d21[0]}, mem[r4]);
      vst1_32({d25[0]}, mem[r8]);
      vst1_32({d29[0]}, mem[r6]);
      break;
    case 4:
      vst1_32({d16, d17}, mem[r11]);
      vst1_32({d20, d21}, mem[r4]);
      vst1_32({d24, d25}, mem[r8]);
      vst1_32({d28, d29}, mem[r6]);
      break;
    case 5:
      vst1_32({d16, d17}, mem[r11]++);
      vst1_32({d20, d21}, mem[r4]++);
      vst1_32({d24, d25}, mem[r8]++);
      vst1_32({d28, d29}, mem[r6]++);
      vst1_32({d18[0]}, mem[r11]);
      vst1_32({d22[0]}, mem[r4]);
      vst1_32({d26[0]}, mem[r8]);
      vst1_32({d30[0]}, mem[r6]);
      break;
    case 6:
      vst1_32({d16-d18}, mem[r11]);
      vst1_32({d20-d22}, mem[r4]);
      vst1_32({d24-d26}, mem[r8]);
      vst1_32({d28-d30}, mem[r6]);
      break;
    case 7:
      vst1_32({d16-d18}, mem[r11]++);
      vst1_32({d20-d22}, mem[r4]++);
      vst1_32({d24-d26}, mem[r8]++);
      vst1_32({d28-d30}, mem[r6]++);
      vst1_32({d19[0]}, mem[r11]);
      vst1_32({d23[0]}, mem[r4]);
      vst1_32({d27[0]}, mem[r8]);
      vst1_32({d31[0]}, mem[r6]);
      break;
    default:
      XNN_UNREACHABLE;
  }

  add(sp, sp, 4);
  pop({r4, r5, r6, r7, r8, r9, r10, r11});
  vpop({d8-d15});
  bx(lr);
}
}  // namespace
}  // aarch32
}  // xnnpack

xnn_status xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53(xnn_code_buffer* code, size_t nc_mod_nr, size_t kc, const void* params) {
  using namespace xnnpack::aarch32;
  Generator g(code);
  auto p = static_cast<const jit_gemm_params*>(params);
  g.generate(nc_mod_nr, kc, p->f32_minmax.min, p->f32_minmax.max);
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
