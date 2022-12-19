// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <limits>

#include <xnnpack.h>
#include <xnnpack/aarch64-assembler.h>
#include <xnnpack/igemm.h>
#include <xnnpack/memory.h>
#include <xnnpack/microparams.h>
#include <xnnpack/post-operation.h>

namespace xnnpack {
namespace aarch64 {
namespace {
class Generator : public MacroAssembler {
  using MacroAssembler::MacroAssembler;

 public:
  void generate(size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const jit_gemm_params* jit_gemm_params);
};

// void xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55(
//     size_t mr,                         x0
//     size_t nc,                         x1
//     size_t kc,                         x2 / x0
//     size_t ks,                         x3 / x9
//     const float**restrict a,           x4
//     const void*restrict w,             x5
//     uint8_t*restrict c,                x6
//     size_t cm_stride,                  x7
//     size_t cn_stride,                  [sp] -> x10
//     size_t a_offset,                   [sp + 8] -> x11
//     const float* zero,                 [sp + 16] -> x12
//     const xnn_f32_minmax_params params [sp + 24] -> (x8)

// d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// A pointers
// x13 a0
// x14 a1
// x15 a2
//  x8 a3

// C pointers
//  x6 c0
// x16 c1
// x17 c2
//  x7 c3

// Vector register usage
// A0  v0     v3
// A1  v0[1]  v3[1]
// A2  v1     v4
// A3  v1[1]  v4[1]
// B   v12 v13 v14 v15 second set of B
// B   v16 v17 v18 v19 first set
// C   v20 v21
// C   v22 v23
// C   v24 v25
// C   v26 v27
// Clamp v6 v7
// temporary vector shadow register x19

// unused A   v8 v9 v10 v11
// x12 a4
//  x4 a5
// x13 c4
//  x7 c5
// A4  v2     v5
// A5  v2[1]  v5[1]
// C   v28 v29
// C   v30 v31

// Converted from: src/f32-igemm/f32-igemm-4x8-minmax-asm-aarch64-neonfma-cortex-a55.S
void Generator::generate(size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const jit_gemm_params* jit_gemm_params)
{
  assert(max_mr <= 4);
  assert(nc_mod_nr < 8);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11;
  const size_t num_post_operations = jit_gemm_params->num_post_operations;
  (void) num_post_operations;  // Silence unused warning.
  const float min = jit_gemm_params->f32_minmax.min;
  const float max = jit_gemm_params->f32_minmax.max;
  const bool clamp_min = min != -std::numeric_limits<float>::infinity();
  const bool clamp_max = max != +std::numeric_limits<float>::infinity();
  assert(num_post_operations == 0 || (!clamp_min && !clamp_max));

  // Clamp C pointers
  if (max_mr > 1) {
    cmp(x0, 2); // if mr < 2
    add(x16, x6, x7); // c1 = c0 + cm_stride
    csel(x16, x6, x16, kLO); //   c1 = c0
  }

  if (max_mr > 2) {
    add(x17, x16, x7); // c2 = c1 + cm_stride
    // if mr <= 2
    csel(x17, x16, x17, kLS); //   c2 = c1
  }

  if (max_mr > 3) {
    cmp(x0, 4); // if mr < 4
    add(x7, x17, x7); // c3 = c2 + cm_stride
    csel(x7, x17, x7, kLO); //   c3 = c2
  }

  // Load cn_stride, a_offset
  ldp(x10, x11, mem[sp]);

  // Load zero, params pointer
  ldp(x12, x8, mem[sp, 16]);

  // Load min/max values
  if (clamp_min || clamp_max) {
    ld2r({v6.v4s(), v7.v4s()}, mem[x8]);
  }

  // Save x19, d12-d15 on stack
  stp(d12, d13, mem[sp, -48]++);
  stp(d14, d15, mem[sp, 16]);
  str(x19, mem[sp, 32]);

  bind(l0);
  // Load initial bias from w into accumulators
  ldp(q20, q21, mem[x5], 32);
  if (max_mr > 1) {
    mov(v22.v16b(), v20.v16b());
    mov(v23.v16b(), v21.v16b());
  }
  if (max_mr > 2) {
    mov(v24.v16b(), v20.v16b());
    mov(v25.v16b(), v21.v16b());
  }
  if (max_mr > 3) {
    mov(v26.v16b(), v20.v16b());
    mov(v27.v16b(), v21.v16b());
  }

  mov(x9, x3); // p = ks

  bind(l1);
  // Load next 4 A pointers
  ldp(x13, x14, mem[x4], 16);
  if (max_mr > 2) {
    ldp(x15, x8, mem[x4], 16);
  }

  cmp(x13, x12); // if a0 == zero
  add(x13, x13, x11); // a0 += a_offset
  csel(x13, x12, x13, kEQ); //   a0 = zero, else += a0 + a_offset
  if (max_mr > 1) {
    cmp(x14, x12); // if a1 == zero
    add(x14, x14, x11); // a1 += a_offset
    csel(x14, x12, x14, kEQ); //   a1 = zero, else += a1 + a_offset
  }
  if (max_mr > 2) {
    cmp(x15, x12); // if a2 == zero
    add(x15, x15, x11); // a2 += a_offset
    csel(x15, x12, x15, kEQ); //   a2 = zero, else += a2 + a_offset
  }
  if (max_mr > 3) {
    cmp(x8, x12); // if a3 == zero
    add(x8, x8, x11); // a3 += a_offset
    csel(x8, x12, x8, kEQ); //   a3 = zero, else += a3 + a_offset
  }

  // Is there at least 4 floats (16 bytes) for prologue + epilogue?
  subs(x0, x2, 16); // k = kc - 16
  b_lo(l4);

  // Prologue - First group loads, no FMA
  ldr(d0, mem[x13], 8); // a0
  ldp(q16, q17, mem[x5], 32); // b
  if (max_mr > 2) {
    ldr(d1, mem[x15], 8); // a2
  }
  if (max_mr > 1) {
    ld1({v0.d()}, 1, mem[x14], 8); // a1
  }
  if (max_mr > 3) {
    ld1({v1.d()}, 1, mem[x8], 8); // a3
  }
  subs(x0, x0, 16);
  ldr(q18, mem[x5], 16);
  ldr(d19, mem[x5], 8);
  ldr(x19, mem[x5], 8); // ins is in BLOCK 0

  // Is there at least 4 floats (16 bytes) for main loop?
  b_lo(l3);

  // Main loop - 4 floats of A (16 bytes)
  // 32 FMA + 8 LD64 A + 8 LDR B
  bind(l2);
  // First group of 16 FMA, Second group loads
  // BLOCK 0
  fmla(v20.v4s(), v16.v4s(), v0.s()[0]);
  ldr(d3, mem[x13], 8); // a0
  if (max_mr > 1) {
    fmla(v22.v4s(), v16.v4s(), v0.s()[2]);
  }
  ins(v19.d()[1], x19); // b from second group
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v1.s()[0]);
  }
  if (max_mr > 1) {
    ldr(x19, mem[x14], 8); // a1
  }

  // BLOCK 1
  if (max_mr > 3) {
    fmla(v26.v4s(), v16.v4s(), v1.s()[2]);
  }
  ldr(d12, mem[x5]);
  fmla(v21.v4s(), v17.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    ins(v3.d()[1], x19); // a1 ins
    fmla(v23.v4s(), v17.v4s(), v0.s()[2]);
  }
  ldr(x19, mem[x5, 8]); // b

  // BLOCK 2
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v1.s()[0]);
    ldr(d4, mem[x15], 8); // a2
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v17.v4s(), v1.s()[2]);
  }
  ins(v12.d()[1], x19); // b  ins
  fmla(v20.v4s(), v18.v4s(), v0.s()[1]);
  if (max_mr > 3) {
    ldr(x19, mem[x8], 8); // a3
  }

  // BLOCK 3
  if (max_mr > 1) {
    fmla(v22.v4s(), v18.v4s(), v0.s()[3]);
  }
  ldr(d13, mem[x5, 16]);
  if (max_mr > 2) {
    fmla(v24.v4s(), v18.v4s(), v1.s()[1]);
  }
  if (max_mr > 3) {
    ins(v4.d()[1], x19); // a3 ins
    fmla(v26.v4s(), v18.v4s(), v1.s()[3]);
  }
  ldr(x19, mem[x5, 24]);

  // BLOCK 4
  fmla(v21.v4s(), v19.v4s(), v0.s()[1]);
  ldr(d14, mem[x5, 32]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v19.v4s(), v0.s()[3]);
  }
  ins(v13.d()[1], x19); // b
  if (max_mr > 2) {
    fmla(v25.v4s(), v19.v4s(), v1.s()[1]);
  }
  ldr(x19, mem[x5, 40]);

  // BLOCK 5
  // NOPs to ensure 4 cycle LDR lands on next LDR
  if (max_mr > 3) {
    fmla(v27.v4s(), v19.v4s(), v1.s()[3]);
  }
  ldr(d15, mem[x5, 48]);
  nop();
  ins(v14.d()[1], x19); // b from previous
  subs(x0, x0, 16);
  ldr(x19, mem[x5, 56]);

  // Second group of 16 FMA, First group of loads
  // BLOCK 0
  fmla(v20.v4s(), v12.v4s(), v3.s()[0]);
  ldr(d0, mem[x13], 8); // a0
  if (max_mr > 1) {
    fmla(v22.v4s(), v12.v4s(), v3.s()[2]);
  }
  ins(v15.d()[1], x19); // b from previous
  if (max_mr > 2) {
    fmla(v24.v4s(), v12.v4s(), v4.s()[0]);
  }
  if (max_mr > 1) {
    ldr(x19, mem[x14], 8); // a1
  }

  // BLOCK 1
  if (max_mr > 3) {
    fmla(v26.v4s(), v12.v4s(), v4.s()[2]);
  }
  ldr(d16, mem[x5, 64]);
  fmla(v21.v4s(), v13.v4s(), v3.s()[0]);
  if (max_mr > 1) {
    ins(v0.d()[1], x19); // a1 ins
    fmla(v23.v4s(), v13.v4s(), v3.s()[2]);
  }
  ldr(x19, mem[x5, 72]); // b

  // BLOCK 2
  if (max_mr > 2) {
    fmla(v25.v4s(), v13.v4s(), v4.s()[0]);
    ldr(d1, mem[x15], 8); // a2
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v13.v4s(), v4.s()[2]);
  }
  ins(v16.d()[1], x19); // b
  fmla(v20.v4s(), v14.v4s(), v3.s()[1]);
  if (max_mr > 3) {
    ldr(x19, mem[x8], 8); // a3
  }

  // BLOCK 3
  if (max_mr > 1) {
    fmla(v22.v4s(), v14.v4s(), v3.s()[3]);
  }
  ldr(d17, mem[x5, 80]);
  if (max_mr > 2) {
    fmla(v24.v4s(), v14.v4s(), v4.s()[1]);
  }
  if (max_mr > 3) {
    ins(v1.d()[1], x19); // a3 ins
    fmla(v26.v4s(), v14.v4s(), v4.s()[3]);
  }
  ldr(x19, mem[x5, 88]);

  // BLOCK 4
  fmla(v21.v4s(), v15.v4s(), v3.s()[1]);
  ldr(d18, mem[x5, 96]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v15.v4s(), v3.s()[3]);
  }
  ins(v17.d()[1], x19); // b
  if (max_mr > 2) {
    fmla(v25.v4s(), v15.v4s(), v4.s()[1]);
  }
  ldr(x19, mem[x5, 104]);

  // BLOCK 5
  // NOTE that block needs to be 4 cycles for LDR not to stall
  if (max_mr > 3) {
    fmla(v27.v4s(), v15.v4s(), v4.s()[3]);
  }
  ldr(d19, mem[x5, 112]);
  ins(v18.d()[1], x19);
  ldr(x19, mem[x5, 120]);
  add(x5, x5, 128);
  b_hs(l2);

  // Epilogue - 4 floats of A (16 bytes)
  // 32 FMA + 8 LD64 A + 8 LDR B
  bind(l3);
  // First group of 16 FMA, Second group loads
  // BLOCK 0
  ldr(d3, mem[x13], 8); // a0
  ins(v19.d()[1], x19); // b from second group
  fmla(v20.v4s(), v16.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    ldr(x19, mem[x14], 8); // a1
    fmla(v22.v4s(), v16.v4s(), v0.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v1.s()[0]);
  }

  // BLOCK 1
  ldr(d12, mem[x5]);
  if (max_mr > 1) {
    ins(v3.d()[1], x19); // a1 ins
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v16.v4s(), v1.s()[2]);
  }
  ldr(x19, mem[x5, 8]); // b
  fmla(v21.v4s(), v17.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v17.v4s(), v0.s()[2]);
  }

  // BLOCK 2
  if (max_mr > 2) {
    ldr(d4, mem[x15], 8); // a2
  }
  ins(v12.d()[1], x19); // b  ins
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v1.s()[0]);
  }
  if (max_mr > 3) {
    ldr(x19, mem[x8], 8); // a3
    fmla(v27.v4s(), v17.v4s(), v1.s()[2]);
  }
  fmla(v20.v4s(), v18.v4s(), v0.s()[1]);

  // BLOCK 3
  ldr(d13, mem[x5, 16]);
  if (max_mr > 3) {
    ins(v4.d()[1], x19); // a3 ins
  }
  if (max_mr > 1) {
    fmla(v22.v4s(), v18.v4s(), v0.s()[3]);
  }
  ldr(x19, mem[x5, 24]);
  if (max_mr > 2) {
    fmla(v24.v4s(), v18.v4s(), v1.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v18.v4s(), v1.s()[3]);
  }

  // BLOCK 4
  ldr(d14, mem[x5, 32]);
  ins(v13.d()[1], x19); // b
  fmla(v21.v4s(), v19.v4s(), v0.s()[1]);
  ldr(x19, mem[x5, 40]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v19.v4s(), v0.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v19.v4s(), v1.s()[1]);
  }

  // BLOCK 5
  // NOPs to ensure 4 cycle LDR lands on next LDR
  ldr(d15, mem[x5, 48]);
  ins(v14.d()[1], x19);
  if (max_mr > 3) {
    fmla(v27.v4s(), v19.v4s(), v1.s()[3]);
  }
  ldr(x19, mem[x5, 56]);
  nop(); // fma
  nop();
  nop(); // fma
  nop();

  // Second group of 16 FMA, no loads
  // BLOCK 0
  ins(v15.d()[1], x19); // b from previous
  fmla(v20.v4s(), v12.v4s(), v3.s()[0]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v12.v4s(), v3.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v12.v4s(), v4.s()[0]);
  }

  // BLOCK 1
  if (max_mr > 3) {
    fmla(v26.v4s(), v12.v4s(), v4.s()[2]);
  }
  fmla(v21.v4s(), v13.v4s(), v3.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v13.v4s(), v3.s()[2]);
  }

  // BLOCK 2
  if (max_mr > 2) {
    fmla(v25.v4s(), v13.v4s(), v4.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v13.v4s(), v4.s()[2]);
  }
  fmla(v20.v4s(), v14.v4s(), v3.s()[1]);

  // BLOCK 3
  if (max_mr > 1) {
    fmla(v22.v4s(), v14.v4s(), v3.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v14.v4s(), v4.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v14.v4s(), v4.s()[3]);
  }

  // BLOCK 4
  fmla(v21.v4s(), v15.v4s(), v3.s()[1]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v15.v4s(), v3.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v15.v4s(), v4.s()[1]);
  }
  add(x5, x5, 64);

  // BLOCK 5
  if (max_mr > 3) {
    fmla(v27.v4s(), v15.v4s(), v4.s()[3]);
  }

  bind(l4);
  // Is there a remainder?- 2 floats of A (8 bytes)
  tbnz(x0, 3, l6);
  // Is there a remainder?- 1 float of A (4 bytes)
  tbnz(x0, 2, l7);
  bind(l5);
  // ks loop
  subs(x9, x9, max_mr * sizeof(void*)); // ks -= MR * sizeof(void*)
  b_hi(l1);

  // Clamp
  if (clamp_min) {
    fmax(v20.v4s(), v20.v4s(), v6.v4s());
    fmax(v21.v4s(), v21.v4s(), v6.v4s());
    if (max_mr > 1) {
      fmax(v22.v4s(), v22.v4s(), v6.v4s());
      fmax(v23.v4s(), v23.v4s(), v6.v4s());
    }
    if (max_mr > 2) {
      fmax(v24.v4s(), v24.v4s(), v6.v4s());
      fmax(v25.v4s(), v25.v4s(), v6.v4s());
    }
    if (max_mr > 3) {
      fmax(v26.v4s(), v26.v4s(), v6.v4s());
      fmax(v27.v4s(), v27.v4s(), v6.v4s());
    }
  }
  if (clamp_max) {
    fmin(v20.v4s(), v20.v4s(), v7.v4s());
    fmin(v21.v4s(), v21.v4s(), v7.v4s());
    if (max_mr > 1) {
      fmin(v22.v4s(), v22.v4s(), v7.v4s());
      fmin(v23.v4s(), v23.v4s(), v7.v4s());
    }
    if (max_mr > 2) {
      fmin(v24.v4s(), v24.v4s(), v7.v4s());
      fmin(v25.v4s(), v25.v4s(), v7.v4s());
    }
    if (max_mr > 3) {
      fmin(v26.v4s(), v26.v4s(), v7.v4s());
      fmin(v27.v4s(), v27.v4s(), v7.v4s());
    }
  }

  // Store full 4 x 8
  subs(x1, x1, 8);
  b_lo(l8);

  if (max_mr > 3) {
    stp(q26, q27, mem[x7]);
    add(x7, x7, x10);
  }
  if (max_mr > 2) {
    stp(q24, q25, mem[x17]);
    add(x17, x17, x10);
  }
  if (max_mr > 1) {
    stp(q22, q23, mem[x16]);
    add(x16, x16, x10);
  }
  stp(q20, q21, mem[x6]);
  add(x6, x6, x10);

  sub(x4, x4, x3); // a -= ks

  // nc loop
  b_hi(l0);

  // Restore x19, d12-d15 from stack
  ldr(x19, mem[sp, 32]);
  ldp(d14, d15, mem[sp, 16]);
  ldp(d12, d13, mem[sp], 48);
  ret();

  // Remainder - 2 floats of A (8 bytes)
  // 16 FMA + 4 LD64 A + 2 LDP B
  bind(l6);
  ldr(d0, mem[x13], 8);
  ldp(q16, q17, mem[x5], 32);
  if (max_mr > 1) {
    ld1({v0.d()}, 1, mem[x14], 8);
  }
  if (max_mr > 2) {
    ldr(d1, mem[x15], 8);
  }
  if (max_mr > 3) {
    ld1({v1.d()}, 1, mem[x8], 8);
  }
  ldp(q18, q19, mem[x5], 32);
  fmla(v20.v4s(), v16.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v16.v4s(), v0.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v1.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v16.v4s(), v1.s()[2]);
  }
  fmla(v21.v4s(), v17.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v17.v4s(), v0.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v1.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v17.v4s(), v1.s()[2]);
  }

  fmla(v20.v4s(), v18.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v18.v4s(), v0.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v18.v4s(), v1.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v18.v4s(), v1.s()[3]);
  }
  fmla(v21.v4s(), v19.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v19.v4s(), v0.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v19.v4s(), v1.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v19.v4s(), v1.s()[3]);
  }

  // Is there a remainder?- 1 float of A (4 bytes)
  tbz(x0, 2, l5);

  bind(l7);
  // Remainder- 1 float of A (4 bytes)
  ldr(s0, mem[x13], 4);
  ldp(q16, q17, mem[x5], 32);
  if (max_mr > 1) {
    ld1({v0.s()}, 2, mem[x14], 4);
  }
  if (max_mr > 2) {
    ldr(s1, mem[x15], 4);
  }
  if (max_mr > 3) {
    ld1({v1.s()}, 2, mem[x8], 4);
  }

  fmla(v20.v4s(), v16.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v16.v4s(), v0.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v1.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v16.v4s(), v1.s()[2]);
  }
  fmla(v21.v4s(), v17.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v17.v4s(), v0.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v1.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v17.v4s(), v1.s()[2]);
  }
  b(l5);

  // Store odd width
  bind(l8);
  tbz(x1, 2, l9);
  if (max_mr > 3) {
    str(q26, mem[x7], 16);
    mov(v26.v16b(), v27.v16b());
  }
  if (max_mr > 2) {
    str(q24, mem[x17], 16);
    mov(v24.v16b(), v25.v16b());
  }
  if (max_mr > 1) {
    str(q22, mem[x16], 16);
    mov(v22.v16b(), v23.v16b());
  }
  str(q20, mem[x6], 16);
  mov(v20.v16b(), v21.v16b());
  bind(l9);
  tbz(x1, 1, l10);
  if (max_mr > 3) {
    str(d26, mem[x7], 8);
  }
  if (max_mr > 2) {
    str(d24, mem[x17], 8);
  }
  if (max_mr > 3) {
    dup(d26, v26.d()[1]);
  }
  if (max_mr > 2) {
    dup(d24, v24.d()[1]);
  }
  if (max_mr > 1) {
    str(d22, mem[x16], 8);
  }
  str(d20, mem[x6], 8);
  if (max_mr > 1) {
    dup(d22, v22.d()[1]);
  }
  dup(d20, v20.d()[1]);

  bind(l10);
  tbz(x1, 0, l11);
  if (max_mr > 3) {
    str(s26, mem[x7]);
  }
  if (max_mr > 2) {
    str(s24, mem[x17]);
  }
  if (max_mr > 1) {
    str(s22, mem[x16]);
  }
  str(s20, mem[x6]);
  bind(l11);
  // Restore x19, d12-d15 from stack
  ldr(x19, mem[sp, 32]);
  ldp(d14, d15, mem[sp, 16]);
  ldp(d12, d13, mem[sp], 48);
  ret();

  align(16, AlignInstruction::kHlt);
}
}  // namespace
}  // namespace aarch64
}  // namespace xnnpack

xnn_status_t xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a55(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
  using namespace xnnpack::aarch64;
  Generator g(code);
  assert(params != nullptr);
  g.generate(max_mr, nc_mod_nr, kc, ks, static_cast<const jit_gemm_params*>(params));
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
