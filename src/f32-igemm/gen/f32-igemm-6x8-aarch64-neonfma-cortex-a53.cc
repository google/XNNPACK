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
  void generate(bool prefetch, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const jit_gemm_params* jit_gemm_params);
};

// void xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_prfm_cortex_a53(
//     size_t mr,                         x0
//     size_t nc,                         x1
//     size_t kc,                         x2 / x0
//     size_t ks,                         x3 / x9
//     const float**restrict a,           x4
//     const void*restrict w,             x5
//     uint8_t*restrict c,                x6
//     size_t cm_stride,                  x7
//     size_t cn_stride,                  [sp] -> (x0)
//     size_t a_offset,                   [sp + 8] -> x11
//     const float* zero,                 [sp + 16] -> x12
//     const xnn_f32_minmax_params params [sp + 24] -> (x8)

// d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x14 v0     v3
// A1  x15 v0[1]  v3[1]
// A2  x20 v1     v4
// A3  x21 v1[1]  v4[1]
// A4  x22 v2     v5
// A5  x23 v2[1]  v5[1]
// B    x5 v12 v13 v14 v15 second set of B
// B       v16 v17 v18 v19 first set
// C0   x6 v20 v21
// C1  x16 v22 v23
// C2  x17 v24 v25
// C3  x10 v26 v27
// C4  x13 v28 v29
// C5   x7 v30 v31
// clamp  v6 v7
// unused A   v8 v9 v10 v11
// temporary vector shadow register x8

// Converted from: src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-prfm-cortex-a53.S
void Generator::generate(bool prefetch, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const jit_gemm_params* jit_gemm_params)
{
  assert(max_mr <= 6);
  assert(nc_mod_nr < 8);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;
  const size_t num_post_operations = jit_gemm_params->num_post_operations;
  (void) num_post_operations;  // Silence unused warning.
  const float min = jit_gemm_params->f32_minmax.min;
  const float max = jit_gemm_params->f32_minmax.max;
  const bool clamp_min = min != -std::numeric_limits<float>::infinity();
  const bool clamp_max = max != +std::numeric_limits<float>::infinity();
  assert(num_post_operations == 0 || (!clamp_min && !clamp_max));

  // Load a_offset
  ldr(x11, mem[sp, 8]);

  // Load zero, params pointer
  ldp(x12, x8, mem[sp, 16]);

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
    add(x10, x17, x7); // c3 = c2 + cm_stride
    csel(x10, x17, x10, kLO); //   c3 = c2
  }

  if (max_mr > 4) {
    add(x13, x10, x7); // c4 = c3 + cm_stride
    // if mr <= 4
    csel(x13, x10, x13, kLS); //   c4 = c3
  }

  if (max_mr > 5) {
    cmp(x0, 6); // if mr < 6
    add(x7, x13, x7); // c5 = c4 + cm_stride
    csel(x7, x13, x7, kLO); //   c5 = c4
  }

  // Load min/max values
  if (clamp_min || clamp_max) {
    ld2r({v6.v4s(), v7.v4s()}, mem[x8]);
  }

  // Save x20-x23, d12-d15 on stack
  stp(d12, d13, mem[sp, -64]++);
  stp(d14, d15, mem[sp, 16]);
  stp(x20, x21, mem[sp, 32]);
  stp(x22, x23, mem[sp, 48]);

  bind(l0);
  // Load initial bias from w into accumulators
  ldp(q20, q21, mem[x5], 32);
  if (max_mr > 1) {
    mov(v22.v16b(), v20.v16b());
    mov(v23.v16b(), v21.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 0]); // Prefetch B
  }
  if (max_mr > 2) {
    mov(v24.v16b(), v20.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 64]);
  }
  if (max_mr > 2) {
    mov(v25.v16b(), v21.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 128]);
  }
  if (max_mr > 3) {
    mov(v26.v16b(), v20.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 192]);
  }
  if (max_mr > 3) {
    mov(v27.v16b(), v21.v16b());
  }
  if (max_mr > 4) {
    mov(v28.v16b(), v20.v16b());
    mov(v29.v16b(), v21.v16b());
  }
  if (max_mr > 5) {
    mov(v30.v16b(), v20.v16b());
    mov(v31.v16b(), v21.v16b());
  }

  mov(x9, x3); // p = ks

  bind(l1);
  // Load next 6 A pointers
  ldp(x14, x15, mem[x4], 16);
  ldp(x20, x21, mem[x4], 16);
  ldp(x22, x23, mem[x4], 16);

  cmp(x14, x12); // if a0 == zero
  add(x14, x14, x11); // A0 += a_offset
  csel(x14, x12, x14, kEQ); //   a0 = zero, else += a0 + a_offset
  if (max_mr > 1) {
    cmp(x15, x12); // if a1 == zero
    add(x15, x15, x11); // A1 += a_offset
    csel(x15, x12, x15, kEQ); //   a1 = zero, else += a1 + a_offset
  }
  if (max_mr > 2) {
    cmp(x20, x12); // if a2 == zero
    add(x20, x20, x11); // A2 += a_offset
    csel(x20, x12, x20, kEQ); //   a2 = zero, else += a2 + a_offset
  }
  if (max_mr > 3) {
    cmp(x21, x12); // if a3 == zero
    add(x21, x21, x11); // A3 += a_offset
    csel(x21, x12, x21, kEQ); //   a3 = zero, else += a3 + a_offset
  }
  if (max_mr > 4) {
    cmp(x22, x12); // if a4 == zero
    add(x22, x22, x11); // A4 += a_offset
    csel(x22, x12, x22, kEQ); //   a4 = zero, else += a4 + a_offset
  }
  if (max_mr > 5) {
    cmp(x23, x12); // if a5 == zero
    add(x23, x23, x11); // A5 += a_offset
    csel(x23, x12, x23, kEQ); //   a5 = zero, else += a5 + a_offset
  }

  // Is there at least 4 floats (16 bytes) for prologue + epilogue?
  subs(x0, x2, 16); // k = kc - 16
  b_lo(l5);

  // Prologue - First group loads, no FMA
  ldr(d0, mem[x14], 8); // A0
  ldp(q16, q17, mem[x5], 32); // B
  if (max_mr > 2) {
    ldr(d1, mem[x20], 8); // A2
  }
  if (max_mr > 4) {
    ldr(d2, mem[x22], 8); // A4
  }
  if (max_mr > 1) {
    ld1({v0.d()}, 1, mem[x15], 8); // A1
  }
  if (max_mr > 3) {
    ld1({v1.d()}, 1, mem[x21], 8); // A3
  }
  if (max_mr > 5) {
    ld1({v2.d()}, 1, mem[x23], 8); // A5
  }
  subs(x0, x0, 16);
  ldr(q18, mem[x5], 16);
  ldr(d19, mem[x5], 8);
  ldr(x8, mem[x5], 8); // ins is in BLOCK 0

  // Is there at least 4 floats (16 bytes) for main loop?
  b_lo(l3);

  // Main loop - 4 floats of A (16 bytes)
  // 48 FMA + 12 LD64 A + 8 LDR B
  bind(l2);
  // First group of 24 FMA, Second group loads
  // BLOCK 0
  ldr(d3, mem[x14], 8); // A0
  ins(v19.d()[1], x8); // B from second group
  fmla(v20.v4s(), v16.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    ldr(x8, mem[x15], 8); // A1
    fmla(v22.v4s(), v16.v4s(), v0.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v1.s()[0]);
  }

  // BLOCK 1
  ldr(d12, mem[x5]);
  if (max_mr > 1) {
    ins(v3.d()[1], x8); // A1 ins
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v16.v4s(), v1.s()[2]);
  }
  ldr(x8, mem[x5, 8]); // B
  if (max_mr > 4) {
    fmla(v28.v4s(), v16.v4s(), v2.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v16.v4s(), v2.s()[2]);
  }

  // BLOCK 2
  if (max_mr > 2) {
    ldr(d4, mem[x20], 8); // A2
  }
  ins(v12.d()[1], x8); // B  ins
  fmla(v21.v4s(), v17.v4s(), v0.s()[0]);
  if (max_mr > 3) {
    ldr(x8, mem[x21], 8); // A3
  }
  if (max_mr > 1) {
    fmla(v23.v4s(), v17.v4s(), v0.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v1.s()[0]);
  }

  // BLOCK 3
  if (max_mr > 4) {
    ldr(d5, mem[x22], 8); // A4
  }
  if (max_mr > 3) {
    ins(v4.d()[1], x8); // A3 ins
    fmla(v27.v4s(), v17.v4s(), v1.s()[2]);
  }
  if (max_mr > 5) {
    ldr(x8, mem[x23], 8); // A5
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v17.v4s(), v2.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v17.v4s(), v2.s()[2]);
  }

  // BLOCK 4
  ldr(d13, mem[x5, 16]);
  if (max_mr > 5) {
    ins(v5.d()[1], x8); // A5 ins
  }
  fmla(v20.v4s(), v18.v4s(), v0.s()[1]);
  ldr(x8, mem[x5, 24]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v18.v4s(), v0.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v18.v4s(), v1.s()[1]);
  }

  // BLOCK 5
  ldr(d14, mem[x5, 32]);
  ins(v13.d()[1], x8); // B
  if (max_mr > 3) {
    fmla(v26.v4s(), v18.v4s(), v1.s()[3]);
  }
  ldr(x8, mem[x5, 40]);
  if (max_mr > 4) {
    fmla(v28.v4s(), v18.v4s(), v2.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v18.v4s(), v2.s()[3]);
  }

  // BLOCK 6
  ldr(d15, mem[x5, 48]);
  ins(v14.d()[1], x8); // B
  fmla(v21.v4s(), v19.v4s(), v0.s()[1]);
  ldr(x8, mem[x5, 56]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v19.v4s(), v0.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v19.v4s(), v1.s()[1]);
  }

  // BLOCK 7
  ins(v15.d()[1], x8);
  if (max_mr > 3) {
    fmla(v27.v4s(), v19.v4s(), v1.s()[3]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v19.v4s(), v2.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v19.v4s(), v2.s()[3]);
  }

  // Second group of 24 FMA, First group of loads
  // BLOCK 0
  ldr(d0, mem[x14], 8); // A0
  fmla(v20.v4s(), v12.v4s(), v3.s()[0]);
  if (max_mr > 1) {
    ldr(x8, mem[x15], 8); // A1
    fmla(v22.v4s(), v12.v4s(), v3.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v12.v4s(), v4.s()[0]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x14, 128]); // Prefetch A0
  }

  // BLOCK 1
  ldr(d16, mem[x5, 64]);
  if (max_mr > 1) {
    ins(v0.d()[1], x8); // A1 ins
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v12.v4s(), v4.s()[2]);
  }
  ldr(x8, mem[x5, 72]); // B
  if (max_mr > 4) {
    fmla(v28.v4s(), v12.v4s(), v5.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v12.v4s(), v5.s()[2]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x15, 128]); // Prefetch A1
  }

  // BLOCK 2
  if (max_mr > 2) {
    ldr(d1, mem[x20], 8); // A2
  }
  ins(v16.d()[1], x8); // B
  fmla(v21.v4s(), v13.v4s(), v3.s()[0]);
  if (max_mr > 3) {
    ldr(x8, mem[x21], 8); // A3
  }
  if (max_mr > 1) {
    fmla(v23.v4s(), v13.v4s(), v3.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v13.v4s(), v4.s()[0]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x20, 128]); // Prefetch A2
  }

  // BLOCK 3
  if (max_mr > 4) {
    ldr(d2, mem[x22], 8); // A4
  }
  if (max_mr > 3) {
    ins(v1.d()[1], x8); // A3 ins
    fmla(v27.v4s(), v13.v4s(), v4.s()[2]);
  }
  if (max_mr > 5) {
    ldr(x8, mem[x23], 8); // A5
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v13.v4s(), v5.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v13.v4s(), v5.s()[2]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x21, 128]); // Prefetch A3
  }

  // BLOCK 4
  ldr(d17, mem[x5, 80]);
  if (max_mr > 5) {
    ins(v2.d()[1], x8); // A5 ins
  }
  fmla(v20.v4s(), v14.v4s(), v3.s()[1]);
  ldr(x8, mem[x5, 88]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v14.v4s(), v3.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v14.v4s(), v4.s()[1]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x22, 128]); // Prefetch A4
  }

  // BLOCK 5
  ldr(d18, mem[x5, 96]);
  ins(v17.d()[1], x8); // B
  if (max_mr > 3) {
    fmla(v26.v4s(), v14.v4s(), v4.s()[3]);
  }
  ldr(x8, mem[x5, 104]);
  if (max_mr > 4) {
    fmla(v28.v4s(), v14.v4s(), v5.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v14.v4s(), v5.s()[3]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x23, 128]); // Prefetch A5
  }

  // BLOCK 6
  ldr(d19, mem[x5, 112]);
  ins(v18.d()[1], x8); // B
  fmla(v21.v4s(), v15.v4s(), v3.s()[1]);
  ldr(x8, mem[x5, 120]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v15.v4s(), v3.s()[3]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 192]); // Prefetch B
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v15.v4s(), v4.s()[1]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 256]); // Prefetch B
  }

  // BLOCK 7
  subs(x0, x0, 16); // LDR lands here
  if (max_mr > 3) {
    fmla(v27.v4s(), v15.v4s(), v4.s()[3]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v15.v4s(), v5.s()[1]);
  }
  add(x5, x5, 128);
  if (max_mr > 5) {
    fmla(v31.v4s(), v15.v4s(), v5.s()[3]);
  }
  b_hs(l2);

  // Epilogue - 4 floats of A (16 bytes)
  // 48 FMA + 12 LD64 A + 8 LDR B
  bind(l3);
  // First group of 24 FMA, Second group loads
  // BLOCK 0
  ldr(d3, mem[x14], 8); // A0
  ins(v19.d()[1], x8); // B from second group
  fmla(v20.v4s(), v16.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    ldr(x8, mem[x15], 8); // A1
    fmla(v22.v4s(), v16.v4s(), v0.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v1.s()[0]);
  }
  if (prefetch) {
    prfm(kPSTL1KEEP, mem[x6]); // Prefetch C0
  }

  // BLOCK 1
  ldr(d12, mem[x5]);
  if (max_mr > 1) {
    ins(v3.d()[1], x8); // A1 ins
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v16.v4s(), v1.s()[2]);
  }
  ldr(x8, mem[x5, 8]); // B
  if (max_mr > 4) {
    fmla(v28.v4s(), v16.v4s(), v2.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v16.v4s(), v2.s()[2]);
  }
  if (prefetch) {
    prfm(kPSTL1KEEP, mem[x16]); // Prefetch C1
  }

  // BLOCK 2
  if (max_mr > 2) {
    ldr(d4, mem[x20], 8); // A2
  }
  ins(v12.d()[1], x8); // B  ins
  fmla(v21.v4s(), v17.v4s(), v0.s()[0]);
  if (max_mr > 3) {
    ldr(x8, mem[x21], 8); // A3
  }
  if (max_mr > 1) {
    fmla(v23.v4s(), v17.v4s(), v0.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v1.s()[0]);
  }
  if (prefetch) {
    prfm(kPSTL1KEEP, mem[x17]); // Prefetch C2
  }

  // BLOCK 3
  if (max_mr > 4) {
    ldr(d5, mem[x22], 8); // A4
  }
  if (max_mr > 3) {
    ins(v4.d()[1], x8); // A3 ins
    fmla(v27.v4s(), v17.v4s(), v1.s()[2]);
  }
  if (max_mr > 5) {
    ldr(x8, mem[x23], 8); // A5
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v17.v4s(), v2.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v17.v4s(), v2.s()[2]);
  }
  if (prefetch) {
    prfm(kPSTL1KEEP, mem[x10]); // Prefetch C3
  }

  // BLOCK 4
  ldr(d13, mem[x5, 16]);
  if (max_mr > 5) {
    ins(v5.d()[1], x8); // A5 ins
  }
  fmla(v20.v4s(), v18.v4s(), v0.s()[1]);
  ldr(x8, mem[x5, 24]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v18.v4s(), v0.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v18.v4s(), v1.s()[1]);
  }
  if (prefetch) {
    prfm(kPSTL1KEEP, mem[x13]); // Prefetch C4
  }

  // BLOCK 5
  ldr(d14, mem[x5, 32]);
  ins(v13.d()[1], x8); // B
  if (max_mr > 3) {
    fmla(v26.v4s(), v18.v4s(), v1.s()[3]);
  }
  ldr(x8, mem[x5, 40]);
  if (max_mr > 4) {
    fmla(v28.v4s(), v18.v4s(), v2.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v18.v4s(), v2.s()[3]);
  }
  if (prefetch) {
    prfm(kPSTL1KEEP, mem[x7]); // Prefetch C5
  }

  // BLOCK 6
  ldr(d15, mem[x5, 48]);
  ins(v14.d()[1], x8); // B
  fmla(v21.v4s(), v19.v4s(), v0.s()[1]);
  ldr(x8, mem[x5, 56]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v19.v4s(), v0.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v19.v4s(), v1.s()[1]);
  }

  // BLOCK 7
  ins(v15.d()[1], x8); // B from previous
  if (max_mr > 3) {
    fmla(v27.v4s(), v19.v4s(), v1.s()[3]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v19.v4s(), v2.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v19.v4s(), v2.s()[3]);
  }

  // Second group of 24 FMA, First group of loads
  // BLOCK 0
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
  if (max_mr > 4) {
    fmla(v28.v4s(), v12.v4s(), v5.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v12.v4s(), v5.s()[2]);
  }

  // BLOCK 2
  fmla(v21.v4s(), v13.v4s(), v3.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v13.v4s(), v3.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v13.v4s(), v4.s()[0]);
  }

  // BLOCK 3
  if (max_mr > 3) {
    fmla(v27.v4s(), v13.v4s(), v4.s()[2]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v13.v4s(), v5.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v13.v4s(), v5.s()[2]);
  }

  // BLOCK 4
  fmla(v20.v4s(), v14.v4s(), v3.s()[1]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v14.v4s(), v3.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v14.v4s(), v4.s()[1]);
  }

  // BLOCK 5
  if (max_mr > 3) {
    fmla(v26.v4s(), v14.v4s(), v4.s()[3]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v14.v4s(), v5.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v14.v4s(), v5.s()[3]);
  }
  tst(x0, 15);

  // BLOCK 6
  fmla(v21.v4s(), v15.v4s(), v3.s()[1]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v15.v4s(), v3.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v15.v4s(), v4.s()[1]);
  }
  add(x5, x5, 64);

  // BLOCK 7
  if (max_mr > 3) {
    fmla(v27.v4s(), v15.v4s(), v4.s()[3]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v15.v4s(), v5.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v15.v4s(), v5.s()[3]);
  }

  // Is there a remainder?- 2 floats of A (8 bytes) or less
  b_ne(l5);

  bind(l4);
  // ks loop
  subs(x9, x9, max_mr * sizeof(void*)); // ks -= MR * sizeof(void*)
  b_hi(l1);

  // Clamp
  if (clamp_min) {
    fmax(v20.v4s(), v20.v4s(), v6.v4s());
  }
  // Load cn_stride
  ldr(x0, mem[sp, 64]);
  if (clamp_min) {
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
    if (max_mr > 4) {
      fmax(v28.v4s(), v28.v4s(), v6.v4s());
      fmax(v29.v4s(), v29.v4s(), v6.v4s());
    }
    if (max_mr > 5) {
      fmax(v30.v4s(), v30.v4s(), v6.v4s());
      fmax(v31.v4s(), v31.v4s(), v6.v4s());
    }
  }
  subs(x1, x1, 8);
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
    if (max_mr > 4) {
      fmin(v28.v4s(), v28.v4s(), v7.v4s());
      fmin(v29.v4s(), v29.v4s(), v7.v4s());
    }
    if (max_mr > 5) {
      fmin(v30.v4s(), v30.v4s(), v7.v4s());
      fmin(v31.v4s(), v31.v4s(), v7.v4s());
    }
  }

  // Store full 6 x 8
  b_lo(l7);

  if (max_mr > 5) {
    stp(q30, q31, mem[x7]);
    add(x7, x7, x0);
  }
  if (max_mr > 4) {
    stp(q28, q29, mem[x13]);
    add(x13, x13, x0);
  }
  if (max_mr > 3) {
    stp(q26, q27, mem[x10]);
    add(x10, x10, x0);
  }
  if (max_mr > 2) {
    stp(q24, q25, mem[x17]);
    add(x17, x17, x0);
  }
  if (max_mr > 1) {
    stp(q22, q23, mem[x16]);
    add(x16, x16, x0);
  }
  stp(q20, q21, mem[x6]);
  add(x6, x6, x0);

  sub(x4, x4, x3); // A -= ks

  // nc loop
  b_hi(l0);

  // Restore x20-x23, d12-d15 from stack
  ldp(x22, x23, mem[sp, 48]);
  ldp(x20, x21, mem[sp, 32]);
  ldp(d14, d15, mem[sp, 16]);
  ldp(d12, d13, mem[sp], 64);
  ret();

  bind(l5);
  // Is there a remainder?- 2 floats of A (8 bytes)
  tbz(x0, 3, l6);

  // Remainder- 2 floats of A (8 bytes)
  ldr(d0, mem[x14], 8);
  ldr(q16, mem[x5], 16);
  if (max_mr > 1) {
    ld1({v0.d()}, 1, mem[x15], 8);
  }
  if (max_mr > 2) {
    ldr(d1, mem[x20], 8);
  }
  if (max_mr > 3) {
    ld1({v1.d()}, 1, mem[x21], 8);
  }
  if (max_mr > 4) {
    ldr(d2, mem[x22], 8);
  }
  if (max_mr > 5) {
    ld1({v2.d()}, 1, mem[x23], 8);
  }
  ldr(q17, mem[x5], 16);
  ldr(q18, mem[x5], 16);
  ldr(q19, mem[x5], 16);
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
  if (max_mr > 4) {
    fmla(v28.v4s(), v16.v4s(), v2.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v16.v4s(), v2.s()[2]);
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
  if (max_mr > 4) {
    fmla(v29.v4s(), v17.v4s(), v2.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v17.v4s(), v2.s()[2]);
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
  if (max_mr > 4) {
    fmla(v28.v4s(), v18.v4s(), v2.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v18.v4s(), v2.s()[3]);
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
  if (max_mr > 4) {
    fmla(v29.v4s(), v19.v4s(), v2.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v19.v4s(), v2.s()[3]);
  }

  // Is there a remainder?- 1 float of A (4 bytes)
  tbz(x0, 2, l4);
  bind(l6);
  // Remainder- 1 float of A (4 bytes)
  ldr(s0, mem[x14], 4);
  ldr(q16, mem[x5], 16);
  if (max_mr > 1) {
    ld1({v0.s()}, 2, mem[x15], 4);
  }
  if (max_mr > 2) {
    ldr(s1, mem[x20], 4);
  }
  if (max_mr > 3) {
    ld1({v1.s()}, 2, mem[x21], 4);
  }
  if (max_mr > 4) {
    ldr(s2, mem[x22], 4);
  }
  if (max_mr > 5) {
    ld1({v2.s()}, 2, mem[x23], 4);
  }
  ldr(q17, mem[x5], 16);

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
  if (max_mr > 4) {
    fmla(v28.v4s(), v16.v4s(), v2.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v16.v4s(), v2.s()[2]);
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
  if (max_mr > 4) {
    fmla(v29.v4s(), v17.v4s(), v2.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v17.v4s(), v2.s()[2]);
  }
  b(l4);

  // Store odd width
  bind(l7);
  tbz(x1, 2, l8);
  if (max_mr > 5) {
    str(q30, mem[x7], 16);
    mov(v30.v16b(), v31.v16b());
  }
  if (max_mr > 4) {
    str(q28, mem[x13], 16);
    mov(v28.v16b(), v29.v16b());
  }
  if (max_mr > 3) {
    str(q26, mem[x10], 16);
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
  bind(l8);
  tbz(x1, 1, l9);
  if (max_mr > 5) {
    str(d30, mem[x7], 8);
  }
  if (max_mr > 4) {
    str(d28, mem[x13], 8);
  }
  if (max_mr > 5) {
    dup(d30, v30.d()[1]);
  }
  if (max_mr > 4) {
    dup(d28, v28.d()[1]);
  }
  if (max_mr > 3) {
    str(d26, mem[x10], 8);
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

  bind(l9);
  tbz(x1, 0, l10);
  if (max_mr > 5) {
    str(s30, mem[x7]);
  }
  if (max_mr > 4) {
    str(s28, mem[x13]);
  }
  if (max_mr > 3) {
    str(s26, mem[x10]);
  }
  if (max_mr > 2) {
    str(s24, mem[x17]);
  }
  if (max_mr > 1) {
    str(s22, mem[x16]);
  }
  str(s20, mem[x6]);
  bind(l10);
  // Restore x20-x23, d12-d15 from stack
  ldp(x22, x23, mem[sp, 48]);
  ldp(x20, x21, mem[sp, 32]);
  ldp(d14, d15, mem[sp, 16]);
  ldp(d12, d13, mem[sp], 64);
  ret();

  align(16, AlignInstruction::kHlt);
}
}  // namespace
}  // namespace aarch64
}  // namespace xnnpack

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_cortex_a53(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
  using namespace xnnpack::aarch64;
  Generator g(code);
  assert(params != nullptr);
  g.generate(false, max_mr, nc_mod_nr, kc, ks, static_cast<const jit_gemm_params*>(params));
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a53(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
  using namespace xnnpack::aarch64;
  Generator g(code);
  assert(params != nullptr);
  g.generate(true, max_mr, nc_mod_nr, kc, ks, static_cast<const jit_gemm_params*>(params));
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
