// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <limits>

#include <xnnpack.h>
#include <xnnpack/aarch64-assembler.h>
#include <xnnpack/gemm.h>
#include <xnnpack/memory.h>
#include <xnnpack/microparams.h>
#include <xnnpack/post-operation.h>

namespace xnnpack {
namespace aarch64 {
namespace {
class Generator : public MacroAssembler {
  using MacroAssembler::MacroAssembler;

 public:
  void generate(size_t max_mr, size_t nc_mod_nr, size_t kc, const jit_gemm_params* jit_gemm_params);
};

// void xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0(
//     size_t mr,                x0
//     size_t nc,                x1
//     size_t kc,                x2 / x0
//     const void*restrict a,    x3
//     size_t a_stride,          x4
//     const void*restrict w,    x5
//     void*restrict c,          x6
//     size_t cm_stride,         x7
//     size_t cn_stride,         [sp] -> (x0)
//     const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])  [sp + 8] -> (x8)

// d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x3 v0
// A1  x9 v1
// A2 x10 v2
// A3 x11 v3
// A4 x12 v4
// A5  x4 v5
// B   x5 v16 v17 v18 v19
// C0  x6  v20 v21
// C1 x16  v22 v23
// C2 x17  v24 v25
// C3 x14  v26 v27
// C4 x13  v28 v29
// C5  x7  v30 v31
// clamp  v6, (v4), (v5)
// unused     v7
// unused A   v8 v9 v10 v11
// unused B   v12 v13 v14 v15

// x8 temporary vector shadow register

// Converted from: src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a55r0.S
void Generator::generate(size_t max_mr, size_t nc_mod_nr, size_t kc, const jit_gemm_params* jit_gemm_params)
{
  assert(max_mr <= 6);
  assert(nc_mod_nr < 16);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;
  const size_t num_post_operations = jit_gemm_params->num_post_operations;
  (void) num_post_operations;  // Silence unused warning.
  const uint16_t min = jit_gemm_params->f16_minmax.min;
  const uint16_t max = jit_gemm_params->f16_minmax.max;
  const bool clamp_min = min != UINT16_C(0xFC00);  // -Inf.
  const bool clamp_max = max != UINT16_C(0x7C00);  // Inf.
  assert(num_post_operations == 0 || (!clamp_min && !clamp_max));

  // Load params pointer
  ldr(x8, mem[sp, 8]);

  // Clamp A and C pointers
  if (max_mr > 1) {
    cmp(x0, 2); // if mr < 2
    add(x9, x3, x4); // a1 = a0 + a_stride
    add(x16, x6, x7); // c1 = c0 + cm_stride
    csel(x9, x3, x9, kLO); //   a1 = a0
    csel(x16, x6, x16, kLO); //   c1 = c0
  }

  // Load params
  ldr(s6, mem[x8]);

  if (max_mr > 2) {
    add(x10, x9, x4); // a2 = a1 + a_stride
    add(x17, x16, x7); // c2 = c1 + cm_stride
    // if mr <= 2
    csel(x10, x9, x10, kLS); //   a2 = a1
    csel(x17, x16, x17, kLS); //   c2 = c1
  }

  if (max_mr > 3) {
    cmp(x0, 4); // if mr < 4
    add(x11, x10, x4); // a3 = a2 + a_stride
    add(x14, x17, x7); // c3 = c2 + cm_stride
    csel(x11, x10, x11, kLO); //   a3 = a2
    csel(x14, x17, x14, kLO); //   c3 = c2
  }

  if (max_mr > 4) {
    add(x12, x11, x4); // a4 = a3 + a_stride
    add(x13, x14, x7); // c4 = c3 + cm_stride
    // if mr <= 4
    csel(x12, x11, x12, kLS); //   a4 = a3
    csel(x13, x14, x13, kLS); //   c4 = c3
  }

  if (max_mr > 5) {
    cmp(x0, 6); // if mr < 6
    add(x4, x12, x4); // a5 = a4 + a_stride
    add(x7, x13, x7); // c5 = c4 + cm_stride
    csel(x4, x12, x4, kLO); //   a5 = a4
    csel(x7, x13, x7, kLO); //   c5 = c4
  }

  // Save d12-d15 on stack
  stp(d12, d13, mem[sp, -32]++);
  stp(d14, d15, mem[sp, 16]);
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
  if (max_mr > 4) {
    mov(v28.v16b(), v20.v16b());
    mov(v29.v16b(), v21.v16b());
  }
  if (max_mr > 5) {
    mov(v30.v16b(), v20.v16b());
    mov(v31.v16b(), v21.v16b());
  }


  // Is there at least 4 halffloats (8 bytes) for prologue + epilogue?
  subs(x0, x2, 8); // k = kc - 8
  b_lo(l4);

  // Prologue - First group loads, no FMA
  ldr(s0, mem[x3], 4); // A0
  ldp(q16, q17, mem[x5], 32); // B
  if (max_mr > 2) {
    ldr(s1, mem[x10], 4); // A2
  }
  if (max_mr > 4) {
    ldr(s2, mem[x12], 4); // A4
  }
  if (max_mr > 1) {
    ld1({v0.s()}, 2, mem[x9], 4); // A1
  }
  if (max_mr > 3) {
    ld1({v1.s()}, 2, mem[x11], 4); // A3
  }
  if (max_mr > 5) {
    ld1({v2.s()}, 2, mem[x4], 4); // A5
  }
  ldr(q18, mem[x5], 16);
  ldr(d19, mem[x5], 8);
  ldr(x8, mem[x5], 8); // ins is in BLOCK 0
  subs(x0, x0, 8);

  // Is there at least 4 halffloats (8 bytes) for main loop?
  b_lo(l2);

  // Main loop - 4 halffloats of A (8 bytes)
  // 48 FMA + 12 LD32 A + 8 LDR B
  bind(l1);
  // First group of 24 FMA, Second group loads
  // BLOCK 0
  ldr(s3, mem[x3], 4); // A0
  ins(v19.d()[1], x8); // B from second group
  fmla(v20.v8h(), v16.v8h(), v0.h()[0]);
  if (max_mr > 1) {
    ldr(w8, mem[x9], 4); // A1
    fmla(v22.v8h(), v16.v8h(), v0.h()[4]);
  }
  if (max_mr > 2) {
    fmla(v24.v8h(), v16.v8h(), v1.h()[0]);
  }

  // BLOCK 1
  ldr(d12, mem[x5]);
  if (max_mr > 1) {
    ins(v3.d()[1], x8); // A1 ins
  }
  if (max_mr > 3) {
    fmla(v26.v8h(), v16.v8h(), v1.h()[4]);
  }
  ldr(x8, mem[x5, 8]); // B
  if (max_mr > 4) {
    fmla(v28.v8h(), v16.v8h(), v2.h()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v8h(), v16.v8h(), v2.h()[4]);
  }

  // BLOCK 2
  if (max_mr > 2) {
    ldr(s4, mem[x10], 4); // A2
  }
  ins(v12.d()[1], x8); // B  ins
  fmla(v21.v8h(), v17.v8h(), v0.h()[0]);
  if (max_mr > 3) {
    ldr(w8, mem[x11], 4); // A3
  }
  if (max_mr > 1) {
    fmla(v23.v8h(), v17.v8h(), v0.h()[4]);
  }
  if (max_mr > 2) {
    fmla(v25.v8h(), v17.v8h(), v1.h()[0]);
  }

  // BLOCK 3
  if (max_mr > 4) {
    ldr(s5, mem[x12], 4); // A4
  }
  if (max_mr > 3) {
    ins(v4.d()[1], x8); // A3 ins
    fmla(v27.v8h(), v17.v8h(), v1.h()[4]);
  }
  if (max_mr > 5) {
    ldr(w8, mem[x4], 4); // A5
  }
  if (max_mr > 4) {
    fmla(v29.v8h(), v17.v8h(), v2.h()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v8h(), v17.v8h(), v2.h()[4]);
  }

  // BLOCK 4
  ldr(d13, mem[x5, 16]);
  if (max_mr > 5) {
    ins(v5.d()[1], x8); // A5 ins
  }
  fmla(v20.v8h(), v18.v8h(), v0.h()[1]);
  ldr(x8, mem[x5, 24]);
  if (max_mr > 1) {
    fmla(v22.v8h(), v18.v8h(), v0.h()[5]);
  }
  if (max_mr > 2) {
    fmla(v24.v8h(), v18.v8h(), v1.h()[1]);
  }

  // BLOCK 5
  ldr(d14, mem[x5, 32]);
  ins(v13.d()[1], x8); // B
  if (max_mr > 3) {
    fmla(v26.v8h(), v18.v8h(), v1.h()[5]);
  }
  ldr(x8, mem[x5, 40]);
  if (max_mr > 4) {
    fmla(v28.v8h(), v18.v8h(), v2.h()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v8h(), v18.v8h(), v2.h()[5]);
  }

  // BLOCK 6
  ldr(d15, mem[x5, 48]);
  ins(v14.d()[1], x8); // B
  fmla(v21.v8h(), v19.v8h(), v0.h()[1]);
  ldr(x8, mem[x5, 56]);
  if (max_mr > 1) {
    fmla(v23.v8h(), v19.v8h(), v0.h()[5]);
  }
  if (max_mr > 2) {
    fmla(v25.v8h(), v19.v8h(), v1.h()[1]);
  }

  // BLOCK 7
  ins(v15.d()[1], x8);
  if (max_mr > 3) {
    fmla(v27.v8h(), v19.v8h(), v1.h()[5]);
  }
  if (max_mr > 4) {
    fmla(v29.v8h(), v19.v8h(), v2.h()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v8h(), v19.v8h(), v2.h()[5]);
  }

  // Second group of 24 FMA, First group of loads
  // BLOCK 0
  ldr(s0, mem[x3], 4); // A0
  fmla(v20.v8h(), v12.v8h(), v3.h()[0]);
  if (max_mr > 1) {
    ldr(w8, mem[x9], 4); // A1
    fmla(v22.v8h(), v12.v8h(), v3.h()[4]);
  }
  if (max_mr > 2) {
    fmla(v24.v8h(), v12.v8h(), v4.h()[0]);
  }

  // BLOCK 1
  ldr(d16, mem[x5, 64]);
  if (max_mr > 1) {
    ins(v0.d()[1], x8); // A1 ins
  }
  if (max_mr > 3) {
    fmla(v26.v8h(), v12.v8h(), v4.h()[4]);
  }
  ldr(x8, mem[x5, 72]); // B
  if (max_mr > 4) {
    fmla(v28.v8h(), v12.v8h(), v5.h()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v8h(), v12.v8h(), v5.h()[4]);
  }

  // BLOCK 2
  if (max_mr > 2) {
    ldr(s1, mem[x10], 4); // A2
  }
  ins(v16.d()[1], x8); // B
  fmla(v21.v8h(), v13.v8h(), v3.h()[0]);
  if (max_mr > 3) {
    ldr(w8, mem[x11], 4); // A3
  }
  if (max_mr > 1) {
    fmla(v23.v8h(), v13.v8h(), v3.h()[4]);
  }
  if (max_mr > 2) {
    fmla(v25.v8h(), v13.v8h(), v4.h()[0]);
  }

  // BLOCK 3
  if (max_mr > 4) {
    ldr(s2, mem[x12], 4); // A4
  }
  if (max_mr > 3) {
    ins(v1.d()[1], x8); // A3 ins
    fmla(v27.v8h(), v13.v8h(), v4.h()[4]);
  }
  if (max_mr > 5) {
    ldr(w8, mem[x4], 4); // A5
  }
  if (max_mr > 4) {
    fmla(v29.v8h(), v13.v8h(), v5.h()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v8h(), v13.v8h(), v5.h()[4]);
  }

  // BLOCK 4
  ldr(d17, mem[x5, 80]);
  if (max_mr > 5) {
    ins(v2.d()[1], x8); // A5 ins
  }
  fmla(v20.v8h(), v14.v8h(), v3.h()[1]);
  ldr(x8, mem[x5, 88]);
  if (max_mr > 1) {
    fmla(v22.v8h(), v14.v8h(), v3.h()[5]);
  }
  if (max_mr > 2) {
    fmla(v24.v8h(), v14.v8h(), v4.h()[1]);
  }

  // BLOCK 5
  ldr(d18, mem[x5, 96]);
  ins(v17.d()[1], x8); // B
  if (max_mr > 3) {
    fmla(v26.v8h(), v14.v8h(), v4.h()[5]);
  }
  ldr(x8, mem[x5, 104]);
  if (max_mr > 4) {
    fmla(v28.v8h(), v14.v8h(), v5.h()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v8h(), v14.v8h(), v5.h()[5]);
  }

  // BLOCK 6
  ldr(d19, mem[x5, 112]);
  ins(v18.d()[1], x8); // B
  fmla(v21.v8h(), v15.v8h(), v3.h()[1]);
  ldr(x8, mem[x5, 120]);
  if (max_mr > 1) {
    fmla(v23.v8h(), v15.v8h(), v3.h()[5]);
  }
  if (max_mr > 2) {
    fmla(v25.v8h(), v15.v8h(), v4.h()[1]);
  }

  // BLOCK 7
  subs(x0, x0, 8); // LDR lands here
  if (max_mr > 3) {
    fmla(v27.v8h(), v15.v8h(), v4.h()[5]);
  }
  if (max_mr > 4) {
    fmla(v29.v8h(), v15.v8h(), v5.h()[1]);
  }
  add(x5, x5, 128);
  if (max_mr > 5) {
    fmla(v31.v8h(), v15.v8h(), v5.h()[5]);
  }
  b_hs(l1);

  // Epilogue - 4 halffloats of A (8 bytes)
  // 48 FMA + 12 LD32 A + 8 LDR B
  bind(l2);
  // First group of 24 FMA, Second group loads
  // BLOCK 0
  ldr(s3, mem[x3], 4); // A0
  ins(v19.d()[1], x8); // B from second group
  fmla(v20.v8h(), v16.v8h(), v0.h()[0]);
  if (max_mr > 1) {
    ldr(w8, mem[x9], 4); // A1
    fmla(v22.v8h(), v16.v8h(), v0.h()[4]);
  }
  if (max_mr > 2) {
    fmla(v24.v8h(), v16.v8h(), v1.h()[0]);
  }

  // BLOCK 1
  ldr(d12, mem[x5]);
  if (max_mr > 1) {
    ins(v3.d()[1], x8); // A1 ins
  }
  if (max_mr > 3) {
    fmla(v26.v8h(), v16.v8h(), v1.h()[4]);
  }
  ldr(x8, mem[x5, 8]); // B
  if (max_mr > 4) {
    fmla(v28.v8h(), v16.v8h(), v2.h()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v8h(), v16.v8h(), v2.h()[4]);
  }

  // BLOCK 2
  if (max_mr > 2) {
    ldr(s4, mem[x10], 4); // A2
  }
  ins(v12.d()[1], x8); // B  ins
  fmla(v21.v8h(), v17.v8h(), v0.h()[0]);
  if (max_mr > 3) {
    ldr(w8, mem[x11], 4); // A3
  }
  if (max_mr > 1) {
    fmla(v23.v8h(), v17.v8h(), v0.h()[4]);
  }
  if (max_mr > 2) {
    fmla(v25.v8h(), v17.v8h(), v1.h()[0]);
  }

  // BLOCK 3
  if (max_mr > 4) {
    ldr(s5, mem[x12], 4); // A4
  }
  if (max_mr > 3) {
    ins(v4.d()[1], x8); // A3 ins
    fmla(v27.v8h(), v17.v8h(), v1.h()[4]);
  }
  if (max_mr > 5) {
    ldr(w8, mem[x4], 4); // A5
  }
  if (max_mr > 4) {
    fmla(v29.v8h(), v17.v8h(), v2.h()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v8h(), v17.v8h(), v2.h()[4]);
  }

  // BLOCK 4
  ldr(d13, mem[x5, 16]);
  if (max_mr > 5) {
    ins(v5.d()[1], x8); // A5 ins
  }
  fmla(v20.v8h(), v18.v8h(), v0.h()[1]);
  ldr(x8, mem[x5, 24]);
  if (max_mr > 1) {
    fmla(v22.v8h(), v18.v8h(), v0.h()[5]);
  }
  if (max_mr > 2) {
    fmla(v24.v8h(), v18.v8h(), v1.h()[1]);
  }

  // BLOCK 5
  ldr(d14, mem[x5, 32]);
  ins(v13.d()[1], x8); // B
  if (max_mr > 3) {
    fmla(v26.v8h(), v18.v8h(), v1.h()[5]);
  }
  ldr(x8, mem[x5, 40]);
  if (max_mr > 4) {
    fmla(v28.v8h(), v18.v8h(), v2.h()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v8h(), v18.v8h(), v2.h()[5]);
  }

  // BLOCK 6
  ldr(d15, mem[x5, 48]);
  ins(v14.d()[1], x8); // B
  fmla(v21.v8h(), v19.v8h(), v0.h()[1]);
  ldr(x8, mem[x5, 56]);
  if (max_mr > 1) {
    fmla(v23.v8h(), v19.v8h(), v0.h()[5]);
  }
  if (max_mr > 2) {
    fmla(v25.v8h(), v19.v8h(), v1.h()[1]);
  }

  // BLOCK 7
  ins(v15.d()[1], x8); // B
  if (max_mr > 3) {
    fmla(v27.v8h(), v19.v8h(), v1.h()[5]);
  }
  if (max_mr > 4) {
    fmla(v29.v8h(), v19.v8h(), v2.h()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v8h(), v19.v8h(), v2.h()[5]);
  }

  // Second group of 24 FMA, First group of loads
  // BLOCK 0
  fmla(v20.v8h(), v12.v8h(), v3.h()[0]);
  if (max_mr > 1) {
    fmla(v22.v8h(), v12.v8h(), v3.h()[4]);
  }
  if (max_mr > 2) {
    fmla(v24.v8h(), v12.v8h(), v4.h()[0]);
  }

  // BLOCK 1
  if (max_mr > 3) {
    fmla(v26.v8h(), v12.v8h(), v4.h()[4]);
  }
  if (max_mr > 4) {
    fmla(v28.v8h(), v12.v8h(), v5.h()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v8h(), v12.v8h(), v5.h()[4]);
  }

  // BLOCK 2
  fmla(v21.v8h(), v13.v8h(), v3.h()[0]);
  if (max_mr > 1) {
    fmla(v23.v8h(), v13.v8h(), v3.h()[4]);
  }
  if (max_mr > 2) {
    fmla(v25.v8h(), v13.v8h(), v4.h()[0]);
  }

  // BLOCK 3
  if (max_mr > 3) {
    fmla(v27.v8h(), v13.v8h(), v4.h()[4]);
  }
  if (max_mr > 4) {
    fmla(v29.v8h(), v13.v8h(), v5.h()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v8h(), v13.v8h(), v5.h()[4]);
  }

  // BLOCK 4
  fmla(v20.v8h(), v14.v8h(), v3.h()[1]);
  if (max_mr > 1) {
    fmla(v22.v8h(), v14.v8h(), v3.h()[5]);
  }
  if (max_mr > 2) {
    fmla(v24.v8h(), v14.v8h(), v4.h()[1]);
  }

  // BLOCK 5
  if (max_mr > 3) {
    fmla(v26.v8h(), v14.v8h(), v4.h()[5]);
  }
  if (max_mr > 4) {
    fmla(v28.v8h(), v14.v8h(), v5.h()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v8h(), v14.v8h(), v5.h()[5]);
  }
  tst(x0, 7);

  // BLOCK 6
  fmla(v21.v8h(), v15.v8h(), v3.h()[1]);
  if (max_mr > 1) {
    fmla(v23.v8h(), v15.v8h(), v3.h()[5]);
  }
  if (max_mr > 2) {
    fmla(v25.v8h(), v15.v8h(), v4.h()[1]);
  }
  add(x5, x5, 64);

  // BLOCK 7
  if (max_mr > 3) {
    fmla(v27.v8h(), v15.v8h(), v4.h()[5]);
  }
  if (max_mr > 4) {
    fmla(v29.v8h(), v15.v8h(), v5.h()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v8h(), v15.v8h(), v5.h()[5]);
  }

  // Is there a remainder?- 2 halffloats of A (4 bytes) or less
  b_ne(l4);

  bind(l3);
  // Clamp
  dup(v4.v8h(), v6.h()[0]);
  dup(v5.v8h(), v6.h()[1]);
  if (clamp_min) {
    fmax(v20.v8h(), v20.v8h(), v4.v8h());
  }
  ldr(x0, mem[sp, 32]); // cn_stride
  if (clamp_min) {
    fmax(v21.v8h(), v21.v8h(), v4.v8h());
    if (max_mr > 1) {
      fmax(v22.v8h(), v22.v8h(), v4.v8h());
      fmax(v23.v8h(), v23.v8h(), v4.v8h());
    }
    if (max_mr > 2) {
      fmax(v24.v8h(), v24.v8h(), v4.v8h());
      fmax(v25.v8h(), v25.v8h(), v4.v8h());
    }
    if (max_mr > 3) {
      fmax(v26.v8h(), v26.v8h(), v4.v8h());
      fmax(v27.v8h(), v27.v8h(), v4.v8h());
    }
    if (max_mr > 4) {
      fmax(v28.v8h(), v28.v8h(), v4.v8h());
      fmax(v29.v8h(), v29.v8h(), v4.v8h());
    }
    if (max_mr > 5) {
      fmax(v30.v8h(), v30.v8h(), v4.v8h());
      fmax(v31.v8h(), v31.v8h(), v4.v8h());
    }
  }
  subs(x1, x1, 16);
  if (clamp_max) {
    fmin(v20.v8h(), v20.v8h(), v5.v8h());
    fmin(v21.v8h(), v21.v8h(), v5.v8h());
    if (max_mr > 1) {
      fmin(v22.v8h(), v22.v8h(), v5.v8h());
      fmin(v23.v8h(), v23.v8h(), v5.v8h());
    }
    if (max_mr > 2) {
      fmin(v24.v8h(), v24.v8h(), v5.v8h());
      fmin(v25.v8h(), v25.v8h(), v5.v8h());
    }
    if (max_mr > 3) {
      fmin(v26.v8h(), v26.v8h(), v5.v8h());
      fmin(v27.v8h(), v27.v8h(), v5.v8h());
    }
    if (max_mr > 4) {
      fmin(v28.v8h(), v28.v8h(), v5.v8h());
      fmin(v29.v8h(), v29.v8h(), v5.v8h());
    }
    if (max_mr > 5) {
      fmin(v30.v8h(), v30.v8h(), v5.v8h());
      fmin(v31.v8h(), v31.v8h(), v5.v8h());
    }
  }

  // Store full 6 x 16
  b_lo(l6);

  st1({v20.v16b(), v21.v16b()}, mem[x6], x0);
  sub(x3, x3, x2); // a0 -= kc
  if (max_mr > 1) {
    st1({v22.v16b(), v23.v16b()}, mem[x16], x0);
    sub(x9, x9, x2); // a1 -= kc
  }
  if (max_mr > 2) {
    st1({v24.v16b(), v25.v16b()}, mem[x17], x0);
    sub(x10, x10, x2); // a2 -= kc
  }
  if (max_mr > 3) {
    st1({v26.v16b(), v27.v16b()}, mem[x14], x0);
    sub(x11, x11, x2); // a3 -= kc
  }
  if (max_mr > 4) {
    st1({v28.v16b(), v29.v16b()}, mem[x13], x0);
    sub(x12, x12, x2); // a4 -= kc
  }
  if (max_mr > 5) {
    st1({v30.v16b(), v31.v16b()}, mem[x7], x0);
    sub(x4, x4, x2); // a5 -= kc
  }

  b_hi(l0);

  // Restore d12-d15 from stack
  ldp(d14, d15, mem[sp, 16]);
  ldp(d12, d13, mem[sp], 32);
  ret();

  bind(l4);
  // Is there a remainder?- 2 halffloats of A (4 bytes)
  tbz(x0, 2, l5);

  // Remainder- 2 halffloats of A (4 bytes)
  ldr(s0, mem[x3], 4); // A0
  ldp(q16, q17, mem[x5], 32); // B
  if (max_mr > 2) {
    ldr(s1, mem[x10], 4); // A2
  }
  if (max_mr > 4) {
    ldr(s2, mem[x12], 4); // A4
  }
  if (max_mr > 1) {
    ld1({v0.s()}, 2, mem[x9], 4); // A1
  }
  if (max_mr > 3) {
    ld1({v1.s()}, 2, mem[x11], 4); // A3
  }
  if (max_mr > 5) {
    ld1({v2.s()}, 2, mem[x4], 4); // A5
  }
  ldr(q18, mem[x5], 16);
  ldr(q19, mem[x5], 16);
  fmla(v20.v8h(), v16.v8h(), v0.h()[0]);
  if (max_mr > 1) {
    fmla(v22.v8h(), v16.v8h(), v0.h()[4]);
  }
  if (max_mr > 2) {
    fmla(v24.v8h(), v16.v8h(), v1.h()[0]);
  }
  if (max_mr > 3) {
    fmla(v26.v8h(), v16.v8h(), v1.h()[4]);
  }
  if (max_mr > 4) {
    fmla(v28.v8h(), v16.v8h(), v2.h()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v8h(), v16.v8h(), v2.h()[4]);
  }
  fmla(v21.v8h(), v17.v8h(), v0.h()[0]);
  if (max_mr > 1) {
    fmla(v23.v8h(), v17.v8h(), v0.h()[4]);
  }
  if (max_mr > 2) {
    fmla(v25.v8h(), v17.v8h(), v1.h()[0]);
  }
  if (max_mr > 3) {
    fmla(v27.v8h(), v17.v8h(), v1.h()[4]);
  }
  if (max_mr > 4) {
    fmla(v29.v8h(), v17.v8h(), v2.h()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v8h(), v17.v8h(), v2.h()[4]);
  }
  fmla(v20.v8h(), v18.v8h(), v0.h()[1]);
  if (max_mr > 1) {
    fmla(v22.v8h(), v18.v8h(), v0.h()[5]);
  }
  if (max_mr > 2) {
    fmla(v24.v8h(), v18.v8h(), v1.h()[1]);
  }
  if (max_mr > 3) {
    fmla(v26.v8h(), v18.v8h(), v1.h()[5]);
  }
  if (max_mr > 4) {
    fmla(v28.v8h(), v18.v8h(), v2.h()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v8h(), v18.v8h(), v2.h()[5]);
  }
  fmla(v21.v8h(), v19.v8h(), v0.h()[1]);
  if (max_mr > 1) {
    fmla(v23.v8h(), v19.v8h(), v0.h()[5]);
  }
  if (max_mr > 2) {
    fmla(v25.v8h(), v19.v8h(), v1.h()[1]);
  }
  if (max_mr > 3) {
    fmla(v27.v8h(), v19.v8h(), v1.h()[5]);
  }
  if (max_mr > 4) {
    fmla(v29.v8h(), v19.v8h(), v2.h()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v8h(), v19.v8h(), v2.h()[5]);
  }

  // Is there a remainder?- 1 halffloat of A (2 bytes)
  tbz(x0, 1, l3);
  bind(l5);

  // Remainder- 1 halffloat of A (2 bytes)
  ldr(h0, mem[x3], 2); // A0
  ldp(q16, q17, mem[x5], 32); // B
  if (max_mr > 2) {
    ldr(h1, mem[x10], 2); // A2
  }
  if (max_mr > 4) {
    ldr(h2, mem[x12], 2); // A4
  }
  if (max_mr > 1) {
    ld1({v0.h()}, 4, mem[x9], 2); // A1
  }
  if (max_mr > 3) {
    ld1({v1.h()}, 4, mem[x11], 2); // A3
  }
  if (max_mr > 5) {
    ld1({v2.h()}, 4, mem[x4], 2); // A5
  }
  fmla(v20.v8h(), v16.v8h(), v0.h()[0]);
  if (max_mr > 1) {
    fmla(v22.v8h(), v16.v8h(), v0.h()[4]);
  }
  if (max_mr > 2) {
    fmla(v24.v8h(), v16.v8h(), v1.h()[0]);
  }
  if (max_mr > 3) {
    fmla(v26.v8h(), v16.v8h(), v1.h()[4]);
  }
  if (max_mr > 4) {
    fmla(v28.v8h(), v16.v8h(), v2.h()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v8h(), v16.v8h(), v2.h()[4]);
  }
  fmla(v21.v8h(), v17.v8h(), v0.h()[0]);
  if (max_mr > 1) {
    fmla(v23.v8h(), v17.v8h(), v0.h()[4]);
  }
  if (max_mr > 2) {
    fmla(v25.v8h(), v17.v8h(), v1.h()[0]);
  }
  if (max_mr > 3) {
    fmla(v27.v8h(), v17.v8h(), v1.h()[4]);
  }
  if (max_mr > 4) {
    fmla(v29.v8h(), v17.v8h(), v2.h()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v8h(), v17.v8h(), v2.h()[4]);
  }
  b(l3);

  // Store odd width
  bind(l6);
  tbz(x1, 3, l7);
  str(q20, mem[x6], 16);
  mov(v20.v16b(), v21.v16b());
  if (max_mr > 1) {
    str(q22, mem[x16], 16);
    mov(v22.v16b(), v23.v16b());
  }
  if (max_mr > 2) {
    str(q24, mem[x17], 16);
    mov(v24.v16b(), v25.v16b());
  }
  if (max_mr > 3) {
    str(q26, mem[x14], 16);
    mov(v26.v16b(), v27.v16b());
  }
  if (max_mr > 4) {
    str(q28, mem[x13], 16);
    mov(v28.v16b(), v29.v16b());
  }
  if (max_mr > 5) {
    str(q30, mem[x7], 16);
    mov(v30.v16b(), v31.v16b());
  }

  bind(l7);
  tbz(x1, 2, l8);
  str(d20, mem[x6], 8);
  if (max_mr > 1) {
    str(d22, mem[x16], 8);
  }
  dup(d20, v20.d()[1]);
  if (max_mr > 1) {
    dup(d22, v22.d()[1]);
  }
  if (max_mr > 2) {
    str(d24, mem[x17], 8);
  }
  if (max_mr > 3) {
    str(d26, mem[x14], 8);
  }
  if (max_mr > 2) {
    dup(d24, v24.d()[1]);
  }
  if (max_mr > 3) {
    dup(d26, v26.d()[1]);
  }
  if (max_mr > 4) {
    str(d28, mem[x13], 8);
  }
  if (max_mr > 5) {
    str(d30, mem[x7], 8);
  }
  if (max_mr > 4) {
    dup(d28, v28.d()[1]);
  }
  if (max_mr > 5) {
    dup(d30, v30.d()[1]);
  }

  bind(l8);
  tbz(x1, 1, l9);
  str(s20, mem[x6], 4);
  if (max_mr > 1) {
    str(s22, mem[x16], 4);
  }
  dup(s20, v20.s()[1]);
  if (max_mr > 1) {
    dup(s22, v22.s()[1]);
  }
  if (max_mr > 2) {
    str(s24, mem[x17], 4);
  }
  if (max_mr > 3) {
    str(s26, mem[x14], 4);
  }
  if (max_mr > 2) {
    dup(s24, v24.s()[1]);
  }
  if (max_mr > 3) {
    dup(s26, v26.s()[1]);
  }
  if (max_mr > 4) {
    str(s28, mem[x13], 4);
  }
  if (max_mr > 5) {
    str(s30, mem[x7], 4);
  }
  if (max_mr > 4) {
    dup(s28, v28.s()[1]);
  }
  if (max_mr > 5) {
    dup(s30, v30.s()[1]);
  }

  bind(l9);
  tbz(x1, 0, l10);
  str(h20, mem[x6]);
  if (max_mr > 1) {
    str(h22, mem[x16]);
  }
  if (max_mr > 2) {
    str(h24, mem[x17]);
  }
  if (max_mr > 3) {
    str(h26, mem[x14]);
  }
  if (max_mr > 4) {
    str(h28, mem[x13]);
  }
  if (max_mr > 5) {
    str(h30, mem[x7]);
  }
  bind(l10);
  // Restore d12-d15 from stack
  ldp(d14, d15, mem[sp, 16]);
  ldp(d12, d13, mem[sp], 32);
  ret();

  align(16, AlignInstruction::kHlt);
}
}  // namespace
}  // namespace aarch64
}  // namespace xnnpack

xnn_status_t xnn_generate_f16_gemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, const void* params) {
  using namespace xnnpack::aarch64;
  Generator g(code);
  assert(params != nullptr);
  g.generate(max_mr, nc_mod_nr, kc, static_cast<const jit_gemm_params*>(params));
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
