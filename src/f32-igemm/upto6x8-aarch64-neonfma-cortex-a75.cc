// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <limits>

#include <xnnpack.h>
#include <xnnpack/aarch64-assembler.h>
#include <xnnpack/allocator.h>
#include <xnnpack/igemm.h>

namespace xnnpack {
namespace aarch64 {
namespace {
class Generator : public Assembler {
  using Assembler::Assembler;

public:
  void generate(bool prefetch, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, float min, float max);
};

// void xnn_f32_igemm_minmax_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75(
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
//     const xnn_f32_minmax_params params [sp + 24] -> x8

// d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// A pointers
// x14 a0
// x15 a1
// x20 a2
// x21 a3
// x22 a4
// x23 a5

// C pointers
//  x6 c0
// x16 c1
// x17 c2
// x10 c3
// x13 c4
//  x7 c5

// Vector register usage
// A0   v0  v6
// A1   v1  v7
// A2   v2  v8
// A3   v3  v9
// A4   v4 v10
// A5   v5 v11
// B   v12 v13 v14 v15
// B   v16 v17 v18 v19
// C   v20 v21
// C   v22 v23
// C   v24 v25
// C   v26 v27
// C   v28 v29
// C   v30 v31
// Clamp v6 v7

// Converted from: src/f32-igemm/gen/6x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
void Generator::generate(bool prefetch, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, float min, float max)
{
  assert(max_mr <= 6);
  assert(nc_mod_nr < 8);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11;
  const bool clamp_min = min != -std::numeric_limits<float>::infinity();
  const bool clamp_max = max != +std::numeric_limits<float>::infinity();

  // Clamp C pointers / Save d8-d15 on stack
  stp(d8, d9, mem[sp, -96]++);
  if (max_mr > 1) {
    cmp(x0, 2);              // if mr < 2
    add(x16, x6, x7);        // c1 = c0 + cm_stride
    csel(x16, x6, x16, kLO); //   c1 = c0
  }

  stp(d10, d11, mem[sp, 16]);
  if (max_mr > 2) {
    add(x17, x16, x7); // c2 = c1 + cm_stride
    // if mr <= 2
    csel(x17, x16, x17, kLS); //   c2 = c1
  }

  stp(d12, d13, mem[sp, 32]);
  if (max_mr > 3) {
    cmp(x0, 4);               // if mr < 4
    add(x10, x17, x7);        // c3 = c2 + cm_stride
    csel(x10, x17, x10, kLO); //   c3 = c2
  }

  stp(d14, d15, mem[sp, 48]);
  if (max_mr > 4) {
    add(x13, x10, x7); // c4 = c3 + cm_stride
                       // if mr <= 4
    csel(x13, x10, x13, kLS); //   c4 = c3
  }

  // Save x20,x21,x22,x23 on stack
  stp(x20, x21, mem[sp, 64]);
  stp(x22, x23, mem[sp, 80]);

  if (max_mr > 5) {
    cmp(x0, 6);             // if mr < 6
    add(x7, x13, x7);       // c5 = c4 + cm_stride
    csel(x7, x13, x7, kLO); //   c5 = c4
  }

  // Load a_offset
  ldr(x11, mem[sp, 104]);

  // Load zero, params pointer
  ldp(x12, x8, mem[sp, 112]);

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
    mov(v25.v16b(), v21.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 64]);
  }
  if (max_mr > 3) {
    mov(v26.v16b(), v20.v16b());
    mov(v27.v16b(), v21.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 128]);
  }
  if (max_mr > 4) {
    mov(v28.v16b(), v20.v16b());
    mov(v29.v16b(), v21.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 192]);
  }
  if (max_mr > 5) {
    mov(v30.v16b(), v20.v16b());
    mov(v31.v16b(), v21.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 256]);
    prfm(kPLDL1KEEP, mem[x5, 320]);
  }

  mov(x9, x3); // p = ks

  bind(l1);
  // Load next 6 A pointers
  switch (max_mr) {
    case 1:
      ldr(x14, mem[x4], 8);
      break;
    case 2:
      ldp(x14, x15, mem[x4], 16);
      break;
    case 3:
      ldp(x14, x15, mem[x4], 16);
      ldr(x20, mem[x4], 8);
      break;
    case 4:
      ldp(x14, x15, mem[x4], 16);
      ldp(x20, x21, mem[x4], 16);
      break;
    case 5:
      ldp(x14, x15, mem[x4], 16);
      ldp(x20, x21, mem[x4], 16);
      ldr(x22, mem[x4], 8);
      break;
    case 6:
      ldp(x14, x15, mem[x4], 16);
      ldp(x20, x21, mem[x4], 16);
      ldp(x22, x23, mem[x4], 16);
      break;
  }

  cmp(x14, x12);            // if a0 == zero
  add(x14, x14, x11);       // a0 += a_offset
  csel(x14, x12, x14, kEQ); //   a0 = zero, else += a0 + a_offset
  if (max_mr > 1) {
    cmp(x15, x12);            // if a1 == zero
    add(x15, x15, x11);       // a1 += a_offset
    csel(x15, x12, x15, kEQ); //   a1 = zero, else += a1 + a_offset
  }
  if (max_mr > 2) {
    cmp(x20, x12);            // if a2 == zero
    add(x20, x20, x11);       // a2 += a_offset
    csel(x20, x12, x20, kEQ); //   a2 = zero, else += a2 + a_offset
  }
  if (max_mr > 3) {
    cmp(x21, x12);            // if a3 == zero
    add(x21, x21, x11);       // a3 += a_offset
    csel(x21, x12, x21, kEQ); //   a3 = zero, else += a3 + a_offset
  }
  if (max_mr > 4) {
    cmp(x22, x12);            // if a4 == zero
    add(x22, x22, x11);       // a4 += a_offset
    csel(x22, x12, x22, kEQ); //   a4 = zero, else += a4 + a_offset
  }
  if (max_mr > 5) {
    cmp(x23, x12);            // if a5 == zero
    add(x23, x23, x11);       // a5 += a_offset
    csel(x23, x12, x23, kEQ); //   a5 = zero, else += a5 + a_offset
  }

  // Is there at least 8 floats (32 bytes) for prologue + epilogue?
  subs(x0, x2, 32); // k = kc - 32
  b_lo(l5);

  // Prologue - loads for main loop of 96 FMA
  ldr(q0, mem[x14], 16);
  ldp(q12, q13, mem[x5], 32); // Fetch 3 B (4th deferred)
  if (max_mr > 1) {
    ldr(q1, mem[x15], 16);
  }
  if (max_mr > 2) {
    ldr(q2, mem[x20], 16);
  }
  if (max_mr > 3) {
    ldr(q3, mem[x21], 16);
  }
  if (max_mr > 4) {
    ldr(q4, mem[x22], 16);
  }
  if (max_mr > 5) {
    ldr(q5, mem[x23], 16);
  }
  ldp(q14, q15, mem[x5], 32);
  ldp(q16, q17, mem[x5], 32);

  // Is there at least 8 floats (32 bytes) for main loop?
  subs(x0, x0, 32);
  b_lo(l3);
  // Main loop - 8 floats of A (32 bytes)
  // 96 FMA + 6 LDP A + 8 LDP B
  bind(l2);
  // First group of 4 A.  48 FMA.
  fmla(v20.v4s(), v12.v4s(), v0.s()[0]);
  ldp(q18, q19, mem[x5], 32); // Load last B
  if (max_mr > 1) {
    fmla(v22.v4s(), v12.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v12.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v12.v4s(), v3.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v12.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v12.v4s(), v5.s()[0]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 256]); // Prefetch B
  }
  fmla(v21.v4s(), v13.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v13.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v13.v4s(), v2.s()[0]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 320]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v13.v4s(), v3.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v13.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v13.v4s(), v5.s()[0]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 384]);
  }
  fmla(v20.v4s(), v14.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v14.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v14.v4s(), v2.s()[1]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 448]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v14.v4s(), v3.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v14.v4s(), v4.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v14.v4s(), v5.s()[1]);
  }
  fmla(v21.v4s(), v15.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v15.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v15.v4s(), v2.s()[1]);
  }
  ldr(q6, mem[x14], 16); // Load next 6 A
  if (max_mr > 3) {
    fmla(v27.v4s(), v15.v4s(), v3.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v15.v4s(), v4.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v15.v4s(), v5.s()[1]);
  }
  if (max_mr > 1) {
    ldr(q7, mem[x15], 16);
  }

  fmla(v20.v4s(), v16.v4s(), v0.s()[2]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v16.v4s(), v1.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v2.s()[2]);
  }
  if (max_mr > 2) {
    ldr(q8, mem[x20], 16);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v16.v4s(), v3.s()[2]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v16.v4s(), v4.s()[2]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v16.v4s(), v5.s()[2]);
  }
  if (max_mr > 3) {
    ldr(q9, mem[x21], 16);
  }
  fmla(v21.v4s(), v17.v4s(), v0.s()[2]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v17.v4s(), v1.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v2.s()[2]);
  }
  if (max_mr > 4) {
    ldr(q10, mem[x22], 16);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v17.v4s(), v3.s()[2]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v17.v4s(), v4.s()[2]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v17.v4s(), v5.s()[2]);
    ldr(q11, mem[x23], 16);
  }

  fmla(v20.v4s(), v18.v4s(), v0.s()[3]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v18.v4s(), v1.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v18.v4s(), v2.s()[3]);
  }
  ldp(q12, q13, mem[x5], 32); // Load 4 B
  if (max_mr > 3) {
    fmla(v26.v4s(), v18.v4s(), v3.s()[3]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v18.v4s(), v4.s()[3]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v18.v4s(), v5.s()[3]);
  }
  ldp(q14, q15, mem[x5], 32);
  fmla(v21.v4s(), v19.v4s(), v0.s()[3]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v19.v4s(), v1.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v19.v4s(), v2.s()[3]);
  }
  ldp(q16, q17, mem[x5], 32);
  if (max_mr > 3) {
    fmla(v27.v4s(), v19.v4s(), v3.s()[3]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v19.v4s(), v4.s()[3]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v19.v4s(), v5.s()[3]);
  }
  ldp(q18, q19, mem[x5], 32);

  // Second group of 4 A.  48 FMA.
  fmla(v20.v4s(), v12.v4s(), v6.s()[0]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v12.v4s(), v7.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v12.v4s(), v8.s()[0]);
  }
  ldr(q0, mem[x14], 16); // Load next 6 A
  if (max_mr > 3) {
    fmla(v26.v4s(), v12.v4s(), v9.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v12.v4s(), v10.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v12.v4s(), v11.s()[0]);
  }
  if (max_mr > 1) {
    ldr(q1, mem[x15], 16);
  }
  fmla(v21.v4s(), v13.v4s(), v6.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v13.v4s(), v7.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v13.v4s(), v8.s()[0]);
  }
  if (max_mr > 2) {
    ldr(q2, mem[x20], 16);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v13.v4s(), v9.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v13.v4s(), v10.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v13.v4s(), v11.s()[0]);
  }
  if (max_mr > 3) {
    ldr(q3, mem[x21], 16);
  }

  fmla(v20.v4s(), v14.v4s(), v6.s()[1]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v14.v4s(), v7.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v14.v4s(), v8.s()[1]);
  }
  if (max_mr > 4) {
    ldr(q4, mem[x22], 16);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v14.v4s(), v9.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v14.v4s(), v10.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v14.v4s(), v11.s()[1]);
  }
  if (max_mr > 5) {
    ldr(q5, mem[x23], 16);
  }
  fmla(v21.v4s(), v15.v4s(), v6.s()[1]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v15.v4s(), v7.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v15.v4s(), v8.s()[1]);
  }
  ldp(q12, q13, mem[x5], 32); // Load next 3 B (not last)
  if (max_mr > 3) {
    fmla(v27.v4s(), v15.v4s(), v9.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v15.v4s(), v10.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v15.v4s(), v11.s()[1]);
  }
  ldp(q14, q15, mem[x5], 32);

  fmla(v20.v4s(), v16.v4s(), v6.s()[2]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v16.v4s(), v7.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v8.s()[2]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v16.v4s(), v9.s()[2]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v16.v4s(), v10.s()[2]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v16.v4s(), v11.s()[2]);
  }
  fmla(v21.v4s(), v17.v4s(), v6.s()[2]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v17.v4s(), v7.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v8.s()[2]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v17.v4s(), v9.s()[2]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v17.v4s(), v10.s()[2]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v17.v4s(), v11.s()[2]);
  }
  ldp(q16, q17, mem[x5], 32);

  fmla(v20.v4s(), v18.v4s(), v6.s()[3]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v18.v4s(), v7.s()[3]);
  }
  subs(x0, x0, 32);
  if (max_mr > 2) {
    fmla(v24.v4s(), v18.v4s(), v8.s()[3]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v18.v4s(), v9.s()[3]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v18.v4s(), v10.s()[3]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v18.v4s(), v11.s()[3]);
  }
  fmla(v21.v4s(), v19.v4s(), v6.s()[3]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v19.v4s(), v7.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v19.v4s(), v8.s()[3]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v19.v4s(), v9.s()[3]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v19.v4s(), v10.s()[3]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v19.v4s(), v11.s()[3]);
  }
  b_hs(l2);

  // Epilogue - 8 floats of A (32 bytes)
  // 96 FMA + 6 LDP A + 8 LDP B
  // First block same as main loop.  Second block has no preloads.
  bind(l3);
  // First group of 4 A.  48 FMA.
  fmla(v20.v4s(), v12.v4s(), v0.s()[0]);
  ldp(q18, q19, mem[x5], 32); // Load last B
  if (max_mr > 1) {
    fmla(v22.v4s(), v12.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v12.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v12.v4s(), v3.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v12.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v12.v4s(), v5.s()[0]);
  }
  fmla(v21.v4s(), v13.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v13.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v13.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v13.v4s(), v3.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v13.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v13.v4s(), v5.s()[0]);
  }
  fmla(v20.v4s(), v14.v4s(), v0.s()[1]);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 128]); // Prefetch B
  }
  if (max_mr > 1) {
    fmla(v22.v4s(), v14.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v14.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v14.v4s(), v3.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v14.v4s(), v4.s()[1]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 256]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v14.v4s(), v5.s()[1]);
  }
  fmla(v21.v4s(), v15.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v15.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v15.v4s(), v2.s()[1]);
  }
  ldr(q6, mem[x14], 16); // Load next 6 A
  if (max_mr > 3) {
    fmla(v27.v4s(), v15.v4s(), v3.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v15.v4s(), v4.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v15.v4s(), v5.s()[1]);
  }
  if (max_mr > 1) {
    ldr(q7, mem[x15], 16);
  }

  fmla(v20.v4s(), v16.v4s(), v0.s()[2]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v16.v4s(), v1.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v2.s()[2]);
  }
  if (max_mr > 2) {
    ldr(q8, mem[x20], 16);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v16.v4s(), v3.s()[2]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v16.v4s(), v4.s()[2]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v16.v4s(), v5.s()[2]);
  }
  if (max_mr > 3) {
    ldr(q9, mem[x21], 16);
  }
  fmla(v21.v4s(), v17.v4s(), v0.s()[2]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v17.v4s(), v1.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v2.s()[2]);
  }
  if (max_mr > 4) {
    ldr(q10, mem[x22], 16);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v17.v4s(), v3.s()[2]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v17.v4s(), v4.s()[2]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v17.v4s(), v5.s()[2]);
  }
  if (max_mr > 5) {
    ldr(q11, mem[x23], 16);
  }

  fmla(v20.v4s(), v18.v4s(), v0.s()[3]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v18.v4s(), v1.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v18.v4s(), v2.s()[3]);
  }
  ldp(q12, q13, mem[x5], 32); // Load 4 B
  if (max_mr > 3) {
    fmla(v26.v4s(), v18.v4s(), v3.s()[3]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v18.v4s(), v4.s()[3]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v18.v4s(), v5.s()[3]);
  }
  ldp(q14, q15, mem[x5], 32);
  fmla(v21.v4s(), v19.v4s(), v0.s()[3]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v19.v4s(), v1.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v19.v4s(), v2.s()[3]);
  }
  ldp(q16, q17, mem[x5], 32);
  if (max_mr > 3) {
    fmla(v27.v4s(), v19.v4s(), v3.s()[3]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v19.v4s(), v4.s()[3]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v19.v4s(), v5.s()[3]);
  }
  ldp(q18, q19, mem[x5], 32);

  // Second group of 4 A.  48 FMA.
  fmla(v20.v4s(), v12.v4s(), v6.s()[0]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v12.v4s(), v7.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v12.v4s(), v8.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v12.v4s(), v9.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v12.v4s(), v10.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v12.v4s(), v11.s()[0]);
  }
  fmla(v21.v4s(), v13.v4s(), v6.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v13.v4s(), v7.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v13.v4s(), v8.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v13.v4s(), v9.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v13.v4s(), v10.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v13.v4s(), v11.s()[0]);
  }

  fmla(v20.v4s(), v14.v4s(), v6.s()[1]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v14.v4s(), v7.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v14.v4s(), v8.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v14.v4s(), v9.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v14.v4s(), v10.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v14.v4s(), v11.s()[1]);
  }
  fmla(v21.v4s(), v15.v4s(), v6.s()[1]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v15.v4s(), v7.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v15.v4s(), v8.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v15.v4s(), v9.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v15.v4s(), v10.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v15.v4s(), v11.s()[1]);
  }

  fmla(v20.v4s(), v16.v4s(), v6.s()[2]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v16.v4s(), v7.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v8.s()[2]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v16.v4s(), v9.s()[2]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v16.v4s(), v10.s()[2]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v16.v4s(), v11.s()[2]);
  }
  fmla(v21.v4s(), v17.v4s(), v6.s()[2]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v17.v4s(), v7.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v8.s()[2]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v17.v4s(), v9.s()[2]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v17.v4s(), v10.s()[2]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v17.v4s(), v11.s()[2]);
  }

  fmla(v20.v4s(), v18.v4s(), v6.s()[3]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v18.v4s(), v7.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v18.v4s(), v8.s()[3]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v18.v4s(), v9.s()[3]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v18.v4s(), v10.s()[3]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v18.v4s(), v11.s()[3]);
  }
  // Is there a remainder?- 4 floats of A (16 bytes) or less
  tst(x0, 31);
  fmla(v21.v4s(), v19.v4s(), v6.s()[3]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v19.v4s(), v7.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v19.v4s(), v8.s()[3]);
  }
  // Load min/max values
  if (clamp_min || clamp_max) {
    ld2r({v6.v4s(), v7.v4s()}, mem[x8]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v19.v4s(), v9.s()[3]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v19.v4s(), v10.s()[3]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v19.v4s(), v11.s()[3]);
  }
  b_ne(l5);

  bind(l4);
  // ks loop
  subs(x9, x9, max_mr * 8); // ks -= MR * sizeof(void*)
  b_hi(l1);

  // Load cn_stride
  ldr(x0, mem[sp, 96]);
  subs(x1, x1, 8);

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
    if (max_mr > 4) {
      fmax(v28.v4s(), v28.v4s(), v6.v4s());
      fmax(v29.v4s(), v29.v4s(), v6.v4s());
    }
    if (max_mr > 5) {
      fmax(v30.v4s(), v30.v4s(), v6.v4s());
      fmax(v31.v4s(), v31.v4s(), v6.v4s());
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
  b_lo(l8);

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

  sub(x4, x4, x3); // a -= ks

  // nc loop
  b_hi(l0);

  // Restore x20,x21,x22,x23 from stack
  if (max_mr > 4) {
    ldp(x22, x23, mem[sp, 80]);
  }
  ldp(x20, x21, mem[sp, 64]);

  // Restore d8-d15 from stack
  ldp(d14, d15, mem[sp, 48]);
  ldp(d12, d13, mem[sp, 32]);
  ldp(d10, d11, mem[sp, 16]);
  ldp(d8, d9, mem[sp], 96);
  ret();

  bind(l5);
  // Load min/max values
  if (clamp_min || clamp_max) {
    ld2r({v6.v4s(), v7.v4s()}, mem[x8]);
  }

  // Is there a remainder?- 4 floats of A (16 bytes)
  tbz(x0, 4, l6);

  // Remainder- 4 floats of A (16 bytes)
  // Load A
  ldr(q0, mem[x14], 16);
  if (max_mr > 1) {
    ldr(q1, mem[x15], 16);
  }
  if (max_mr > 2) {
    ldr(q2, mem[x20], 16);
  }
  if (max_mr > 3) {
    ldr(q3, mem[x21], 16);
  }
  if (max_mr > 4) {
    ldr(q4, mem[x22], 16);
  }
  if (max_mr > 5) {
    ldr(q5, mem[x23], 16);
  }
  // Load B
  ldp(q12, q13, mem[x5], 32);
  ldp(q14, q15, mem[x5], 32);
  ldp(q16, q17, mem[x5], 32);
  ldp(q18, q19, mem[x5], 32);

  fmla(v20.v4s(), v12.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v12.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v12.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v12.v4s(), v3.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v12.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v12.v4s(), v5.s()[0]);
  }
  fmla(v21.v4s(), v13.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v13.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v13.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v13.v4s(), v3.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v13.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v13.v4s(), v5.s()[0]);
  }

  fmla(v20.v4s(), v14.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v14.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v14.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v14.v4s(), v3.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v14.v4s(), v4.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v14.v4s(), v5.s()[1]);
  }
  fmla(v21.v4s(), v15.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v15.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v15.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v15.v4s(), v3.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v15.v4s(), v4.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v15.v4s(), v5.s()[1]);
  }

  fmla(v20.v4s(), v16.v4s(), v0.s()[2]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v16.v4s(), v1.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v2.s()[2]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v16.v4s(), v3.s()[2]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v16.v4s(), v4.s()[2]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v16.v4s(), v5.s()[2]);
  }
  fmla(v21.v4s(), v17.v4s(), v0.s()[2]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v17.v4s(), v1.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v2.s()[2]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v17.v4s(), v3.s()[2]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v17.v4s(), v4.s()[2]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v17.v4s(), v5.s()[2]);
  }

  fmla(v20.v4s(), v18.v4s(), v0.s()[3]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v18.v4s(), v1.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v18.v4s(), v2.s()[3]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v18.v4s(), v3.s()[3]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v18.v4s(), v4.s()[3]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v18.v4s(), v5.s()[3]);
  }
  fmla(v21.v4s(), v19.v4s(), v0.s()[3]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v19.v4s(), v1.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v19.v4s(), v2.s()[3]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v19.v4s(), v3.s()[3]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v19.v4s(), v4.s()[3]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v19.v4s(), v5.s()[3]);
  }

  // Is there a remainder?- 2 floats of A (8 bytes)
  bind(l6);
  tbz(x0, 3, l7);

  // Remainder- 2 floats of A (8 bytes)
  // Load A
  ldr(d0, mem[x14], 8);
  if (max_mr > 1) {
    ldr(d1, mem[x15], 8);
  }
  if (max_mr > 2) {
    ldr(d2, mem[x20], 8);
  }
  if (max_mr > 3) {
    ldr(d3, mem[x21], 8);
  }
  if (max_mr > 4) {
    ldr(d4, mem[x22], 8);
  }
  if (max_mr > 5) {
    ldr(d5, mem[x23], 8);
  }
  // Load B
  ldp(q12, q13, mem[x5], 32);
  ldp(q14, q15, mem[x5], 32);

  fmla(v20.v4s(), v12.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v12.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v12.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v12.v4s(), v3.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v12.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v12.v4s(), v5.s()[0]);
  }
  fmla(v21.v4s(), v13.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v13.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v13.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v13.v4s(), v3.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v13.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v13.v4s(), v5.s()[0]);
  }

  fmla(v20.v4s(), v14.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v14.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v14.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v14.v4s(), v3.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v14.v4s(), v4.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v14.v4s(), v5.s()[1]);
  }
  fmla(v21.v4s(), v15.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v15.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v15.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v15.v4s(), v3.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v15.v4s(), v4.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v15.v4s(), v5.s()[1]);
  }

  // Is there a remainder?- 1 float of A (4 bytes)
  bind(l7);
  tbz(x0, 2, l4);

  // Remainder- 1 float of A (4 bytes)
  // Load A
  ldr(s0, mem[x14], 4);
  if (max_mr > 1) {
    ldr(s1, mem[x15], 4);
  }
  if (max_mr > 2) {
    ldr(s2, mem[x20], 4);
  }
  if (max_mr > 3) {
    ldr(s3, mem[x21], 4);
  }
  if (max_mr > 4) {
    ldr(s4, mem[x22], 4);
  }
  if (max_mr > 5) {
    ldr(s5, mem[x23], 4);
  }
  // Load B
  ldp(q12, q13, mem[x5], 32);

  fmla(v20.v4s(), v12.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v12.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v12.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v12.v4s(), v3.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v12.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v12.v4s(), v5.s()[0]);
  }
  fmla(v21.v4s(), v13.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v13.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v13.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v13.v4s(), v3.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v13.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v13.v4s(), v5.s()[0]);
  }
  b(l4);

  // Store odd width
  bind(l8);
  tbz(x1, 2, l9);
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
  bind(l9);
  tbz(x1, 1, l10);
  if (max_mr > 5) {
    str(d30, mem[x7], 8);
    dup(d30, v30.d()[1]);
  }
  if (max_mr > 4) {
    str(d28, mem[x13], 8);
    dup(d28, v28.d()[1]);
  }
  if (max_mr > 3) {
    str(d26, mem[x10], 8);
    dup(d26, v26.d()[1]);
  }
  if (max_mr > 2) {
    str(d24, mem[x17], 8);
    dup(d24, v24.d()[1]);
  }
  if (max_mr > 1) {
    str(d22, mem[x16], 8);
    dup(d22, v22.d()[1]);
  }
  str(d20, mem[x6], 8);
  dup(d20, v20.d()[1]);

  bind(l10);
  tbz(x1, 0, l11);
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
  bind(l11);
  // Restore x20,x21,x22,x23 from stack
  ldp(x22, x23, mem[sp, 80]);
  ldp(x20, x21, mem[sp, 64]);

  // Restore d8-d15 from stack
  ldp(d14, d15, mem[sp, 48]);
  ldp(d12, d13, mem[sp, 32]);
  ldp(d10, d11, mem[sp, 16]);
  ldp(d8, d9, mem[sp], 96);
  ret();

  align(16, AlignInstruction::kHlt);
}
} // namespace
} // namespace aarch64
} // namespace xnnpack

xnn_status_t xnn_generate_f32_igemm_ukernel_upto6x8__aarch64_neonfma_cortex_a75(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
  using namespace xnnpack::aarch64;
  Generator g(code);
  assert(params != nullptr);
  const jit_gemm_params* gemm_params = static_cast<const jit_gemm_params*>(params);
  g.generate(false, max_mr, nc_mod_nr, kc, ks, gemm_params->f32_minmax.min, gemm_params->f32_minmax.max);
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;

}

xnn_status_t xnn_generate_f32_igemm_ukernel_upto6x8__aarch64_neonfma_prfm_cortex_a75(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
  using namespace xnnpack::aarch64;
  Generator g(code);
  assert(params != nullptr);
  const jit_gemm_params* gemm_params = static_cast<const jit_gemm_params*>(params);
  g.generate(true, max_mr, nc_mod_nr, kc, ks, gemm_params->f32_minmax.min, gemm_params->f32_minmax.max);
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
