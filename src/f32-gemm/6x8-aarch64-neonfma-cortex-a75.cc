// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <limits>

#include <xnnpack/aarch64-assembler.h>
#include <xnnpack/allocator.h>
#include <xnnpack/gemm.h>
#include <xnnpack/params.h>

namespace xnnpack {
namespace aarch64 {
namespace {
class Generator : public Assembler {
  using Assembler::Assembler;
 public:
  void generate(bool prefetch, size_t nc_mod_nr, size_t kc, float min, float max);
};

// void xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75(
//     size_t mr,                x0
//     size_t nc,                x1
//     size_t kc,                x2 / x0
//     const uint8_t*restrict a, x3
//     size_t a_stride,          x4
//     const void*restrict w,    x5
//     uint8_t*restrict c,       x6
//     size_t cm_stride,         x7
//     size_t cn_stride,         [sp] -> (x0)
//     const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])  [sp + 8] -> x8

// d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// A pointers
//  x3 a0
//  x9 a1
// x10 a2
// x11 a3
// x12 a4
//  x4 a5

// C pointers
//  x6 c0
// x16 c1
// x17 c2
// x14 c3
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

// Converted from: src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
void Generator::generate(bool prefetch, size_t nc_mod_nr, size_t kc, float min, float max) {
  Label l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;
  const bool clamp_min = min != -std::numeric_limits<float>::infinity();
  const bool clamp_max = max != +std::numeric_limits<float>::infinity();

  // Load params pointer
  ldr(x8, mem[sp, 8]);

  // Clamp A and C pointers / Save d8-d15 on stack
  stp(d8, d9, mem[sp, -64]++);
  cmp(x0, 2); // if mr < 2
  add(x9, x3, x4); // a1 = a0 + a_stride
  add(x16, x6, x7); // c1 = c0 + cm_stride
  csel(x9, x3, x9, kLO); //   a1 = a0
  csel(x16, x6, x16, kLO); //   c1 = c0

  stp(d10, d11, mem[sp, 16]);
  add(x10, x9, x4); // a2 = a1 + a_stride
  add(x17, x16, x7); // c2 = c1 + cm_stride
  // if mr <= 2
  csel(x10, x9, x10, kLS); //   a2 = a1
  csel(x17, x16, x17, kLS); //   c2 = c1

  stp(d12, d13, mem[sp, 32]);
  cmp(x0, 4); // if mr < 4
  add(x11, x10, x4); // a3 = a2 + a_stride
  add(x14, x17, x7); // c3 = c2 + cm_stride
  csel(x11, x10, x11, kLO); //   a3 = a2
  csel(x14, x17, x14, kLO); //   c3 = c2

  stp(d14, d15, mem[sp, 48]);
  add(x12, x11, x4); // a4 = a3 + a_stride
  add(x13, x14, x7); // c4 = c3 + cm_stride
  // if mr <= 4
  csel(x12, x11, x12, kLS); //   a4 = a3
  csel(x13, x14, x13, kLS); //   c4 = c3

  cmp(x0, 6); // if mr < 6
  add(x4, x12, x4); // a5 = a4 + a_stride
  add(x7, x13, x7); // c5 = c4 + cm_stride
  csel(x4, x12, x4, kLO); //   a5 = a4
  csel(x7, x13, x7, kLO); //   c5 = c4

  bind(l0);
  // Load initial bias from w into accumulators
  ldp(q20, q21, mem[x5], 32);
  mov(v22.v16b(), v20.v16b());
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 0]); // Prefetch B
  }
  mov(v23.v16b(), v21.v16b());
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 64]);
  }
  mov(v24.v16b(), v20.v16b());
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 128]);
  }
  mov(v25.v16b(), v21.v16b());
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 192]);
  }
  mov(v26.v16b(), v20.v16b());
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x3]); // Prefetch A
  }
  mov(v27.v16b(), v21.v16b());
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x9]);
  }
  mov(v28.v16b(), v20.v16b());
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x10]);
  }
  mov(v29.v16b(), v21.v16b());
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x11]);
  }
  mov(v30.v16b(), v20.v16b());
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x12]);
  }
  mov(v31.v16b(), v21.v16b());
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x4]);
  }

  // Is there at least 8 floats (32 bytes) for prologue + epilogue?
  subs(x0, x2, 32); // k = kc - 32
  b_lo(l4);

  // Prologue - loads for main loop of 96 FMA
  ldr(q0, mem[x3], 16);
  ldr(q1, mem[x9], 16);
  ldr(q2, mem[x10], 16);
  ldr(q3, mem[x11], 16);
  ldr(q4, mem[x12], 16);
  ldr(q5, mem[x4], 16);
  ldp(q12, q13, mem[x5], 32); // Fetch 3 B (4th deferred)
  ldp(q14, q15, mem[x5], 32);
  ldp(q16, q17, mem[x5], 32);

  // Is there at least 8 floats (32 bytes) for main loop?
  subs(x0, x0, 32);
  b_lo(l2);

  // Main loop - 8 floats of A (32 bytes)
  // 96 FMA + 6 LDP A + 8 LDP B
  bind(l1);
  // First group of 4 A.  48 FMA.
  fmla(v20.v4s(), v12.v4s(), v0.s()[0]);
  ldp(q18, q19, mem[x5], 32); // Load last B
  fmla(v22.v4s(), v12.v4s(), v1.s()[0]);
  fmla(v24.v4s(), v12.v4s(), v2.s()[0]);
  fmla(v26.v4s(), v12.v4s(), v3.s()[0]);
  fmla(v28.v4s(), v12.v4s(), v4.s()[0]);
  fmla(v30.v4s(), v12.v4s(), v5.s()[0]);
  fmla(v21.v4s(), v13.v4s(), v0.s()[0]);
  fmla(v23.v4s(), v13.v4s(), v1.s()[0]);
  fmla(v25.v4s(), v13.v4s(), v2.s()[0]);
  fmla(v27.v4s(), v13.v4s(), v3.s()[0]);
  fmla(v29.v4s(), v13.v4s(), v4.s()[0]);

  fmla(v31.v4s(), v13.v4s(), v5.s()[0]);
  fmla(v20.v4s(), v14.v4s(), v0.s()[1]);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 128]); // Prefetch B
  }
  fmla(v22.v4s(), v14.v4s(), v1.s()[1]);
  fmla(v24.v4s(), v14.v4s(), v2.s()[1]);
  fmla(v26.v4s(), v14.v4s(), v3.s()[1]);
  fmla(v28.v4s(), v14.v4s(), v4.s()[1]);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 256]);
  }
  fmla(v30.v4s(), v14.v4s(), v5.s()[1]);
  fmla(v21.v4s(), v15.v4s(), v0.s()[1]);
  fmla(v23.v4s(), v15.v4s(), v1.s()[1]);
  fmla(v25.v4s(), v15.v4s(), v2.s()[1]);
  ldr(q6, mem[x3], 16); // Load next 6 A
  fmla(v27.v4s(), v15.v4s(), v3.s()[1]);
  fmla(v29.v4s(), v15.v4s(), v4.s()[1]);
  fmla(v31.v4s(), v15.v4s(), v5.s()[1]);
  ldr(q7, mem[x9], 16);

  fmla(v20.v4s(), v16.v4s(), v0.s()[2]);
  fmla(v22.v4s(), v16.v4s(), v1.s()[2]);
  fmla(v24.v4s(), v16.v4s(), v2.s()[2]);
  ldr(q8, mem[x10], 16);
  fmla(v26.v4s(), v16.v4s(), v3.s()[2]);
  fmla(v28.v4s(), v16.v4s(), v4.s()[2]);
  fmla(v30.v4s(), v16.v4s(), v5.s()[2]);
  ldr(q9, mem[x11], 16);
  fmla(v21.v4s(), v17.v4s(), v0.s()[2]);
  fmla(v23.v4s(), v17.v4s(), v1.s()[2]);
  fmla(v25.v4s(), v17.v4s(), v2.s()[2]);
  ldr(q10, mem[x12], 16);
  fmla(v27.v4s(), v17.v4s(), v3.s()[2]);
  fmla(v29.v4s(), v17.v4s(), v4.s()[2]);
  fmla(v31.v4s(), v17.v4s(), v5.s()[2]);
  ldr(q11, mem[x4], 16);

  fmla(v20.v4s(), v18.v4s(), v0.s()[3]);
  fmla(v22.v4s(), v18.v4s(), v1.s()[3]);
  fmla(v24.v4s(), v18.v4s(), v2.s()[3]);
  ldp(q12, q13, mem[x5], 32); // Load 4 B
  fmla(v26.v4s(), v18.v4s(), v3.s()[3]);
  fmla(v28.v4s(), v18.v4s(), v4.s()[3]);
  fmla(v30.v4s(), v18.v4s(), v5.s()[3]);
  ldp(q14, q15, mem[x5], 32);
  fmla(v21.v4s(), v19.v4s(), v0.s()[3]);
  fmla(v23.v4s(), v19.v4s(), v1.s()[3]);
  fmla(v25.v4s(), v19.v4s(), v2.s()[3]);
  ldp(q16, q17, mem[x5], 32);
  fmla(v27.v4s(), v19.v4s(), v3.s()[3]);
  fmla(v29.v4s(), v19.v4s(), v4.s()[3]);
  fmla(v31.v4s(), v19.v4s(), v5.s()[3]);
  ldp(q18, q19, mem[x5], 32);

  // Second group of 4 A.  48 FMA.
  fmla(v20.v4s(), v12.v4s(), v6.s()[0]);
  fmla(v22.v4s(), v12.v4s(), v7.s()[0]);
  fmla(v24.v4s(), v12.v4s(), v8.s()[0]);
  ldr(q0, mem[x3], 16); // Load next 6 A
  fmla(v26.v4s(), v12.v4s(), v9.s()[0]);
  fmla(v28.v4s(), v12.v4s(), v10.s()[0]);
  fmla(v30.v4s(), v12.v4s(), v11.s()[0]);
  ldr(q1, mem[x9], 16);
  fmla(v21.v4s(), v13.v4s(), v6.s()[0]);
  fmla(v23.v4s(), v13.v4s(), v7.s()[0]);
  fmla(v25.v4s(), v13.v4s(), v8.s()[0]);
  ldr(q2, mem[x10], 16);
  fmla(v27.v4s(), v13.v4s(), v9.s()[0]);
  fmla(v29.v4s(), v13.v4s(), v10.s()[0]);
  fmla(v31.v4s(), v13.v4s(), v11.s()[0]);
  ldr(q3, mem[x11], 16);

  fmla(v20.v4s(), v14.v4s(), v6.s()[1]);
  fmla(v22.v4s(), v14.v4s(), v7.s()[1]);
  fmla(v24.v4s(), v14.v4s(), v8.s()[1]);
  ldr(q4, mem[x12], 16);
  fmla(v26.v4s(), v14.v4s(), v9.s()[1]);
  fmla(v28.v4s(), v14.v4s(), v10.s()[1]);
  fmla(v30.v4s(), v14.v4s(), v11.s()[1]);
  ldr(q5, mem[x4], 16);
  fmla(v21.v4s(), v15.v4s(), v6.s()[1]);
  fmla(v23.v4s(), v15.v4s(), v7.s()[1]);
  fmla(v25.v4s(), v15.v4s(), v8.s()[1]);
  ldp(q12, q13, mem[x5], 32); // Load next 3 B (not last)
  fmla(v27.v4s(), v15.v4s(), v9.s()[1]);
  fmla(v29.v4s(), v15.v4s(), v10.s()[1]);
  fmla(v31.v4s(), v15.v4s(), v11.s()[1]);
  ldp(q14, q15, mem[x5], 32);

  fmla(v20.v4s(), v16.v4s(), v6.s()[2]);
  fmla(v22.v4s(), v16.v4s(), v7.s()[2]);
  fmla(v24.v4s(), v16.v4s(), v8.s()[2]);
  fmla(v26.v4s(), v16.v4s(), v9.s()[2]);
  fmla(v28.v4s(), v16.v4s(), v10.s()[2]);
  fmla(v30.v4s(), v16.v4s(), v11.s()[2]);
  fmla(v21.v4s(), v17.v4s(), v6.s()[2]);
  fmla(v23.v4s(), v17.v4s(), v7.s()[2]);
  fmla(v25.v4s(), v17.v4s(), v8.s()[2]);
  fmla(v27.v4s(), v17.v4s(), v9.s()[2]);
  fmla(v29.v4s(), v17.v4s(), v10.s()[2]);
  fmla(v31.v4s(), v17.v4s(), v11.s()[2]);
  ldp(q16, q17, mem[x5], 32);

  fmla(v20.v4s(), v18.v4s(), v6.s()[3]);
  fmla(v22.v4s(), v18.v4s(), v7.s()[3]);
  subs(x0, x0, 32);
  fmla(v24.v4s(), v18.v4s(), v8.s()[3]);
  fmla(v26.v4s(), v18.v4s(), v9.s()[3]);
  fmla(v28.v4s(), v18.v4s(), v10.s()[3]);
  fmla(v30.v4s(), v18.v4s(), v11.s()[3]);
  fmla(v21.v4s(), v19.v4s(), v6.s()[3]);
  fmla(v23.v4s(), v19.v4s(), v7.s()[3]);
  fmla(v25.v4s(), v19.v4s(), v8.s()[3]);
  fmla(v27.v4s(), v19.v4s(), v9.s()[3]);
  fmla(v29.v4s(), v19.v4s(), v10.s()[3]);
  fmla(v31.v4s(), v19.v4s(), v11.s()[3]);
  b_hs(l1);

  // Epilogue - 8 floats of A (32 bytes)
  // 96 FMA + 6 LDP A + 8 LDP B
  // First block same as main loop.  Second block has no preloads.
  bind(l2);
  // First group of 4 A.  48 FMA.
  fmla(v20.v4s(), v12.v4s(), v0.s()[0]);
  ldp(q18, q19, mem[x5], 32); // Load last B
  fmla(v22.v4s(), v12.v4s(), v1.s()[0]);
  fmla(v24.v4s(), v12.v4s(), v2.s()[0]);
  fmla(v26.v4s(), v12.v4s(), v3.s()[0]);
  fmla(v28.v4s(), v12.v4s(), v4.s()[0]);
  fmla(v30.v4s(), v12.v4s(), v5.s()[0]);
  fmla(v21.v4s(), v13.v4s(), v0.s()[0]);
  fmla(v23.v4s(), v13.v4s(), v1.s()[0]);
  fmla(v25.v4s(), v13.v4s(), v2.s()[0]);
  fmla(v27.v4s(), v13.v4s(), v3.s()[0]);
  fmla(v29.v4s(), v13.v4s(), v4.s()[0]);

  fmla(v31.v4s(), v13.v4s(), v5.s()[0]);
  fmla(v20.v4s(), v14.v4s(), v0.s()[1]);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 128]); // Prefetch B
  }
  fmla(v22.v4s(), v14.v4s(), v1.s()[1]);
  fmla(v24.v4s(), v14.v4s(), v2.s()[1]);
  fmla(v26.v4s(), v14.v4s(), v3.s()[1]);
  fmla(v28.v4s(), v14.v4s(), v4.s()[1]);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 256]);
  }
  fmla(v30.v4s(), v14.v4s(), v5.s()[1]);
  fmla(v21.v4s(), v15.v4s(), v0.s()[1]);
  fmla(v23.v4s(), v15.v4s(), v1.s()[1]);
  fmla(v25.v4s(), v15.v4s(), v2.s()[1]);
  ldr(q6, mem[x3], 16); // Load next 6 A
  fmla(v27.v4s(), v15.v4s(), v3.s()[1]);
  fmla(v29.v4s(), v15.v4s(), v4.s()[1]);
  fmla(v31.v4s(), v15.v4s(), v5.s()[1]);
  ldr(q7, mem[x9], 16);

  fmla(v20.v4s(), v16.v4s(), v0.s()[2]);
  fmla(v22.v4s(), v16.v4s(), v1.s()[2]);
  fmla(v24.v4s(), v16.v4s(), v2.s()[2]);
  ldr(q8, mem[x10], 16);
  fmla(v26.v4s(), v16.v4s(), v3.s()[2]);
  fmla(v28.v4s(), v16.v4s(), v4.s()[2]);
  fmla(v30.v4s(), v16.v4s(), v5.s()[2]);
  ldr(q9, mem[x11], 16);
  fmla(v21.v4s(), v17.v4s(), v0.s()[2]);
  fmla(v23.v4s(), v17.v4s(), v1.s()[2]);
  fmla(v25.v4s(), v17.v4s(), v2.s()[2]);
  ldr(q10, mem[x12], 16);
  fmla(v27.v4s(), v17.v4s(), v3.s()[2]);
  fmla(v29.v4s(), v17.v4s(), v4.s()[2]);
  fmla(v31.v4s(), v17.v4s(), v5.s()[2]);
  ldr(q11, mem[x4], 16);

  fmla(v20.v4s(), v18.v4s(), v0.s()[3]);
  fmla(v22.v4s(), v18.v4s(), v1.s()[3]);
  fmla(v24.v4s(), v18.v4s(), v2.s()[3]);
  ldp(q12, q13, mem[x5], 32); // Load 4 B
  fmla(v26.v4s(), v18.v4s(), v3.s()[3]);
  fmla(v28.v4s(), v18.v4s(), v4.s()[3]);
  fmla(v30.v4s(), v18.v4s(), v5.s()[3]);
  ldp(q14, q15, mem[x5], 32);
  fmla(v21.v4s(), v19.v4s(), v0.s()[3]);
  fmla(v23.v4s(), v19.v4s(), v1.s()[3]);
  fmla(v25.v4s(), v19.v4s(), v2.s()[3]);
  ldp(q16, q17, mem[x5], 32);
  fmla(v27.v4s(), v19.v4s(), v3.s()[3]);
  fmla(v29.v4s(), v19.v4s(), v4.s()[3]);
  fmla(v31.v4s(), v19.v4s(), v5.s()[3]);
  ldp(q18, q19, mem[x5], 32);

  // Second group of 4 A.  48 FMA.
  fmla(v20.v4s(), v12.v4s(), v6.s()[0]);
  fmla(v22.v4s(), v12.v4s(), v7.s()[0]);
  fmla(v24.v4s(), v12.v4s(), v8.s()[0]);
  fmla(v26.v4s(), v12.v4s(), v9.s()[0]);
  fmla(v28.v4s(), v12.v4s(), v10.s()[0]);
  fmla(v30.v4s(), v12.v4s(), v11.s()[0]);
  fmla(v21.v4s(), v13.v4s(), v6.s()[0]);
  fmla(v23.v4s(), v13.v4s(), v7.s()[0]);
  fmla(v25.v4s(), v13.v4s(), v8.s()[0]);
  fmla(v27.v4s(), v13.v4s(), v9.s()[0]);
  fmla(v29.v4s(), v13.v4s(), v10.s()[0]);
  fmla(v31.v4s(), v13.v4s(), v11.s()[0]);

  fmla(v20.v4s(), v14.v4s(), v6.s()[1]);
  fmla(v22.v4s(), v14.v4s(), v7.s()[1]);
  fmla(v24.v4s(), v14.v4s(), v8.s()[1]);
  fmla(v26.v4s(), v14.v4s(), v9.s()[1]);
  fmla(v28.v4s(), v14.v4s(), v10.s()[1]);
  fmla(v30.v4s(), v14.v4s(), v11.s()[1]);
  fmla(v21.v4s(), v15.v4s(), v6.s()[1]);
  fmla(v23.v4s(), v15.v4s(), v7.s()[1]);
  fmla(v25.v4s(), v15.v4s(), v8.s()[1]);
  fmla(v27.v4s(), v15.v4s(), v9.s()[1]);
  fmla(v29.v4s(), v15.v4s(), v10.s()[1]);
  fmla(v31.v4s(), v15.v4s(), v11.s()[1]);

  fmla(v20.v4s(), v16.v4s(), v6.s()[2]);
  fmla(v22.v4s(), v16.v4s(), v7.s()[2]);
  fmla(v24.v4s(), v16.v4s(), v8.s()[2]);
  fmla(v26.v4s(), v16.v4s(), v9.s()[2]);
  fmla(v28.v4s(), v16.v4s(), v10.s()[2]);
  fmla(v30.v4s(), v16.v4s(), v11.s()[2]);
  fmla(v21.v4s(), v17.v4s(), v6.s()[2]);
  fmla(v23.v4s(), v17.v4s(), v7.s()[2]);
  fmla(v25.v4s(), v17.v4s(), v8.s()[2]);
  fmla(v27.v4s(), v17.v4s(), v9.s()[2]);
  fmla(v29.v4s(), v17.v4s(), v10.s()[2]);
  fmla(v31.v4s(), v17.v4s(), v11.s()[2]);

  fmla(v20.v4s(), v18.v4s(), v6.s()[3]);
  fmla(v22.v4s(), v18.v4s(), v7.s()[3]);
  fmla(v24.v4s(), v18.v4s(), v8.s()[3]);
  fmla(v26.v4s(), v18.v4s(), v9.s()[3]);
  fmla(v28.v4s(), v18.v4s(), v10.s()[3]);
  fmla(v30.v4s(), v18.v4s(), v11.s()[3]);
  fmla(v21.v4s(), v19.v4s(), v6.s()[3]);
  fmla(v23.v4s(), v19.v4s(), v7.s()[3]);

  // Load min/max values
  if (clamp_min || clamp_max) {
    ld2r({v6.v4s(), v7.v4s()}, mem[x8]);
  }

  fmla(v25.v4s(), v19.v4s(), v8.s()[3]);
  fmla(v27.v4s(), v19.v4s(), v9.s()[3]);
  // Is there a remainder?- 4 floats of A (16 bytes) or less
  tst(x0, 31);
  fmla(v29.v4s(), v19.v4s(), v10.s()[3]);
  fmla(v31.v4s(), v19.v4s(), v11.s()[3]);
  b_ne(l4);

  // Clamp
  bind(l3);
  // Load cn_stride
  ldr(x0, mem[sp, 64]);
  subs(x1, x1, 8);
  if (clamp_min) {
    fmax(v20.v4s(), v20.v4s(), v6.v4s());
    fmax(v21.v4s(), v21.v4s(), v6.v4s());
    fmax(v22.v4s(), v22.v4s(), v6.v4s());
    fmax(v23.v4s(), v23.v4s(), v6.v4s());
    fmax(v24.v4s(), v24.v4s(), v6.v4s());
    fmax(v25.v4s(), v25.v4s(), v6.v4s());
    fmax(v26.v4s(), v26.v4s(), v6.v4s());
    fmax(v27.v4s(), v27.v4s(), v6.v4s());
    fmax(v28.v4s(), v28.v4s(), v6.v4s());
    fmax(v29.v4s(), v29.v4s(), v6.v4s());
    fmax(v30.v4s(), v30.v4s(), v6.v4s());
    fmax(v31.v4s(), v31.v4s(), v6.v4s());
  }
  if (clamp_max) {
    fmin(v20.v4s(), v20.v4s(), v7.v4s());
    fmin(v21.v4s(), v21.v4s(), v7.v4s());
    fmin(v22.v4s(), v22.v4s(), v7.v4s());
    fmin(v23.v4s(), v23.v4s(), v7.v4s());
    fmin(v24.v4s(), v24.v4s(), v7.v4s());
    fmin(v25.v4s(), v25.v4s(), v7.v4s());
    fmin(v26.v4s(), v26.v4s(), v7.v4s());
    fmin(v27.v4s(), v27.v4s(), v7.v4s());
    fmin(v28.v4s(), v28.v4s(), v7.v4s());
    fmin(v29.v4s(), v29.v4s(), v7.v4s());
    fmin(v30.v4s(), v30.v4s(), v7.v4s());
    fmin(v31.v4s(), v31.v4s(), v7.v4s());
  }

  // Store full 6 x 8
  b_lo(l7);

  stp(q20, q21, mem[x6]);
  add(x6, x6, x0);
  sub(x3, x3, x2); // a0 -= kc
  stp(q22, q23, mem[x16]);
  add(x16, x16, x0);
  sub(x9, x9, x2); // a1 -= kc
  stp(q24, q25, mem[x17]);
  add(x17, x17, x0);
  sub(x10, x10, x2); // a2 -= kc
  stp(q26, q27, mem[x14]);
  add(x14, x14, x0);
  sub(x11, x11, x2); // a3 -= kc
  stp(q28, q29, mem[x13]);
  add(x13, x13, x0);
  sub(x12, x12, x2); // a4 -= kc
  stp(q30, q31, mem[x7]);
  add(x7, x7, x0);
  sub(x4, x4, x2); // a5 -= kc

  b_hi(l0);

  // Restore d8-d15 from stack
  ldp(d14, d15, mem[sp, 48]);
  ldp(d12, d13, mem[sp, 32]);
  ldp(d10, d11, mem[sp, 16]);
  ldp(d8, d9, mem[sp], 64);
  ret();

  bind(l4);
  // Load min/max values
  ld2r({v6.v4s(), v7.v4s()}, mem[x8]);

  // Is there a remainder?- 4 floats of A (16 bytes)
  tbz(x0, 4, l5);

  // Remainder- 4 floats of A (16 bytes)
  // Load A
  ldr(q0, mem[x3], 16);
  ldr(q1, mem[x9], 16);
  ldr(q2, mem[x10], 16);
  ldr(q3, mem[x11], 16);
  ldr(q4, mem[x12], 16);
  ldr(q5, mem[x4], 16);
  // Load B
  ldp(q12, q13, mem[x5], 32);
  ldp(q14, q15, mem[x5], 32);
  ldp(q16, q17, mem[x5], 32);
  ldp(q18, q19, mem[x5], 32);

  fmla(v20.v4s(), v12.v4s(), v0.s()[0]);
  fmla(v22.v4s(), v12.v4s(), v1.s()[0]);
  fmla(v24.v4s(), v12.v4s(), v2.s()[0]);
  fmla(v26.v4s(), v12.v4s(), v3.s()[0]);
  fmla(v28.v4s(), v12.v4s(), v4.s()[0]);
  fmla(v30.v4s(), v12.v4s(), v5.s()[0]);
  fmla(v21.v4s(), v13.v4s(), v0.s()[0]);
  fmla(v23.v4s(), v13.v4s(), v1.s()[0]);
  fmla(v25.v4s(), v13.v4s(), v2.s()[0]);
  fmla(v27.v4s(), v13.v4s(), v3.s()[0]);
  fmla(v29.v4s(), v13.v4s(), v4.s()[0]);
  fmla(v31.v4s(), v13.v4s(), v5.s()[0]);

  fmla(v20.v4s(), v14.v4s(), v0.s()[1]);
  fmla(v22.v4s(), v14.v4s(), v1.s()[1]);
  fmla(v24.v4s(), v14.v4s(), v2.s()[1]);
  fmla(v26.v4s(), v14.v4s(), v3.s()[1]);
  fmla(v28.v4s(), v14.v4s(), v4.s()[1]);
  fmla(v30.v4s(), v14.v4s(), v5.s()[1]);
  fmla(v21.v4s(), v15.v4s(), v0.s()[1]);
  fmla(v23.v4s(), v15.v4s(), v1.s()[1]);
  fmla(v25.v4s(), v15.v4s(), v2.s()[1]);
  fmla(v27.v4s(), v15.v4s(), v3.s()[1]);
  fmla(v29.v4s(), v15.v4s(), v4.s()[1]);
  fmla(v31.v4s(), v15.v4s(), v5.s()[1]);

  fmla(v20.v4s(), v16.v4s(), v0.s()[2]);
  fmla(v22.v4s(), v16.v4s(), v1.s()[2]);
  fmla(v24.v4s(), v16.v4s(), v2.s()[2]);
  fmla(v26.v4s(), v16.v4s(), v3.s()[2]);
  fmla(v28.v4s(), v16.v4s(), v4.s()[2]);
  fmla(v30.v4s(), v16.v4s(), v5.s()[2]);
  fmla(v21.v4s(), v17.v4s(), v0.s()[2]);
  fmla(v23.v4s(), v17.v4s(), v1.s()[2]);
  fmla(v25.v4s(), v17.v4s(), v2.s()[2]);
  fmla(v27.v4s(), v17.v4s(), v3.s()[2]);
  fmla(v29.v4s(), v17.v4s(), v4.s()[2]);
  fmla(v31.v4s(), v17.v4s(), v5.s()[2]);

  fmla(v20.v4s(), v18.v4s(), v0.s()[3]);
  fmla(v22.v4s(), v18.v4s(), v1.s()[3]);
  fmla(v24.v4s(), v18.v4s(), v2.s()[3]);
  fmla(v26.v4s(), v18.v4s(), v3.s()[3]);
  fmla(v28.v4s(), v18.v4s(), v4.s()[3]);
  fmla(v30.v4s(), v18.v4s(), v5.s()[3]);
  fmla(v21.v4s(), v19.v4s(), v0.s()[3]);
  fmla(v23.v4s(), v19.v4s(), v1.s()[3]);
  fmla(v25.v4s(), v19.v4s(), v2.s()[3]);
  fmla(v27.v4s(), v19.v4s(), v3.s()[3]);
  fmla(v29.v4s(), v19.v4s(), v4.s()[3]);
  fmla(v31.v4s(), v19.v4s(), v5.s()[3]);

  // Is there a remainder?- 2 floats of A (8 bytes)
  bind(l5);
  tbz(x0, 3, l6);

  // Remainder- 2 floats of A (8 bytes)
  // Load A
  ldr(d0, mem[x3], 8);
  ldr(d1, mem[x9], 8);
  ldr(d2, mem[x10], 8);
  ldr(d3, mem[x11], 8);
  ldr(d4, mem[x12], 8);
  ldr(d5, mem[x4], 8);
  // Load B
  ldp(q12, q13, mem[x5], 32);
  ldp(q14, q15, mem[x5], 32);

  fmla(v20.v4s(), v12.v4s(), v0.s()[0]);
  fmla(v22.v4s(), v12.v4s(), v1.s()[0]);
  fmla(v24.v4s(), v12.v4s(), v2.s()[0]);
  fmla(v26.v4s(), v12.v4s(), v3.s()[0]);
  fmla(v28.v4s(), v12.v4s(), v4.s()[0]);
  fmla(v30.v4s(), v12.v4s(), v5.s()[0]);
  fmla(v21.v4s(), v13.v4s(), v0.s()[0]);
  fmla(v23.v4s(), v13.v4s(), v1.s()[0]);
  fmla(v25.v4s(), v13.v4s(), v2.s()[0]);
  fmla(v27.v4s(), v13.v4s(), v3.s()[0]);
  fmla(v29.v4s(), v13.v4s(), v4.s()[0]);
  fmla(v31.v4s(), v13.v4s(), v5.s()[0]);

  fmla(v20.v4s(), v14.v4s(), v0.s()[1]);
  fmla(v22.v4s(), v14.v4s(), v1.s()[1]);
  fmla(v24.v4s(), v14.v4s(), v2.s()[1]);
  fmla(v26.v4s(), v14.v4s(), v3.s()[1]);
  fmla(v28.v4s(), v14.v4s(), v4.s()[1]);
  fmla(v30.v4s(), v14.v4s(), v5.s()[1]);
  fmla(v21.v4s(), v15.v4s(), v0.s()[1]);
  fmla(v23.v4s(), v15.v4s(), v1.s()[1]);
  fmla(v25.v4s(), v15.v4s(), v2.s()[1]);
  fmla(v27.v4s(), v15.v4s(), v3.s()[1]);
  fmla(v29.v4s(), v15.v4s(), v4.s()[1]);
  fmla(v31.v4s(), v15.v4s(), v5.s()[1]);

  // Is there a remainder?- 1 float of A (4 bytes)
  bind(l6);
  tbz(x0, 2, l3);

  // Remainder- 1 float of A (4 bytes)
  // Load A
  ldr(s0, mem[x3], 4);
  ldr(s1, mem[x9], 4);
  ldr(s2, mem[x10], 4);
  ldr(s3, mem[x11], 4);
  ldr(s4, mem[x12], 4);
  ldr(s5, mem[x4], 4);
  // Load B
  ldp(q12, q13, mem[x5], 32);

  fmla(v20.v4s(), v12.v4s(), v0.s()[0]);
  fmla(v22.v4s(), v12.v4s(), v1.s()[0]);
  fmla(v24.v4s(), v12.v4s(), v2.s()[0]);
  fmla(v26.v4s(), v12.v4s(), v3.s()[0]);
  fmla(v28.v4s(), v12.v4s(), v4.s()[0]);
  fmla(v30.v4s(), v12.v4s(), v5.s()[0]);
  fmla(v21.v4s(), v13.v4s(), v0.s()[0]);
  fmla(v23.v4s(), v13.v4s(), v1.s()[0]);
  fmla(v25.v4s(), v13.v4s(), v2.s()[0]);
  fmla(v27.v4s(), v13.v4s(), v3.s()[0]);
  fmla(v29.v4s(), v13.v4s(), v4.s()[0]);
  fmla(v31.v4s(), v13.v4s(), v5.s()[0]);
  b(l3);

  // Store odd width
  bind(l7);
  tbz(x1, 2, l8);
  str(q20, mem[x6], 16);
  mov(v20.v16b(), v21.v16b());
  str(q22, mem[x16], 16);
  mov(v22.v16b(), v23.v16b());
  str(q24, mem[x17], 16);
  mov(v24.v16b(), v25.v16b());
  str(q26, mem[x14], 16);
  mov(v26.v16b(), v27.v16b());
  str(q28, mem[x13], 16);
  mov(v28.v16b(), v29.v16b());
  str(q30, mem[x7], 16);
  mov(v30.v16b(), v31.v16b());
  bind(l8);
  tbz(x1, 1, l9);
  str(d20, mem[x6], 8);
  str(d22, mem[x16], 8);
  dup(d20, v20.d()[1]);
  dup(d22, v22.d()[1]);
  str(d24, mem[x17], 8);
  str(d26, mem[x14], 8);
  dup(d24, v24.d()[1]);
  dup(d26, v26.d()[1]);
  str(d28, mem[x13], 8);
  str(d30, mem[x7], 8);
  dup(d28, v28.d()[1]);
  dup(d30, v30.d()[1]);

  bind(l9);
  tbz(x1, 0, l10);
  str(s20, mem[x6]);
  str(s22, mem[x16]);
  str(s24, mem[x17]);
  str(s26, mem[x14]);
  str(s28, mem[x13]);
  str(s30, mem[x7]);
  bind(l10);
  // Restore d8-d15 from stack
  ldp(d14, d15, mem[sp, 48]);
  ldp(d12, d13, mem[sp, 32]);
  ldp(d10, d11, mem[sp, 16]);
  ldp(d8, d9, mem[sp], 64);
  ret();

  align(16, AlignInstruction::kHlt);
}
}  // namespace
}  // aarch64
}  // xnnpack

xnn_status xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75(xnn_code_buffer* code, size_t nc_mod_nr, size_t kc, const void* params) {
  using namespace xnnpack::aarch64;
  Generator g(code);
  assert(params != nullptr);
  const jit_gemm_params* gemm_params = static_cast<const jit_gemm_params*>(params);
  g.generate(false, nc_mod_nr, kc, gemm_params->f32_minmax.min, gemm_params->f32_minmax.max);
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}

xnn_status xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_prfm_cortex_a75(xnn_code_buffer* code, size_t nc_mod_nr, size_t kc, const void* params) {
  using namespace xnnpack::aarch64;
  Generator g(code);
  assert(params != nullptr);
  const jit_gemm_params* gemm_params = static_cast<const jit_gemm_params*>(params);
  g.generate(true, nc_mod_nr, kc, gemm_params->f32_minmax.min, gemm_params->f32_minmax.max);
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
