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
#include <xnnpack/gemm.h>

namespace xnnpack {
namespace aarch64 {
namespace {
class Generator : public Assembler {
  using Assembler::Assembler;
 public:
  void generate(bool prefetch, size_t max_mr, size_t nc_mod_nr, size_t kc, float min, float max);
};

// void xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_prfm_cortex_a75(
//     size_t mr,                x0
//     size_t nc,                x1
//     size_t kc,                x2 / x0
//     const uint8_t*restrict a, x3
//     size_t a_stride,          x4
//     const void*restrict w,    x5
//     uint8_t*restrict c,       x6
//     size_t cm_stride,         x7
//     size_t cn_stride,         [sp] -> x14
//     const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])  [sp + 8] -> x8

// d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.
// x13 used to store pointer to params->max if (!clamp_min && clamp_max).

// A pointers
// x3  a0
// x11 a1
// x12 a2
// x4  a3 / a_stride

// C pointers
// x6  c0
// x9  c1
// x10 c2
// x7  c3 / cm_stride

// Vector register usage
// A0  v0  v4
// A1  v1  v5
// A2  v2  v6
// A3  v3  v7
// B   v8  v9 v10 v11
// B  v12 v13 v14 v15
// B  v16 v17 v18 v19
// B  v20 v21 v22 v23
// C  v24 v25
// C  v26 v27
// C  v28 v29
// C  v30 v31
// Clamp v4 v5

// Converted from: src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
void Generator::generate(bool prefetch, size_t max_mr, size_t nc_mod_nr, size_t kc, float min, float max) {
  assert(nc_mod_nr < 8);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;
  const bool clamp_min = min != -std::numeric_limits<float>::infinity();
  const bool clamp_max = max != +std::numeric_limits<float>::infinity();

  // Load cn_stride, params pointer
  ldp(x14, x8, mem[sp]);

  // Load min/max values
  if (clamp_min && clamp_max) {
    ld2r({v4.v4s(), v5.v4s()}, mem[x8]);
  } else if (clamp_min) {
    ld1r({v4.v4s()}, mem[x8]);
  } else if (clamp_max) {
    add(x13, x8, 4);
    ld1r({v5.v4s()}, mem[x13]);
  }

  // Save d8-d15 on stack
  stp(d8, d9, mem[sp, -64]++);
  stp(d10, d11, mem[sp, 16]);
  stp(d12, d13, mem[sp, 32]);
  stp(d14, d15, mem[sp, 48]);

  // Clamp A and C pointers
  cmp(x0, 2); // if mr < 2
  add(x11, x3, x4); // a1 = a0 + a_stride
  add(x9, x6, x7); // c1 = c0 + cm_stride
  csel(x11, x3, x11, kLO); //   a1 = a0
  csel(x9, x6, x9, kLO); //   c1 = c0

  add(x12, x11, x4); // a2 = a1 + a_stride
  add(x10, x9, x7); // c2 = c1 + cm_stride
  // if mr <= 2
  csel(x12, x11, x12, kLS); //   a2 = a1
  csel(x10, x9, x10, kLS); //   c2 = c1

  cmp(x0, 4); // if mr < 4
  add(x4, x12, x4); // a3 = a2 + a_stride
  add(x7, x10, x7); // c3 = c2 + cm_stride
  csel(x4, x12, x4, kLO); //   a3 = a2
  csel(x7, x10, x7, kLO); //   c3 = c2

  bind(l0);
  // Load initial bias from w into accumulators
  ldp(q24, q25, mem[x5], 32);
  mov(v26.v16b(), v24.v16b());
  mov(v27.v16b(), v25.v16b());
  mov(v28.v16b(), v24.v16b());
  mov(v29.v16b(), v25.v16b());
  mov(v30.v16b(), v24.v16b());
  mov(v31.v16b(), v25.v16b());

  // Is there at least 8 floats (32 bytes) for prologue + epilogue?
  subs(x0, x2, 32); // k = kc - 32
  b_lo(l3);

  // 16 prologue
  // Read first block of 4 A and B.
  ldr(q0, mem[x3], 16);
  ldp(q16, q17, mem[x5], 32);
  ldr(q1, mem[x11], 16);
  ldr(q2, mem[x12], 16);
  ldr(q3, mem[x4], 16);
  ldp(q18, q19, mem[x5], 32);
  ldp(q20, q21, mem[x5], 32);
  ldp(q22, q23, mem[x5], 32);

  // Is there at least 32.  yes do main loop
  subs(x0, x0, 32);
  b_lo(l2);

  // Main loop - 8 floats of A (32 bytes)
  bind(l1);
  // First block of 4.  FMA for first 4, loads for 2nd block of 4.
  fmla(v24.v4s(), v16.v4s(), v0.s()[0]);
  ldp(q8, q9, mem[x5], 32);
  fmla(v25.v4s(), v17.v4s(), v0.s()[0]);
  fmla(v26.v4s(), v16.v4s(), v1.s()[0]);
  ldp(q10, q11, mem[x5], 32);
  fmla(v27.v4s(), v17.v4s(), v1.s()[0]);
  fmla(v28.v4s(), v16.v4s(), v2.s()[0]);
  ldp(q12, q13, mem[x5], 32);
  fmla(v29.v4s(), v17.v4s(), v2.s()[0]);
  fmla(v30.v4s(), v16.v4s(), v3.s()[0]);
  ldp(q14, q15, mem[x5], 32);
  fmla(v31.v4s(), v17.v4s(), v3.s()[0]);
  fmla(v24.v4s(), v18.v4s(), v0.s()[1]);
  ldr(q4, mem[x3], 16);
  fmla(v25.v4s(), v19.v4s(), v0.s()[1]);
  fmla(v26.v4s(), v18.v4s(), v1.s()[1]);
  ldr(q5, mem[x11], 16);
  fmla(v27.v4s(), v19.v4s(), v1.s()[1]);
  fmla(v28.v4s(), v18.v4s(), v2.s()[1]);
  ldr(q6, mem[x12], 16);
  fmla(v29.v4s(), v19.v4s(), v2.s()[1]);
  fmla(v30.v4s(), v18.v4s(), v3.s()[1]);
  ldr(q7, mem[x4], 16);
  fmla(v31.v4s(), v19.v4s(), v3.s()[1]);
  fmla(v24.v4s(), v20.v4s(), v0.s()[2]);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 128]);
  }
  fmla(v25.v4s(), v21.v4s(), v0.s()[2]);
  fmla(v26.v4s(), v20.v4s(), v1.s()[2]);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 192]);
  }
  fmla(v27.v4s(), v21.v4s(), v1.s()[2]);
  fmla(v28.v4s(), v20.v4s(), v2.s()[2]);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 256]);
  }
  fmla(v29.v4s(), v21.v4s(), v2.s()[2]);
  fmla(v30.v4s(), v20.v4s(), v3.s()[2]);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 320]);
  }
  fmla(v31.v4s(), v21.v4s(), v3.s()[2]);
  fmla(v24.v4s(), v22.v4s(), v0.s()[3]);
  fmla(v25.v4s(), v23.v4s(), v0.s()[3]);
  fmla(v26.v4s(), v22.v4s(), v1.s()[3]);
  fmla(v27.v4s(), v23.v4s(), v1.s()[3]);
  fmla(v28.v4s(), v22.v4s(), v2.s()[3]);
  fmla(v29.v4s(), v23.v4s(), v2.s()[3]);
  fmla(v30.v4s(), v22.v4s(), v3.s()[3]);
  fmla(v31.v4s(), v23.v4s(), v3.s()[3]);

  // Second block of 4.  FMA for second 4, loads for 1st block of 4.
  fmla(v24.v4s(), v8.v4s(), v4.s()[0]);
  ldp(q16, q17, mem[x5], 32);
  fmla(v25.v4s(), v9.v4s(), v4.s()[0]);
  fmla(v26.v4s(), v8.v4s(), v5.s()[0]);
  ldp(q18, q19, mem[x5], 32);
  fmla(v27.v4s(), v9.v4s(), v5.s()[0]);
  fmla(v28.v4s(), v8.v4s(), v6.s()[0]);
  ldp(q20, q21, mem[x5], 32);
  fmla(v29.v4s(), v9.v4s(), v6.s()[0]);
  fmla(v30.v4s(), v8.v4s(), v7.s()[0]);
  ldp(q22, q23, mem[x5], 32);
  fmla(v31.v4s(), v9.v4s(), v7.s()[0]);
  fmla(v24.v4s(), v10.v4s(), v4.s()[1]);
  ldr(q0, mem[x3], 16);
  fmla(v25.v4s(), v11.v4s(), v4.s()[1]);
  fmla(v26.v4s(), v10.v4s(), v5.s()[1]);
  ldr(q1, mem[x11], 16);
  fmla(v27.v4s(), v11.v4s(), v5.s()[1]);
  fmla(v28.v4s(), v10.v4s(), v6.s()[1]);
  ldr(q2, mem[x12], 16);
  fmla(v29.v4s(), v11.v4s(), v6.s()[1]);
  fmla(v30.v4s(), v10.v4s(), v7.s()[1]);
  ldr(q3, mem[x4], 16);
  fmla(v31.v4s(), v11.v4s(), v7.s()[1]);
  fmla(v24.v4s(), v12.v4s(), v4.s()[2]);
  fmla(v25.v4s(), v13.v4s(), v4.s()[2]);
  fmla(v26.v4s(), v12.v4s(), v5.s()[2]);
  fmla(v27.v4s(), v13.v4s(), v5.s()[2]);
  fmla(v28.v4s(), v12.v4s(), v6.s()[2]);
  fmla(v29.v4s(), v13.v4s(), v6.s()[2]);
  fmla(v30.v4s(), v12.v4s(), v7.s()[2]);
  fmla(v31.v4s(), v13.v4s(), v7.s()[2]);
  fmla(v24.v4s(), v14.v4s(), v4.s()[3]);
  fmla(v25.v4s(), v15.v4s(), v4.s()[3]);
  fmla(v26.v4s(), v14.v4s(), v5.s()[3]);
  fmla(v27.v4s(), v15.v4s(), v5.s()[3]);
  fmla(v28.v4s(), v14.v4s(), v6.s()[3]);
  fmla(v29.v4s(), v15.v4s(), v6.s()[3]);
  subs(x0, x0, 32);
  fmla(v30.v4s(), v14.v4s(), v7.s()[3]);
  fmla(v31.v4s(), v15.v4s(), v7.s()[3]);
  b_hs(l1);

  bind(l2);
  // Epilogue
  // First block of 4.  FMA for first 4, loads for 2nd block of 4.
  fmla(v24.v4s(), v16.v4s(), v0.s()[0]);
  ldp(q8, q9, mem[x5], 32);
  fmla(v25.v4s(), v17.v4s(), v0.s()[0]);
  fmla(v26.v4s(), v16.v4s(), v1.s()[0]);
  ldp(q10, q11, mem[x5], 32);
  fmla(v27.v4s(), v17.v4s(), v1.s()[0]);
  fmla(v28.v4s(), v16.v4s(), v2.s()[0]);
  ldp(q12, q13, mem[x5], 32);
  fmla(v29.v4s(), v17.v4s(), v2.s()[0]);
  fmla(v30.v4s(), v16.v4s(), v3.s()[0]);
  ldp(q14, q15, mem[x5], 32);
  fmla(v31.v4s(), v17.v4s(), v3.s()[0]);
  fmla(v24.v4s(), v18.v4s(), v0.s()[1]);
  ldr(q4, mem[x3], 16);
  fmla(v25.v4s(), v19.v4s(), v0.s()[1]);
  fmla(v26.v4s(), v18.v4s(), v1.s()[1]);
  ldr(q5, mem[x11], 16);
  fmla(v27.v4s(), v19.v4s(), v1.s()[1]);
  fmla(v28.v4s(), v18.v4s(), v2.s()[1]);
  ldr(q6, mem[x12], 16);
  fmla(v29.v4s(), v19.v4s(), v2.s()[1]);
  fmla(v30.v4s(), v18.v4s(), v3.s()[1]);
  ldr(q7, mem[x4], 16);
  fmla(v31.v4s(), v19.v4s(), v3.s()[1]);
  fmla(v24.v4s(), v20.v4s(), v0.s()[2]);
  fmla(v25.v4s(), v21.v4s(), v0.s()[2]);
  fmla(v26.v4s(), v20.v4s(), v1.s()[2]);
  fmla(v27.v4s(), v21.v4s(), v1.s()[2]);
  fmla(v28.v4s(), v20.v4s(), v2.s()[2]);
  fmla(v29.v4s(), v21.v4s(), v2.s()[2]);
  fmla(v30.v4s(), v20.v4s(), v3.s()[2]);
  fmla(v31.v4s(), v21.v4s(), v3.s()[2]);
  fmla(v24.v4s(), v22.v4s(), v0.s()[3]);
  fmla(v25.v4s(), v23.v4s(), v0.s()[3]);
  fmla(v26.v4s(), v22.v4s(), v1.s()[3]);
  fmla(v27.v4s(), v23.v4s(), v1.s()[3]);
  fmla(v28.v4s(), v22.v4s(), v2.s()[3]);
  fmla(v29.v4s(), v23.v4s(), v2.s()[3]);
  fmla(v30.v4s(), v22.v4s(), v3.s()[3]);
  fmla(v31.v4s(), v23.v4s(), v3.s()[3]);

  // Second block of 4.  FMA for second 4, noloads
  fmla(v24.v4s(), v8.v4s(), v4.s()[0]);
  fmla(v25.v4s(), v9.v4s(), v4.s()[0]);
  fmla(v26.v4s(), v8.v4s(), v5.s()[0]);
  fmla(v27.v4s(), v9.v4s(), v5.s()[0]);
  fmla(v28.v4s(), v8.v4s(), v6.s()[0]);
  fmla(v29.v4s(), v9.v4s(), v6.s()[0]);
  fmla(v30.v4s(), v8.v4s(), v7.s()[0]);
  fmla(v31.v4s(), v9.v4s(), v7.s()[0]);

  fmla(v24.v4s(), v10.v4s(), v4.s()[1]);
  fmla(v25.v4s(), v11.v4s(), v4.s()[1]);
  fmla(v26.v4s(), v10.v4s(), v5.s()[1]);
  fmla(v27.v4s(), v11.v4s(), v5.s()[1]);
  fmla(v28.v4s(), v10.v4s(), v6.s()[1]);
  fmla(v29.v4s(), v11.v4s(), v6.s()[1]);
  fmla(v30.v4s(), v10.v4s(), v7.s()[1]);
  fmla(v31.v4s(), v11.v4s(), v7.s()[1]);

  fmla(v24.v4s(), v12.v4s(), v4.s()[2]);
  fmla(v25.v4s(), v13.v4s(), v4.s()[2]);
  fmla(v26.v4s(), v12.v4s(), v5.s()[2]);
  fmla(v27.v4s(), v13.v4s(), v5.s()[2]);
  fmla(v28.v4s(), v12.v4s(), v6.s()[2]);
  fmla(v29.v4s(), v13.v4s(), v6.s()[2]);
  fmla(v30.v4s(), v12.v4s(), v7.s()[2]);
  fmla(v31.v4s(), v13.v4s(), v7.s()[2]);

  fmla(v24.v4s(), v14.v4s(), v4.s()[3]);
  fmla(v25.v4s(), v15.v4s(), v4.s()[3]);
  fmla(v26.v4s(), v14.v4s(), v5.s()[3]);
  fmla(v27.v4s(), v15.v4s(), v5.s()[3]);

  // Load min/max values
  if (clamp_min && clamp_max) {
    ld2r({v4.v4s(), v5.v4s()}, mem[x8]);
  } else if (clamp_min) {
    ld1r({v4.v4s()}, mem[x8]);
  } else if (clamp_max) {
    add(x13, x8, 4);
    ld1r({v5.v4s()}, mem[x13]);
  }

  fmla(v28.v4s(), v14.v4s(), v6.s()[3]);
  fmla(v29.v4s(), v15.v4s(), v6.s()[3]);
  fmla(v30.v4s(), v14.v4s(), v7.s()[3]);
  fmla(v31.v4s(), v15.v4s(), v7.s()[3]);

  bind(l3);
  // Remainder- 4 floats of A (16 bytes)
  tbz(x0, 4, l4);

  ldr(q0, mem[x3], 16);
  ldp(q16, q17, mem[x5], 32);
  ldr(q1, mem[x11], 16);
  ldr(q2, mem[x12], 16);
  ldr(q3, mem[x4], 16);
  fmla(v24.v4s(), v16.v4s(), v0.s()[0]);
  fmla(v25.v4s(), v17.v4s(), v0.s()[0]);
  ldp(q18, q19, mem[x5], 32);
  fmla(v26.v4s(), v16.v4s(), v1.s()[0]);
  fmla(v27.v4s(), v17.v4s(), v1.s()[0]);
  ldp(q20, q21, mem[x5], 32);
  fmla(v28.v4s(), v16.v4s(), v2.s()[0]);
  fmla(v29.v4s(), v17.v4s(), v2.s()[0]);
  ldp(q22, q23, mem[x5], 32);
  fmla(v30.v4s(), v16.v4s(), v3.s()[0]);
  fmla(v31.v4s(), v17.v4s(), v3.s()[0]);
  fmla(v24.v4s(), v18.v4s(), v0.s()[1]);
  fmla(v25.v4s(), v19.v4s(), v0.s()[1]);
  fmla(v26.v4s(), v18.v4s(), v1.s()[1]);
  fmla(v27.v4s(), v19.v4s(), v1.s()[1]);
  fmla(v28.v4s(), v18.v4s(), v2.s()[1]);
  fmla(v29.v4s(), v19.v4s(), v2.s()[1]);
  fmla(v30.v4s(), v18.v4s(), v3.s()[1]);
  fmla(v31.v4s(), v19.v4s(), v3.s()[1]);
  fmla(v24.v4s(), v20.v4s(), v0.s()[2]);
  fmla(v25.v4s(), v21.v4s(), v0.s()[2]);
  fmla(v26.v4s(), v20.v4s(), v1.s()[2]);
  fmla(v27.v4s(), v21.v4s(), v1.s()[2]);
  fmla(v28.v4s(), v20.v4s(), v2.s()[2]);
  fmla(v29.v4s(), v21.v4s(), v2.s()[2]);
  fmla(v30.v4s(), v20.v4s(), v3.s()[2]);
  fmla(v31.v4s(), v21.v4s(), v3.s()[2]);
  fmla(v24.v4s(), v22.v4s(), v0.s()[3]);
  fmla(v25.v4s(), v23.v4s(), v0.s()[3]);
  fmla(v26.v4s(), v22.v4s(), v1.s()[3]);
  fmla(v27.v4s(), v23.v4s(), v1.s()[3]);
  fmla(v28.v4s(), v22.v4s(), v2.s()[3]);
  fmla(v29.v4s(), v23.v4s(), v2.s()[3]);
  fmla(v30.v4s(), v22.v4s(), v3.s()[3]);
  fmla(v31.v4s(), v23.v4s(), v3.s()[3]);

  bind(l4);
  // Remainder- 2 floats of A (8 bytes)
  tbz(x0, 3, l5);

  ldr(d0, mem[x3], 8);
  ldp(q16, q17, mem[x5], 32);
  ldr(d1, mem[x11], 8);
  ldr(d2, mem[x12], 8);
  ldr(d3, mem[x4], 8);
  fmla(v24.v4s(), v16.v4s(), v0.s()[0]);
  fmla(v25.v4s(), v17.v4s(), v0.s()[0]);
  ldp(q18, q19, mem[x5], 32);
  fmla(v26.v4s(), v16.v4s(), v1.s()[0]);
  fmla(v27.v4s(), v17.v4s(), v1.s()[0]);
  fmla(v28.v4s(), v16.v4s(), v2.s()[0]);
  fmla(v29.v4s(), v17.v4s(), v2.s()[0]);
  fmla(v30.v4s(), v16.v4s(), v3.s()[0]);
  fmla(v31.v4s(), v17.v4s(), v3.s()[0]);
  fmla(v24.v4s(), v18.v4s(), v0.s()[1]);
  fmla(v25.v4s(), v19.v4s(), v0.s()[1]);
  fmla(v26.v4s(), v18.v4s(), v1.s()[1]);
  fmla(v27.v4s(), v19.v4s(), v1.s()[1]);
  fmla(v28.v4s(), v18.v4s(), v2.s()[1]);
  fmla(v29.v4s(), v19.v4s(), v2.s()[1]);
  fmla(v30.v4s(), v18.v4s(), v3.s()[1]);
  fmla(v31.v4s(), v19.v4s(), v3.s()[1]);

  bind(l5);
  // Remainder- 1 float of A (4 bytes)
  tbz(x0, 2, l6);

  ldr(s0, mem[x3], 4);
  ldp(q16, q17, mem[x5], 32);
  ldr(s1, mem[x11], 4);
  ldr(s2, mem[x12], 4);
  ldr(s3, mem[x4], 4);
  fmla(v24.v4s(), v16.v4s(), v0.s()[0]);
  fmla(v25.v4s(), v17.v4s(), v0.s()[0]);
  fmla(v26.v4s(), v16.v4s(), v1.s()[0]);
  fmla(v27.v4s(), v17.v4s(), v1.s()[0]);
  fmla(v28.v4s(), v16.v4s(), v2.s()[0]);
  fmla(v29.v4s(), v17.v4s(), v2.s()[0]);
  fmla(v30.v4s(), v16.v4s(), v3.s()[0]);
  fmla(v31.v4s(), v17.v4s(), v3.s()[0]);

  bind(l6);
  // Clamp
  subs(x1, x1, 8);
  if (clamp_min) {
    fmax(v24.v4s(), v24.v4s(), v4.v4s());
    fmax(v25.v4s(), v25.v4s(), v4.v4s());
    fmax(v26.v4s(), v26.v4s(), v4.v4s());
    fmax(v27.v4s(), v27.v4s(), v4.v4s());
    fmax(v28.v4s(), v28.v4s(), v4.v4s());
    fmax(v29.v4s(), v29.v4s(), v4.v4s());
    fmax(v30.v4s(), v30.v4s(), v4.v4s());
    fmax(v31.v4s(), v31.v4s(), v4.v4s());
  }
  if (clamp_max) {
    fmin(v24.v4s(), v24.v4s(), v5.v4s());
    fmin(v25.v4s(), v25.v4s(), v5.v4s());
    fmin(v26.v4s(), v26.v4s(), v5.v4s());
    fmin(v27.v4s(), v27.v4s(), v5.v4s());
    fmin(v28.v4s(), v28.v4s(), v5.v4s());
    fmin(v29.v4s(), v29.v4s(), v5.v4s());
    fmin(v30.v4s(), v30.v4s(), v5.v4s());
    fmin(v31.v4s(), v31.v4s(), v5.v4s());
  }

  // Store full 4 x 8
  b_lo(l7);

  stp(q24, q25, mem[x6]);
  sub(x3, x3, x2); // a0 -= kc
  add(x6, x6, x14);
  stp(q26, q27, mem[x9]);
  sub(x11, x11, x2); // a1 -= kc
  add(x9, x9, x14);
  stp(q28, q29, mem[x10]);
  sub(x12, x12, x2); // a2 -= kc
  add(x10, x10, x14);
  stp(q30, q31, mem[x7]);
  sub(x4, x4, x2); // a3 -= kc
  add(x7, x7, x14);

  b_hi(l0);

  // Restore d8-d15 from stack
  ldp(d14, d15, mem[sp, 48]);
  ldp(d12, d13, mem[sp, 32]);
  ldp(d10, d11, mem[sp, 16]);
  ldp(d8, d9, mem[sp], 64);
  ret();

  // Store odd width
  bind(l7);
  tbz(x1, 2, l8);
  str(q24, mem[x6], 16);
  mov(v24.v16b(), v25.v16b());
  str(q26, mem[x9], 16);
  mov(v26.v16b(), v27.v16b());
  str(q28, mem[x10], 16);
  mov(v28.v16b(), v29.v16b());
  str(q30, mem[x7], 16);
  mov(v30.v16b(), v31.v16b());

  bind(l8);
  tbz(x1, 1, l9);
  str(d24, mem[x6], 8);
  str(d26, mem[x9], 8);
  dup(d24, v24.d()[1]);
  dup(d26, v26.d()[1]);
  str(d28, mem[x10], 8);
  str(d30, mem[x7], 8);
  dup(d28, v28.d()[1]);
  dup(d30, v30.d()[1]);

  bind(l9);
  tbz(x1, 0, l10);
  str(s24, mem[x6]);
  str(s26, mem[x9]);
  str(s28, mem[x10]);
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

xnn_status_t xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, const void* params) {
  using namespace xnnpack::aarch64;
  Generator g(code);
  assert(params != nullptr);
  const jit_gemm_params* gemm_params = static_cast<const jit_gemm_params*>(params);
  g.generate(false, max_mr, nc_mod_nr, kc, gemm_params->f32_minmax.min, gemm_params->f32_minmax.max);
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}

xnn_status_t xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_prfm_cortex_a75(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, const void* params) {
  using namespace xnnpack::aarch64;
  Generator g(code);
  assert(params != nullptr);
  const jit_gemm_params* gemm_params = static_cast<const jit_gemm_params*>(params);
  g.generate(true, max_mr, nc_mod_nr, kc, gemm_params->f32_minmax.min, gemm_params->f32_minmax.max);
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
