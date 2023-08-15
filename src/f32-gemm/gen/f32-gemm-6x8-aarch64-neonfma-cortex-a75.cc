// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <limits>

#include <xnnpack.h>
#include <xnnpack/aarch64-assembler.h>
#include <xnnpack/gemm.h>
#include <xnnpack/log.h>
#include <xnnpack/memory.h>
#include <xnnpack/microparams.h>
#include <xnnpack/post-operation.h>

namespace xnnpack {
namespace aarch64 {
namespace {
class Generator : public MacroAssembler {
  using MacroAssembler::MacroAssembler;

 public:
  void generate(bool prefetch, size_t max_mr, size_t nc_mod_nr, size_t kc, const jit_gemm_params* jit_gemm_params);
  void perform_post_operations(size_t max_mr, size_t num_post_operations, const xnn_post_operation* post_operations);
};

// void xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm(
//     size_t mr,                         x0
//     size_t nc,                         x1
//     size_t kc,                         x2 / x0
//     const uint8_t* a,                  x3
//     size_t a_stride,                   x4
//     const void* w,                     x5
//     uint8_t* c,                        x6
//     size_t cm_stride,                  x7
//     size_t cn_stride,                  [sp] -> (x0)
//     const xnn_f32_minmax_params params [sp + 8] -> x8

// d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Vector register usage
// A0   x3 v0  v6
// A1   x9 v1  v7
// A2  x10 v2  v8
// A3  x11 v3  v9
// A4  x12 v4 v10
// A5   x4 v5 v11
// B    x5 v12 v13 v14 v15
// B       v16 v17 v18 v19
// C    x6 v20 v21
// C   x16 v22 v23
// C   x17 v24 v25
// C   x14 v26 v27
// C   x13 v28 v29
// C    x7 v30 v31
// Clamp v6 v7

// Converted from: src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S
void Generator::generate(bool prefetch, size_t max_mr, size_t nc_mod_nr, size_t kc, const jit_gemm_params* jit_gemm_params)
{
  assert(max_mr <= 6);
  assert(nc_mod_nr < 8 || nc_mod_nr == SIZE_MAX);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;
  const size_t num_post_operations = jit_gemm_params->num_post_operations;
  const xnn_post_operation* post_operations = jit_gemm_params->post_operations;
  const float min = jit_gemm_params->f32_minmax.min;
  const float max = jit_gemm_params->f32_minmax.max;
  const bool clamp_min = min != -std::numeric_limits<float>::infinity();
  const bool clamp_max = max != +std::numeric_limits<float>::infinity();
  assert(num_post_operations == 0 || (!clamp_min && !clamp_max));

  // Clamp A and C pointers / Save d8-d15 on stack
  if (max_mr > 1) {
    cmp(x0, 2); // if mr < 2
  }
  stp(d8, d9, mem[sp, -64]++);
  if (max_mr > 1) {
    add(x9, x3, x4); // a1 = a0 + a_stride
    add(x16, x6, x7); // c1 = c0 + cm_stride
    csel(x9, x3, x9, kLO); //   a1 = a0
    csel(x16, x6, x16, kLO); //   c1 = c0
  }

  stp(d10, d11, mem[sp, 16]);
  if (max_mr > 2) {
    add(x10, x9, x4); // a2 = a1 + a_stride
    add(x17, x16, x7); // c2 = c1 + cm_stride
    // if mr <= 2
    csel(x10, x9, x10, kLS); //   a2 = a1
    csel(x17, x16, x17, kLS); //   c2 = c1
  }

  stp(d12, d13, mem[sp, 32]);
  if (max_mr > 3) {
    cmp(x0, 4); // if mr < 4
    add(x11, x10, x4); // a3 = a2 + a_stride
    add(x14, x17, x7); // c3 = c2 + cm_stride
    csel(x11, x10, x11, kLO); //   a3 = a2
    csel(x14, x17, x14, kLO); //   c3 = c2
  }

  stp(d14, d15, mem[sp, 48]);
  if (max_mr > 4) {
    add(x12, x11, x4); // a4 = a3 + a_stride
    add(x13, x14, x7); // c4 = c3 + cm_stride
    // if mr <= 4
    csel(x12, x11, x12, kLS); //   a4 = a3
    csel(x13, x14, x13, kLS); //   c4 = c3
  }

  // Load params pointer
  ldr(x8, mem[sp, 72]);

  if (max_mr > 5) {
    cmp(x0, 6); // if mr < 6
    add(x4, x12, x4); // a5 = a4 + a_stride
    add(x7, x13, x7); // c5 = c4 + cm_stride
    csel(x4, x12, x4, kLO); //   a5 = a4
    csel(x7, x13, x7, kLO); //   c5 = c4
  }

  bind(l0);
  // Load initial bias from w into accumulators
  ldp(q20, q21, mem[x5], 32);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 0]); // Prefetch B
  }
  subs(x0, x2, 32); // k = kc - 32
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 64]);
  }
  if (max_mr > 1) {
    mov(v22.v16b(), v20.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 128]);
  }
  if (max_mr > 1) {
    mov(v23.v16b(), v21.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 192]);
  }
  if (max_mr > 2) {
    mov(v24.v16b(), v20.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 256]);
  }
  if (max_mr > 2) {
    mov(v25.v16b(), v21.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 320]);
  }
  if (max_mr > 3) {
    mov(v26.v16b(), v20.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x3]); // Prefetch A
  }
  if (max_mr > 3) {
    mov(v27.v16b(), v21.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x9]);
  }
  if (max_mr > 4) {
    mov(v28.v16b(), v20.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x10]);
  }
  if (max_mr > 4) {
    mov(v29.v16b(), v21.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x11]);
  }
  if (max_mr > 5) {
    mov(v30.v16b(), v20.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x12]);
  }
  if (max_mr > 5) {
    mov(v31.v16b(), v21.v16b());
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x4]);
  }
  b_lo(l4);

  // Prologue - loads for main loop of 96 FMA
  ldr(q0, mem[x3], 16);
  ldp(q12, q13, mem[x5], 32); // Fetch 3 B (4th deferred)
  if (max_mr > 1) {
    ldr(q1, mem[x9], 16);
  }
  if (max_mr > 2) {
    ldr(q2, mem[x10], 16);
  }
  if (max_mr > 3) {
    ldr(q3, mem[x11], 16);
  }
  if (max_mr > 4) {
    ldr(q4, mem[x12], 16);
  }
  if (max_mr > 5) {
    ldr(q5, mem[x4], 16);
  }
  ldp(q14, q15, mem[x5], 32);
  ldp(q16, q17, mem[x5], 32);

  // Is there at least 8 floats (32 bytes) for main loop?
  subs(x0, x0, 32);
  b_lo(l2);

  // Main loop - 8 floats of A (32 bytes)
  // 96 FMA + 6 LDP A + 8 LDP B
  // 64 float weights = 256 bytes.  4 cache lines.
  bind(l1);
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
  ldr(q6, mem[x3], 16); // Load next 6 A
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
    ldr(q7, mem[x9], 16);
  }

  fmla(v20.v4s(), v16.v4s(), v0.s()[2]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v16.v4s(), v1.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v2.s()[2]);
    ldr(q8, mem[x10], 16);
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
    ldr(q9, mem[x11], 16);
  }
  fmla(v21.v4s(), v17.v4s(), v0.s()[2]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v17.v4s(), v1.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v2.s()[2]);
  }
  if (max_mr > 4) {
    ldr(q10, mem[x12], 16);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v17.v4s(), v3.s()[2]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v17.v4s(), v4.s()[2]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v17.v4s(), v5.s()[2]);
    ldr(q11, mem[x4], 16);
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
  ldr(q0, mem[x3], 16); // Load next 6 A
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
    ldr(q1, mem[x9], 16);
  }
  fmla(v21.v4s(), v13.v4s(), v6.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v13.v4s(), v7.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v13.v4s(), v8.s()[0]);
    ldr(q2, mem[x10], 16);
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
    ldr(q3, mem[x11], 16);
  }

  fmla(v20.v4s(), v14.v4s(), v6.s()[1]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v14.v4s(), v7.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v14.v4s(), v8.s()[1]);
  }
  if (max_mr > 4) {
    ldr(q4, mem[x12], 16);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v14.v4s(), v9.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v14.v4s(), v10.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v14.v4s(), v11.s()[1]);
    ldr(q5, mem[x4], 16);
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

  fmla(v20.v4s(), v18.v4s(), v6.s()[3]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v18.v4s(), v7.s()[3]);
  }
  ldp(q16, q17, mem[x5], 32);
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
  subs(x0, x0, 32);
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
  b_hs(l1);

  // Epilogue - 8 floats of A (32 bytes)
  // 96 FMA + 6 LDP A + 8 LDP B
  // First block same as main loop.  Second block has no preloads.
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
  ldr(q6, mem[x3], 16); // Load next 6 A
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
    ldr(q7, mem[x9], 16);
  }

  fmla(v20.v4s(), v16.v4s(), v0.s()[2]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v16.v4s(), v1.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v2.s()[2]);
    ldr(q8, mem[x10], 16);
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
    ldr(q9, mem[x11], 16);
  }
  fmla(v21.v4s(), v17.v4s(), v0.s()[2]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v17.v4s(), v1.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v2.s()[2]);
  }
  if (max_mr > 4) {
    ldr(q10, mem[x12], 16);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v17.v4s(), v3.s()[2]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v17.v4s(), v4.s()[2]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v17.v4s(), v5.s()[2]);
    ldr(q11, mem[x4], 16);
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
  if (clamp_min || clamp_max) {
    ld2r({v6.v4s(), v7.v4s()}, mem[x8]); // Load min/max values
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
  b_ne(l4);

  // Clamp
  bind(l3);
  if (clamp_min) {
    fmax(v20.v4s(), v20.v4s(), v6.v4s());
    fmax(v21.v4s(), v21.v4s(), v6.v4s());
    if (max_mr > 1) {
      fmax(v22.v4s(), v22.v4s(), v6.v4s());
      fmax(v23.v4s(), v23.v4s(), v6.v4s());
    }
    if (max_mr > 2) {
      fmax(v24.v4s(), v24.v4s(), v6.v4s());
    }
  }
  ldr(x0, mem[sp, 64]); // Load cn_stride
  if (clamp_min) {
    if (max_mr > 2) {
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
  perform_post_operations(max_mr, num_post_operations, post_operations);

  // Store full 6 x 8
  b_lo(l7);

  stp(q20, q21, mem[x6]);
  add(x6, x6, x0);
  sub(x3, x3, x2); // a0 -= kc
  if (max_mr > 1) {
    stp(q22, q23, mem[x16]);
    add(x16, x16, x0);
    sub(x9, x9, x2); // a1 -= kc
  }
  if (max_mr > 2) {
    stp(q24, q25, mem[x17]);
    add(x17, x17, x0);
    sub(x10, x10, x2); // a2 -= kc
  }
  if (max_mr > 3) {
    stp(q26, q27, mem[x14]);
    add(x14, x14, x0);
    sub(x11, x11, x2); // a3 -= kc
  }
  if (max_mr > 4) {
    stp(q28, q29, mem[x13]);
    add(x13, x13, x0);
    sub(x12, x12, x2); // a4 -= kc
  }
  if (max_mr > 5) {
    stp(q30, q31, mem[x7]);
    add(x7, x7, x0);
    sub(x4, x4, x2); // a5 -= kc
  }

  b_hi(l0);

  // Restore d8-d15 from stack
  ldp(d14, d15, mem[sp, 48]);
  ldp(d12, d13, mem[sp, 32]);
  ldp(d10, d11, mem[sp, 16]);
  ldp(d8, d9, mem[sp], 64);
  ret();

  bind(l4);
  // Load min/max values
  if (clamp_min || clamp_max) {
    ld2r({v6.v4s(), v7.v4s()}, mem[x8]);
  }

  // Is there a remainder?- 4 floats of A (16 bytes)
  tbz(x0, 4, l5);

  // Remainder- 4 floats of A (16 bytes)
  // Load A
  ldr(q0, mem[x3], 16);
  if (max_mr > 1) {
    ldr(q1, mem[x9], 16);
  }
  if (max_mr > 2) {
    ldr(q2, mem[x10], 16);
  }
  if (max_mr > 3) {
    ldr(q3, mem[x11], 16);
  }
  if (max_mr > 4) {
    ldr(q4, mem[x12], 16);
  }
  if (max_mr > 5) {
    ldr(q5, mem[x4], 16);
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
  bind(l5);
  tbz(x0, 3, l6);

  // Remainder- 2 floats of A (8 bytes)
  // Load A
  ldr(d0, mem[x3], 8);
  if (max_mr > 1) {
    ldr(d1, mem[x9], 8);
  }
  if (max_mr > 2) {
    ldr(d2, mem[x10], 8);
  }
  if (max_mr > 3) {
    ldr(d3, mem[x11], 8);
  }
  if (max_mr > 4) {
    ldr(d4, mem[x12], 8);
  }
  if (max_mr > 5) {
    ldr(d5, mem[x4], 8);
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
  bind(l6);
  tbz(x0, 2, l3);

  // Remainder- 1 float of A (4 bytes)
  // Load A
  ldr(s0, mem[x3], 4);
  if (max_mr > 1) {
    ldr(s1, mem[x9], 4);
  }
  if (max_mr > 2) {
    ldr(s2, mem[x10], 4);
  }
  if (max_mr > 3) {
    ldr(s3, mem[x11], 4);
  }
  if (max_mr > 4) {
    ldr(s4, mem[x12], 4);
  }
  if (max_mr > 5) {
    ldr(s5, mem[x4], 4);
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
  b(l3);

  // Store odd width
  bind(l7);
  tbz(x1, 2, l8);
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
  bind(l8);
  tbz(x1, 1, l9);
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

  bind(l9);
  tbz(x1, 0, l10);
  str(s20, mem[x6]);
  if (max_mr > 1) {
    str(s22, mem[x16]);
  }
  if (max_mr > 2) {
    str(s24, mem[x17]);
  }
  if (max_mr > 3) {
    str(s26, mem[x14]);
  }
  if (max_mr > 4) {
    str(s28, mem[x13]);
  }
  if (max_mr > 5) {
    str(s30, mem[x7]);
  }
  bind(l10);
  // Restore d8-d15 from stack
  ldp(d14, d15, mem[sp, 48]);
  ldp(d12, d13, mem[sp, 32]);
  ldp(d10, d11, mem[sp, 16]);
  ldp(d8, d9, mem[sp], 64);
  ret();

  align(16, AlignInstruction::kHlt);
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
        // Reuse A pointers (don't use v8-v15 as they are callee saved).
        const auto sixth = v0.v4s();
        const auto three = v1.v4s();
        const auto six = v2.v4s();
        const auto zero = v3.v4s();
        // v4, v5, v6, v7 available for temporaries.
        ld3r({sixth, three, six}, mem[x8]++);
        movi(zero, 0);
        const VRegister accs[] = {
          v20.v4s(), v21.v4s(), v22.v4s(), v23.v4s(),
          v24.v4s(), v25.v4s(), v26.v4s(), v27.v4s(),
          v28.v4s(), v29.v4s(), v30.v4s(), v31.v4s(),
        };
        const VRegister tmps[] = {v4.v4s(), v5.v4s(), v6.v4s(), v7.v4s()};
        f32_hardswish(sixth, three, six, zero, &accs[0], XNN_COUNT_OF(accs), &tmps[0], XNN_COUNT_OF(tmps));
        break;
      }
      default:
        XNN_LOG_UNREACHABLE("unsupported post operation: %u", post_operations[i].op_type);
    }
  }
}

}  // namespace
}  // namespace aarch64
}  // namespace xnnpack

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, const void* params) {
  using namespace xnnpack::aarch64;
  Generator g(code);
  assert(params != nullptr);
  g.generate(false, max_mr, nc_mod_nr, kc, static_cast<const jit_gemm_params*>(params));
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, const void* params) {
  using namespace xnnpack::aarch64;
  Generator g(code);
  assert(params != nullptr);
  g.generate(true, max_mr, nc_mod_nr, kc, static_cast<const jit_gemm_params*>(params));
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
