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
  void perform_post_operations(size_t max_mr, size_t num_post_operations, const xnn_post_operation* post_operations);
};

// void xnn_f32_igemm_minmax_ukernel_4x8__asm_aarch64_neonfma_prfm_cortex_a75(
//     size_t mr,                         x0
//     size_t nc,                         x1
//     size_t kc,                         x2 / x0
//     size_t ks,                         x3 / x9
//     const float**restrict a,           x4
//     const float*restrict w,            x5
//     float*restrict c,                  x6
//     size_t cm_stride,                  x7
//     size_t cn_stride,                  [sp] -> x10
//     size_t a_offset,                   [sp + 8] -> x11
//     const float* zero,                 [sp + 16] -> x12
//     const xnn_f32_minmax_params params [sp + 24] -> x8

// d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0 x20  v0  v4
// A1 x13  v1  v5
// A2 x14  v2  v6
// A3 x15  v3  v7
// B   x5  v8  v9 v10 v11
// B       v12 v13 v14 v15
// B       v16 v17 v18 v19
// B       v20 v21 v22 v23
// C0  x6  v24 v25
// C1 x16  v26 v27
// C2 x17  v28 v29
// C3  x7  v30 v31
// Clamp v4 v5

// Converted from: src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-prfm-cortex-a75.S
void Generator::generate(bool prefetch, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const jit_gemm_params* jit_gemm_params)
{
  assert(max_mr <= 4);
  assert(nc_mod_nr < 8);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11;
  const size_t num_post_operations = jit_gemm_params->num_post_operations;
  const xnn_post_operation* post_operations = jit_gemm_params->post_operations;
  const float min = jit_gemm_params->f32_minmax.min;
  const float max = jit_gemm_params->f32_minmax.max;
  const bool clamp_min = min != -std::numeric_limits<float>::infinity();
  const bool clamp_max = max != +std::numeric_limits<float>::infinity();
  assert(num_post_operations == 0 || (!clamp_min && !clamp_max));

  // Load cn_stride, a_offset
  ldp(x10, x11, mem[sp]);

  // Load zero, params pointer
  ldp(x12, x8, mem[sp, 16]);

  // Load min/max values
  if (clamp_min || clamp_max) {
    ld2r({v4.v4s(), v5.v4s()}, mem[x8]);
  }

  // Save x20 on stack
  str(x20, mem[sp, -80]++);

  // Save d8-d15 on stack
  stp(d8, d9, mem[sp, 16]);
  stp(d10, d11, mem[sp, 32]);
  stp(d12, d13, mem[sp, 48]);
  stp(d14, d15, mem[sp, 64]);

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

  bind(l0);
  // Load initial bias from w into accumulators
  ldp(q24, q25, mem[x5], 32);
  if (max_mr > 1) {
    mov(v26.v16b(), v24.v16b());
    mov(v27.v16b(), v25.v16b());
  }
  if (max_mr > 2) {
    mov(v28.v16b(), v24.v16b());
    mov(v29.v16b(), v25.v16b());
  }
  if (max_mr > 3) {
    mov(v30.v16b(), v24.v16b());
    mov(v31.v16b(), v25.v16b());
  }

  mov(x9, x3); // p = ks

  bind(l1);
  // Load next 4 A pointers
  ldr(x20, mem[x4], 8);
  if (max_mr > 1) {
    ldr(x13, mem[x4], 8);
  }
  if (max_mr > 2) {
    ldr(x14, mem[x4], 8);
  }
  if (max_mr > 3) {
    ldr(x15, mem[x4], 8);
  }

  cmp(x20, x12); // if a0 == zero
  add(x20, x20, x11); // a0 += a_offset
  csel(x20, x12, x20, kEQ); //   a0 = zero, else += a0 + a_offset
  if (max_mr > 1) {
    cmp(x13, x12); // if a1 == zero
    add(x13, x13, x11); // a1 += a_offset
    csel(x13, x12, x13, kEQ); //   a1 = zero, else += a1 + a_offset
  }
  if (max_mr > 2) {
    cmp(x14, x12); // if a2 == zero
    add(x14, x14, x11); // a2 += a_offset
    csel(x14, x12, x14, kEQ); //   a2 = zero, else += a2 + a_offset
  }
  if (max_mr > 3) {
    cmp(x15, x12); // if a3 == zero
    add(x15, x15, x11); // a3 += a_offset
    csel(x15, x12, x15, kEQ); //   a3 = zero, else += a3 + a_offset
  }

  // Is there at least 8 floats (32 bytes) for prologue + epilogue?
  subs(x0, x2, 32); // k = kc - 32
  b_lo(l4);

  // 16 prologue
  // Read first block of 4 A and B.
  ldr(q0, mem[x20], 16);
  ldp(q16, q17, mem[x5], 32);
  if (max_mr > 1) {
    ldr(q1, mem[x13], 16);
  }
  if (max_mr > 2) {
    ldr(q2, mem[x14], 16);
  }
  if (max_mr > 3) {
    ldr(q3, mem[x15], 16);
  }
  ldp(q18, q19, mem[x5], 32);
  ldp(q20, q21, mem[x5], 32);
  ldp(q22, q23, mem[x5], 32);

  // Is there at least 32.  yes do main loop
  subs(x0, x0, 32);
  b_lo(l3);

  // Main loop - 8 floats of A
  bind(l2);
  // First block of 4.  FMA for first 4, loads for 2nd block of 4.
  fmla(v24.v4s(), v16.v4s(), v0.s()[0]);
  ldp(q8, q9, mem[x5], 32);
  fmla(v25.v4s(), v17.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v16.v4s(), v1.s()[0]);
  }
  ldp(q10, q11, mem[x5], 32);
  if (max_mr > 1) {
    fmla(v27.v4s(), v17.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v16.v4s(), v2.s()[0]);
  }
  ldp(q12, q13, mem[x5], 32);
  if (max_mr > 2) {
    fmla(v29.v4s(), v17.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v16.v4s(), v3.s()[0]);
  }
  ldp(q14, q15, mem[x5], 32);
  if (max_mr > 3) {
    fmla(v31.v4s(), v17.v4s(), v3.s()[0]);
  }
  fmla(v24.v4s(), v18.v4s(), v0.s()[1]);
  ldr(q4, mem[x20], 16);
  fmla(v25.v4s(), v19.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v18.v4s(), v1.s()[1]);
    ldr(q5, mem[x13], 16);
    fmla(v27.v4s(), v19.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v18.v4s(), v2.s()[1]);
    ldr(q6, mem[x14], 16);
    fmla(v29.v4s(), v19.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v18.v4s(), v3.s()[1]);
    ldr(q7, mem[x15], 16);
    fmla(v31.v4s(), v19.v4s(), v3.s()[1]);
  }
  fmla(v24.v4s(), v20.v4s(), v0.s()[2]);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 128]);
  }
  fmla(v25.v4s(), v21.v4s(), v0.s()[2]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v20.v4s(), v1.s()[2]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 192]);
  }
  if (max_mr > 1) {
    fmla(v27.v4s(), v21.v4s(), v1.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v20.v4s(), v2.s()[2]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 256]);
  }
  if (max_mr > 2) {
    fmla(v29.v4s(), v21.v4s(), v2.s()[2]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v20.v4s(), v3.s()[2]);
  }
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 320]);
  }
  if (max_mr > 3) {
    fmla(v31.v4s(), v21.v4s(), v3.s()[2]);
  }
  fmla(v24.v4s(), v22.v4s(), v0.s()[3]);
  fmla(v25.v4s(), v23.v4s(), v0.s()[3]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v22.v4s(), v1.s()[3]);
    fmla(v27.v4s(), v23.v4s(), v1.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v22.v4s(), v2.s()[3]);
    fmla(v29.v4s(), v23.v4s(), v2.s()[3]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v22.v4s(), v3.s()[3]);
    fmla(v31.v4s(), v23.v4s(), v3.s()[3]);
  }

  // Second block of 4.  FMA for second 4, loads for 1st block of 4.
  fmla(v24.v4s(), v8.v4s(), v4.s()[0]);
  ldp(q16, q17, mem[x5], 32);
  fmla(v25.v4s(), v9.v4s(), v4.s()[0]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v8.v4s(), v5.s()[0]);
  }
  ldp(q18, q19, mem[x5], 32);
  if (max_mr > 1) {
    fmla(v27.v4s(), v9.v4s(), v5.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v8.v4s(), v6.s()[0]);
  }
  ldp(q20, q21, mem[x5], 32);
  if (max_mr > 2) {
    fmla(v29.v4s(), v9.v4s(), v6.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v8.v4s(), v7.s()[0]);
  }
  ldp(q22, q23, mem[x5], 32);
  if (max_mr > 3) {
    fmla(v31.v4s(), v9.v4s(), v7.s()[0]);
  }
  fmla(v24.v4s(), v10.v4s(), v4.s()[1]);
  ldr(q0, mem[x20], 16);
  fmla(v25.v4s(), v11.v4s(), v4.s()[1]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v10.v4s(), v5.s()[1]);
    ldr(q1, mem[x13], 16);
    fmla(v27.v4s(), v11.v4s(), v5.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v10.v4s(), v6.s()[1]);
    ldr(q2, mem[x14], 16);
    fmla(v29.v4s(), v11.v4s(), v6.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v10.v4s(), v7.s()[1]);
    ldr(q3, mem[x15], 16);
    fmla(v31.v4s(), v11.v4s(), v7.s()[1]);
  }
  fmla(v24.v4s(), v12.v4s(), v4.s()[2]);
  fmla(v25.v4s(), v13.v4s(), v4.s()[2]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v12.v4s(), v5.s()[2]);
    fmla(v27.v4s(), v13.v4s(), v5.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v12.v4s(), v6.s()[2]);
    fmla(v29.v4s(), v13.v4s(), v6.s()[2]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v12.v4s(), v7.s()[2]);
    fmla(v31.v4s(), v13.v4s(), v7.s()[2]);
  }
  fmla(v24.v4s(), v14.v4s(), v4.s()[3]);
  fmla(v25.v4s(), v15.v4s(), v4.s()[3]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v14.v4s(), v5.s()[3]);
    fmla(v27.v4s(), v15.v4s(), v5.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v14.v4s(), v6.s()[3]);
    fmla(v29.v4s(), v15.v4s(), v6.s()[3]);
  }
  subs(x0, x0, 32);
  if (max_mr > 3) {
    fmla(v30.v4s(), v14.v4s(), v7.s()[3]);
    fmla(v31.v4s(), v15.v4s(), v7.s()[3]);
  }

  b_hs(l2);

  bind(l3);
  // Epilogue
  // First block of 4.  FMA for first 4, loads for 2nd block of 4.
  fmla(v24.v4s(), v16.v4s(), v0.s()[0]);
  ldp(q8, q9, mem[x5], 32);
  fmla(v25.v4s(), v17.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v16.v4s(), v1.s()[0]);
  }
  ldp(q10, q11, mem[x5], 32);
  if (max_mr > 1) {
    fmla(v27.v4s(), v17.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v16.v4s(), v2.s()[0]);
  }
  ldp(q12, q13, mem[x5], 32);
  if (max_mr > 2) {
    fmla(v29.v4s(), v17.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v16.v4s(), v3.s()[0]);
  }
  ldp(q14, q15, mem[x5], 32);
  if (max_mr > 3) {
    fmla(v31.v4s(), v17.v4s(), v3.s()[0]);
  }
  fmla(v24.v4s(), v18.v4s(), v0.s()[1]);
  ldr(q4, mem[x20], 16);
  fmla(v25.v4s(), v19.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v18.v4s(), v1.s()[1]);
    ldr(q5, mem[x13], 16);
    fmla(v27.v4s(), v19.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v18.v4s(), v2.s()[1]);
    ldr(q6, mem[x14], 16);
    fmla(v29.v4s(), v19.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v18.v4s(), v3.s()[1]);
    ldr(q7, mem[x15], 16);
    fmla(v31.v4s(), v19.v4s(), v3.s()[1]);
  }
  fmla(v24.v4s(), v20.v4s(), v0.s()[2]);
  fmla(v25.v4s(), v21.v4s(), v0.s()[2]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v20.v4s(), v1.s()[2]);
    fmla(v27.v4s(), v21.v4s(), v1.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v20.v4s(), v2.s()[2]);
    fmla(v29.v4s(), v21.v4s(), v2.s()[2]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v20.v4s(), v3.s()[2]);
    fmla(v31.v4s(), v21.v4s(), v3.s()[2]);
  }
  fmla(v24.v4s(), v22.v4s(), v0.s()[3]);
  fmla(v25.v4s(), v23.v4s(), v0.s()[3]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v22.v4s(), v1.s()[3]);
    fmla(v27.v4s(), v23.v4s(), v1.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v22.v4s(), v2.s()[3]);
    fmla(v29.v4s(), v23.v4s(), v2.s()[3]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v22.v4s(), v3.s()[3]);
    fmla(v31.v4s(), v23.v4s(), v3.s()[3]);
  }

  // Second block of 4.  FMA for second 4, noloads
  fmla(v24.v4s(), v8.v4s(), v4.s()[0]);
  fmla(v25.v4s(), v9.v4s(), v4.s()[0]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v8.v4s(), v5.s()[0]);
    fmla(v27.v4s(), v9.v4s(), v5.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v8.v4s(), v6.s()[0]);
    fmla(v29.v4s(), v9.v4s(), v6.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v8.v4s(), v7.s()[0]);
    fmla(v31.v4s(), v9.v4s(), v7.s()[0]);
  }
  fmla(v24.v4s(), v10.v4s(), v4.s()[1]);
  fmla(v25.v4s(), v11.v4s(), v4.s()[1]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v10.v4s(), v5.s()[1]);
    fmla(v27.v4s(), v11.v4s(), v5.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v10.v4s(), v6.s()[1]);
    fmla(v29.v4s(), v11.v4s(), v6.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v10.v4s(), v7.s()[1]);
    fmla(v31.v4s(), v11.v4s(), v7.s()[1]);
  }
  fmla(v24.v4s(), v12.v4s(), v4.s()[2]);
  fmla(v25.v4s(), v13.v4s(), v4.s()[2]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v12.v4s(), v5.s()[2]);
    fmla(v27.v4s(), v13.v4s(), v5.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v12.v4s(), v6.s()[2]);
    fmla(v29.v4s(), v13.v4s(), v6.s()[2]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v12.v4s(), v7.s()[2]);
    fmla(v31.v4s(), v13.v4s(), v7.s()[2]);
  }

  fmla(v24.v4s(), v14.v4s(), v4.s()[3]);
  fmla(v25.v4s(), v15.v4s(), v4.s()[3]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v14.v4s(), v5.s()[3]);
    fmla(v27.v4s(), v15.v4s(), v5.s()[3]);
  }

  // Load min/max values
  if (clamp_min || clamp_max) {
    ld2r({v4.v4s(), v5.v4s()}, mem[x8]);
  }

  if (max_mr > 2) {
    fmla(v28.v4s(), v14.v4s(), v6.s()[3]);
    fmla(v29.v4s(), v15.v4s(), v6.s()[3]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v14.v4s(), v7.s()[3]);
    fmla(v31.v4s(), v15.v4s(), v7.s()[3]);
  }

  bind(l4);
  // Remainder- 4 floats of A
  tbz(x0, 4, l5);

  ldr(q0, mem[x20], 16);
  ldp(q16, q17, mem[x5], 32);
  if (max_mr > 1) {
    ldr(q1, mem[x13], 16);
  }
  if (max_mr > 2) {
    ldr(q2, mem[x14], 16);
  }
  if (max_mr > 3) {
    ldr(q3, mem[x15], 16);
  }
  fmla(v24.v4s(), v16.v4s(), v0.s()[0]);
  fmla(v25.v4s(), v17.v4s(), v0.s()[0]);
  ldp(q18, q19, mem[x5], 32);
  if (max_mr > 1) {
    fmla(v26.v4s(), v16.v4s(), v1.s()[0]);
    fmla(v27.v4s(), v17.v4s(), v1.s()[0]);
  }
  ldp(q20, q21, mem[x5], 32);
  if (max_mr > 2) {
    fmla(v28.v4s(), v16.v4s(), v2.s()[0]);
    fmla(v29.v4s(), v17.v4s(), v2.s()[0]);
  }
  ldp(q22, q23, mem[x5], 32);
  if (max_mr > 3) {
    fmla(v30.v4s(), v16.v4s(), v3.s()[0]);
    fmla(v31.v4s(), v17.v4s(), v3.s()[0]);
  }
  fmla(v24.v4s(), v18.v4s(), v0.s()[1]);
  fmla(v25.v4s(), v19.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v18.v4s(), v1.s()[1]);
    fmla(v27.v4s(), v19.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v18.v4s(), v2.s()[1]);
    fmla(v29.v4s(), v19.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v18.v4s(), v3.s()[1]);
    fmla(v31.v4s(), v19.v4s(), v3.s()[1]);
  }
  fmla(v24.v4s(), v20.v4s(), v0.s()[2]);
  fmla(v25.v4s(), v21.v4s(), v0.s()[2]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v20.v4s(), v1.s()[2]);
    fmla(v27.v4s(), v21.v4s(), v1.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v20.v4s(), v2.s()[2]);
    fmla(v29.v4s(), v21.v4s(), v2.s()[2]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v20.v4s(), v3.s()[2]);
    fmla(v31.v4s(), v21.v4s(), v3.s()[2]);
  }
  fmla(v24.v4s(), v22.v4s(), v0.s()[3]);
  fmla(v25.v4s(), v23.v4s(), v0.s()[3]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v22.v4s(), v1.s()[3]);
    fmla(v27.v4s(), v23.v4s(), v1.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v22.v4s(), v2.s()[3]);
    fmla(v29.v4s(), v23.v4s(), v2.s()[3]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v22.v4s(), v3.s()[3]);
    fmla(v31.v4s(), v23.v4s(), v3.s()[3]);
  }

  bind(l5);
  // Remainder- 2 floats of A
  tbz(x0, 3, l6);

  ldr(d0, mem[x20], 8);
  ldp(q16, q17, mem[x5], 32);
  if (max_mr > 1) {
    ldr(d1, mem[x13], 8);
  }
  if (max_mr > 2) {
    ldr(d2, mem[x14], 8);
  }
  if (max_mr > 3) {
    ldr(d3, mem[x15], 8);
  }
  fmla(v24.v4s(), v16.v4s(), v0.s()[0]);
  fmla(v25.v4s(), v17.v4s(), v0.s()[0]);
  ldp(q18, q19, mem[x5], 32);
  if (max_mr > 1) {
    fmla(v26.v4s(), v16.v4s(), v1.s()[0]);
    fmla(v27.v4s(), v17.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v16.v4s(), v2.s()[0]);
    fmla(v29.v4s(), v17.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v16.v4s(), v3.s()[0]);
    fmla(v31.v4s(), v17.v4s(), v3.s()[0]);
  }
  fmla(v24.v4s(), v18.v4s(), v0.s()[1]);
  fmla(v25.v4s(), v19.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v18.v4s(), v1.s()[1]);
    fmla(v27.v4s(), v19.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v18.v4s(), v2.s()[1]);
    fmla(v29.v4s(), v19.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v18.v4s(), v3.s()[1]);
    fmla(v31.v4s(), v19.v4s(), v3.s()[1]);
  }

  bind(l6);
  // Remainder- 1 float of A
  tbz(x0, 2, l7);

  ldr(s0, mem[x20], 4);
  ldp(q16, q17, mem[x5], 32);
  if (max_mr > 1) {
    ldr(s1, mem[x13], 4);
  }
  if (max_mr > 2) {
    ldr(s2, mem[x14], 4);
  }
  if (max_mr > 3) {
    ldr(s3, mem[x15], 4);
  }
  fmla(v24.v4s(), v16.v4s(), v0.s()[0]);
  fmla(v25.v4s(), v17.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v26.v4s(), v16.v4s(), v1.s()[0]);
    fmla(v27.v4s(), v17.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v16.v4s(), v2.s()[0]);
    fmla(v29.v4s(), v17.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v16.v4s(), v3.s()[0]);
    fmla(v31.v4s(), v17.v4s(), v3.s()[0]);
  }

  bind(l7);
  // ks loop
  subs(x9, x9, max_mr * sizeof(void*)); // ks -= MR * sizeof(void*)
  b_hi(l1);

  // Clamp
  if (clamp_min) {
    fmax(v24.v4s(), v24.v4s(), v4.v4s());
    fmax(v25.v4s(), v25.v4s(), v4.v4s());
    if (max_mr > 1) {
      fmax(v26.v4s(), v26.v4s(), v4.v4s());
      fmax(v27.v4s(), v27.v4s(), v4.v4s());
    }
    if (max_mr > 2) {
      fmax(v28.v4s(), v28.v4s(), v4.v4s());
      fmax(v29.v4s(), v29.v4s(), v4.v4s());
    }
    if (max_mr > 3) {
      fmax(v30.v4s(), v30.v4s(), v4.v4s());
      fmax(v31.v4s(), v31.v4s(), v4.v4s());
    }
  }
  if (clamp_max) {
    fmin(v24.v4s(), v24.v4s(), v5.v4s());
    fmin(v25.v4s(), v25.v4s(), v5.v4s());
    if (max_mr > 1) {
      fmin(v26.v4s(), v26.v4s(), v5.v4s());
      fmin(v27.v4s(), v27.v4s(), v5.v4s());
    }
    if (max_mr > 2) {
      fmin(v28.v4s(), v28.v4s(), v5.v4s());
      fmin(v29.v4s(), v29.v4s(), v5.v4s());
    }
    if (max_mr > 3) {
      fmin(v30.v4s(), v30.v4s(), v5.v4s());
      fmin(v31.v4s(), v31.v4s(), v5.v4s());
    }
  }
  perform_post_operations(max_mr, num_post_operations, post_operations);

  // Store full 4 x 8
  subs(x1, x1, 8);
  b_lo(l8);

  if (max_mr > 3) {
    stp(q30, q31, mem[x7]);
    add(x7, x7, x10);
  }
  if (max_mr > 2) {
    stp(q28, q29, mem[x17]);
    add(x17, x17, x10);
  }
  if (max_mr > 1) {
    stp(q26, q27, mem[x16]);
    add(x16, x16, x10);
  }
  stp(q24, q25, mem[x6]);
  add(x6, x6, x10);

  sub(x4, x4, x3); // a -= ks

  // nc loop
  b_hi(l0);

  // Restore d8-d15 from stack
  ldp(d14, d15, mem[sp, 64]);
  ldp(d12, d13, mem[sp, 48]);
  ldp(d10, d11, mem[sp, 32]);
  ldp(d8, d9, mem[sp, 16]);

  // Restore x20 from stack
  ldr(x20, mem[sp], 80);
  ret();

  // Store odd width
  bind(l8);
  tbz(x1, 2, l9);
  if (max_mr > 3) {
    str(q30, mem[x7], 16);
    mov(v30.v16b(), v31.v16b());
  }
  if (max_mr > 2) {
    str(q28, mem[x17], 16);
    mov(v28.v16b(), v29.v16b());
  }
  if (max_mr > 1) {
    str(q26, mem[x16], 16);
    mov(v26.v16b(), v27.v16b());
  }
  str(q24, mem[x6], 16);
  mov(v24.v16b(), v25.v16b());

  bind(l9);
  tbz(x1, 1, l10);
  if (max_mr > 3) {
    str(d30, mem[x7], 8);
  }
  if (max_mr > 2) {
    str(d28, mem[x17], 8);
  }
  if (max_mr > 3) {
    dup(d30, v30.d()[1]);
  }
  if (max_mr > 2) {
    dup(d28, v28.d()[1]);
  }
  if (max_mr > 1) {
    str(d26, mem[x16], 8);
  }
  str(d24, mem[x6], 8);
  if (max_mr > 1) {
    dup(d26, v26.d()[1]);
  }
  dup(d24, v24.d()[1]);

  bind(l10);
  tbz(x1, 0, l11);
  if (max_mr > 3) {
    str(s30, mem[x7]);
  }
  if (max_mr > 2) {
    str(s28, mem[x17]);
  }
  if (max_mr > 1) {
    str(s26, mem[x16]);
  }
  str(s24, mem[x6]);
  bind(l11);
  // Restore d8-d15 from stack
  ldp(d14, d15, mem[sp, 64]);
  ldp(d12, d13, mem[sp, 48]);
  ldp(d10, d11, mem[sp, 32]);
  ldp(d8, d9, mem[sp, 16]);

  // Restore x20 from stack
  ldr(x20, mem[sp], 80);
  ret();

  align(16, AlignInstruction::kHlt);
}

void Generator::perform_post_operations(
  size_t max_mr,
  size_t num_post_operations,
  const xnn_post_operation* post_operations)
{
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
          v24.v4s(), v25.v4s(),
          v26.v4s(), v27.v4s(),
          v28.v4s(), v29.v4s(),
          v30.v4s(), v31.v4s(),
        };
        const VRegister tmps[] = {v4.v4s(), v5.v4s(), v6.v4s(), v7.v4s()};
        f32_hardswish(sixth, three, six, zero, &accs[0], XNN_COUNT_OF(accs), &tmps[0], XNN_COUNT_OF(tmps));
        break;
      }
      default:
        XNN_UNREACHABLE;
    }
  }
}

}  // namespace
}  // namespace aarch64
}  // namespace xnnpack

xnn_status_t xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_cortex_a75(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
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

xnn_status_t xnn_generate_f32_igemm_ukernel_4x8__aarch64_neonfma_prfm_cortex_a75(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
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
