// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <limits>

#include <xnnpack.h>
#include <xnnpack/aarch64-assembler.h>
#include <xnnpack/igemm.h>
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
  void generate(size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const jit_gemm_params* jit_gemm_params);
  void perform_post_operations(size_t max_mr, size_t num_post_operations, const xnn_post_operation* post_operations);
};

// void xnn_f32_igemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128(
//     size_t mr,                         x0
//     size_t nc,                         x1
//     size_t kc,                         x2 / x0
//     size_t ks,                         x3 / x9
//     const float** restrict a,           x4
//     const void* restrict w,             x5
//     uint8_t* restrict c,                x6
//     size_t cm_stride,                  x7
//     size_t cn_stride,                  [sp] -> (x0)
//     size_t a_offset,                   [sp + 8] -> x11
//     const float* zero,                 [sp + 16] -> x12
//     const xnn_f32_minmax_params params [sp + 24] -> x8

// d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x14 v0
// A1  x15 v1
// A2  x20 v2
// A3  x21 v3
// A4  x22 v4
// A5  x23 v5
// B    x5 v16 v17 v18 v19
// C0   x6 v20 v21
// C1  x16 v22 v23
// C2  x17 v24 v25
// C3  x10 v26 v27
// C4  x13 v28 v29
// C5   x7 v30 v31
// Clamp v6 v7
// unused A   v8 v9 v10 v11
// unused B   v12 v13 v14 v15

// Converted from: src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-ld128.S
void Generator::generate(size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const jit_gemm_params* jit_gemm_params)
{
  assert(max_mr <= 6);
  assert(nc_mod_nr < 8 || nc_mod_nr == SIZE_MAX);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8, l9;
  const size_t num_post_operations = jit_gemm_params->num_post_operations;
  const xnn_post_operation* post_operations = jit_gemm_params->post_operations;
  const float min = jit_gemm_params->f32_minmax.min;
  const float max = jit_gemm_params->f32_minmax.max;
  const bool clamp_min = min != -std::numeric_limits<float>::infinity();
  const bool clamp_max = max != +std::numeric_limits<float>::infinity();
  assert(num_post_operations == 0 || (!clamp_min && !clamp_max));

  // Load zero, params pointer
  ldp(x12, x8, mem[sp, 16]);

  // Clamp C pointers
  if (max_mr > 1) {
    cmp(x0, 2); // if mr < 2
    add(x16, x6, x7); // c1 = c0 + cm_stride
    csel(x16, x6, x16, kLO); //   c1 = c0
  }

  // Load min/max values
  if (clamp_min || clamp_max) {
    ld2r({v6.v4s(), v7.v4s()}, mem[x8]);
  }

  if (max_mr > 2) {
    add(x17, x16, x7); // c2 = c1 + cm_stride
    // if mr <= 2
    csel(x17, x16, x17, kLS); //   c2 = c1
  }

  // Save x20,x21,x22,x23 on stack
  stp(x20, x21, mem[sp, -32]++);

  if (max_mr > 3) {
    cmp(x0, 4); // if mr < 4
    add(x10, x17, x7); // c3 = c2 + cm_stride
    csel(x10, x17, x10, kLO); //   c3 = c2
  }

  stp(x22, x23, mem[sp, 16]);

  if (max_mr > 4) {
    add(x13, x10, x7); // c4 = c3 + cm_stride
    // if mr <= 4
    csel(x13, x10, x13, kLS); //   c4 = c3
  }

  // Load a_offset
  ldr(x11, mem[sp, 40]);

  if (max_mr > 5) {
    cmp(x0, 6); // if mr < 6
    add(x7, x13, x7); // c5 = c4 + cm_stride
    csel(x7, x13, x7, kLO); //   c5 = c4
  }

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

  mov(x9, x3); // p = ks

  bind(l1);
  // Load next 6 A pointers
  ldr(x14, mem[x4], 8);
  if (max_mr > 1) {
    ldr(x15, mem[x4], 8);
  }
  if (max_mr > 2) {
    ldr(x20, mem[x4], 8);
  }
  if (max_mr > 3) {
    ldr(x21, mem[x4], 8);
  }
  if (max_mr > 4) {
    ldr(x22, mem[x4], 8);
  }
  if (max_mr > 5) {
    ldr(x23, mem[x4], 8);
  }

  cmp(x14, x12); // if a0 == zero
  add(x14, x14, x11); // a0 += a_offset
  csel(x14, x12, x14, kEQ); //   a0 = zero, else += a0 + a_offset
  if (max_mr > 1) {
    cmp(x15, x12); // if a1 == zero
    add(x15, x15, x11); // a1 += a_offset
    csel(x15, x12, x15, kEQ); //   a1 = zero, else += a1 + a_offset
  }
  if (max_mr > 2) {
    cmp(x20, x12); // if a2 == zero
    add(x20, x20, x11); // a2 += a_offset
    csel(x20, x12, x20, kEQ); //   a2 = zero, else += a2 + a_offset
  }
  if (max_mr > 3) {
    cmp(x21, x12); // if a3 == zero
    add(x21, x21, x11); // a3 += a_offset
    csel(x21, x12, x21, kEQ); //   a3 = zero, else += a3 + a_offset
  }
  if (max_mr > 4) {
    cmp(x22, x12); // if a4 == zero
    add(x22, x22, x11); // a4 += a_offset
    csel(x22, x12, x22, kEQ); //   a4 = zero, else += a4 + a_offset
  }
  if (max_mr > 5) {
    cmp(x23, x12); // if a5 == zero
    add(x23, x23, x11); // a5 += a_offset
    csel(x23, x12, x23, kEQ); //   a5 = zero, else += a5 + a_offset
  }

  // Is there at least 4 floats (16 bytes)?
  subs(x0, x2, 16); // k = kc - 16
  b_lo(l4);

  // Main loop - 4 floats of A (16 bytes)
  // 48 FMA + 6 ld128 A + 4 LDP B
  bind(l2);
  ldp(q16, q17, mem[x5], 32);
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
  fmla(v20.v4s(), v16.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v16.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v16.v4s(), v3.s()[0]);
  }
  ldp(q18, q19, mem[x5], 32);
  if (max_mr > 4) {
    fmla(v28.v4s(), v16.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v16.v4s(), v5.s()[0]);
  }
  fmla(v21.v4s(), v17.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v17.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v17.v4s(), v3.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v17.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v17.v4s(), v5.s()[0]);
  }

  fmla(v20.v4s(), v18.v4s(), v0.s()[1]);
  ldp(q16, q17, mem[x5], 32);
  if (max_mr > 1) {
    fmla(v22.v4s(), v18.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v18.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v18.v4s(), v3.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v18.v4s(), v4.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v18.v4s(), v5.s()[1]);
  }
  fmla(v21.v4s(), v19.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v19.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v19.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v19.v4s(), v3.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v19.v4s(), v4.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v19.v4s(), v5.s()[1]);
  }

  fmla(v20.v4s(), v16.v4s(), v0.s()[2]);
  ldp(q18, q19, mem[x5], 32);
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
  subs(x0, x0, 16);
  if (max_mr > 4) {
    fmla(v29.v4s(), v19.v4s(), v4.s()[3]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v19.v4s(), v5.s()[3]);
  }
  b_hs(l2);

  // Is there a remainder?- 2 floats of A (8 bytes) or less
  tst(x0, 15);
  b_ne(l4);

  bind(l3);
  // ks loop
  subs(x9, x9, max_mr * sizeof(void*)); // ks -= MR * sizeof(void*)
  b_hi(l1);

  // Clamp
  if (clamp_min) {
    fmax(v20.v4s(), v20.v4s(), v6.v4s());
  }
  // Load cn_stride
  ldr(x0, mem[sp, 32]);
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
  perform_post_operations(max_mr, num_post_operations, post_operations);

  // Store full 6 x 8
  b_lo(l6);

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
  ldp(x22, x23, mem[sp, 16]);
  ldp(x20, x21, mem[sp], 32);
  ret();

  bind(l4);
  // Is there a remainder?- 2 floats of A (8 bytes)
  tbz(x0, 3, l5);

  // Remainder- 2 floats of A (8 bytes)
  ldr(d0, mem[x14], 8);
  ldp(q16, q17, mem[x5], 32);
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
  fmla(v20.v4s(), v16.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v16.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v16.v4s(), v3.s()[0]);
  }
  ldp(q18, q19, mem[x5], 32);
  if (max_mr > 4) {
    fmla(v28.v4s(), v16.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v16.v4s(), v5.s()[0]);
  }
  fmla(v21.v4s(), v17.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v17.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v17.v4s(), v3.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v17.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v17.v4s(), v5.s()[0]);
  }

  fmla(v20.v4s(), v18.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v18.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v18.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v18.v4s(), v3.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v18.v4s(), v4.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v18.v4s(), v5.s()[1]);
  }
  fmla(v21.v4s(), v19.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v19.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v19.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v19.v4s(), v3.s()[1]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v19.v4s(), v4.s()[1]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v19.v4s(), v5.s()[1]);
  }

  // Is there a remainder?- 1 float of A (4 bytes)
  tbz(x0, 2, l3);

  // Remainder- 1 float of A (4 bytes)
  bind(l5);
  ldr(s0, mem[x14], 4);
  ldp(q16, q17, mem[x5], 32);
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
  fmla(v20.v4s(), v16.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v22.v4s(), v16.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v24.v4s(), v16.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v26.v4s(), v16.v4s(), v3.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v28.v4s(), v16.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v30.v4s(), v16.v4s(), v5.s()[0]);
  }
  fmla(v21.v4s(), v17.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v23.v4s(), v17.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v25.v4s(), v17.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v27.v4s(), v17.v4s(), v3.s()[0]);
  }
  if (max_mr > 4) {
    fmla(v29.v4s(), v17.v4s(), v4.s()[0]);
  }
  if (max_mr > 5) {
    fmla(v31.v4s(), v17.v4s(), v5.s()[0]);
  }
  b(l3);

  // Store odd width
  bind(l6);
  tbz(x1, 2, l7);
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
  bind(l7);
  tbz(x1, 1, l8);
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

  bind(l8);
  tbz(x1, 0, l9);
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
  bind(l9);
  // Restore x20,x21,x22,x23 from stack
  ldp(x22, x23, mem[sp, 16]);
  ldp(x20, x21, mem[sp], 32);
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

xnn_status_t xnn_generate_f32_igemm_ukernel_6x8__aarch64_neonfma_ld128(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
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
