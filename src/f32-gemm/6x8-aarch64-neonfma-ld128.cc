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
  void generate(size_t max_mr, size_t nc_mod_nr, size_t kc, float min, float max);
};

// void xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_ld128(
//     size_t mr,                x0
//     size_t nc,                x1
//     size_t kc,                x2 / x0
//     const uint8_t*restrict a, x3
//     size_t a_stride,          x4
//     const void*restrict w,    x5
//     uint8_t*restrict c,       x6
//     size_t cm_stride,         x7
//     size_t cn_stride,         [sp] -> (x0)
//     const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])  [sp + 8] -> (x8)

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
// A0   v0
// A1   v1
// A2   v2
// A3   v3
// A4   v4
// A5   v5
// B   v16 v17 v18 v19
// C   v20 v21
// C   v22 v23
// C   v24 v25
// C   v26 v27
// C   v28 v29
// C   v30 v31
// Clamp v6 v7
// unused A   v8 v9 v10 v11
// unused B   v12 v13 v14 v15

// Converted from: src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-ld128.S
void Generator::generate(size_t max_mr, size_t nc_mod_nr, size_t kc, float min, float max) {
  assert(max_mr <= 6);
  assert(nc_mod_nr < 8);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8;

  const bool clamp_min = min != -std::numeric_limits<float>::infinity();
  const bool clamp_max = max != +std::numeric_limits<float>::infinity();

  // Load params pointer
  ldr(x8, mem[sp, 8]);

  // Clamp A and C pointers
  cmp(x0, 2); // if mr < 2
  add(x9, x3, x4); // a1 = a0 + a_stride
  add(x16, x6, x7); // c1 = c0 + cm_stride
  csel(x9, x3, x9, kLO); //   a1 = a0
  csel(x16, x6, x16, kLO); //   c1 = c0

  add(x10, x9, x4); // a2 = a1 + a_stride
  add(x17, x16, x7); // c2 = c1 + cm_stride
  // if mr <= 2
  csel(x10, x9, x10, kLS); //   a2 = a1
  csel(x17, x16, x17, kLS); //   c2 = c1

  cmp(x0, 4); // if mr < 4
  add(x11, x10, x4); // a3 = a2 + a_stride
  add(x14, x17, x7); // c3 = c2 + cm_stride
  csel(x11, x10, x11, kLO); //   a3 = a2
  csel(x14, x17, x14, kLO); //   c3 = c2

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

  // Load min/max values
  if (clamp_min || clamp_max) {
    ld2r({v6.v4s(), v7.v4s()}, mem[x8]);
  }

  bind(l0);
  // Load initial bias from w into accumulators
  ldp(q20, q21, mem[x5], 32);
  mov(v22.v16b(), v20.v16b());
  prfm(kPLDL1KEEP, mem[x5, 0]); // Prefetch B
  mov(v23.v16b(), v21.v16b());
  prfm(kPLDL1KEEP, mem[x5, 64]);
  mov(v24.v16b(), v20.v16b());
  prfm(kPLDL1KEEP, mem[x5, 128]);
  mov(v25.v16b(), v21.v16b());
  prfm(kPLDL1KEEP, mem[x5, 192]);
  mov(v26.v16b(), v20.v16b());
  prfm(kPLDL1KEEP, mem[x3]); // Prefetch A
  mov(v27.v16b(), v21.v16b());
  prfm(kPLDL1KEEP, mem[x9]);
  mov(v28.v16b(), v20.v16b());
  prfm(kPLDL1KEEP, mem[x10]);
  mov(v29.v16b(), v21.v16b());
  prfm(kPLDL1KEEP, mem[x11]);
  mov(v30.v16b(), v20.v16b());
  prfm(kPLDL1KEEP, mem[x12]);
  mov(v31.v16b(), v21.v16b());
  prfm(kPLDL1KEEP, mem[x4]);

  // Is there at least 4 floats (16 bytes)?
  subs(x0, x2, 16); // k = kc - 16
  b_lo(l3);

  // Main loop - 4 floats of A (16 bytes)
  // 48 FMA + 6 ld128 A + 4 LDP B
  bind(l1);
  ldr(q0, mem[x3], 16);
  ldp(q16, q17, mem[x5], 32);
  ldr(q1, mem[x9], 16);
  ldr(q2, mem[x10], 16);
  ldr(q3, mem[x11], 16);
  ldr(q4, mem[x12], 16);
  ldr(q5, mem[x4], 16);
  fmla(v20.v4s(), v16.v4s(), v0.s()[0]);
  fmla(v22.v4s(), v16.v4s(), v1.s()[0]);
  fmla(v24.v4s(), v16.v4s(), v2.s()[0]);
  fmla(v26.v4s(), v16.v4s(), v3.s()[0]);
  ldp(q18, q19, mem[x5], 32);
  fmla(v28.v4s(), v16.v4s(), v4.s()[0]);
  fmla(v30.v4s(), v16.v4s(), v5.s()[0]);
  fmla(v21.v4s(), v17.v4s(), v0.s()[0]);
  fmla(v23.v4s(), v17.v4s(), v1.s()[0]);
  fmla(v25.v4s(), v17.v4s(), v2.s()[0]);
  fmla(v27.v4s(), v17.v4s(), v3.s()[0]);
  fmla(v29.v4s(), v17.v4s(), v4.s()[0]);
  fmla(v31.v4s(), v17.v4s(), v5.s()[0]);

  fmla(v20.v4s(), v18.v4s(), v0.s()[1]);
  ldp(q16, q17, mem[x5], 32);
  fmla(v22.v4s(), v18.v4s(), v1.s()[1]);
  fmla(v24.v4s(), v18.v4s(), v2.s()[1]);
  fmla(v26.v4s(), v18.v4s(), v3.s()[1]);
  fmla(v28.v4s(), v18.v4s(), v4.s()[1]);
  fmla(v30.v4s(), v18.v4s(), v5.s()[1]);
  fmla(v21.v4s(), v19.v4s(), v0.s()[1]);
  fmla(v23.v4s(), v19.v4s(), v1.s()[1]);
  fmla(v25.v4s(), v19.v4s(), v2.s()[1]);
  fmla(v27.v4s(), v19.v4s(), v3.s()[1]);
  fmla(v29.v4s(), v19.v4s(), v4.s()[1]);
  fmla(v31.v4s(), v19.v4s(), v5.s()[1]);

  fmla(v20.v4s(), v16.v4s(), v0.s()[2]);
  ldp(q18, q19, mem[x5], 32);
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
  subs(x0, x0, 16);
  fmla(v29.v4s(), v19.v4s(), v4.s()[3]);
  fmla(v31.v4s(), v19.v4s(), v5.s()[3]);
  b_hs(l1);

  // Is there a remainder?- 2 floats of A (8 bytes) or less
  tst(x0, 15);
  b_ne(l3);

  bind(l2);
  // Clamp
  // Load cn_stride
  ldr(x0, mem[sp]);
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
  subs(x1, x1, 8);
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
  b_lo(l5);

  st1({v20.v16b(), v21.v16b()}, mem[x6], x0);
  sub(x3, x3, x2); // a0 -= kc
  st1({v22.v16b(), v23.v16b()}, mem[x16], x0);
  sub(x9, x9, x2); // a1 -= kc
  st1({v24.v16b(), v25.v16b()}, mem[x17], x0);
  sub(x10, x10, x2); // a2 -= kc
  st1({v26.v16b(), v27.v16b()}, mem[x14], x0);
  sub(x11, x11, x2); // a3 -= kc
  st1({v28.v16b(), v29.v16b()}, mem[x13], x0);
  sub(x12, x12, x2); // a4 -= kc
  st1({v30.v16b(), v31.v16b()}, mem[x7], x0);
  sub(x4, x4, x2); // a5 -= kc

  b_hi(l0);
  ret();

  bind(l3);
  // Is there a remainder?- 2 floats of A (8 bytes)
  tbz(x0, 3, l4);

  // Remainder- 2 floats of A (8 bytes)
  ldr(d0, mem[x3], 8);
  ldp(q16, q17, mem[x5], 32);
  ldr(d1, mem[x9], 8);
  ldr(d2, mem[x10], 8);
  ldr(d3, mem[x11], 8);
  ldr(d4, mem[x12], 8);
  ldr(d5, mem[x4], 8);
  fmla(v20.v4s(), v16.v4s(), v0.s()[0]);
  fmla(v22.v4s(), v16.v4s(), v1.s()[0]);
  fmla(v24.v4s(), v16.v4s(), v2.s()[0]);
  fmla(v26.v4s(), v16.v4s(), v3.s()[0]);
  ldp(q18, q19, mem[x5], 32);
  fmla(v28.v4s(), v16.v4s(), v4.s()[0]);
  fmla(v30.v4s(), v16.v4s(), v5.s()[0]);
  fmla(v21.v4s(), v17.v4s(), v0.s()[0]);
  fmla(v23.v4s(), v17.v4s(), v1.s()[0]);
  fmla(v25.v4s(), v17.v4s(), v2.s()[0]);
  fmla(v27.v4s(), v17.v4s(), v3.s()[0]);
  fmla(v29.v4s(), v17.v4s(), v4.s()[0]);
  fmla(v31.v4s(), v17.v4s(), v5.s()[0]);

  fmla(v20.v4s(), v18.v4s(), v0.s()[1]);
  fmla(v22.v4s(), v18.v4s(), v1.s()[1]);
  fmla(v24.v4s(), v18.v4s(), v2.s()[1]);
  fmla(v26.v4s(), v18.v4s(), v3.s()[1]);
  fmla(v28.v4s(), v18.v4s(), v4.s()[1]);
  fmla(v30.v4s(), v18.v4s(), v5.s()[1]);
  fmla(v21.v4s(), v19.v4s(), v0.s()[1]);
  fmla(v23.v4s(), v19.v4s(), v1.s()[1]);
  fmla(v25.v4s(), v19.v4s(), v2.s()[1]);
  fmla(v27.v4s(), v19.v4s(), v3.s()[1]);
  fmla(v29.v4s(), v19.v4s(), v4.s()[1]);
  fmla(v31.v4s(), v19.v4s(), v5.s()[1]);

  // Is there a remainder?- 1 float of A (4 bytes)
  tbz(x0, 2, l2);

  // Remainder- 1 float of A (4 bytes)
  bind(l4);
  ldr(s0, mem[x3], 4);
  ldp(q16, q17, mem[x5], 32);
  ldr(s1, mem[x9], 4);
  ldr(s2, mem[x10], 4);
  ldr(s3, mem[x11], 4);
  ldr(s4, mem[x12], 4);
  ldr(s5, mem[x4], 4);
  fmla(v20.v4s(), v16.v4s(), v0.s()[0]);
  fmla(v22.v4s(), v16.v4s(), v1.s()[0]);
  fmla(v24.v4s(), v16.v4s(), v2.s()[0]);
  fmla(v26.v4s(), v16.v4s(), v3.s()[0]);
  fmla(v28.v4s(), v16.v4s(), v4.s()[0]);
  fmla(v30.v4s(), v16.v4s(), v5.s()[0]);
  fmla(v21.v4s(), v17.v4s(), v0.s()[0]);
  fmla(v23.v4s(), v17.v4s(), v1.s()[0]);
  fmla(v25.v4s(), v17.v4s(), v2.s()[0]);
  fmla(v27.v4s(), v17.v4s(), v3.s()[0]);
  fmla(v29.v4s(), v17.v4s(), v4.s()[0]);
  fmla(v31.v4s(), v17.v4s(), v5.s()[0]);
  b(l2);

  // Store odd width
  bind(l5);
  tbz(x1, 2, l6);
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

  bind(l6);
  tbz(x1, 1, l7);
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

  bind(l7);
  tbz(x1, 0, l8);
  str(s20, mem[x6]);
  str(s22, mem[x16]);
  str(s24, mem[x17]);
  str(s26, mem[x14]);
  str(s28, mem[x13]);
  str(s30, mem[x7]);
  bind(l8);
  ret();

  align(16, AlignInstruction::kHlt);
}
}  // namespace
}  // aarch64
}  // xnnpack

xnn_status_t xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_ld128(
    xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, const void* params) {
  using namespace xnnpack::aarch64;
  Generator g(code);
  assert(params != nullptr);
  const jit_gemm_params* gemm_params = static_cast<const jit_gemm_params*>(params);
  g.generate(max_mr, nc_mod_nr, kc, gemm_params->f32_minmax.min, gemm_params->f32_minmax.max);
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
