// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <limits>

#include <xnnpack/aarch64-assembler.h>
#include <xnnpack/allocator.h>
#include <xnnpack/igemm.h>


namespace xnnpack {
namespace aarch64 {
namespace {
class Generator : public Assembler {
  using Assembler::Assembler;

public:
  void generate(bool prefetch, size_t nc_mod_nr, size_t kc, size_t ks, float min, float max);
};

// void xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_prfm_cortex_a75(
//     size_t mr,                         (x0) - unused.  mr = 1
//     size_t nc,                         x1
//     size_t kc,                         x2 / x0
//     size_t ks,                         x3 / x9
//     const float**restrict a,           x4
//     const float*restrict w,            x5
//     float*restrict c,                  x6
//     size_t cm_stride,                  (x7) - unused
//     size_t cn_stride,                  [sp] -> x10
//     size_t a_offset,                   [sp + 8] -> x11
//     const float* zero,                 [sp + 16] -> x12
//     const xnn_f32_minmax_params params [sp + 24] -> (x8)

// d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// A pointer
// x8  a0

// C pointer
// x6  c0

// Converted from: src/f32-igemm/gen/1x8-minmax-aarch64-neonfma-prfm-cortex-a75.S
void Generator::generate(bool prefetch, size_t nc_mod_nr, size_t kc, size_t ks, float min, float max)
{
  assert(ks != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13;
  const bool clamp_min = min != -std::numeric_limits<float>::infinity();
  const bool clamp_max = max != +std::numeric_limits<float>::infinity();

  // Load cn_stride, a_offset
  ldp(x10, x11, mem[sp]);

  // Load zero, params pointer
  ldp(x12, x8, mem[sp, 16]);

  // Load min/max values
  if (clamp_max) {
    ld2r({v30.v4s(), v31.v4s()}, mem[x8]);
  } else if (clamp_min) {
    if (min == 0.f) {
      movi(v30.v4s(), 0);
    } else {
      ld1r({v30.v4s()}, mem[x8]);
    }
  }

  bind(l0);
  // Load initial bias from w into accumulators
  ldp(q16, q17, mem[x5], 32);
  movi(v18.v4s(), 0); // second set of C for pipelining FMLA
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5]);
  }
  movi(v19.v4s(), 0);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 64]);
    prfm(kPLDL1KEEP, mem[x5, 128]);
    prfm(kPLDL1KEEP, mem[x5, 192]);
  }

  mov(x9, x3); // p = ks

  bind(l1);
  // Load next A pointer
  ldr(x8, mem[x4], 8);

  cmp(x8, x12);           // if a0 == zero
  add(x8, x8, x11);       // a0 += a_offset
  csel(x8, x12, x8, kEQ); //   a0 = zero, else += a0 + a_offset

  // Is there at least 8 floats (32 bytes) for prologue + epilogue?
  subs(x0, x2, 32); // k = kc - 32
  b_lo(l4);

  // 16 prologue
  // Read first block of A and B.
  ldp(q20, q21, mem[x5], 32);
  ldp(q22, q23, mem[x5], 32);
  ldp(q24, q25, mem[x5], 32);
  ldp(q26, q27, mem[x5], 32);
  ldr(q0, mem[x8], 16);

  // Is there at least 8.  yes do main loop
  subs(x0, x0, 32);
  b_lo(l3);

  // Main loop - 8 floats of A (32 bytes)
  bind(l2);
  // First block of 4.  FMA for first 4, loads for 2nd block of 4.
  fmla(v16.v4s(), v20.v4s(), v0.s()[0]);
  ldr(q1, mem[x8], 16);
  fmla(v17.v4s(), v21.v4s(), v0.s()[0]);
  ldp(q20, q21, mem[x5], 32);
  fmla(v18.v4s(), v22.v4s(), v0.s()[1]);
  fmla(v19.v4s(), v23.v4s(), v0.s()[1]);
  ldp(q22, q23, mem[x5], 32);
  fmla(v16.v4s(), v24.v4s(), v0.s()[2]);
  fmla(v17.v4s(), v25.v4s(), v0.s()[2]);
  ldp(q24, q25, mem[x5], 32);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 128]);
  }
  fmla(v18.v4s(), v26.v4s(), v0.s()[3]);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 256]);
  }
  fmla(v19.v4s(), v27.v4s(), v0.s()[3]);
  ldp(q26, q27, mem[x5], 32);

  // Second block of 4.  FMA for second 4, loads for 1st block of 4.
  fmla(v16.v4s(), v20.v4s(), v1.s()[0]);
  ldr(q0, mem[x8], 16);
  fmla(v17.v4s(), v21.v4s(), v1.s()[0]);
  ldp(q20, q21, mem[x5], 32);
  fmla(v18.v4s(), v22.v4s(), v1.s()[1]);
  fmla(v19.v4s(), v23.v4s(), v1.s()[1]);
  ldp(q22, q23, mem[x5], 32);
  fmla(v16.v4s(), v24.v4s(), v1.s()[2]);
  fmla(v17.v4s(), v25.v4s(), v1.s()[2]);
  ldp(q24, q25, mem[x5], 32);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 128]);
  }
  fmla(v18.v4s(), v26.v4s(), v1.s()[3]);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 256]);
  }
  fmla(v19.v4s(), v27.v4s(), v1.s()[3]);
  subs(x0, x0, 32);
  ldp(q26, q27, mem[x5], 32);
  b_hs(l2);

  bind(l3);
  // Epilogue

  // First block of 4.  FMA for first 4, loads for 2nd block of 4.
  fmla(v16.v4s(), v20.v4s(), v0.s()[0]);
  ldr(q1, mem[x8], 16);
  fmla(v17.v4s(), v21.v4s(), v0.s()[0]);
  ldp(q20, q21, mem[x5], 32);
  fmla(v18.v4s(), v22.v4s(), v0.s()[1]);
  fmla(v19.v4s(), v23.v4s(), v0.s()[1]);
  ldp(q22, q23, mem[x5], 32);
  fmla(v16.v4s(), v24.v4s(), v0.s()[2]);
  fmla(v17.v4s(), v25.v4s(), v0.s()[2]);
  ldp(q24, q25, mem[x5], 32);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 128]);
  }
  fmla(v18.v4s(), v26.v4s(), v0.s()[3]);
  if (prefetch) {
    prfm(kPLDL1KEEP, mem[x5, 256]);
  }
  fmla(v19.v4s(), v27.v4s(), v0.s()[3]);
  ldp(q26, q27, mem[x5], 32);

  // Second block of 4.  no loads
  fmla(v16.v4s(), v20.v4s(), v1.s()[0]);
  fmla(v17.v4s(), v21.v4s(), v1.s()[0]);
  fmla(v18.v4s(), v22.v4s(), v1.s()[1]);
  fmla(v19.v4s(), v23.v4s(), v1.s()[1]);
  fmla(v16.v4s(), v24.v4s(), v1.s()[2]);
  fmla(v17.v4s(), v25.v4s(), v1.s()[2]);
  fmla(v18.v4s(), v26.v4s(), v1.s()[3]);
  fmla(v19.v4s(), v27.v4s(), v1.s()[3]);

  bind(l4);
  // Is there a remainder?- 4 floats of A (16 bytes)
  tbnz(x0, 4, l6);
  // Is there a remainder?- 2 floats of A (8 bytes)
  tbnz(x0, 3, l7);
  // Is there a remainder?- 1 float of A (4 bytes)
  tbnz(x0, 2, l9);

  bind(l5);
  // ks loop
  subs(x9, x9, 8); // ks -= MR * sizeof(void*)
  b_hi(l1);

  fadd(v16.v4s(), v16.v4s(), v18.v4s());
  fadd(v17.v4s(), v17.v4s(), v19.v4s());

  // Clamp
  if (clamp_min) {
    fmax(v16.v4s(), v16.v4s(), v30.v4s());
    fmax(v17.v4s(), v17.v4s(), v30.v4s());
  }
  if (clamp_max) {
    fmin(v16.v4s(), v16.v4s(), v31.v4s());
    fmin(v17.v4s(), v17.v4s(), v31.v4s());
  }

  // Store full 1 x 8
  subs(x1, x1, 8);
  b_lo(l10);

  stp(q16, q17, mem[x6]);
  add(x6, x6, x10);

  sub(x4, x4, x3); // a -= ks

  // nc loop
  b_hi(l0);

  ret();

  bind(l6);
  // Remainder- 4 floats of A (16 bytes)
  ldp(q20, q21, mem[x5], 32);
  ldr(q0, mem[x8], 16);
  fmla(v16.v4s(), v20.v4s(), v0.s()[0]);
  fmla(v17.v4s(), v21.v4s(), v0.s()[0]);
  ldp(q22, q23, mem[x5], 32);
  ldp(q24, q25, mem[x5], 32);
  ldp(q26, q27, mem[x5], 32);
  fmla(v18.v4s(), v22.v4s(), v0.s()[1]);
  fmla(v19.v4s(), v23.v4s(), v0.s()[1]);
  fmla(v16.v4s(), v24.v4s(), v0.s()[2]);
  fmla(v17.v4s(), v25.v4s(), v0.s()[2]);
  fmla(v18.v4s(), v26.v4s(), v0.s()[3]);
  fmla(v19.v4s(), v27.v4s(), v0.s()[3]);

  tbz(x0, 3, l8);
  bind(l7);
  // Remainder- 2 floats of A (8 bytes)
  ldp(q20, q21, mem[x5], 32);
  ldr(d0, mem[x8], 8);
  fmla(v16.v4s(), v20.v4s(), v0.s()[0]);
  fmla(v17.v4s(), v21.v4s(), v0.s()[0]);
  ldp(q22, q23, mem[x5], 32);
  fmla(v18.v4s(), v22.v4s(), v0.s()[1]);
  fmla(v19.v4s(), v23.v4s(), v0.s()[1]);
  bind(l8);
  tbz(x0, 2, l5);
  bind(l9);
  // Remainder- 1 float of A (4 bytes)
  ldp(q20, q21, mem[x5], 32);
  ldr(s0, mem[x8], 4);
  fmla(v16.v4s(), v20.v4s(), v0.s()[0]);
  fmla(v17.v4s(), v21.v4s(), v0.s()[0]);
  b(l5);

  bind(l10);
  // Store odd channels
  tbz(x1, 2, l11);
  str(q16, mem[x6], 16);
  mov(v16.v16b(), v17.v16b());

  bind(l11);
  tbz(x1, 1, l12);
  str(d16, mem[x6], 8);
  dup(d16, v16.d()[1]);

  bind(l12);
  tbz(x1, 0, l13);
  str(s16, mem[x6], 4);
  bind(l13);
  ret();

  align(16, AlignInstruction::kHlt);
}
} // namespace
} // namespace aarch64
} // namespace xnnpack

xnn_status xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_cortex_a75(
    xnn_code_buffer* code, size_t nc_mod_nr, size_t kc, size_t ks, const void* params)
{
  using namespace xnnpack::aarch64;
  Generator g(code);
  assert(params != nullptr);
  auto jit_params = static_cast<const jit_gemm_params*>(params);
  g.generate(false, nc_mod_nr, kc, ks, jit_params->f32_minmax.min, jit_params->f32_minmax.max);
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}

xnn_status xnn_generate_f32_igemm_ukernel_1x8__aarch64_neonfma_prfm_cortex_a75(
    xnn_code_buffer* code, size_t nc_mod_nr, size_t kc, size_t ks, const void* params)
{
  using namespace xnnpack::aarch64;
  Generator g(code);
  assert(params != nullptr);
  auto jit_params = static_cast<const jit_gemm_params*>(params);
  g.generate(true, nc_mod_nr, kc, ks, jit_params->f32_minmax.min, jit_params->f32_minmax.max);
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
