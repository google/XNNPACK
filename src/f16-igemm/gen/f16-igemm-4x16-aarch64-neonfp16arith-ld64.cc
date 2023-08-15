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
  void generate(size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const jit_gemm_params* jit_gemm_params);
};

// void xnn_f16_igemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64(
//     size_t mr,                         x0
//     size_t nc,                         x1
//     size_t kc,                         x2 / x0
//     size_t ks,                         x3 / x9
//     const void** restrict a,            x4
//     const void* restrict w,             x5
//     void* restrict c,                   x6
//     size_t cm_stride,                  x7
//     size_t cn_stride,                  [sp] -> x10
//     size_t a_offset,                   [sp + 8] -> x11
//     const void* zero,                  [sp + 16] -> x12
//     const xnn_f16_minmax_params params [sp + 24] -> (x8)

// d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x8 v0
// A1 x13 v1
// A2 x14 v2
// A3 x15 v3
// B   x5 v20 v21 v22 v23 v16 v17 v18 v19
// C0  x6 v24 v25
// C1 x16 v26 v27
// C2 x17 v28 v29
// C3  x7 v30 v31
// clamp  v4, v5

// Converted from: src/f16-igemm/f16-igemm-4x16-minmax-asm-aarch64-neonfp16arith-ld64.S
void Generator::generate(size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const jit_gemm_params* jit_gemm_params)
{
  assert(max_mr <= 4);
  assert(nc_mod_nr < 16 || nc_mod_nr == SIZE_MAX);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);
  assert(ks != 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10;
  const size_t num_post_operations = jit_gemm_params->num_post_operations;
  (void) num_post_operations;  // Silence unused warning.
  const uint16_t min = jit_gemm_params->f16_minmax.min;
  const uint16_t max = jit_gemm_params->f16_minmax.max;
  const bool clamp_min = min != UINT16_C(0xFC00);  // -Inf.
  const bool clamp_max = max != UINT16_C(0x7C00);  // Inf.
  assert(num_post_operations == 0 || (!clamp_min && !clamp_max));

  // Load cn_stride, a_offset
  ldp(x10, x11, mem[sp]);

  // Load zero, params pointer
  ldp(x12, x8, mem[sp, 16]);

  // Load params values
  if (clamp_min || clamp_max) {
    ld2r({v4.v8h(), v5.v8h()}, mem[x8]);
  }

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
  ldr(q24, mem[x5], 16);
  ldr(q25, mem[x5], 16);
  if (max_mr > 1) {
    mov(v26.v16b(), v24.v16b());
  }
  if (max_mr > 2) {
    mov(v28.v16b(), v24.v16b());
  }
  if (max_mr > 3) {
    mov(v30.v16b(), v24.v16b());
  }
  if (max_mr > 1) {
    mov(v27.v16b(), v25.v16b());
  }
  if (max_mr > 2) {
    mov(v29.v16b(), v25.v16b());
  }
  if (max_mr > 3) {
    mov(v31.v16b(), v25.v16b());
  }

  mov(x9, x3); // p = ks

  bind(l1);
  // Load next 4 A pointers
  if (max_mr == 1) {
    ldr(x8, mem[x4], 8);
  }
  if (max_mr > 1) {
    ldp(x8, x13, mem[x4], 16);
  }
  if (max_mr == 3) {
    ldr(x14, mem[x4], 8);
  }
  if (max_mr > 3) {
    ldp(x14, x15, mem[x4], 16);
  }

  cmp(x8, x12); // if a0 == zero
  add(x8, x8, x11); // a0 += a_offset
  csel(x8, x12, x8, kEQ); //   a0 = zero, else += a0 + a_offset
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

  // Is there at least 4 halffloats (8 bytes)?
  subs(x0, x2, 8); // k = kc - 8
  b_lo(l4);

  align(8);
  // Main loop - 2 halffloats of A (4 bytes)
  bind(l2);
  ldr(d0, mem[x8], 8);
  ldr(q20, mem[x5], 16);
  ldr(q21, mem[x5], 16);
  if (max_mr > 1) {
    ldr(d1, mem[x13], 8);
  }
  if (max_mr > 2) {
    ldr(d2, mem[x14], 8);
  }
  if (max_mr > 3) {
    ldr(d3, mem[x15], 8);
  }
  ldr(q22, mem[x5], 16);
  ldr(q23, mem[x5], 16);
  ldr(q16, mem[x5], 16);
  ldr(q17, mem[x5], 16);
  ldr(q18, mem[x5], 16);
  ldr(q19, mem[x5], 16);
  subs(x0, x0, 8);
  fmla(v24.v8h(), v20.v8h(), v0.h()[0]);
  fmla(v25.v8h(), v21.v8h(), v0.h()[0]);
  if (max_mr > 1) {
    fmla(v26.v8h(), v20.v8h(), v1.h()[0]);
    fmla(v27.v8h(), v21.v8h(), v1.h()[0]);
  }
  if (max_mr > 2) {
    fmla(v28.v8h(), v20.v8h(), v2.h()[0]);
    fmla(v29.v8h(), v21.v8h(), v2.h()[0]);
  }
  if (max_mr > 3) {
    fmla(v30.v8h(), v20.v8h(), v3.h()[0]);
    fmla(v31.v8h(), v21.v8h(), v3.h()[0]);
  }
  fmla(v24.v8h(), v22.v8h(), v0.h()[1]);
  fmla(v25.v8h(), v23.v8h(), v0.h()[1]);
  if (max_mr > 1) {
    fmla(v26.v8h(), v22.v8h(), v1.h()[1]);
    fmla(v27.v8h(), v23.v8h(), v1.h()[1]);
  }
  if (max_mr > 2) {
    fmla(v28.v8h(), v22.v8h(), v2.h()[1]);
    fmla(v29.v8h(), v23.v8h(), v2.h()[1]);
  }
  if (max_mr > 3) {
    fmla(v30.v8h(), v22.v8h(), v3.h()[1]);
    fmla(v31.v8h(), v23.v8h(), v3.h()[1]);
  }

  fmla(v24.v8h(), v16.v8h(), v0.h()[2]);
  fmla(v25.v8h(), v17.v8h(), v0.h()[2]);
  if (max_mr > 1) {
    fmla(v26.v8h(), v16.v8h(), v1.h()[2]);
    fmla(v27.v8h(), v17.v8h(), v1.h()[2]);
  }
  if (max_mr > 2) {
    fmla(v28.v8h(), v16.v8h(), v2.h()[2]);
    fmla(v29.v8h(), v17.v8h(), v2.h()[2]);
  }
  if (max_mr > 3) {
    fmla(v30.v8h(), v16.v8h(), v3.h()[2]);
    fmla(v31.v8h(), v17.v8h(), v3.h()[2]);
  }
  fmla(v24.v8h(), v18.v8h(), v0.h()[3]);
  fmla(v25.v8h(), v19.v8h(), v0.h()[3]);
  if (max_mr > 1) {
    fmla(v26.v8h(), v18.v8h(), v1.h()[3]);
    fmla(v27.v8h(), v19.v8h(), v1.h()[3]);
  }
  if (max_mr > 2) {
    fmla(v28.v8h(), v18.v8h(), v2.h()[3]);
    fmla(v29.v8h(), v19.v8h(), v2.h()[3]);
  }
  if (max_mr > 3) {
    fmla(v30.v8h(), v18.v8h(), v3.h()[3]);
    fmla(v31.v8h(), v19.v8h(), v3.h()[3]);
  }
  b_hs(l2);

  // Is there a remainder?- 1 halffloat of A (2 bytes)
  ands(x0, x0, 7);
  b_ne(l4);

  bind(l3);
  // ks loop
  subs(x9, x9, max_mr * sizeof(void*)); // ks -= MR * sizeof(void*)
  b_hi(l1);

  // Clamp
  if (clamp_min) {
    fmax(v24.v8h(), v24.v8h(), v4.v8h());
    fmax(v25.v8h(), v25.v8h(), v4.v8h());
    if (max_mr > 1) {
      fmax(v26.v8h(), v26.v8h(), v4.v8h());
      fmax(v27.v8h(), v27.v8h(), v4.v8h());
    }
    if (max_mr > 2) {
      fmax(v28.v8h(), v28.v8h(), v4.v8h());
      fmax(v29.v8h(), v29.v8h(), v4.v8h());
    }
    if (max_mr > 3) {
      fmax(v30.v8h(), v30.v8h(), v4.v8h());
      fmax(v31.v8h(), v31.v8h(), v4.v8h());
    }
  }
  if (clamp_max) {
    fmin(v24.v8h(), v24.v8h(), v5.v8h());
    fmin(v25.v8h(), v25.v8h(), v5.v8h());
    if (max_mr > 1) {
      fmin(v26.v8h(), v26.v8h(), v5.v8h());
      fmin(v27.v8h(), v27.v8h(), v5.v8h());
    }
    if (max_mr > 2) {
      fmin(v28.v8h(), v28.v8h(), v5.v8h());
      fmin(v29.v8h(), v29.v8h(), v5.v8h());
    }
    if (max_mr > 3) {
      fmin(v30.v8h(), v30.v8h(), v5.v8h());
      fmin(v31.v8h(), v31.v8h(), v5.v8h());
    }
  }

  // Store full 4 x 16
  subs(x1, x1, 16);
  b_lo(l6);

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
  ret();


  // Remainder- 1 to 3 halffloats of A (2 to 6 bytes)
  bind(l4);
  tbz(x0, 2, l5);
  ldr(s0, mem[x8], 4);
  ldr(q20, mem[x5], 16);
  ldr(q21, mem[x5], 16);
  if (max_mr > 1) {
    ldr(s1, mem[x13], 4);
  }
  if (max_mr > 2) {
    ldr(s2, mem[x14], 4);
  }
  if (max_mr > 3) {
    ldr(s3, mem[x15], 4);
  }
  ldr(q22, mem[x5], 16);
  ldr(q23, mem[x5], 16);
  fmla(v24.v8h(), v20.v8h(), v0.h()[0]);
  fmla(v25.v8h(), v21.v8h(), v0.h()[0]);
  if (max_mr > 1) {
    fmla(v26.v8h(), v20.v8h(), v1.h()[0]);
    fmla(v27.v8h(), v21.v8h(), v1.h()[0]);
  }
  if (max_mr > 2) {
    fmla(v28.v8h(), v20.v8h(), v2.h()[0]);
    fmla(v29.v8h(), v21.v8h(), v2.h()[0]);
  }
  if (max_mr > 3) {
    fmla(v30.v8h(), v20.v8h(), v3.h()[0]);
    fmla(v31.v8h(), v21.v8h(), v3.h()[0]);
  }
  fmla(v24.v8h(), v22.v8h(), v0.h()[1]);
  fmla(v25.v8h(), v23.v8h(), v0.h()[1]);
  if (max_mr > 1) {
    fmla(v26.v8h(), v22.v8h(), v1.h()[1]);
    fmla(v27.v8h(), v23.v8h(), v1.h()[1]);
  }
  if (max_mr > 2) {
    fmla(v28.v8h(), v22.v8h(), v2.h()[1]);
    fmla(v29.v8h(), v23.v8h(), v2.h()[1]);
  }
  if (max_mr > 3) {
    fmla(v30.v8h(), v22.v8h(), v3.h()[1]);
    fmla(v31.v8h(), v23.v8h(), v3.h()[1]);
  }
  tbz(x0, 1, l3);

  bind(l5);
  ldr(h0, mem[x8], 2);
  ldr(q20, mem[x5], 16);
  ldr(q21, mem[x5], 16);
  if (max_mr > 1) {
    ldr(h1, mem[x13], 2);
  }
  if (max_mr > 2) {
    ldr(h2, mem[x14], 2);
  }
  if (max_mr > 3) {
    ldr(h3, mem[x15], 2);
  }
  fmla(v24.v8h(), v20.v8h(), v0.h()[0]);
  fmla(v25.v8h(), v21.v8h(), v0.h()[0]);
  if (max_mr > 1) {
    fmla(v26.v8h(), v20.v8h(), v1.h()[0]);
    fmla(v27.v8h(), v21.v8h(), v1.h()[0]);
  }
  if (max_mr > 2) {
    fmla(v28.v8h(), v20.v8h(), v2.h()[0]);
    fmla(v29.v8h(), v21.v8h(), v2.h()[0]);
  }
  if (max_mr > 3) {
    fmla(v30.v8h(), v20.v8h(), v3.h()[0]);
    fmla(v31.v8h(), v21.v8h(), v3.h()[0]);
  }
  b(l3);

  // Store odd width
  bind(l6);
  tbz(x1, 3, l7);
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

  bind(l7);
  tbz(x1, 2, l8);
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

  bind(l8);
  tbz(x1, 1, l9);
  if (max_mr > 3) {
    str(s30, mem[x7], 4);
  }
  if (max_mr > 2) {
    str(s28, mem[x17], 4);
  }
  if (max_mr > 3) {
    dup(s30, v30.s()[1]);
  }
  if (max_mr > 2) {
    dup(s28, v28.s()[1]);
  }
  if (max_mr > 1) {
    str(s26, mem[x16], 4);
  }
  str(s24, mem[x6], 4);
  if (max_mr > 1) {
    dup(s26, v26.s()[1]);
  }
  dup(s24, v24.s()[1]);
  bind(l9);
  tbz(x1, 0, l10);
  if (max_mr > 3) {
    str(h30, mem[x7]);
  }
  if (max_mr > 2) {
    str(h28, mem[x17]);
  }
  if (max_mr > 1) {
    str(h26, mem[x16]);
  }
  str(h24, mem[x6]);
  bind(l10);
  ret();

  align(16, AlignInstruction::kHlt);
}
}  // namespace
}  // namespace aarch64
}  // namespace xnnpack

xnn_status_t xnn_generate_f16_igemm_ukernel_4x16__aarch64_neonfp16arith_ld64(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
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
