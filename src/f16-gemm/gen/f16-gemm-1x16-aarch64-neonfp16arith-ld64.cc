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
  void generate(size_t max_mr, size_t nc_mod_nr, size_t kc, const jit_gemm_params* jit_gemm_params);
  void perform_post_operations(size_t max_mr, size_t num_post_operations, const xnn_post_operation* post_operations);
};

// void xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64(
//     size_t mr,                (x0) - unused.  mr = 1
//     size_t nc,                x1
//     size_t kc,                x2 / x0
//     const void* restrict a,    x3
//     size_t a_stride,          (x4) - unused
//     const void* restrict w,    x5
//     void* restrict c,          x6
//     size_t cm_stride,         (x7) - unused
//     size_t cn_stride,         [sp] -> x14
//     const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])  [sp + 8] -> (x8)

// d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x8 v0
// B   x5 v24 v25 v26 v27 v28 v29 v30 v31
// C0  x6 v16 v17 v18 v19 v20 v21 v22 v23
// clamp  v4, v5

// Converted from: src/f16-gemm/gen/f16-gemm-1x16-minmax-asm-aarch64-neonfp16arith-ld64.S
void Generator::generate(size_t max_mr, size_t nc_mod_nr, size_t kc, const jit_gemm_params* jit_gemm_params)
{
  assert(max_mr <= 1);
  assert(nc_mod_nr < 16 || nc_mod_nr == SIZE_MAX);
  assert(kc != 0);
  assert(kc % sizeof(uint16_t) == 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8, l9;
  const size_t num_post_operations = jit_gemm_params->num_post_operations;
  const xnn_post_operation* post_operations = jit_gemm_params->post_operations;
  const uint16_t min = jit_gemm_params->f16_minmax.min;
  const uint16_t max = jit_gemm_params->f16_minmax.max;
  const bool clamp_min = min != UINT16_C(0xFC00);  // -Inf.
  const bool clamp_max = max != UINT16_C(0x7C00);  // Inf.
  assert(num_post_operations == 0 || (!clamp_min && !clamp_max));
  // Load cn_stride, params pointer
  ldp(x14, x8, mem[sp]);

  // Load params values
  if (clamp_min || clamp_max) {
    ld2r({v4.v8h(), v5.v8h()}, mem[x8]);
  }
  bind(l0);
  // Load initial bias from w into accumulators
  ldp(q16, q17, mem[x5], 32);
  movi(v18.v8h(), 0); // 4 sets of C for pipelining FMLA
  movi(v19.v8h(), 0);
  movi(v20.v8h(), 0);
  movi(v21.v8h(), 0);
  movi(v22.v8h(), 0);
  movi(v23.v8h(), 0);

  // Is there at least 4 halffloats (8 bytes)
  subs(x0, x2, 8); // k = kc - 8
  b_lo(l3);

  align(8);
  // Main loop - 4 halffloats of A (8 bytes)
  bind(l1);
  ldr(d0, mem[x3], 8);
  ldr(q24, mem[x5, 0]);
  ldr(q25, mem[x5, 16]);
  ldr(q26, mem[x5, 32]);
  ldr(q27, mem[x5, 48]);
  ldr(q28, mem[x5, 64]);
  ldr(q29, mem[x5, 80]);
  ldr(q30, mem[x5, 96]);
  ldr(q31, mem[x5, 112]);
  subs(x0, x0, 8);
  fmla(v16.v8h(), v24.v8h(), v0.h()[0]);
  fmla(v17.v8h(), v25.v8h(), v0.h()[0]);
  fmla(v18.v8h(), v26.v8h(), v0.h()[1]);
  fmla(v19.v8h(), v27.v8h(), v0.h()[1]);
  fmla(v20.v8h(), v28.v8h(), v0.h()[2]);
  fmla(v21.v8h(), v29.v8h(), v0.h()[2]);
  fmla(v22.v8h(), v30.v8h(), v0.h()[3]);
  fmla(v23.v8h(), v31.v8h(), v0.h()[3]);
  add(x5, x5, 128);
  b_hs(l1);

  // Is there a remainder- 1 to 3 halffloats of A (2 to 6 bytes)
  ands(x0, x0, 7);
  b_ne(l3);

  bind(l2);
  fadd(v16.v8h(), v16.v8h(), v18.v8h());
  fadd(v17.v8h(), v17.v8h(), v19.v8h());
  fadd(v20.v8h(), v20.v8h(), v22.v8h());
  fadd(v21.v8h(), v21.v8h(), v23.v8h());
  fadd(v16.v8h(), v16.v8h(), v20.v8h());
  fadd(v17.v8h(), v17.v8h(), v21.v8h());
  subs(x1, x1, 16);

  // Clamp
  if (clamp_min) {
    fmax(v16.v8h(), v16.v8h(), v4.v8h());
    fmax(v17.v8h(), v17.v8h(), v4.v8h());
  }
  if (clamp_max) {
    fmin(v16.v8h(), v16.v8h(), v5.v8h());
    fmin(v17.v8h(), v17.v8h(), v5.v8h());
  }
  perform_post_operations(max_mr, num_post_operations, post_operations);

  // Store full 1 x 16
  b_lo(l5);

  stp(q16, q17, mem[x6]);
  add(x6, x6, x14);

  sub(x3, x3, x2); // a0 -= kc

  b_hi(l0);

  ret();

  // Remainder- 1 to 3 halffloats of A (2 to 6 bytes)
  bind(l3);
  tbz(x0, 2, l4);
  ldr(s0, mem[x3], 4);
  ldr(q24, mem[x5, 0]);
  ldr(q25, mem[x5, 16]);
  ldr(q26, mem[x5, 32]);
  ldr(q27, mem[x5, 48]);
  fmla(v16.v8h(), v24.v8h(), v0.h()[0]);
  fmla(v17.v8h(), v25.v8h(), v0.h()[0]);
  fmla(v18.v8h(), v26.v8h(), v0.h()[1]);
  fmla(v19.v8h(), v27.v8h(), v0.h()[1]);
  add(x5, x5, 64);
  tbz(x0, 1, l2);

  bind(l4);
  ldr(h0, mem[x3], 2);
  ldr(q24, mem[x5, 0]);
  ldr(q25, mem[x5, 16]);
  fmla(v16.v8h(), v24.v8h(), v0.h()[0]);
  fmla(v17.v8h(), v25.v8h(), v0.h()[0]);
  add(x5, x5, 32);
  b(l2);

  // Store odd channels
  bind(l5);
  tbz(x1, 3, l6);
  str(q16, mem[x6], 16);
  mov(v16.v16b(), v17.v16b());
  bind(l6);
  tbz(x1, 2, l7);
  str(d16, mem[x6], 8);
  dup(d16, v16.d()[1]);
  bind(l7);
  tbz(x1, 1, l8);
  str(s16, mem[x6], 4);
  dup(s16, v16.s()[1]);
  bind(l8);
  tbz(x1, 0, l9);
  str(h16, mem[x6]);
  bind(l9);
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
          v16.v4s(), v17.v4s(),
        };
        const VRegister tmps[] = {v4.v4s(), v5.v4s()};
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

xnn_status_t xnn_generate_f16_gemm_ukernel_1x16__aarch64_neonfp16arith_ld64(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, const void* params) {
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
