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

// void xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64(
//     size_t mr,                (x0) - unused.  mr = 1
//     size_t nc,                x1
//     size_t kc,                x2 / x0
//     const float* a,           x3
//     size_t a_stride,          (x4) - unused
//     const void* w,            x5
//     float* c,                 x6
//     size_t cm_stride,         (x7) - unused
//     size_t cn_stride,         [sp] -> x14
//     const xnn_f32_minmax_params* params)  [sp + 8] -> (x8)

// d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x3 v0
// B   x5 v20 v21 v22 v23
// C0  x6 v16 v17
// Clamp v4 v5

// Converted from: src/f32-gemm/gen/f32-gemm-1x8-minmax-asm-aarch64-neonfma-ld64.S
void Generator::generate(size_t max_mr, size_t nc_mod_nr, size_t kc, const jit_gemm_params* jit_gemm_params)
{
  assert(max_mr <= 1);
  assert(nc_mod_nr < 8 || nc_mod_nr == SIZE_MAX);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7;
  const size_t num_post_operations = jit_gemm_params->num_post_operations;
  const xnn_post_operation* post_operations = jit_gemm_params->post_operations;
  const float min = jit_gemm_params->f32_minmax.min;
  const float max = jit_gemm_params->f32_minmax.max;
  const bool clamp_min = min != -std::numeric_limits<float>::infinity();
  const bool clamp_max = max != +std::numeric_limits<float>::infinity();
  assert(num_post_operations == 0 || (!clamp_min && !clamp_max));

  // Load cn_stride, params pointer
  ldp(x14, x8, mem[sp]);

  // Load min/max values
  if (clamp_min || clamp_max) {
    ld2r({v4.v4s(), v5.v4s()}, mem[x8]);
  }
  bind(l0);
  // Load initial bias from w into accumulators
  ldp(q16, q17, mem[x5], 32);
  subs(x0, x2, 8); // k = kc - 8
  // Is there at least 2 floats (8 bytes)
  b_lo(l3);

  // Main loop - 2 floats of A (8 bytes)
  bind(l1);
  ldr(d0, mem[x3], 8);
  ldp(q20, q21, mem[x5], 32); // 16 F32 weights
  ldp(q22, q23, mem[x5], 32);
  subs(x0, x0, 8);
  fmla(v16.v4s(), v20.v4s(), v0.s()[0]);
  fmla(v17.v4s(), v21.v4s(), v0.s()[0]);
  fmla(v16.v4s(), v22.v4s(), v0.s()[1]);
  fmla(v17.v4s(), v23.v4s(), v0.s()[1]);
  b_hs(l1);

  // Is there a remainder?- 1 float of A (4 bytes)
  tbnz(x0, 2, l3);

  bind(l2);
  subs(x1, x1, 8);

  // Clamp
  if (clamp_min) {
    fmax(v16.v4s(), v16.v4s(), v4.v4s());
    fmax(v17.v4s(), v17.v4s(), v4.v4s());
  }
  if (clamp_max) {
    fmin(v16.v4s(), v16.v4s(), v5.v4s());
    fmin(v17.v4s(), v17.v4s(), v5.v4s());
  }
  perform_post_operations(max_mr, num_post_operations, post_operations);

  // Store full 1 x 8
  b_lo(l4);

  stp(q16, q17, mem[x6]);
  add(x6, x6, x14);

  sub(x3, x3, x2); // a0 -= kc
  b_hi(l0);
  ret();

  bind(l3);
  // Remainder- 1 float of A (4 bytes)
  ldr(s0, mem[x3], 4);
  ldp(q20, q21, mem[x5], 32); // 8 F32 weights
  fmla(v16.v4s(), v20.v4s(), v0.s()[0]);
  fmla(v17.v4s(), v21.v4s(), v0.s()[0]);
  b(l2);

  // Store odd channels
  bind(l4);
  tbz(x1, 2, l5);
  str(q16, mem[x6], 16);
  mov(v16.v16b(), v17.v16b());

  bind(l5);
  tbz(x1, 1, l6);
  str(d16, mem[x6], 8);
  dup(d16, v16.d()[1]);

  bind(l6);
  tbz(x1, 0, l7);
  str(s16, mem[x6]);
  bind(l7);
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

xnn_status_t xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_ld64(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, const void* params) {
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
