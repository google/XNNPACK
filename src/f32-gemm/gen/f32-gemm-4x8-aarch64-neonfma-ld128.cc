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
  void generate(size_t max_mr, size_t nc_mod_nr, size_t kc, const jit_gemm_params* jit_gemm_params);
  void perform_post_operations(size_t max_mr, size_t num_post_operations, const xnn_post_operation* post_operations);
};

// void xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128(
//     size_t mr,                x0
//     size_t nc,                x1
//     size_t kc,                x2 / x0
//     const float* a,           x3
//     size_t a_stride,          x4
//     const void* w,            x5
//     float* c,                 x6
//     size_t cm_stride,         x7
//     size_t cn_stride,         [sp] -> x14
//     const xnn_f32_minmax_params* params)  [sp + 8] -> (x8)

// d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x3  v0
// A1  x11 v1
// A2  x12 v2
// A3  x4  v3
// B   x5  v20 v24 v21 v25 v22 v26 v23 v27
// C0  x6  v16 v17
// C1  x9  v18 v19
// C2  x10 v28 v29
// C3  x7  v30 v31
// Clamp v4 v5

// Converted from: src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-ld128.S
void Generator::generate(size_t max_mr, size_t nc_mod_nr, size_t kc, const jit_gemm_params* jit_gemm_params)
{
  assert(max_mr <= 4);
  assert(nc_mod_nr < 8 || nc_mod_nr == SIZE_MAX);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8;
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

  // Clamp A and C pointers
  if (max_mr > 1) {
    cmp(x0, 2); // if mr < 2
    add(x11, x3, x4); // a1 = a0 + a_stride
    add(x9, x6, x7); // c1 = c0 + cm_stride
    csel(x11, x3, x11, kLO); //   a1 = a0
    csel(x9, x6, x9, kLO); //   c1 = c0
  }

  if (max_mr > 2) {
    add(x12, x11, x4); // a2 = a1 + a_stride
    add(x10, x9, x7); // c2 = c1 + cm_stride
    // if mr <= 2
    csel(x12, x11, x12, kLS); //   a2 = a1
    csel(x10, x9, x10, kLS); //   c2 = c1
  }

  if (max_mr > 3) {
    cmp(x0, 4); // if mr < 4
    add(x4, x12, x4); // a3 = a2 + a_stride
    add(x7, x10, x7); // c3 = c2 + cm_stride
    csel(x4, x12, x4, kLO); //   a3 = a2
    csel(x7, x10, x7, kLO); //   c3 = c2
  }
  bind(l0);
  // Load initial bias from w into accumulators
  ldp(q16, q17, mem[x5], 32);
  if (max_mr > 1) {
    mov(v18.v16b(), v16.v16b());
    mov(v19.v16b(), v17.v16b());
  }
  if (max_mr > 2) {
    mov(v28.v16b(), v16.v16b());
    mov(v29.v16b(), v17.v16b());
  }
  if (max_mr > 3) {
    mov(v30.v16b(), v16.v16b());
    mov(v31.v16b(), v17.v16b());
  }

  // Is there at least 4 floats (16 bytes)?
  subs(x0, x2, 16); // k = kc - 16
  b_lo(l3);

  // Main loop - 4 floats of A (16 bytes)
  bind(l1);
  ldr(q0, mem[x3], 16);
  ldp(q20, q24, mem[x5], 32); // 8 F32 weights
  if (max_mr > 1) {
    ldr(q1, mem[x11], 16);
  }
  if (max_mr > 2) {
    ldr(q2, mem[x12], 16);
  }
  if (max_mr > 3) {
    ldr(q3, mem[x4], 16);
  }
  fmla(v16.v4s(), v20.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v18.v4s(), v20.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v20.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v20.v4s(), v3.s()[0]);
  }
  ldp(q21, q25, mem[x5], 32); // 8 F32 weights
  fmla(v17.v4s(), v24.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v19.v4s(), v24.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v29.v4s(), v24.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v31.v4s(), v24.v4s(), v3.s()[0]);
  }
  ldp(q22, q26, mem[x5], 32); // 8 F32 weights
  fmla(v16.v4s(), v21.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v18.v4s(), v21.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v21.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v21.v4s(), v3.s()[1]);
  }
  ldp(q23, q27, mem[x5], 32); // 8 F32 weights
  fmla(v17.v4s(), v25.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v19.v4s(), v25.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v29.v4s(), v25.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v31.v4s(), v25.v4s(), v3.s()[1]);
  }
  fmla(v16.v4s(), v22.v4s(), v0.s()[2]);
  if (max_mr > 1) {
    fmla(v18.v4s(), v22.v4s(), v1.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v22.v4s(), v2.s()[2]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v22.v4s(), v3.s()[2]);
  }
  fmla(v17.v4s(), v26.v4s(), v0.s()[2]);
  if (max_mr > 1) {
    fmla(v19.v4s(), v26.v4s(), v1.s()[2]);
  }
  if (max_mr > 2) {
    fmla(v29.v4s(), v26.v4s(), v2.s()[2]);
  }
  if (max_mr > 3) {
    fmla(v31.v4s(), v26.v4s(), v3.s()[2]);
  }
  fmla(v16.v4s(), v23.v4s(), v0.s()[3]);
  if (max_mr > 1) {
    fmla(v18.v4s(), v23.v4s(), v1.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v23.v4s(), v2.s()[3]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v23.v4s(), v3.s()[3]);
  }
  subs(x0, x0, 16);
  fmla(v17.v4s(), v27.v4s(), v0.s()[3]);
  if (max_mr > 1) {
    fmla(v19.v4s(), v27.v4s(), v1.s()[3]);
  }
  if (max_mr > 2) {
    fmla(v29.v4s(), v27.v4s(), v2.s()[3]);
  }
  if (max_mr > 3) {
    fmla(v31.v4s(), v27.v4s(), v3.s()[3]);
  }
  b_hs(l1);

  tst(x0, 15);
  b_ne(l3);

  bind(l2);
  // Clamp
  if (clamp_min) {
    fmax(v16.v4s(), v16.v4s(), v4.v4s());
  }
  subs(x1, x1, 8);
  if (clamp_min) {
    fmax(v17.v4s(), v17.v4s(), v4.v4s());
    if (max_mr > 1) {
      fmax(v18.v4s(), v18.v4s(), v4.v4s());
      fmax(v19.v4s(), v19.v4s(), v4.v4s());
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
    fmin(v16.v4s(), v16.v4s(), v5.v4s());
    fmin(v17.v4s(), v17.v4s(), v5.v4s());
    if (max_mr > 1) {
      fmin(v18.v4s(), v18.v4s(), v5.v4s());
      fmin(v19.v4s(), v19.v4s(), v5.v4s());
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
  b_lo(l5);


  st1({v16.v16b(), v17.v16b()}, mem[x6], x14);
  sub(x3, x3, x2); // a0 -= kc
  if (max_mr > 1) {
    st1({v18.v16b(), v19.v16b()}, mem[x9], x14);
    sub(x11, x11, x2); // a1 -= kc
  }
  if (max_mr > 2) {
    st1({v28.v16b(), v29.v16b()}, mem[x10], x14);
    sub(x12, x12, x2); // a2 -= kc
  }
  if (max_mr > 3) {
    st1({v30.v16b(), v31.v16b()}, mem[x7], x14);
    sub(x4, x4, x2); // a3 -= kc
  }

  b_hi(l0);
  ret();

  // Remainder- 2 floats of A (8 bytes)
  bind(l3);
  // Is there a remainder?- 2 floats of A (8 bytes)
  tbz(x0, 3, l4);

  // Remainder- 2 floats of A (8 bytes)
  ldp(q20, q24, mem[x5], 32); // 16 F32 weights
  ldp(q21, q25, mem[x5], 32);
  ldr(d0, mem[x3], 8);
  if (max_mr > 1) {
    ldr(d1, mem[x11], 8);
  }
  if (max_mr > 2) {
    ldr(d2, mem[x12], 8);
  }
  if (max_mr > 3) {
    ldr(d3, mem[x4], 8);
  }
  fmla(v16.v4s(), v20.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v18.v4s(), v20.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v20.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v20.v4s(), v3.s()[0]);
  }
  fmla(v17.v4s(), v24.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v19.v4s(), v24.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v29.v4s(), v24.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v31.v4s(), v24.v4s(), v3.s()[0]);
  }
  fmla(v16.v4s(), v21.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v18.v4s(), v21.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v21.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v21.v4s(), v3.s()[1]);
  }
  fmla(v17.v4s(), v25.v4s(), v0.s()[1]);
  if (max_mr > 1) {
    fmla(v19.v4s(), v25.v4s(), v1.s()[1]);
  }
  if (max_mr > 2) {
    fmla(v29.v4s(), v25.v4s(), v2.s()[1]);
  }
  if (max_mr > 3) {
    fmla(v31.v4s(), v25.v4s(), v3.s()[1]);
  }

  // Is there a remainder?- 1 float of A (4 bytes)
  tbz(x0, 2, l2);

  // Remainder- 1 float of A (4 bytes)
  bind(l4);
  // Remainder- 2 floats of A (8 bytes)
  ldp(q20, q24, mem[x5], 32); // 8 F32 weights
  ldr(s0, mem[x3], 4);
  if (max_mr > 1) {
    ldr(s1, mem[x11], 4);
  }
  if (max_mr > 2) {
    ldr(s2, mem[x12], 4);
  }
  if (max_mr > 3) {
    ldr(s3, mem[x4], 4);
  }
  fmla(v16.v4s(), v20.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v18.v4s(), v20.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v28.v4s(), v20.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v30.v4s(), v20.v4s(), v3.s()[0]);
  }
  fmla(v17.v4s(), v24.v4s(), v0.s()[0]);
  if (max_mr > 1) {
    fmla(v19.v4s(), v24.v4s(), v1.s()[0]);
  }
  if (max_mr > 2) {
    fmla(v29.v4s(), v24.v4s(), v2.s()[0]);
  }
  if (max_mr > 3) {
    fmla(v31.v4s(), v24.v4s(), v3.s()[0]);
  }
  b(l2);

  // Store odd width
  bind(l5);
  tbz(x1, 2, l6);
  str(q16, mem[x6], 16);
  mov(v16.v16b(), v17.v16b());
  if (max_mr > 1) {
    str(q18, mem[x9], 16);
    mov(v18.v16b(), v19.v16b());
  }
  if (max_mr > 2) {
    str(q28, mem[x10], 16);
    mov(v28.v16b(), v29.v16b());
  }
  if (max_mr > 3) {
    str(q30, mem[x7], 16);
    mov(v30.v16b(), v31.v16b());
  }

  bind(l6);
  tbz(x1, 1, l7);
  str(d16, mem[x6], 8);
  if (max_mr > 1) {
    str(d18, mem[x9], 8);
  }
  dup(d16, v16.d()[1]);
  if (max_mr > 1) {
    dup(d18, v18.d()[1]);
  }
  if (max_mr > 2) {
    str(d28, mem[x10], 8);
  }
  if (max_mr > 3) {
    str(d30, mem[x7], 8);
  }
  if (max_mr > 2) {
    dup(d28, v28.d()[1]);
  }
  if (max_mr > 3) {
    dup(d30, v30.d()[1]);
  }

  bind(l7);
  tbz(x1, 0, l8);
  str(s16, mem[x6]);
  if (max_mr > 1) {
    str(s18, mem[x9]);
  }
  if (max_mr > 2) {
    str(s28, mem[x10]);
  }
  if (max_mr > 3) {
    str(s30, mem[x7]);
  }

  bind(l8);
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
          v18.v4s(), v19.v4s(),
          v28.v4s(), v29.v4s(),
          v30.v4s(), v31.v4s(),
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

xnn_status_t xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_ld128(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, const void* params) {
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
