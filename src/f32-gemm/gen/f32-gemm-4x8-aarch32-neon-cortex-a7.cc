// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <limits>

#include <xnnpack.h>
#include <xnnpack/aarch32-assembler.h>
#include <xnnpack/gemm.h>
#include <xnnpack/memory.h>
#include <xnnpack/microparams.h>
#include <xnnpack/post-operation.h>

namespace xnnpack {
namespace aarch32 {
namespace {
class Generator : public MacroAssembler {
  using MacroAssembler::MacroAssembler;

 public:
  void generate(size_t max_mr, size_t nc_mod_nr, size_t kc, const jit_gemm_params* jit_gemm_params);
  void perform_post_operations(size_t max_mr, size_t num_post_operations, const xnn_post_operation* post_operations);
};


// void xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7(
//     size_t mr,                            r0
//     size_t nc,                            r1
//     size_t kc,                            r2 -> r5
//     const uint8_t*restrict a,             r3
//     size_t a_stride,          sp + 96  -> (r7)
//     const void*restrict w,    sp + 100 -> r9
//     uint8_t*restrict c,       sp + 104 -> r11
//     size_t cm_stride,         sp + 108 -> (r6)
//     size_t cn_stride,         sp + 112 -> r7
//     const union xnn_f32_minmax_params params)  sp + 116 -> (r5)

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// A0   r3  d0
// A1  r12  d1
// A2  r10  d2
// A3   r0  d3
// B    r9  d8,  d9, d10, d11
// B       d12, d13, d14, d15
// C0  r11 d16-d17  q8  d18-d19  q9
// C1   r4 d20-d21 q10  d22-d23 q11
// C2   r8 d24-d25 q12  d26-d27 q13
// C3   r6 d28-d29 q14  d30-d31 q15
// clamp  (r5) d4 d5 d6 d7

// Converted from: src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch32-neon-cortex-a7.S
void Generator::generate(size_t max_mr, size_t nc_mod_nr, size_t kc, const jit_gemm_params* jit_gemm_params)
{
  assert(max_mr <= 4);
  assert(nc_mod_nr < 8);
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
  // Push 96 bytes
  push({r4, r5, r6, r7, r8, r9, r10, r11}); // 32
  vpush({d8-d15}); // +64 = 96

  ldr(r7, mem[sp, 96]); // a_stride
  ldr(r11, mem[sp, 104]); // c
  ldr(r6, mem[sp, 108]); // cm_stride
  ldr(r9, mem[sp, 100]); // w
  ldr(r5, mem[sp, 116]); // params

  // Clamp A and C pointers
  if (max_mr > 1) {
    cmp(r0, 2); // if mr >= 2
    add(r12, r3, r7); //   a1 = a0 + a_stride
    add(r4, r11, r6); //   c1 = c0 + cm_stride
    movlo(r12, r3); // a1
    movlo(r4, r11); // c1
  }
  if (max_mr > 2) {
    // if mr > 2
    add(r10, r12, r7); //   a2 = a1 + a_stride
    add(r8, r4, r6); //   c2 = c1 + cm_stride
    movls(r10, r12); // a2
    movls(r8, r4); // c2
  }

  if (max_mr > 3) {
    cmp(r0, 4); // if mr >=4
    add(r0, r10, r7); //   a3 = a2 + a_stride
    add(r6, r8, r6); //   c3 = c2 + cm_stride
    movlo(r0, r10); // a3
    movlo(r6, r8); // c3
  }

  // Load min/max values
  if (clamp_min || clamp_max) {
    vld1r_32({d4, d5}, mem[r5]++);
  }
  ldr(r7, mem[sp, 112]); // cn_stride
  if (clamp_min || clamp_max) {
    vld1r_32({d6, d7}, mem[r5]);
  }

  bind(l0);
  // Load initial bias from w into accumulators
  vldm(mem[r9]++, {d16-d19}); // Bias
  subs(r5, r2, 8);
  if (max_mr > 1) {
    vmov(q10, q8);
    vmov(q11, q9);
  }
  if (max_mr > 2) {
    vmov(q12, q8);
    vmov(q13, q9);
  }
  if (max_mr > 3) {
    vmov(q14, q8);
    vmov(q15, q9);
  }

  blo(l3); // less than 2 channels?

  // Main loop - 2 floats of A (8 bytes)
  bind(l1);
  vld1_32({d0}, mem[r3]++); // A0
  vldm(mem[r9]++, {d8-d11}); // B0
  if (max_mr > 1) {
    vld1_32({d1}, mem[r12]++); // A1
  }
  if (max_mr > 2) {
    vld1_32({d2}, mem[r10]++); // A2
  }
  if (max_mr > 3) {
    vld1_32({d3}, mem[r0]++); // A3
  }
  vldm(mem[r9]++, {d12-d15}); // B1
  vmla_f32(q8, q4, d0[0]);
  vmla_f32(q9, q5, d0[0]);
  if (max_mr > 1) {
    vmla_f32(q10, q4, d1[0]);
  }
  if (max_mr > 2) {
    vmla_f32(q13, q5, d2[0]);
  }
  if (max_mr > 1) {
    vmla_f32(q11, q5, d1[0]);
  }
  if (max_mr > 2) {
    vmla_f32(q12, q4, d2[0]);
  }
  if (max_mr > 3) {
    vmla_f32(q14, q4, d3[0]);
    vmla_f32(q15, q5, d3[0]);
  }
  vmla_f32(q8, q6, d0[1]);
  vmla_f32(q9, q7, d0[1]);
  if (max_mr > 1) {
    vmla_f32(q10, q6, d1[1]);
    vmla_f32(q11, q7, d1[1]);
  }
  subs(r5, r5, 8);
  if (max_mr > 2) {
    vmla_f32(q12, q6, d2[1]);
    vmla_f32(q13, q7, d2[1]);
  }
  if (max_mr > 3) {
    vmla_f32(q14, q6, d3[1]);
    vmla_f32(q15, q7, d3[1]);
  }
  bhs(l1);

  // Is there a remainder?- 1 float of A (4 bytes)
  tst(r5, 4);
  bne(l3);

  bind(l2);
  // Clamp
  if (clamp_min) {
    vmax_f32(q8, q8, q2);
  }
  subs(r1, r1, 8);
  if (clamp_min) {
    vmax_f32(q9, q9, q2);
    if (max_mr > 1) {
      vmax_f32(q10, q10, q2);
      vmax_f32(q11, q11, q2);
    }
    if (max_mr > 2) {
      vmax_f32(q12, q12, q2);
      vmax_f32(q13, q13, q2);
    }
    if (max_mr > 3) {
      vmax_f32(q14, q14, q2);
      vmax_f32(q15, q15, q2);
    }
  }
  if (clamp_max) {
    vmin_f32(q8, q8, q3);
    vmin_f32(q9, q9, q3);
    if (max_mr > 1) {
      vmin_f32(q10, q10, q3);
      vmin_f32(q11, q11, q3);
    }
    if (max_mr > 2) {
      vmin_f32(q12, q12, q3);
      vmin_f32(q13, q13, q3);
    }
    if (max_mr > 3) {
      vmin_f32(q14, q14, q3);
      vmin_f32(q15, q15, q3);
    }
  }
  perform_post_operations(max_mr, num_post_operations, post_operations);

  // Store full 4 x 8
  blo(l4);
  vst1_32({d16-d19}, mem[r11], r7);
  if (max_mr > 3) {
    sub(r0, r0, r2);
  }
  if (max_mr > 1) {
    vst1_32({d20-d23}, mem[r4], r7);
  }
  if (max_mr > 2) {
    sub(r10, r10, r2);
    vst1_32({d24-d27}, mem[r8], r7);
  }
  if (max_mr > 1) {
    sub(r12, r12, r2);
  }
  if (max_mr > 3) {
    vst1_32({d28-d31}, mem[r6], r7);
  }
  sub(r3, r3, r2);
  bhi(l0);

  vpop({d8-d15});
  pop({r4, r5, r6, r7, r8, r9, r10, r11});
  bx(lr);

  bind(l3);
  // Remainder- 1 float of A (4 bytes)
  vldm(mem[r3]++, {s0}); // A0
  vldm(mem[r9]++, {d8-d11}); // B0
  if (max_mr > 1) {
    vldm(mem[r12]++, {s2}); // A1
  }
  if (max_mr > 2) {
    vldm(mem[r10]++, {s4}); // A2
  }
  if (max_mr > 3) {
    vldm(mem[r0]++, {s6}); // A3
  }
  vmla_f32(q8, q4, d0[0]);
  vmla_f32(q9, q5, d0[0]);
  if (max_mr > 1) {
    vmla_f32(q10, q4, d1[0]);
    vmla_f32(q11, q5, d1[0]);
  }
  if (max_mr > 2) {
    vmla_f32(q12, q4, d2[0]);
    vmla_f32(q13, q5, d2[0]);
  }
  if (max_mr > 3) {
    vmla_f32(q14, q4, d3[0]);
    vmla_f32(q15, q5, d3[0]);
  }
  b(l2);

  // Store odd width
  bind(l4);
  tst(r1, 4);
  beq(l5);
  vst1_32({d16-d17}, mem[r11]++);
  if (max_mr > 1) {
    vst1_32({d20-d21}, mem[r4]++);
  }
  vmov(q8, q9);
  if (max_mr > 1) {
    vmov(q10, q11);
  }
  if (max_mr > 2) {
    vst1_32({d24-d25}, mem[r8]++);
  }
  if (max_mr > 3) {
    vst1_32({d28-d29}, mem[r6]++);
  }
  if (max_mr > 2) {
    vmov(q12, q13);
  }
  if (max_mr > 3) {
    vmov(q14, q15);
  }

  bind(l5);
  tst(r1, 2);
  beq(l6);
  vst1_32({d16}, mem[r11]++);
  if (max_mr > 1) {
    vst1_32({d20}, mem[r4]++);
  }
  vmov(d16, d17);
  if (max_mr > 1) {
    vmov(d20, d21);
  }
  if (max_mr > 2) {
    vst1_32({d24}, mem[r8]++);
  }
  if (max_mr > 3) {
    vst1_32({d28}, mem[r6]++);
  }
  if (max_mr > 2) {
    vmov(d24, d25);
  }
  if (max_mr > 3) {
    vmov(d28, d29);
  }

  bind(l6);
  tst(r1, 1);
  beq(l7);
  vst1_32({d16[0]}, mem[r11]);
  if (max_mr > 1) {
    vst1_32({d20[0]}, mem[r4]);
  }
  if (max_mr > 2) {
    vst1_32({d24[0]}, mem[r8]);
  }
  if (max_mr > 3) {
    vst1_32({d28[0]}, mem[r6]);
  }

  bind(l7);
  vpop({d8-d15});
  pop({r4, r5, r6, r7, r8, r9, r10, r11});
  bx(lr);

  align(16);
}

void Generator::perform_post_operations(
  size_t max_mr,
  size_t num_post_operations,
  const xnn_post_operation* post_operations)
{
  ldr(r5, mem[sp, 116]);  // params
  for (size_t i = 0; i < num_post_operations; i++) {
    switch (post_operations[i].op_type) {
      case xnn_post_operation_type_hardswish: {
        const auto sixth = q0;
        const auto three = q1;
        const auto six = q2;
        const auto zero = q3;
        vld3r_32({sixth.low(), three.low(), six.low()}, mem[r5]++);
        vmov(zero, 0);
        vmov(three.high(), three.low());
        vmov(six.high(), six.low());
        const QRegister accs[] = {q8, q9, q10, q11, q12, q13, q14, q15};
        const QRegister tmps[] = {q4, q5, q6, q7};
        f32_hardswish(sixth, three, six, zero, &accs[0], XNN_COUNT_OF(accs), &tmps[0], XNN_COUNT_OF(tmps));
        break;
      }
      default:
        XNN_UNREACHABLE;
    }
  }
}

}  // namespace
}  // namespace aarch32
}  // namespace xnnpack

xnn_status_t xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a7(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, const void* params) {
  using namespace xnnpack::aarch32;
  Generator g(code);
  assert(params != nullptr);
  g.generate(max_mr, nc_mod_nr, kc, static_cast<const jit_gemm_params*>(params));
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
