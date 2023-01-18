// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <limits>

#include <xnnpack.h>
#include <xnnpack/aarch32-assembler.h>
#include <xnnpack/igemm.h>
#include <xnnpack/memory.h>
#include <xnnpack/microparams.h>
#include <xnnpack/post-operation.h>

namespace xnnpack {
namespace aarch32 {
namespace {
class Generator : public MacroAssembler {
  using MacroAssembler::MacroAssembler;

 public:
  void generate(bool prefetch, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const jit_gemm_params* jit_gemm_params);
  void perform_post_operations(size_t max_mr, size_t num_post_operations, const xnn_post_operation* post_operations);
};


// void xnn_f32_igemm_minmax_ukernel_1x8__asm_aarch32_neon_prfm_cortex_a53(
//     size_t mr,                            (unused)
//     size_t nc,                            r1
//     size_t kc,                            r2 -> r0
//     size_t ks,                            (r3) -> sp + 4 -> r14
//     const float**restrict a,   sp + 24 -> r4
//     const void*restrict w,     sp + 28 -> r9
//     uint8_t*restrict c,        sp + 32 -> r12
//     size_t cm_stride,          sp + 36 -> (unused)
//     size_t cn_stride,          sp + 40 -> (r7)
//     size_t a_offset,           sp + 44 -> (r0)
//     const float* zero,         sp + 48 -> (r7)
//     minmax_params*params,      sp + 52 -> (r0)

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// A0   r3  d0
// B    r9 d24, d25, d26, d27
// B       d28, d29, d30, d31
// C0  r12 d16-d17  q8  d18-d19  q9
// clamp  (r0) d4 d5 d6 d7

// Converted from: src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch32-neon-prfm-cortex-a53.S
void Generator::generate(bool prefetch, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const jit_gemm_params* jit_gemm_params)
{
  assert(max_mr <= 1);
  assert(nc_mod_nr < 8);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);

  Label l0, l1, l2, l3, l4, l5, l6, l7, l8;
  const size_t num_post_operations = jit_gemm_params->num_post_operations;
  const xnn_post_operation* post_operations = jit_gemm_params->post_operations;
  const float min = jit_gemm_params->f32_minmax.min;
  const float max = jit_gemm_params->f32_minmax.max;
  const bool clamp_min = min != -std::numeric_limits<float>::infinity();
  const bool clamp_max = max != +std::numeric_limits<float>::infinity();
  assert(num_post_operations == 0 || (!clamp_min && !clamp_max));
  // Push 24 bytes
  // r3 is ks
  push({r3, r4, r7, r9, lr}); // 20
  sub(sp, sp, 4); // +4 = 24

  ldr(r4, mem[sp, 24]); // a
  ldr(r9, mem[sp, 28]); // w
  ldr(r12, mem[sp, 32]); // c
  ldr(r0, mem[sp, 52]); // params
  mov(r14, r3); // p = ks

  // Load min/max values
  if (clamp_min || clamp_max) {
    vld1r_32({d4, d5}, mem[r0]++);
    vld1r_32({d6, d7}, mem[r0]);
  }

  bind(l0);
  // Load initial bias from w into accumulators
  vldm(mem[r9]++, {d16-d19}); // Bias
  vmov_i32(q10, 0); // second set of C for pipelining VMLA
  vmov_i32(q11, 0);

  if (prefetch) {
    pld(mem[r9, 0]); // Prefetch B
    pld(mem[r9, 64]);
    pld(mem[r9, 128]);
    pld(mem[r9, 192]);
    pld(mem[r9, 256]);
    pld(mem[r9, 320]);
    pld(mem[r9, 384]);
    pld(mem[r9, 448]);
    pld(mem[r9, 512]);
    pld(mem[r9, 576]);
  }
  bind(l1);
  // Load next A pointer
  ldr(r3, mem[r4], 4);

  // Add a_offset
  ldr(r0, mem[sp, 44]); // a_offset
  ldr(r7, mem[sp, 48]); // zero
  cmp(r3, r7); // if a0 == zero
  add(r3, r3, r0); // a0 += a_offset
  moveq(r3, r7); //   a0 = zero, else += a0 + a_offset

  subs(r0, r2, 8); // kc - 8

  if (prefetch) {
    pld(mem[r3, 0]); // Prefetch A
    pld(mem[r3, 64]);
  }

  blo(l4); // less than 2 channels?

  // Main loop - 2 floats of A (8 bytes)
  bind(l2);
  vldm(mem[r9]++, {d24-d27}); // B0
  vld1_32({d0}, mem[r3]++); // A0
  vldm(mem[r9]++, {d28-d31}); // B1

  vmla_f32(q8, q12, d0[0]);
  vmla_f32(q9, q13, d0[0]);
  if (prefetch) {
    pld(mem[r9, 576]); // Prefetch B
  }
  vmla_f32(q10, q14, d0[1]);
  vmla_f32(q11, q15, d0[1]);
  subs(r0, r0, 8);
  if (prefetch) {
    pld(mem[r3, 128]); // Prefetch A0
  }
  bhs(l2);

  // Is there a remainder?- 1 float of A (4 bytes)
  tst(r0, 4);
  bne(l4);

  bind(l3);
  // ks loop
  subs(r14, r14, max_mr * sizeof(void*)); // ks -= MR * sizeof(void*)
  bhi(l1);

  ldr(r7, mem[sp, 40]); // cn_stride
  vadd_f32(q8, q8, q10);
  ldr(r14, mem[sp, 4]); // p = ks
  vadd_f32(q9, q9, q11);

  // Clamp
  if (clamp_min) {
    vmax_f32(q8, q8, q2);
  }
  subs(r1, r1, 8);
  if (clamp_min) {
    vmax_f32(q9, q9, q2);
  }
  if (clamp_max) {
    vmin_f32(q8, q8, q3);
    vmin_f32(q9, q9, q3);
  }
  perform_post_operations(max_mr, num_post_operations, post_operations);

  // Store full 1 x 8
  blo(l5);
  vst1_32({d16-d19}, mem[r12], r7);
  sub(r4, r4, r14); // a -= ks
  bhi(l0);

  add(sp, sp, 8); // skip pad, r3
  pop({r4, r7, r9, pc});

  bind(l4);
  // Remainder- 1 float of A (4 bytes)
  vldm(mem[r3]++, {s0}); // A0
  vldm(mem[r9]++, {d24-d27}); // B0
  vmla_f32(q8, q12, d0[0]);
  vmla_f32(q9, q13, d0[0]);
  b(l3);

  // Store odd width
  bind(l5);
  tst(r1, 4);
  beq(l6);
  vst1_32({d16-d17}, mem[r12]++);
  vmov(q8, q9);

  bind(l6);
  tst(r1, 2);
  beq(l7);
  vst1_32({d16}, mem[r12]++);
  vmov(d16, d17);

  bind(l7);
  tst(r1, 1);
  beq(l8);
  vst1_32({d16[0]}, mem[r12]++);

  bind(l8);
  add(sp, sp, 8); // skip pad, r3
  pop({r4, r7, r9, pc});

  align(16);
}

void Generator::perform_post_operations(
  size_t max_mr,
  size_t num_post_operations,
  const xnn_post_operation* post_operations)
{
  ldr(r0, mem[sp, 52]);  // params
  for (size_t i = 0; i < num_post_operations; i++) {
    switch (post_operations[i].op_type) {
      case xnn_post_operation_type_hardswish: {
        const auto sixth = q0;
        const auto three = q1;
        const auto six = q2;
        const auto zero = q3;
        vld3r_32({sixth.low(), three.low(), six.low()}, mem[r0]++);
        vmov(zero, 0);
        vmov(three.high(), three.low());
        vmov(six.high(), six.low());
        const QRegister accs[] = {q8, q9};
        const QRegister tmps[] = {q12, q13};
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

xnn_status_t xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_cortex_a53(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
  using namespace xnnpack::aarch32;
  Generator g(code);
  assert(params != nullptr);
  g.generate(false, max_mr, nc_mod_nr, kc, ks, static_cast<const jit_gemm_params*>(params));
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}

xnn_status_t xnn_generate_f32_igemm_ukernel_1x8__aarch32_neon_prfm_cortex_a53(xnn_code_buffer* code, size_t max_mr, size_t nc_mod_nr, size_t kc, size_t ks, const void* params) {
  using namespace xnnpack::aarch32;
  Generator g(code);
  assert(params != nullptr);
  g.generate(true, max_mr, nc_mod_nr, kc, ks, static_cast<const jit_gemm_params*>(params));
  g.finalize();
  if (g.error() != xnnpack::Error::kNoError) {
    return xnn_status_invalid_state;
  }
  return xnn_status_success;
}
