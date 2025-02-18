// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_2

      # Free up GP registers.
      stp x19, x20, [sp, -64]
      stp x21, x22, [sp, -48]
      stp x23, x24, [sp, -32]
      stp x25, x26, [sp, -16]

      # Preserve callee saved q8-q15 registers.
      stp d8, d9, [sp, -128]
      stp d10, d11, [sp, -112]
      stp d12, d13, [sp, -96]
      stp d14, d15, [sp, -80]

      # Load params.
      ldr x13, [sp, 8]

      # Load min/max values.
      ld2r {v0.4s, v1.4s}, [x13]

outer_loop:
      # Initialize k counter.
      mov x20, x2

      # Initialize accumulators with the biases.
      ldp q11, q12, [x5, 0]
      add x5, x5, 32

      # Are there at least 16 bytes?
      cmp x20, 16
      blt inner_loop_tail
      sub x20, x20, 16

inner_loop:
      ldr q2, [x3], 16
      ldp q7, q8, [x5], 32
      fmla  v11.4s, v7.4s, v2.s[0]
      fmla  v12.4s, v8.4s, v2.s[0]
      ldp q7, q8, [x5], 32
      fmla  v11.4s, v7.4s, v2.s[1]
      fmla  v12.4s, v8.4s, v2.s[1]
      ldp q7, q8, [x5], 32
      fmla  v11.4s, v7.4s, v2.s[2]
      fmla  v12.4s, v8.4s, v2.s[2]
      ldp q7, q8, [x5], 32
      fmla  v11.4s, v7.4s, v2.s[3]
      fmla  v12.4s, v8.4s, v2.s[3]
      subs x20, x20, 16
      bhs inner_loop

      add x20, x20, 16
      cmp x20, 4
      blt inner_loop_end

inner_loop_tail:
      ldr s2, [x3], 4
      ldp q7, q8, [x5], 32
      fmla  v11.4s, v7.4s, v2.s[0]
      fmla  v12.4s, v8.4s, v2.s[0]
      subs x20, x20, 4
      bne inner_loop_tail

inner_loop_end:
      # Min/max clamping.
      fmin  v11.4s, v1.4s, v11.4s
      fmin  v12.4s, v1.4s, v12.4s
      fmax  v11.4s, v0.4s, v11.4s
      fmax  v12.4s, v0.4s, v12.4s

      # Check whether full or partial store.
      cmp x1, 8
      b.lo tail_4
      stp  q11, q12, [x6], 32
      sub x3, x3, x2

      sub x1, x1, 8
      b.ne outer_loop
      b return

tail_4:
      tbz w1, 2, tail_2
      str  q11, [x6], 16
      mov  v11.16b, v12.16b


tail_2:
      tbz w1, 1, tail_1
      str  d11, [x6], 8
      dup d11, v11.d[1]


tail_1:
      tbz w1, 0, return
      str  s11, [x6]

return:
      # Restore the callee saved GP registers.
      ldp x19, x20, [sp, -64]
      ldp x21, x22, [sp, -48]
      ldp x23, x24, [sp, -32]
      ldp x25, x26, [sp, -16]

      # Restore callee saved q8-q15 registers.
      ldp d8, d9, [sp, -128]
      ldp d10, d11, [sp, -112]
      ldp d12, d13, [sp, -96]
      ldp d14, d15, [sp, -80]
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_2