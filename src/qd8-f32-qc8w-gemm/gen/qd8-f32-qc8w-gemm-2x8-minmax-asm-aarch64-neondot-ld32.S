// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld32_2

      # Free up GP registers.
      sub sp, sp, 256
      stp x19, x20, [sp, 192]
      stp x21, x22, [sp, 160]
      stp x23, x24, [sp, 128]
      stp x25, x26, [sp, 96]

      # Preserve callee saved q8-q15 registers.
      stp d8, d9, [sp, 64]
      stp d10, d11, [sp, 48]
      stp d12, d13, [sp, 32]
      stp d14, d15, [sp, 16]

      # Load params.
      ldr x13, [sp, 264]

      # Load min/max values.
      ld2r {v0.4s, v1.4s}, [x13]
      ldr x24, [sp, 272]
      # Round kc up to channels.
      add x2, x2, #3
      and x2, x2, #0xFFFFFFFFFFFFFFFC

      # Setup and alias a & c pointers.
      add x9, x3, x4
      add x13, x6, x7

      cmp x0, 2
      csel  x9, x3, x9, LO
      csel  x13, x6, x13, LO

.Louter_loop:
      # Initialize k counter.
      mov x20, x2
      # Initialize accumulators with k_sum * input zero point.
      ldp q10, q11, [x24]
      ldp  q2, q3, [x5, 0]
      mul v12.4s, v2.4s, v10.s[0]
      mul v13.4s, v2.4s, v10.s[2]
      mul v14.4s, v3.4s, v10.s[0]
      mul v15.4s, v3.4s, v10.s[2]
      add x5, x5, 32

.Linner_loop:
      ldr s2, [x3], 4
      ldr s3, [x9], 4
      ldp q6, q7, [x5], 32
      sdot  v12.4s, v6.16b, v2.4b[0]
      sdot  v13.4s, v6.16b, v3.4b[0]
      sdot  v14.4s, v7.16b, v2.4b[0]
      sdot  v15.4s, v7.16b, v3.4b[0]
      subs x20, x20, 4
      bne .Linner_loop

.Linner_loop_end:

      # Convert from int32 to float.
      scvtf v12.4s, v12.4s
      scvtf v13.4s, v13.4s
      scvtf v14.4s, v14.4s
      scvtf v15.4s, v15.4s
      # Multiply by input scale.
      fmul v12.4s, v12.4s, v10.s[1]
      fmul v13.4s, v13.4s, v10.s[3]
      fmul v14.4s, v14.4s, v10.s[1]
      fmul v15.4s, v15.4s, v10.s[3]
      # Load weights scale.
      ldp q2, q3, [x5, 0]
      add x5, x5, 32
      # Load biases.
      ldp q6, q7, [x5, 0]
      add x5, x5, 32
      # Multiply by weight's scale.
      fmul v12.4s, v12.4s, v2.4s
      fmul v13.4s, v13.4s, v2.4s
      fmul v14.4s, v14.4s, v3.4s
      fmul v15.4s, v15.4s, v3.4s
      # Add bias.
      fadd v12.4s, v12.4s, v6.4s
      fadd v13.4s, v13.4s, v6.4s
      fadd v14.4s, v14.4s, v7.4s
      fadd v15.4s, v15.4s, v7.4s
      # Min/max clamping.
      fmin  v12.4s, v1.4s, v12.4s
      fmin  v13.4s, v1.4s, v13.4s
      fmin  v14.4s, v1.4s, v14.4s
      fmin  v15.4s, v1.4s, v15.4s
      fmax  v12.4s, v0.4s, v12.4s
      fmax  v13.4s, v0.4s, v13.4s
      fmax  v14.4s, v0.4s, v14.4s
      fmax  v15.4s, v0.4s, v15.4s

      # Check whether full or partial store.
      cmp x1, 8
      b.lo .Ltail_4
      stp  q12, q14, [x6], 32
      stp  q13, q15, [x13], 32
      sub x3, x3, x2
      sub x9, x9, x2

      sub x1, x1, 8
      b.ne .Louter_loop
      b .Lreturn

.Ltail_4:
      tbz w1, 2, .Ltail_2
      str  q12, [x6], 16
      str  q13, [x13], 16
      mov  v12.16b, v14.16b
      mov  v13.16b, v15.16b


.Ltail_2:
      tbz w1, 1, .Ltail_1
      str  d12, [x6], 8
      str  d13, [x13], 8
      dup d12, v12.d[1]
      dup d13, v13.d[1]


.Ltail_1:
      tbz w1, 0, .Lreturn
      str  s12, [x6]
      str  s13, [x13]

.Lreturn:
      # Restore the callee saved GP registers.
      ldp x19, x20, [sp, 192]
      ldp x21, x22, [sp, 160]
      ldp x23, x24, [sp, 128]
      ldp x25, x26, [sp, 96]

      # Restore callee saved q8-q15 registers.
      ldp d8, d9, [sp, 64]
      ldp d10, d11, [sp, 48]
      ldp d12, d13, [sp, 32]
      ldp d14, d15, [sp, 16]
      add sp, sp, 256
      ret
END_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__asm_aarch64_neondot_ld32_2