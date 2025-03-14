// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld128_2

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
      # Load 0xF0 for masking the weights
      ldr x24, [sp, 272]
      movi v28.16b, #240
      # Round kc up to channels.
      add x2, x2, #3
      and x2, x2, #0xFFFFFFFFFFFFFFFC


.Louter_loop:
      # Initialize k counter.
      mov x20, x2
      # Initialize accumulators with k_sum * input zero point.
      ldr q10, [x24]
      ldp  q2, q3, [x5, 0]
      mul v12.4s, v2.4s, v10.s[0]
      mul v13.4s, v3.4s, v10.s[0]
      add x5, x5, 32

      # Are there at least 16 bytes?
      cmp x20, 16
      blt .Linner_loop_tail
      sub x20, x20, 16

.Linner_loop:
      ldr q2, [x3], 16
      ldr q29, [x5], 16
      shl v6.16b, v29.16b, #4
      and v7.16b, v29.16b, v28.16b
      sdot  v12.4s, v6.16b, v2.4b[0]
      sdot  v13.4s, v7.16b, v2.4b[0]
      ldr q29, [x5], 16
      shl v6.16b, v29.16b, #4
      and v7.16b, v29.16b, v28.16b
      sdot  v12.4s, v6.16b, v2.4b[1]
      sdot  v13.4s, v7.16b, v2.4b[1]
      ldr q29, [x5], 16
      shl v6.16b, v29.16b, #4
      and v7.16b, v29.16b, v28.16b
      sdot  v12.4s, v6.16b, v2.4b[2]
      sdot  v13.4s, v7.16b, v2.4b[2]
      ldr q29, [x5], 16
      shl v6.16b, v29.16b, #4
      and v7.16b, v29.16b, v28.16b
      sdot  v12.4s, v6.16b, v2.4b[3]
      sdot  v13.4s, v7.16b, v2.4b[3]
      subs x20, x20, 16
      bhs .Linner_loop

      add x20, x20, 16
      cmp x20, 4
      blt .Linner_loop_end

.Linner_loop_tail:
      ldr s2, [x3], 4
      ldr q29, [x5], 16
      shl v6.16b, v29.16b, #4
      and v7.16b, v29.16b, v28.16b
      sdot  v12.4s, v6.16b, v2.4b[0]
      sdot  v13.4s, v7.16b, v2.4b[0]
      subs x20, x20, 4
      bne .Linner_loop_tail

.Linner_loop_end:

      # Convert from int32 to float.
      scvtf v12.4s, v12.4s, #4
      scvtf v13.4s, v13.4s, #4
      # Multiply by input scale.
      fmul v12.4s, v12.4s, v10.s[1]
      fmul v13.4s, v13.4s, v10.s[1]
      # Load weights scale.
      ldp q2, q3, [x5, 0]
      add x5, x5, 32
      # Load biases.
      ldp q6, q7, [x5, 0]
      add x5, x5, 32
      # Multiply by weight's scale.
      fmul v12.4s, v12.4s, v2.4s
      fmul v13.4s, v13.4s, v3.4s
      # Add bias.
      fadd v12.4s, v12.4s, v6.4s
      fadd v13.4s, v13.4s, v7.4s
      # Min/max clamping.
      fmin  v12.4s, v1.4s, v12.4s
      fmin  v13.4s, v1.4s, v13.4s
      fmax  v12.4s, v0.4s, v12.4s
      fmax  v13.4s, v0.4s, v13.4s

      # Check whether full or partial store.
      cmp x1, 8
      b.lo .Ltail_4
      stp  q12, q13, [x6], 32
      sub x3, x3, x2

      sub x1, x1, 8
      b.ne .Louter_loop
      b .Lreturn

.Ltail_4:
      tbz w1, 2, .Ltail_2
      str  q12, [x6], 16
      mov  v12.16b, v13.16b


.Ltail_2:
      tbz w1, 1, .Ltail_1
      str  d12, [x6], 8
      dup d12, v12.d[1]


.Ltail_1:
      tbz w1, 0, .Lreturn
      str  s12, [x6]

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
END_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__asm_aarch64_neondot_ld128_2