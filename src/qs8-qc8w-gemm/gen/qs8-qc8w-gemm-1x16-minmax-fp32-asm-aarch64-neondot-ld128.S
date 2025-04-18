// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld128_2

      # Free up GP registers.
      sub sp, sp, 256
      stp x27, x28, [sp, 224]
      stp x25, x26, [sp, 192]
      stp x23, x24, [sp, 160]
      stp x21, x22, [sp, 128]
      stp x19, x20, [sp, 96]

      # Preserve callee saved q8-q15 registers.
      stp d8, d9, [sp, 64]
      stp d10, d11, [sp, 48]
      stp d12, d13, [sp, 32]
      stp d14, d15, [sp, 16]

      # Load params.
      ldr x13, [sp, 264]

      # Load min/max values.
      add x13, x13, 2
      ld2r {v0.16b, v1.16b}, [x13]
      sub x13, x13, 2
      ldr x24, [sp, 272]
      # Round kc up to channels.
      add x2, x2, #3
      and x2, x2, #0xFFFFFFFFFFFFFFFC


.Louter_loop:
      # Initialize k counter.
      mov x20, x2

      # Initialize accumulators with the biases.
      ldp q12, q13, [x5, 0]
      ldp q14, q15, [x5, 32]
      add x5, x5, 64

      # Are there at least 16 bytes?
      cmp x20, 16
      blt .Linner_loop_tail
      sub x20, x20, 16

.Linner_loop:
      ldr q2, [x3], 16
      ldp q6, q7, [x5], 32
      ldp q8, q9, [x5], 32
      sdot  v12.4s, v6.16b, v2.4b[0]
      sdot  v13.4s, v7.16b, v2.4b[0]
      sdot  v14.4s, v8.16b, v2.4b[0]
      sdot  v15.4s, v9.16b, v2.4b[0]
      ldp q6, q7, [x5], 32
      ldp q8, q9, [x5], 32
      sdot  v12.4s, v6.16b, v2.4b[1]
      sdot  v13.4s, v7.16b, v2.4b[1]
      sdot  v14.4s, v8.16b, v2.4b[1]
      sdot  v15.4s, v9.16b, v2.4b[1]
      ldp q6, q7, [x5], 32
      ldp q8, q9, [x5], 32
      sdot  v12.4s, v6.16b, v2.4b[2]
      sdot  v13.4s, v7.16b, v2.4b[2]
      sdot  v14.4s, v8.16b, v2.4b[2]
      sdot  v15.4s, v9.16b, v2.4b[2]
      ldp q6, q7, [x5], 32
      ldp q8, q9, [x5], 32
      sdot  v12.4s, v6.16b, v2.4b[3]
      sdot  v13.4s, v7.16b, v2.4b[3]
      sdot  v14.4s, v8.16b, v2.4b[3]
      sdot  v15.4s, v9.16b, v2.4b[3]
      subs x20, x20, 16
      bhs .Linner_loop

      add x20, x20, 16
      cmp x20, 4
      blt .Linner_loop_end

.Linner_loop_tail:
      ldr s2, [x3], 4
      ldp q6, q7, [x5], 32
      ldp q8, q9, [x5], 32
      sdot  v12.4s, v6.16b, v2.4b[0]
      sdot  v13.4s, v7.16b, v2.4b[0]
      sdot  v14.4s, v8.16b, v2.4b[0]
      sdot  v15.4s, v9.16b, v2.4b[0]
      subs x20, x20, 4
      bne .Linner_loop_tail


.Linner_loop_end:
      # Convert from int32 to float.
      scvtf v12.4s, v12.4s
      scvtf v13.4s, v13.4s
      scvtf v14.4s, v14.4s
      scvtf v15.4s, v15.4s
      # Load weights scale.
      ldp q2, q3, [x5, 0]
      ldp q4, q5, [x5, 32]
      add x5, x5, 64
      # Multiply by weight's scale.
      fmul v12.4s, v12.4s, v2.4s
      fmul v13.4s, v13.4s, v3.4s
      fmul v14.4s, v14.4s, v4.4s
      fmul v15.4s, v15.4s, v5.4s
      # Reconvert to int32.
      fcvtns v12.4s, v12.4s
      fcvtns v13.4s, v13.4s
      fcvtns v14.4s, v14.4s
      fcvtns v15.4s, v15.4s
      # Convert to int16.
      sqxtn v12.4h, v12.4s
      sqxtn v14.4h, v14.4s
      sqxtn2 v12.8h, v13.4s
      sqxtn2 v14.8h, v15.4s
      ld1r {v9.8h}, [x13]
      # Add output zero point.
      sqadd v12.8h, v12.8h, v9.8h
      sqadd v14.8h, v14.8h, v9.8h
      # Convert to int8.
      sqxtn v12.8b, v12.8h
      sqxtn2 v12.16b, v14.8h
      # Min/max clamping.
      smin  v12.16b, v1.16b, v12.16b
      smax  v12.16b, v0.16b, v12.16b

      # Check whether full or partial store.
      cmp x1, 16
      b.lo .Ltail_8
      str q12, [x6], #16
      sub x3, x3, x2

      sub x1, x1, 16
      b.ne .Louter_loop
      b .Lreturn

.Ltail_8:
      tbz w1, 3, .Ltail_4
      str d12, [x6], #8
      ext v12.16b, v12.16b, v12.16b, 8


.Ltail_4:
      tbz w1, 2, .Ltail_2
      st1 {v12.s}[0], [x6], #4
      ext v12.16b, v12.16b, v12.16b, 4


.Ltail_2:
      tbz w1, 1, .Ltail_1
      st1 {v12.h}[0], [x6], #2
      ext v12.16b, v12.16b, v12.16b, 2


.Ltail_1:
      tbz w1, 0, .Lreturn
      st1 {v12.b}[0], [x6]

.Lreturn:
      # Restore the callee saved GP registers.
      ldp x27, x28, [sp, 224]
      ldp x25, x26, [sp, 192]
      ldp x23, x24, [sp, 160]
      ldp x21, x22, [sp, 128]
      ldp x19, x20, [sp, 96]

      # Restore callee saved q8-q15 registers.
      ldp d8, d9, [sp, 64]
      ldp d10, d11, [sp, 48]
      ldp d12, d13, [sp, 32]
      ldp d14, d15, [sp, 16]
      add sp, sp, 256
      ret
END_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld128_2