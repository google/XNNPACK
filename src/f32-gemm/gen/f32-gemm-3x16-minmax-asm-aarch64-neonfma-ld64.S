// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_3x16__asm_aarch64_neonfma_ld64_2

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
      ld2r {v0.4s, v1.4s}, [x13]
      # Setup and alias a & c pointers.
      add x9, x3, x4
      add x10, x9, x4
      add x14, x6, x7
      add x15, x14, x7

      cmp x0, 2
      csel  x9, x3, x9, LO
      csel  x14, x6, x14, LO
      csel  x10, x9, x10, LS
      csel  x15, x14, x15, LS

.Louter_loop:
      # Initialize k counter.
      mov x20, x2

      # Initialize accumulators with the biases.
      ldp q11, q12, [x5, 0]
      ldp q13, q14, [x5, 32]
      mov v15.16b, v11.16b
      mov v19.16b, v11.16b
      mov v16.16b, v12.16b
      mov v20.16b, v12.16b
      mov v17.16b, v13.16b
      mov v21.16b, v13.16b
      mov v18.16b, v14.16b
      mov v22.16b, v14.16b
      add x5, x5, 64

      # Are there at least 8 bytes?
      cmp x20, 8
      blt .Linner_loop_tail
      sub x20, x20, 8

.Linner_loop:
      ldr d2, [x3], 8
      ldr d3, [x9], 8
      ldr d4, [x10], 8
      ldp q7, q8, [x5], 32
      ldp q9, q10, [x5], 32
      fmla v11.4s, v7.4s, v2.s[0]
      fmla v15.4s, v7.4s, v3.s[0]
      fmla v19.4s, v7.4s, v4.s[0]
      fmla v12.4s, v8.4s, v2.s[0]
      fmla v16.4s, v8.4s, v3.s[0]
      fmla v20.4s, v8.4s, v4.s[0]
      fmla v13.4s, v9.4s, v2.s[0]
      fmla v17.4s, v9.4s, v3.s[0]
      fmla v21.4s, v9.4s, v4.s[0]
      fmla v14.4s, v10.4s, v2.s[0]
      fmla v18.4s, v10.4s, v3.s[0]
      fmla v22.4s, v10.4s, v4.s[0]
      ldp q7, q8, [x5], 32
      ldp q9, q10, [x5], 32
      fmla v11.4s, v7.4s, v2.s[1]
      fmla v15.4s, v7.4s, v3.s[1]
      fmla v19.4s, v7.4s, v4.s[1]
      fmla v12.4s, v8.4s, v2.s[1]
      fmla v16.4s, v8.4s, v3.s[1]
      fmla v20.4s, v8.4s, v4.s[1]
      fmla v13.4s, v9.4s, v2.s[1]
      fmla v17.4s, v9.4s, v3.s[1]
      fmla v21.4s, v9.4s, v4.s[1]
      fmla v14.4s, v10.4s, v2.s[1]
      fmla v18.4s, v10.4s, v3.s[1]
      fmla v22.4s, v10.4s, v4.s[1]
      subs x20, x20, 8
      bhs .Linner_loop

      add x20, x20, 8
      cmp x20, 4
      blt .Linner_loop_end

.Linner_loop_tail:
      ldr s2, [x3], 4
      ldr s3, [x9], 4
      ldr s4, [x10], 4
      ldp q7, q8, [x5], 32
      ldp q9, q10, [x5], 32
      fmla v11.4s, v7.4s, v2.s[0]
      fmla v15.4s, v7.4s, v3.s[0]
      fmla v19.4s, v7.4s, v4.s[0]
      fmla v12.4s, v8.4s, v2.s[0]
      fmla v16.4s, v8.4s, v3.s[0]
      fmla v20.4s, v8.4s, v4.s[0]
      fmla v13.4s, v9.4s, v2.s[0]
      fmla v17.4s, v9.4s, v3.s[0]
      fmla v21.4s, v9.4s, v4.s[0]
      fmla v14.4s, v10.4s, v2.s[0]
      fmla v18.4s, v10.4s, v3.s[0]
      fmla v22.4s, v10.4s, v4.s[0]
      subs x20, x20, 4
      bne .Linner_loop_tail


.Linner_loop_end:
      # Min/max clamping.
      fmin v11.4s, v1.4s, v11.4s
      fmin v15.4s, v1.4s, v15.4s
      fmin v19.4s, v1.4s, v19.4s
      fmin v12.4s, v1.4s, v12.4s
      fmin v16.4s, v1.4s, v16.4s
      fmin v20.4s, v1.4s, v20.4s
      fmin v13.4s, v1.4s, v13.4s
      fmin v17.4s, v1.4s, v17.4s
      fmin v21.4s, v1.4s, v21.4s
      fmin v14.4s, v1.4s, v14.4s
      fmin v18.4s, v1.4s, v18.4s
      fmin v22.4s, v1.4s, v22.4s
      fmax v11.4s, v0.4s, v11.4s
      fmax v15.4s, v0.4s, v15.4s
      fmax v19.4s, v0.4s, v19.4s
      fmax v12.4s, v0.4s, v12.4s
      fmax v16.4s, v0.4s, v16.4s
      fmax v20.4s, v0.4s, v20.4s
      fmax v13.4s, v0.4s, v13.4s
      fmax v17.4s, v0.4s, v17.4s
      fmax v21.4s, v0.4s, v21.4s
      fmax v14.4s, v0.4s, v14.4s
      fmax v18.4s, v0.4s, v18.4s
      fmax v22.4s, v0.4s, v22.4s

      # Check whether full or partial store.
      cmp x1, 16
      b.lo .Ltail_8
      stp q11, q12, [x6], #32
      stp q13, q14, [x6], #32
      stp q15, q16, [x14], #32
      stp q17, q18, [x14], #32
      stp q19, q20, [x15], #32
      stp q21, q22, [x15], #32
      sub x3, x3, x2
      sub x9, x9, x2
      sub x10, x10, x2

      sub x1, x1, 16
      b.ne .Louter_loop
      b .Lreturn

.Ltail_8:
      tbz w1, 3, .Ltail_4
      stp q11, q12, [x6], #32
      stp q15, q16, [x14], #32
      stp q19, q20, [x15], #32
      mov v11.16b, v13.16b
      mov v12.16b, v14.16b
      mov v15.16b, v17.16b
      mov v16.16b, v18.16b
      mov v19.16b, v21.16b
      mov v20.16b, v22.16b


.Ltail_4:
      tbz w1, 2, .Ltail_2
      str q11, [x6], #16
      str q15, [x14], #16
      str q19, [x15], #16
      mov v11.16b, v12.16b
      mov v15.16b, v16.16b
      mov v19.16b, v20.16b


.Ltail_2:
      tbz w1, 1, .Ltail_1
      str d11, [x6], #8
      str d15, [x14], #8
      str d19, [x15], #8
      dup d11, v11.d[1]
      dup d15, v15.d[1]
      dup d19, v19.d[1]


.Ltail_1:
      tbz w1, 0, .Lreturn
      str s11, [x6], #0
      str s15, [x14], #0
      str s19, [x15], #0

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
END_FUNCTION xnn_f32_gemm_minmax_ukernel_3x16__asm_aarch64_neonfma_ld64_2