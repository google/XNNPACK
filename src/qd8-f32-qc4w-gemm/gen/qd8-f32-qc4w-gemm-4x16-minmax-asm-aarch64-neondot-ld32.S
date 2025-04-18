// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld32_2

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
      # Load 0xF0 for masking the weights
      ldr x24, [sp, 272]
      movi v10.16b, #240
      # Round kc up to channels.
      add x2, x2, #3
      and x2, x2, #0xFFFFFFFFFFFFFFFC

      # Setup and alias a & c pointers.
      add x9, x3, x4
      add x10, x9, x4
      add x11, x10, x4
      add x14, x6, x7
      add x15, x14, x7
      add x19, x15, x7

      cmp x0, 2
      csel  x9, x3, x9, LO
      csel  x14, x6, x14, LO
      csel  x10, x9, x10, LS
      csel  x15, x14, x15, LS

      cmp x0, 4
      csel  x11, x10, x11, LO
      csel  x19, x15, x19, LO

.Louter_loop:
      # Initialize k counter.
      mov x20, x2
      # Initialize accumulators with k_sum * input zero point.
      ldp q30, q31, [x24, 0]
      ldp q2, q3, [x5, 0]
      ldp q4, q5, [x5, 32]
      mul v12.4s, v2.4s, v30.s[0]
      mul v16.4s, v2.4s, v30.s[2]
      mul v20.4s, v2.4s, v31.s[0]
      mul v24.4s, v2.4s, v31.s[2]
      mul v13.4s, v3.4s, v30.s[0]
      mul v17.4s, v3.4s, v30.s[2]
      mul v21.4s, v3.4s, v31.s[0]
      mul v25.4s, v3.4s, v31.s[2]
      mul v14.4s, v4.4s, v30.s[0]
      mul v18.4s, v4.4s, v30.s[2]
      mul v22.4s, v4.4s, v31.s[0]
      mul v26.4s, v4.4s, v31.s[2]
      mul v15.4s, v5.4s, v30.s[0]
      mul v19.4s, v5.4s, v30.s[2]
      mul v23.4s, v5.4s, v31.s[0]
      mul v27.4s, v5.4s, v31.s[2]
      add x5, x5, 64

.Linner_loop:
      ldr s2, [x3], 4
      ldr s3, [x9], 4
      ldr s4, [x10], 4
      ldr s5, [x11], 4
      ldr q9, [x5], 16
      shl v6.16b, v9.16b, #4
      and v7.16b, v9.16b, v10.16b
      ldr q9, [x5], 16
      shl v8.16b, v9.16b, #4
      and v9.16b, v9.16b, v10.16b
      sdot  v12.4s, v6.16b, v2.4b[0]
      sdot  v16.4s, v6.16b, v3.4b[0]
      sdot  v20.4s, v6.16b, v4.4b[0]
      sdot  v24.4s, v6.16b, v5.4b[0]
      sdot  v13.4s, v7.16b, v2.4b[0]
      sdot  v17.4s, v7.16b, v3.4b[0]
      sdot  v21.4s, v7.16b, v4.4b[0]
      sdot  v25.4s, v7.16b, v5.4b[0]
      sdot  v14.4s, v8.16b, v2.4b[0]
      sdot  v18.4s, v8.16b, v3.4b[0]
      sdot  v22.4s, v8.16b, v4.4b[0]
      sdot  v26.4s, v8.16b, v5.4b[0]
      sdot  v15.4s, v9.16b, v2.4b[0]
      sdot  v19.4s, v9.16b, v3.4b[0]
      sdot  v23.4s, v9.16b, v4.4b[0]
      sdot  v27.4s, v9.16b, v5.4b[0]
      subs x20, x20, 4
      bne .Linner_loop


.Linner_loop_end:
      # Convert from int32 to float.
      scvtf v12.4s, v12.4s, #4
      scvtf v13.4s, v13.4s, #4
      scvtf v14.4s, v14.4s, #4
      scvtf v15.4s, v15.4s, #4
      scvtf v16.4s, v16.4s, #4
      scvtf v17.4s, v17.4s, #4
      scvtf v18.4s, v18.4s, #4
      scvtf v19.4s, v19.4s, #4
      scvtf v20.4s, v20.4s, #4
      scvtf v21.4s, v21.4s, #4
      scvtf v22.4s, v22.4s, #4
      scvtf v23.4s, v23.4s, #4
      scvtf v24.4s, v24.4s, #4
      scvtf v25.4s, v25.4s, #4
      scvtf v26.4s, v26.4s, #4
      scvtf v27.4s, v27.4s, #4
      # Multiply by input scale.
      fmul v12.4s, v12.4s, v30.s[1]
      fmul v16.4s, v16.4s, v30.s[3]
      fmul v20.4s, v20.4s, v31.s[1]
      fmul v24.4s, v24.4s, v31.s[3]
      fmul v13.4s, v13.4s, v30.s[1]
      fmul v17.4s, v17.4s, v30.s[3]
      fmul v21.4s, v21.4s, v31.s[1]
      fmul v25.4s, v25.4s, v31.s[3]
      fmul v14.4s, v14.4s, v30.s[1]
      fmul v18.4s, v18.4s, v30.s[3]
      fmul v22.4s, v22.4s, v31.s[1]
      fmul v26.4s, v26.4s, v31.s[3]
      fmul v15.4s, v15.4s, v30.s[1]
      fmul v19.4s, v19.4s, v30.s[3]
      fmul v23.4s, v23.4s, v31.s[1]
      fmul v27.4s, v27.4s, v31.s[3]
      # Load weights scale.
      ldp q2, q3, [x5, 0]
      ldp q4, q5, [x5, 32]
      add x5, x5, 64
      # Load biases.
      ldp q6, q7, [x5, 0]
      ldp q8, q9, [x5, 32]
      add x5, x5, 64
      # Multiply by weight's scale.
      fmul v12.4s, v12.4s, v2.4s
      fmul v16.4s, v16.4s, v2.4s
      fmul v20.4s, v20.4s, v2.4s
      fmul v24.4s, v24.4s, v2.4s
      fmul v13.4s, v13.4s, v3.4s
      fmul v17.4s, v17.4s, v3.4s
      fmul v21.4s, v21.4s, v3.4s
      fmul v25.4s, v25.4s, v3.4s
      fmul v14.4s, v14.4s, v4.4s
      fmul v18.4s, v18.4s, v4.4s
      fmul v22.4s, v22.4s, v4.4s
      fmul v26.4s, v26.4s, v4.4s
      fmul v15.4s, v15.4s, v5.4s
      fmul v19.4s, v19.4s, v5.4s
      fmul v23.4s, v23.4s, v5.4s
      fmul v27.4s, v27.4s, v5.4s
      # Add bias.
      fadd v12.4s, v12.4s, v6.4s
      fadd v16.4s, v16.4s, v6.4s
      fadd v20.4s, v20.4s, v6.4s
      fadd v24.4s, v24.4s, v6.4s
      fadd v13.4s, v13.4s, v7.4s
      fadd v17.4s, v17.4s, v7.4s
      fadd v21.4s, v21.4s, v7.4s
      fadd v25.4s, v25.4s, v7.4s
      fadd v14.4s, v14.4s, v8.4s
      fadd v18.4s, v18.4s, v8.4s
      fadd v22.4s, v22.4s, v8.4s
      fadd v26.4s, v26.4s, v8.4s
      fadd v15.4s, v15.4s, v9.4s
      fadd v19.4s, v19.4s, v9.4s
      fadd v23.4s, v23.4s, v9.4s
      fadd v27.4s, v27.4s, v9.4s
      # Min/max clamping.
      fmin v12.4s, v1.4s, v12.4s
      fmin v16.4s, v1.4s, v16.4s
      fmin v20.4s, v1.4s, v20.4s
      fmin v24.4s, v1.4s, v24.4s
      fmin v13.4s, v1.4s, v13.4s
      fmin v17.4s, v1.4s, v17.4s
      fmin v21.4s, v1.4s, v21.4s
      fmin v25.4s, v1.4s, v25.4s
      fmin v14.4s, v1.4s, v14.4s
      fmin v18.4s, v1.4s, v18.4s
      fmin v22.4s, v1.4s, v22.4s
      fmin v26.4s, v1.4s, v26.4s
      fmin v15.4s, v1.4s, v15.4s
      fmin v19.4s, v1.4s, v19.4s
      fmin v23.4s, v1.4s, v23.4s
      fmin v27.4s, v1.4s, v27.4s
      fmax v12.4s, v0.4s, v12.4s
      fmax v16.4s, v0.4s, v16.4s
      fmax v20.4s, v0.4s, v20.4s
      fmax v24.4s, v0.4s, v24.4s
      fmax v13.4s, v0.4s, v13.4s
      fmax v17.4s, v0.4s, v17.4s
      fmax v21.4s, v0.4s, v21.4s
      fmax v25.4s, v0.4s, v25.4s
      fmax v14.4s, v0.4s, v14.4s
      fmax v18.4s, v0.4s, v18.4s
      fmax v22.4s, v0.4s, v22.4s
      fmax v26.4s, v0.4s, v26.4s
      fmax v15.4s, v0.4s, v15.4s
      fmax v19.4s, v0.4s, v19.4s
      fmax v23.4s, v0.4s, v23.4s
      fmax v27.4s, v0.4s, v27.4s

      # Check whether full or partial store.
      cmp x1, 16
      b.lo .Ltail_8
      stp q12, q13, [x6], #32
      stp q14, q15, [x6], #32
      stp q16, q17, [x14], #32
      stp q18, q19, [x14], #32
      stp q20, q21, [x15], #32
      stp q22, q23, [x15], #32
      stp q24, q25, [x19], #32
      stp q26, q27, [x19], #32
      sub x3, x3, x2
      sub x9, x9, x2
      sub x10, x10, x2
      sub x11, x11, x2

      sub x1, x1, 16
      b.ne .Louter_loop
      b .Lreturn

.Ltail_8:
      tbz w1, 3, .Ltail_4
      stp q12, q13, [x6], #32
      stp q16, q17, [x14], #32
      stp q20, q21, [x15], #32
      stp q24, q25, [x19], #32
      mov v12.16b, v14.16b
      mov v13.16b, v15.16b
      mov v16.16b, v18.16b
      mov v17.16b, v19.16b
      mov v20.16b, v22.16b
      mov v21.16b, v23.16b
      mov v24.16b, v26.16b
      mov v25.16b, v27.16b


.Ltail_4:
      tbz w1, 2, .Ltail_2
      str q12, [x6], #16
      str q16, [x14], #16
      str q20, [x15], #16
      str q24, [x19], #16
      mov v12.16b, v13.16b
      mov v16.16b, v17.16b
      mov v20.16b, v21.16b
      mov v24.16b, v25.16b


.Ltail_2:
      tbz w1, 1, .Ltail_1
      str d12, [x6], #8
      str d16, [x14], #8
      str d20, [x15], #8
      str d24, [x19], #8
      dup d12, v12.d[1]
      dup d16, v16.d[1]
      dup d20, v20.d[1]
      dup d24, v24.d[1]


.Ltail_1:
      tbz w1, 0, .Lreturn
      str s12, [x6], #0
      str s16, [x14], #0
      str s20, [x15], #0
      str s24, [x19], #0

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
END_FUNCTION xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld32_2