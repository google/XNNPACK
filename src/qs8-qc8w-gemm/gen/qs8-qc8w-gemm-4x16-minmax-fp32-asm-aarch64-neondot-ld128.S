// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "src/xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128_2

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
      ld1r {v10.8h}, [x13]
      add x13, x13, 2
      ld2r {v0.16b, v1.16b}, [x13]
      ldr x24, [sp, 272]
      # Round kc up to channels.
      add x2, x2, #3
      and x2, x2, #0xFFFFFFFFFFFFFFFC

      # Setup and alias a & c pointers.
      add x9, x3, x4
      add x10, x9, x4
      add x11, x10, x4
      add x13, x6, x7
      add x14, x13, x7
      add x15, x14, x7

      cmp x0, 2
      csel  x9, x3, x9, LO
      csel  x13, x6, x13, LO
      csel  x10, x9, x10, LS
      csel  x14, x13, x14, LS

      cmp x0, 4
      csel  x11, x10, x11, LO
      csel  x15, x14, x15, LO

.Louter_loop:
      # Initialize k counter.
      mov x20, x2

      # Initialize accumulators with the biases.
      ldp q12, q16, [x5, 0]
      ldp q20, q24, [x5, 32]
      mov v13.16b, v12.16b
      mov v14.16b, v12.16b
      mov v15.16b, v12.16b
      mov v17.16b, v16.16b
      mov v18.16b, v16.16b
      mov v19.16b, v16.16b
      mov v21.16b, v20.16b
      mov v22.16b, v20.16b
      mov v23.16b, v20.16b
      mov v25.16b, v24.16b
      mov v26.16b, v24.16b
      mov v27.16b, v24.16b
      add x5, x5, 64

      # Are there at least 16 bytes?
      cmp x20, 16
      blt .Linner_loop_tail
      sub x20, x20, 16

.Linner_loop:
      ldr q2, [x3], 16
      ldr q3, [x9], 16
      ldr q4, [x10], 16
      ldr q5, [x11], 16
      ldp q6, q7, [x5], 32
      ldp q8, q9, [x5], 32
      sdot  v12.4s, v6.16b, v2.4b[0]
      sdot  v13.4s, v6.16b, v3.4b[0]
      sdot  v14.4s, v6.16b, v4.4b[0]
      sdot  v15.4s, v6.16b, v5.4b[0]
      sdot  v16.4s, v7.16b, v2.4b[0]
      sdot  v17.4s, v7.16b, v3.4b[0]
      sdot  v18.4s, v7.16b, v4.4b[0]
      sdot  v19.4s, v7.16b, v5.4b[0]
      sdot  v20.4s, v8.16b, v2.4b[0]
      sdot  v21.4s, v8.16b, v3.4b[0]
      sdot  v22.4s, v8.16b, v4.4b[0]
      sdot  v23.4s, v8.16b, v5.4b[0]
      sdot  v24.4s, v9.16b, v2.4b[0]
      sdot  v25.4s, v9.16b, v3.4b[0]
      sdot  v26.4s, v9.16b, v4.4b[0]
      sdot  v27.4s, v9.16b, v5.4b[0]
      ldp q6, q7, [x5], 32
      ldp q8, q9, [x5], 32
      sdot  v12.4s, v6.16b, v2.4b[1]
      sdot  v13.4s, v6.16b, v3.4b[1]
      sdot  v14.4s, v6.16b, v4.4b[1]
      sdot  v15.4s, v6.16b, v5.4b[1]
      sdot  v16.4s, v7.16b, v2.4b[1]
      sdot  v17.4s, v7.16b, v3.4b[1]
      sdot  v18.4s, v7.16b, v4.4b[1]
      sdot  v19.4s, v7.16b, v5.4b[1]
      sdot  v20.4s, v8.16b, v2.4b[1]
      sdot  v21.4s, v8.16b, v3.4b[1]
      sdot  v22.4s, v8.16b, v4.4b[1]
      sdot  v23.4s, v8.16b, v5.4b[1]
      sdot  v24.4s, v9.16b, v2.4b[1]
      sdot  v25.4s, v9.16b, v3.4b[1]
      sdot  v26.4s, v9.16b, v4.4b[1]
      sdot  v27.4s, v9.16b, v5.4b[1]
      ldp q6, q7, [x5], 32
      ldp q8, q9, [x5], 32
      sdot  v12.4s, v6.16b, v2.4b[2]
      sdot  v13.4s, v6.16b, v3.4b[2]
      sdot  v14.4s, v6.16b, v4.4b[2]
      sdot  v15.4s, v6.16b, v5.4b[2]
      sdot  v16.4s, v7.16b, v2.4b[2]
      sdot  v17.4s, v7.16b, v3.4b[2]
      sdot  v18.4s, v7.16b, v4.4b[2]
      sdot  v19.4s, v7.16b, v5.4b[2]
      sdot  v20.4s, v8.16b, v2.4b[2]
      sdot  v21.4s, v8.16b, v3.4b[2]
      sdot  v22.4s, v8.16b, v4.4b[2]
      sdot  v23.4s, v8.16b, v5.4b[2]
      sdot  v24.4s, v9.16b, v2.4b[2]
      sdot  v25.4s, v9.16b, v3.4b[2]
      sdot  v26.4s, v9.16b, v4.4b[2]
      sdot  v27.4s, v9.16b, v5.4b[2]
      ldp q6, q7, [x5], 32
      ldp q8, q9, [x5], 32
      sdot  v12.4s, v6.16b, v2.4b[3]
      sdot  v13.4s, v6.16b, v3.4b[3]
      sdot  v14.4s, v6.16b, v4.4b[3]
      sdot  v15.4s, v6.16b, v5.4b[3]
      sdot  v16.4s, v7.16b, v2.4b[3]
      sdot  v17.4s, v7.16b, v3.4b[3]
      sdot  v18.4s, v7.16b, v4.4b[3]
      sdot  v19.4s, v7.16b, v5.4b[3]
      sdot  v20.4s, v8.16b, v2.4b[3]
      sdot  v21.4s, v8.16b, v3.4b[3]
      sdot  v22.4s, v8.16b, v4.4b[3]
      sdot  v23.4s, v8.16b, v5.4b[3]
      sdot  v24.4s, v9.16b, v2.4b[3]
      sdot  v25.4s, v9.16b, v3.4b[3]
      sdot  v26.4s, v9.16b, v4.4b[3]
      sdot  v27.4s, v9.16b, v5.4b[3]
      subs x20, x20, 16
      bhs .Linner_loop

      add x20, x20, 16
      cmp x20, 4
      blt .Linner_loop_end

.Linner_loop_tail:
      ldr s2, [x3], 4
      ldr s3, [x9], 4
      ldr s4, [x10], 4
      ldr s5, [x11], 4
      ldp q6, q7, [x5], 32
      ldp q8, q9, [x5], 32
      sdot  v12.4s, v6.16b, v2.4b[0]
      sdot  v13.4s, v6.16b, v3.4b[0]
      sdot  v14.4s, v6.16b, v4.4b[0]
      sdot  v15.4s, v6.16b, v5.4b[0]
      sdot  v16.4s, v7.16b, v2.4b[0]
      sdot  v17.4s, v7.16b, v3.4b[0]
      sdot  v18.4s, v7.16b, v4.4b[0]
      sdot  v19.4s, v7.16b, v5.4b[0]
      sdot  v20.4s, v8.16b, v2.4b[0]
      sdot  v21.4s, v8.16b, v3.4b[0]
      sdot  v22.4s, v8.16b, v4.4b[0]
      sdot  v23.4s, v8.16b, v5.4b[0]
      sdot  v24.4s, v9.16b, v2.4b[0]
      sdot  v25.4s, v9.16b, v3.4b[0]
      sdot  v26.4s, v9.16b, v4.4b[0]
      sdot  v27.4s, v9.16b, v5.4b[0]
      subs x20, x20, 4
      bne .Linner_loop_tail

.Linner_loop_end:

      # Convert from int32 to float.
      scvtf v12.4s, v12.4s
      scvtf v13.4s, v13.4s
      scvtf v14.4s, v14.4s
      scvtf v15.4s, v15.4s
      scvtf v16.4s, v16.4s
      scvtf v17.4s, v17.4s
      scvtf v18.4s, v18.4s
      scvtf v19.4s, v19.4s
      scvtf v20.4s, v20.4s
      scvtf v21.4s, v21.4s
      scvtf v22.4s, v22.4s
      scvtf v23.4s, v23.4s
      scvtf v24.4s, v24.4s
      scvtf v25.4s, v25.4s
      scvtf v26.4s, v26.4s
      scvtf v27.4s, v27.4s
      # Load weights scale.
      ldp q2, q3, [x5, 0]
      ldp q4, q5, [x5, 32]
      add x5, x5, 64
      # Multiply by weight's scale.
      fmul v12.4s, v12.4s, v2.4s
      fmul v13.4s, v13.4s, v2.4s
      fmul v14.4s, v14.4s, v2.4s
      fmul v15.4s, v15.4s, v2.4s
      fmul v16.4s, v16.4s, v3.4s
      fmul v17.4s, v17.4s, v3.4s
      fmul v18.4s, v18.4s, v3.4s
      fmul v19.4s, v19.4s, v3.4s
      fmul v20.4s, v20.4s, v4.4s
      fmul v21.4s, v21.4s, v4.4s
      fmul v22.4s, v22.4s, v4.4s
      fmul v23.4s, v23.4s, v4.4s
      fmul v24.4s, v24.4s, v5.4s
      fmul v25.4s, v25.4s, v5.4s
      fmul v26.4s, v26.4s, v5.4s
      fmul v27.4s, v27.4s, v5.4s
      # Reconvert to int32.
      fcvtns v12.4s, v12.4s
      fcvtns v13.4s, v13.4s
      fcvtns v14.4s, v14.4s
      fcvtns v15.4s, v15.4s
      fcvtns v16.4s, v16.4s
      fcvtns v17.4s, v17.4s
      fcvtns v18.4s, v18.4s
      fcvtns v19.4s, v19.4s
      fcvtns v20.4s, v20.4s
      fcvtns v21.4s, v21.4s
      fcvtns v22.4s, v22.4s
      fcvtns v23.4s, v23.4s
      fcvtns v24.4s, v24.4s
      fcvtns v25.4s, v25.4s
      fcvtns v26.4s, v26.4s
      fcvtns v27.4s, v27.4s
      # Convert to int16.
      sqxtn v12.4h, v12.4s
      sqxtn v13.4h, v13.4s
      sqxtn v14.4h, v14.4s
      sqxtn v15.4h, v15.4s
      sqxtn v20.4h, v20.4s
      sqxtn v21.4h, v21.4s
      sqxtn v22.4h, v22.4s
      sqxtn v23.4h, v23.4s
      sqxtn2 v12.8h, v16.4s
      sqxtn2 v13.8h, v17.4s
      sqxtn2 v14.8h, v18.4s
      sqxtn2 v15.8h, v19.4s
      sqxtn2 v20.8h, v24.4s
      sqxtn2 v21.8h, v25.4s
      sqxtn2 v22.8h, v26.4s
      sqxtn2 v23.8h, v27.4s
      # Add output zero point.
      sqadd v12.8h, v12.8h, v10.8h
      sqadd v13.8h, v13.8h, v10.8h
      sqadd v14.8h, v14.8h, v10.8h
      sqadd v15.8h, v15.8h, v10.8h
      sqadd v20.8h, v20.8h, v10.8h
      sqadd v21.8h, v21.8h, v10.8h
      sqadd v22.8h, v22.8h, v10.8h
      sqadd v23.8h, v23.8h, v10.8h
      # Convert to int8.
      sqxtn v12.8b, v12.8h
      sqxtn v13.8b, v13.8h
      sqxtn v14.8b, v14.8h
      sqxtn v15.8b, v15.8h
      sqxtn2 v12.16b, v20.8h
      sqxtn2 v13.16b, v21.8h
      sqxtn2 v14.16b, v22.8h
      sqxtn2 v15.16b, v23.8h
      # Min/max clamping.
      smin  v12.16b, v1.16b, v12.16b
      smin  v13.16b, v1.16b, v13.16b
      smin  v14.16b, v1.16b, v14.16b
      smin  v15.16b, v1.16b, v15.16b
      smax  v12.16b, v0.16b, v12.16b
      smax  v13.16b, v0.16b, v13.16b
      smax  v14.16b, v0.16b, v14.16b
      smax  v15.16b, v0.16b, v15.16b

      # Check whether full or partial store.
      cmp x1, 16
      b.lo .Ltail_8
      str q12, [x6], 16
      str q13, [x13], 16
      str q14, [x14], 16
      str q15, [x15], 16
      sub x3, x3, x2
      sub x9, x9, x2
      sub x10, x10, x2
      sub x11, x11, x2

      sub x1, x1, 16
      b.ne .Louter_loop
      b .Lreturn

.Ltail_8:
      tbz w1, 3, .Ltail_4
      str d12, [x6], 8
      str d13, [x13], 8
      str d14, [x14], 8
      str d15, [x15], 8
      ext v12.16b, v12.16b, v12.16b, 8
      ext v13.16b, v13.16b, v13.16b, 8
      ext v14.16b, v14.16b, v14.16b, 8
      ext v15.16b, v15.16b, v15.16b, 8


.Ltail_4:
      tbz w1, 2, .Ltail_2
      st1 {v12.s}[0], [x6], 4
      st1 {v13.s}[0], [x13], 4
      st1 {v14.s}[0], [x14], 4
      st1 {v15.s}[0], [x15], 4
      ext v12.16b, v12.16b, v12.16b, 4
      ext v13.16b, v13.16b, v13.16b, 4
      ext v14.16b, v14.16b, v14.16b, 4
      ext v15.16b, v15.16b, v15.16b, 4


.Ltail_2:
      tbz w1, 1, .Ltail_1
      st1 {v12.h}[0], [x6], 2
      st1 {v13.h}[0], [x13], 2
      st1 {v14.h}[0], [x14], 2
      st1 {v15.h}[0], [x15], 2
      ext v12.16b, v12.16b, v12.16b, 2
      ext v13.16b, v13.16b, v13.16b, 2
      ext v14.16b, v14.16b, v14.16b, 2
      ext v15.16b, v15.16b, v15.16b, 2


.Ltail_1:
      tbz w1, 0, .Lreturn
      st1 {v12.b}[0], [x6]
      st1 {v13.b}[0], [x13]
      st1 {v14.b}[0], [x14]
      st1 {v15.b}[0], [x15]

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
END_FUNCTION xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128_2