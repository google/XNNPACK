#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_lane

      # Free up GP registers.
      stp x19, x20, [sp, -48]
      stp x21, x22, [sp, -32]
      stp x23, x24, [sp, -16]

      # Preserve callee saved q8-q15 registers.
      stp q8, q9, [sp, -176]
      stp q10, q11, [sp, -144]
      stp q12, q13, [sp, -112]
      stp q14, q15, [sp, -80]

      # Load params.
      ldr x13, [sp, 8]

      # Load min/max values.
      ld2r {v0.4s, v1.4s}, [x13]
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

outer_loop:
      # Zero k counter.
      eor x20, x20, x20
      # Initialize accumulators with the biases.
      ldp q11, q15, [x5, 0]
      mov v12.16b, v11.16b
      mov v13.16b, v11.16b
      mov v14.16b, v11.16b
      mov v16.16b, v15.16b
      mov v17.16b, v15.16b
      mov v18.16b, v15.16b
      add x5, x5, 32

inner_loop:
      ldr d2, [x3, x20]
      ldr d3, [x9, x20]
      ldr d4, [x10, x20]
      ldr d5, [x11, x20]
      ldp q7, q8, [x5, 0]
      add x5, x5, 32
      fmla  v11.4s, v7.4s, v2.s[0]
      fmla  v12.4s, v7.4s, v3.s[0]
      fmla  v13.4s, v7.4s, v4.s[0]
      fmla  v14.4s, v7.4s, v5.s[0]
      fmla  v15.4s, v8.4s, v2.s[0]
      fmla  v16.4s, v8.4s, v3.s[0]
      fmla  v17.4s, v8.4s, v4.s[0]
      fmla  v18.4s, v8.4s, v5.s[0]
      add x20, x20, 4
      cmp x2, x20
      bne inner_loop
      # Min/max clamping..
      fmin  v11.4s, v1.4s, v11.4s
      fmin  v12.4s, v1.4s, v12.4s
      fmin  v13.4s, v1.4s, v13.4s
      fmin  v14.4s, v1.4s, v14.4s
      fmin  v15.4s, v1.4s, v15.4s
      fmin  v16.4s, v1.4s, v16.4s
      fmin  v17.4s, v1.4s, v17.4s
      fmin  v18.4s, v1.4s, v18.4s
      fmax  v11.4s, v0.4s, v11.4s
      fmax  v12.4s, v0.4s, v12.4s
      fmax  v13.4s, v0.4s, v13.4s
      fmax  v14.4s, v0.4s, v14.4s
      fmax  v15.4s, v0.4s, v15.4s
      fmax  v16.4s, v0.4s, v16.4s
      fmax  v17.4s, v0.4s, v17.4s
      fmax  v18.4s, v0.4s, v18.4s

      # Check whether full or partial store.
      cmp x1, 8
      b.lo tail_4
      stp  q11, q15, [x6]
      stp  q12, q16, [x13]
      stp  q13, q17, [x14]
      stp  q14, q18, [x15]
      add x6, x6, 32
      add x13, x13, 32
      add x14, x14, 32
      add x15, x15, 32

      sub x1, x1, 8
      b.ne outer_loop
      b return

tail_4:
      tbz x1, 2, tail_2
      str  q11, [x6]
      str  q12, [x13]
      str  q13, [x14]
      str  q14, [x15]
      mov  v11.16b, v15.16b
      mov  v12.16b, v16.16b
      mov  v13.16b, v17.16b
      mov  v14.16b, v18.16b
      add x6, x6, 16
      add x13, x13, 16
      add x14, x14, 16
      add x15, x15, 16


tail_2:
      tbz x1, 1, tail_1
      str  d11, [x6]
      str  d12, [x13]
      str  d13, [x14]
      str  d14, [x15]
      add  x6, x6, 8
      add  x13, x13, 8
      add  x14, x14, 8
      add  x15, x15, 8
      dup d11, v11.d[1]
      dup d12, v12.d[1]
      dup d13, v13.d[1]
      dup d14, v14.d[1]


tail_1:
      tbz x1, 0, return
      str  s11, [x6]
      str  s12, [x13]
      str  s13, [x14]
      str  s14, [x15]

return:
      # Restore the callee saved GP registers.
      ldp x19, x20, [sp, -48]
      ldp x21, x22, [sp, -32]
      ldp x23, x24, [sp, -16]

      # Restore callee saved q8-q15 registers.
      ldp q8, q9, [sp, -176]
      ldp q10, q11, [sp, -144]
      ldp q12, q13, [sp, -112]
      ldp q14, q15, [sp, -80]
      ret
END_FUNCTION xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_lane