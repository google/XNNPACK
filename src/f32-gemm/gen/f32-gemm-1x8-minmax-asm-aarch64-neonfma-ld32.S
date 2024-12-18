#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_lane

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

outer_loop:
      # Zero k counter.
      eor x20, x20, x20
      # Initialize accumulators with the biases.
      ldp q11, q12, [x5, 0]
      add x5, x5, 32

inner_loop:
      ldr d2, [x3, x20]
      ldp q7, q8, [x5], 32
      fmla  v11.4s, v7.4s, v2.s[0]
      fmla  v12.4s, v8.4s, v2.s[0]
      add x20, x20, 4
      cmp x2, x20
      bne inner_loop
      # Min/max clamping..
      fmin  v11.4s, v1.4s, v11.4s
      fmin  v12.4s, v1.4s, v12.4s
      fmax  v11.4s, v0.4s, v11.4s
      fmax  v12.4s, v0.4s, v12.4s

      # Check whether full or partial store.
      cmp x1, 8
      b.lo tail_4
      stp  q11, q12, [x6], 32

      sub x1, x1, 8
      b.ne outer_loop
      b return

tail_4:
      tbz x1, 2, tail_2
      str  q11, [x6], 16
      mov  v11.16b, v12.16b


tail_2:
      tbz x1, 1, tail_1
      str  d11, [x6], 8
      dup d11, v11.d[1]


tail_1:
      tbz x1, 0, return
      str  s11, [x6]

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
END_FUNCTION xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_lane