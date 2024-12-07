#include "xnnpack/assembly.h"

BEGIN_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_asm_5x8c4__aarch64_neondot_lane

      # free up GP registers
      stp x19, x20, [sp, -48]
      stp x21, x22, [sp, -32]
      stp x23, x24, [sp, -16]

      # load params
      ldr x13, [sp, 8]
      # Load min/max values
      LD2R        {v0.4s, v1.4s}, [x13]
    
    ldr x24, [sp, 16]
    add x2, x2, #3
    and x2, x2, #0xFFFFFFFFFFFFFFFC
    
      add x9, x3, x4
    
      add x10, x9, x4
    
      add x11, x10, x4
    
      add x12, x11, x4
    
      add x13, x6, x7
    
      add x14, x13, x7
    
      add x15, x14, x7
    
      add x19, x15, x7
    
        cmp x0, 2
        csel  x9, x3, x9, LO
        csel  x13, x6, x13, LO
        csel  x10, x9, x10, LS
        csel  x14, x13, x14, LS

        cmp x0, 4
        csel  x11, x10, x11, LO
        csel  x15, x14, x15, LO
        csel  x12, x11, x12, LS
        csel  x19, x15, x19, LS
five: # outer loop
eor x20, x20, x20
ldp q10, q11, [x24]
 ldp  q2, q3, [x5, 0]
 mul v12.4s, v2.4s, v10.s[0]
 mul v13.4s, v2.4s, v10.s[2]
 mul v14.4s, v2.4s, v11.s[0]
 mul v15.4s, v2.4s, v11.s[2]
 mul v16.4s, v2.4s, v31.s[0]
 mul v17.4s, v3.4s, v10.s[0]
 mul v18.4s, v3.4s, v10.s[2]
 mul v19.4s, v3.4s, v11.s[0]
 mul v20.4s, v3.4s, v11.s[2]
 mul v21.4s, v3.4s, v31.s[0]
add x5, x5, 32
inner_loop:
ldr d2, [x3, x20]
ldr d3, [x9, x20]
ldr d4, [x10, x20]
ldr d5, [x11, x20]
ldr d6, [x12, x20]
ldp  q7, q8, [x5, 0]
ldr  q7, [x5, 0]
add x5, x5, 32
sdot  v12.4s, v7.16b, v2.4b[0]
sdot  v13.4s, v7.16b, v3.4b[0]
sdot  v14.4s, v7.16b, v4.4b[0]
sdot  v15.4s, v7.16b, v5.4b[0]
sdot  v16.4s, v7.16b, v6.4b[0]
sdot  v17.4s, v8.16b, v2.4b[0]
sdot  v18.4s, v8.16b, v3.4b[0]
sdot  v19.4s, v8.16b, v4.4b[0]
sdot  v20.4s, v8.16b, v5.4b[0]
sdot  v21.4s, v8.16b, v6.4b[0]
add x20, x20, 4

    cmp x2, x20

    bne inner_loop
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
fmul v12.4s, v12.4s, v10.s[1]
fmul v13.4s, v13.4s, v10.s[3]
fmul v14.4s, v14.4s, v11.s[1]
fmul v15.4s, v15.4s, v11.s[3]
fmul v16.4s, v16.4s, v31.s[1]
fmul v17.4s, v17.4s, v10.s[1]
fmul v18.4s, v18.4s, v10.s[3]
fmul v19.4s, v19.4s, v11.s[1]
fmul v20.4s, v20.4s, v11.s[3]
fmul v21.4s, v21.4s, v31.s[1]
 ldp q2, q3, [x5, 0]
add x5, x5, 32
 ldp q7, q8, [x5, 0]
add x5, x5, 32
fmul v12.4s, v12.4s, v2.4s
fmul v13.4s, v13.4s, v2.4s
fmul v14.4s, v14.4s, v2.4s
fmul v15.4s, v15.4s, v2.4s
fmul v16.4s, v16.4s, v2.4s
fmul v17.4s, v17.4s, v3.4s
fmul v18.4s, v18.4s, v3.4s
fmul v19.4s, v19.4s, v3.4s
fmul v20.4s, v20.4s, v3.4s
fmul v21.4s, v21.4s, v3.4s
fadd v12.4s, v12.4s, v7.4s
fadd v13.4s, v13.4s, v7.4s
fadd v14.4s, v14.4s, v7.4s
fadd v15.4s, v15.4s, v7.4s
fadd v16.4s, v16.4s, v7.4s
fadd v17.4s, v17.4s, v8.4s
fadd v18.4s, v18.4s, v8.4s
fadd v19.4s, v19.4s, v8.4s
fadd v20.4s, v20.4s, v8.4s
fadd v21.4s, v21.4s, v8.4s
fmin  v12.4s, v1.4s, v12.4s
fmin  v13.4s, v1.4s, v13.4s
fmin  v14.4s, v1.4s, v14.4s
fmin  v15.4s, v1.4s, v15.4s
fmin  v16.4s, v1.4s, v16.4s
fmin  v17.4s, v1.4s, v17.4s
fmin  v18.4s, v1.4s, v18.4s
fmin  v19.4s, v1.4s, v19.4s
fmin  v20.4s, v1.4s, v20.4s
fmin  v21.4s, v1.4s, v21.4s
fmax  v12.4s, v0.4s, v12.4s
fmax  v13.4s, v0.4s, v13.4s
fmax  v14.4s, v0.4s, v14.4s
fmax  v15.4s, v0.4s, v15.4s
fmax  v16.4s, v0.4s, v16.4s
fmax  v17.4s, v0.4s, v17.4s
fmax  v18.4s, v0.4s, v18.4s
fmax  v19.4s, v0.4s, v19.4s
fmax  v20.4s, v0.4s, v20.4s
fmax  v21.4s, v0.4s, v21.4s

    cmp x1, 8
    b.lo tail_4
    stp  q12, q17, [x6]
stp  q13, q18, [x13]
stp  q14, q19, [x14]
stp  q15, q20, [x15]
stp  q16, q21, [x19]
add x6, x6, 32
add x13, x13, 32
add x14, x14, 32
add x15, x15, 32
add x19, x19, 32

    sub x1, x1, 8
    b.ne five
    b return
      tail_4:
      tbz x1, 2, tail_2
str  q12, [x6]
str  q13, [x13]
str  q14, [x14]
str  q15, [x15]
str  q16, [x19]
mov  v12.16b, v17.16b
mov  v13.16b, v18.16b
mov  v14.16b, v19.16b
mov  v15.16b, v20.16b
mov  v16.16b, v21.16b
add x6, x6, 16
add x13, x13, 16
add x14, x14, 16
add x15, x15, 16
add x19, x19, 16

  tail_2:
    tbz x1, 1, tail_1
str  d12, [x6]
str  d13, [x13]
str  d14, [x14]
str  d15, [x15]
str  d16, [x19]
add  x6, x6, 8
add  x13, x13, 8
add  x14, x14, 8
add  x15, x15, 8
add  x19, x19, 8
dup d12, v12.d[1]
dup d13, v13.d[1]
dup d14, v14.d[1]
dup d15, v15.d[1]
dup d16, v16.d[1]

  tail_1:
    tbz x1, 0, return
str  s12, [x6]
str  s13, [x13]
str  s14, [x14]
str  s15, [x15]
str  s16, [x19]

  return:
    # restore the callee saved registers
    ldp x19, x20, [sp, -48]
    ldp x21, x22, [sp, -32]
    ldp x23, x24, [sp, -16]
    ret
    END_FUNCTION xnn_qd8_f32_qc8w_gemm_minmax_ukernel_asm_5x8c4__aarch64_v_lane

    