// Copyright (c) 2025 Qualcomm Innovation Center, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause-Clear

// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
 
#include "src/xnnpack/assembly.h"
 
#define fpack_rhs_head_rows                    4
#define fpack_rhs_tail_rows                    1
#define fpack_lhs_zero_offset                  0
#define fpack_lhs_one_offset                   1
#define fpack_lhs_eight_offset                 8
#define fpack_input                            x0
#define fpack_output                           x1

BEGIN_FUNCTION xnn_pf32_get_word_sme_vl__asm_aarch64_neonsme
        SMSTART
        #Count the number of words in SVL
        cntw x0
        SMSTOP          
        ret
END_FUNCTION xnn_pf32_get_word_sme_vl__asm_aarch64_neonsme
        
BEGIN_FUNCTION xnn_pf32_pack_lhs__asm_aarch64_neonsme
        stp x29, x30, [sp, #-96]!
        mov x29, sp
        stp x23, x24, [sp, #16]
        stp x21, x22, [sp, #32]
        stp x19, x20, [sp, #48]
        stp x25, x26, [sp, #64]
        stp x27, x28, [sp, #80]
        SMSTART
        mov x20, x4
        mov x21, x4
        incw x20
        cntw x16
        sub x20, x20, #0x1
        sub x17, x16, #0x1
        #No of passes will be the division followed by ceil of width and vector length
        udiv x20, x20, x16
        ands x17, x21, x17
        sub x21, x20, #0x1
        sub x14, x16, #0x2
        mov x15, #0x0
        mov x9, fpack_input
        add x27, fpack_input, x16, LSL #3
        cntw x11, ALL, MUL #2
        ldr x25, [x9, fpack_lhs_zero_offset]
        cntw x10, ALL, MUL #3
        #No of iterations = (no of passes - 1)/2
        lsr x21, x21, #0x1
        ldr x23, [x27, fpack_lhs_zero_offset]
        #If no of passes is odd, then odd tail exist
        and x28, x20, #0x1
        csel x17, x17, x16, NE
        ldr x22, [x9, fpack_lhs_eight_offset]
        ptrue p2.s
        whilelt p3.s, XZR, x2
        ldr x20, [x27, fpack_lhs_eight_offset]
        whilelt p10.s, x16, x2
        mov x26, x3
        mov x24, fpack_output
        whilelt p0.s, x15, x4
        whilelt p1.s, x15, x4
        add x9, x9, #0x10
        add x27, x27, #0x10
        mov x13, #0x0
        #If the number of active vector elements is 2, then move to end condition
        cbz x14, K_Charge_End
        K_Charge_Start:
        psel p5, p1, p3.s[w13, fpack_lhs_zero_offset]
        psel p4, p1, p10.s[w13, fpack_lhs_zero_offset]
        psel p6, p1, p3.s[w13, fpack_lhs_one_offset]
        psel p7, p1, p10.s[w13, fpack_lhs_one_offset]
        ld1w { za0h.s[w13, fpack_lhs_zero_offset] }, p5/Z, [x25, x26, LSL #2]
        ldr x25, [x9, fpack_lhs_zero_offset]
        ld1w { za1h.s[w13, fpack_lhs_zero_offset] }, p4/Z, [x23, x26, LSL #2]
        ldr x23, [x27, fpack_lhs_zero_offset]
        ld1w { za0h.s[w13, fpack_lhs_one_offset] }, p6/Z, [x22, x26, LSL #2]
        ldr x22, [x9, fpack_lhs_eight_offset]
        add x9, x9, #0x10
        ld1w { za1h.s[w13, fpack_lhs_one_offset] }, p7/Z, [x20, x26, LSL #2]
        add x13, x13, #0x2
        ldr x20, [x27, fpack_lhs_eight_offset]
        add x27, x27, #0x10
        cmp x13, x14
        blt K_Charge_Start
        K_Charge_End:
        psel p5, p1, p3.s[w13, fpack_lhs_zero_offset]
        psel p4, p1, p10.s[w13, fpack_lhs_zero_offset]
        psel p6, p1, p3.s[w13, fpack_lhs_one_offset]
        psel p7, p1, p10.s[w13, fpack_lhs_one_offset]
        mov x9, fpack_input
        add x27, fpack_input, x16, LSL #3
        ld1w { za0h.s[w13, fpack_lhs_zero_offset] }, p5/Z, [x25, x26, LSL #2]
        ldr x25, [x9, fpack_lhs_zero_offset]
        incw x15
        ld1w { za1h.s[w13, fpack_lhs_zero_offset] }, p4/Z, [x23, x26, LSL #2]
        ldr x23, [x27, fpack_lhs_zero_offset]
        ld1w { za0h.s[w13, fpack_lhs_one_offset] }, p6/Z, [x22, x26, LSL #2]
        ldr x22, [x9, fpack_lhs_eight_offset]
        add x9, x9, #0x10
        ld1w { za1h.s[w13, fpack_lhs_one_offset] }, p7/Z, [x20, x26, LSL #2]
        ldr x20, [x27, fpack_lhs_eight_offset]
        add x27, x27, #0x10
        incw x26
        #Check for loop end condition
        cbz x21, K_Tails_Main
        mov x21, x21
        K_Main_Start:
        whilelt p1.s, x15, x4
        mov x12, #0x0
        #If the number of active vector elements is 2, then move to end condition
        cbz x14, K_Main_First_Tail
        K_Main_First:
        psel p7, p1, p3.s[w12, fpack_lhs_zero_offset]
        psel p4, p1, p10.s[w12, fpack_lhs_zero_offset]
        psel p6, p1, p3.s[w12, fpack_lhs_one_offset]
        psel p5, p1, p10.s[w12, fpack_lhs_one_offset]
        ld1w { za2h.s[w12, fpack_lhs_zero_offset] }, p7/Z, [x25, x26, LSL #2]
        psel p7, p2, p0.s[w12, fpack_lhs_zero_offset]
        ldr x25, [x9, fpack_lhs_zero_offset]
        ld1w { za3h.s[w12, fpack_lhs_zero_offset] }, p4/Z, [x23, x26, LSL #2]
        psel p4, p2, p0.s[w12, fpack_lhs_zero_offset]
        ldr x23, [x27, fpack_lhs_zero_offset]
        ld1w { za2h.s[w12, fpack_lhs_one_offset] }, p6/Z, [x22, x26, LSL #2]
        psel p6, p2, p0.s[w12, fpack_lhs_one_offset]
        ldr x22, [x9, fpack_lhs_eight_offset]
        add x9, x9, #0x10
        ld1w { za3h.s[w12, fpack_lhs_one_offset] }, p5/Z, [x20, x26, LSL #2]
        ldr x20, [x27, fpack_lhs_eight_offset]
        st1w { za0v.s[w12, fpack_lhs_zero_offset] }, p7, [x24, XZR, LSL #2]
        psel p7, p2, p0.s[w12, fpack_lhs_one_offset]
        st1w { za1v.s[w12, fpack_lhs_zero_offset] }, p4, [x24, x16, LSL #2]
        add x27, x27, #0x10
        st1w { za0v.s[w12, fpack_lhs_one_offset] }, p6, [x24, x11, LSL #2]
        st1w { za1v.s[w12, fpack_lhs_one_offset] }, p7, [x24, x10, LSL #2]
        add x12, x12, #0x2
        addvl x24, x24, #4
        cmp x12, x14
        blt K_Main_First
        K_Main_First_Tail:
        psel p7, p1, p3.s[w12, fpack_lhs_zero_offset]
        psel p4, p1, p10.s[w12, fpack_lhs_zero_offset]
        psel p6, p1, p3.s[w12, fpack_lhs_one_offset]
        psel p5, p1, p10.s[w12, fpack_lhs_one_offset]
        mov x9, fpack_input
        add x27, fpack_input, x16, LSL #3
        ld1w { za2h.s[w12, fpack_lhs_zero_offset] }, p7/Z, [x25, x26, LSL #2]
        psel p7, p2, p0.s[w12, fpack_lhs_zero_offset]
        ldr x25, [x9, fpack_lhs_zero_offset]
        mov x13, #0x0
        ld1w { za3h.s[w12, fpack_lhs_zero_offset] }, p4/Z, [x23, x26, LSL #2]
        psel p4, p2, p0.s[w12, fpack_lhs_zero_offset]
        ldr x23, [x27, fpack_lhs_zero_offset]
        ld1w { za2h.s[w12, fpack_lhs_one_offset] }, p6/Z, [x22, x26, LSL #2]
        psel p6, p2, p0.s[w12, fpack_lhs_one_offset]
        ldr x22, [x9, fpack_lhs_eight_offset]
        add x9, x9, #0x10
        ld1w { za3h.s[w12, fpack_lhs_one_offset] }, p5/Z, [x20, x26, LSL #2]
        ldr x20, [x27, fpack_lhs_eight_offset]
        st1w { za0v.s[w12, fpack_lhs_zero_offset] }, p7, [x24, XZR, LSL #2]
        psel p7, p2, p0.s[w12, fpack_lhs_one_offset]
        st1w { za1v.s[w12, fpack_lhs_zero_offset] }, p4, [x24, x16, LSL #2]
        whilelt p0.s, x15, x4
        incw x15
        st1w { za0v.s[w12, fpack_lhs_one_offset] }, p6, [x24, x11, LSL #2]
        add x27, x27, #0x10
        incw x26
        st1w { za1v.s[w12, fpack_lhs_one_offset] }, p7, [x24, x10, LSL #2]
        addvl x24, x24, #4
        whilelt p1.s, x15, x4
        #If the number of active vector elements is 2, then move to end condition
        cbz x14, K_Main_Second_Tail
        K_Main_Second:
        psel p7, p1, p3.s[w13, fpack_lhs_zero_offset]
        psel p4, p1, p10.s[w13, fpack_lhs_zero_offset]
        psel p6, p1, p3.s[w13, fpack_lhs_one_offset]
        psel p5, p1, p10.s[w13, fpack_lhs_one_offset]
        ld1w { za0h.s[w13, fpack_lhs_zero_offset] }, p7/Z, [x25, x26, LSL #2]
        psel p7, p2, p0.s[w13, fpack_lhs_zero_offset]
        ldr x25, [x9, fpack_lhs_zero_offset]
        ld1w { za1h.s[w13, fpack_lhs_zero_offset] }, p4/Z, [x23, x26, LSL #2]
        psel p4, p2, p0.s[w13, fpack_lhs_zero_offset]
        ldr x23, [x27, fpack_lhs_zero_offset]
        ld1w { za0h.s[w13, fpack_lhs_one_offset] }, p6/Z, [x22, x26, LSL #2]
        psel p6, p2, p0.s[w13, fpack_lhs_one_offset]
        ldr x22, [x9, fpack_lhs_eight_offset]
        add x9, x9, #0x10
        ld1w { za1h.s[w13, fpack_lhs_one_offset] }, p5/Z, [x20, x26, LSL #2]
        ldr x20, [x27, fpack_lhs_eight_offset]
        st1w { za2v.s[w13, fpack_lhs_zero_offset] }, p7, [x24, XZR, LSL #2]
        psel p7, p2, p0.s[w13, fpack_lhs_one_offset]
        st1w { za3v.s[w13, fpack_lhs_zero_offset] }, p4, [x24, x16, LSL #2]
        add x27, x27, #0x10
        st1w { za2v.s[w13, fpack_lhs_one_offset] }, p6, [x24, x11, LSL #2]
        st1w { za3v.s[w13, fpack_lhs_one_offset] }, p7, [x24, x10, LSL #2]
        add x13, x13, #0x2
        addvl x24, x24, #4
        cmp x13, x14
        blt K_Main_Second
        K_Main_Second_Tail:
        psel p7, p1, p3.s[w13, fpack_lhs_zero_offset]
        psel p4, p1, p10.s[w13, fpack_lhs_zero_offset]
        psel p6, p1, p3.s[w13, fpack_lhs_one_offset]
        psel p5, p1, p10.s[w13, fpack_lhs_one_offset]
        mov x9, fpack_input
        add x27, fpack_input, x16, LSL #3
        ld1w { za0h.s[w13, fpack_lhs_zero_offset] }, p7/Z, [x25, x26, LSL #2]
        psel p7, p2, p0.s[w13, fpack_lhs_zero_offset]
        ldr x25, [x9, fpack_lhs_zero_offset]
        ld1w { za1h.s[w13, fpack_lhs_zero_offset] }, p4/Z, [x23, x26, LSL #2]
        psel p4, p2, p0.s[w13, fpack_lhs_zero_offset]
        ldr x23, [x27, fpack_lhs_zero_offset]
        ld1w { za0h.s[w13, fpack_lhs_one_offset] }, p6/Z, [x22, x26, LSL #2]
        psel p6, p2, p0.s[w13, fpack_lhs_one_offset]
        ldr x22, [x9, fpack_lhs_eight_offset]
        add x9, x9, #0x10
        ld1w { za1h.s[w13, fpack_lhs_one_offset] }, p5/Z, [x20, x26, LSL #2]
        ldr x20, [x27, fpack_lhs_eight_offset]
        st1w { za2v.s[w13, fpack_lhs_zero_offset] }, p7, [x24, XZR, LSL #2]
        psel p7, p2, p0.s[w13, fpack_lhs_one_offset]
        st1w { za3v.s[w13, fpack_lhs_zero_offset] }, p4, [x24, x16, LSL #2]
        whilelt p0.s, x15, x4
        subs x21, x21, #0x1
        st1w { za2v.s[w13, fpack_lhs_one_offset] }, p6, [x24, x11, LSL #2]
        add x27, x27, #0x10
        incw x15
        st1w { za3v.s[w13, fpack_lhs_one_offset] }, p7, [x24, x10, LSL #2]
        addvl x24, x24, #4
        incw x26
        bgt K_Main_Start
        K_Tails_Main:
        #Check if odd tail exists
        cbnz x28, K_Tails_Odd
        mov x9, fpack_input
        whilelt p1.s, x15, x4
        mov x13, #0x0
        #Loops to store the even tails
        K_Tails_Even_First:
        psel p5, p2, p0.s[w13, fpack_lhs_zero_offset]
        psel p4, p2, p0.s[w13, fpack_lhs_zero_offset]
        psel p6, p1, p3.s[w13, fpack_lhs_zero_offset]
        psel p7, p1, p10.s[w13, fpack_lhs_zero_offset]
        st1w { za0v.s[w13, fpack_lhs_zero_offset] }, p5, [x24, XZR, LSL #2]
        st1w { za1v.s[w13, fpack_lhs_zero_offset] }, p4, [x24, x16, LSL #2]
        addvl x24, x24, #2
        ldr x20, [x9, fpack_lhs_zero_offset]
        ldr x21, [x9, x16, LSL #0x3]
        add x9, x9, #0x8
        ld1w { za2h.s[w13, fpack_lhs_zero_offset] }, p6/Z, [x20, x26, LSL #2]
        ld1w { za3h.s[w13, fpack_lhs_zero_offset] }, p7/Z, [x21, x26, LSL #2]
        add x13, x13, #0x1
        cmp x13, x16
        blt K_Tails_Even_First
        whilelt p0.s, x15, x4
        whilelt p1.s, x15, x4
        mov x13, #0x0
        K_Tails_Even_Second:
        psel p6, p2, p0.s[w13, fpack_lhs_zero_offset]
        psel p7, p2, p0.s[w13, fpack_lhs_zero_offset]
        st1w { za2v.s[w13, fpack_lhs_zero_offset] }, p6, [x24, XZR, LSL #2]
        st1w { za3v.s[w13, fpack_lhs_zero_offset] }, p7, [x24, x16, LSL #2]
        add x13, x13, #0x1
        addvl x24, x24, #2
        cmp x13, x17
        blt K_Tails_Even_Second
        whilelt p1.s, x15, x4
        #Jump to end as odd tail does not exist
        b K_End
        #Store the odd tail
        K_Tails_Odd:
        mov x13, #0x0
        K_Tails_Odd_Loop:
        psel p6, p2, p0.s[w13, fpack_lhs_zero_offset]
        psel p7, p2, p0.s[w13, fpack_lhs_zero_offset]
        st1w { za0v.s[w13, fpack_lhs_zero_offset] }, p6, [x24, XZR, LSL #2]
        st1w { za1v.s[w13, fpack_lhs_zero_offset] }, p7, [x24, x16, LSL #2]
        add x13, x13, #0x1
        addvl x24, x24, #2
        cmp x13, x17
        blt K_Tails_Odd_Loop
        K_End:
        mov fpack_output, x24
        SMSTOP
        ldp x19, x20, [sp, #48]
        ldp x21, x22, [sp, #32]
        ldp x23, x24, [sp, #16]
        ldp x25, x26, [sp, #64]
        ldp x27, x28, [sp, #80]
        ldp x29, x30, [sp], #96
        ret
END_FUNCTION xnn_pf32_pack_lhs__asm_aarch64_neonsme

BEGIN_FUNCTION xnn_pf32_pack_rhs__asm_aarch64_neonsme
        stp x29, x30, [sp, #-96]!
        mov x29, sp
        stp x23, x24, [sp, #16]
        stp x21, x22, [sp, #32]
        stp x19, x20, [sp, #48]
        stp x25, x26, [sp, #64]
        stp x27, x28, [sp, #80]
        SMSTART
        mov x15, fpack_output
        mov x14, #0
        ptrue p0.b
        #Load the entire bias into the output
        Store_Bias:
        whilelt p2.s, x14, x6
        incw x14
        whilelt p3.s, x14, x6
        incw x14
        cmp x14, x6
        ld1w { z10.s }, p2/Z, [x4]
        ld1w { z11.s }, p3/Z, [x4, #1, MUL VL]
        incb x4, ALL, MUL #2
        st1w { z10.s }, p0, [x15]
        st1w { z11.s }, p0, [x15, #1, MUL VL]
        add x15, x15, x5
        blt Store_Bias
        cmp x2, fpack_rhs_head_rows
        incb fpack_output, ALL, MUL #2
        blt SkipProcess4xRows
        #Process 4 rows of input and store it into output
        Process_4xRows:
        mov x15, #0
        mov x10, fpack_output
        mov x11, fpack_input
        add x12, x11, x3
        add x13, x12, x3
        add x14, x13, x3
        sub x2, x2, fpack_rhs_head_rows
        #Re-initialize input index for next iteration
        add fpack_input, x14, x3
        #Store 4 rows into output
        Store4xRows:
        whilelt p2.s, x15, x6
        incw x15
        whilelt p3.s, x15, x6
        incw x15
        cmp x15, x6
        ld1w { z10.s }, p2/Z, [x11]
        ld1w { z11.s }, p3/Z, [x11, #1, MUL VL]
        addvl x11, x11, #2
        ld1w { z12.s }, p2/Z, [x12]
        ld1w { z13.s }, p3/Z, [x12, #1, MUL VL]
        addvl x12, x12, #2
        ld1w { z14.s }, p2/Z, [x13]
        ld1w { z15.s }, p3/Z, [x13, #1, MUL VL]
        addvl x13, x13, #2
        ld1w { z16.s }, p2/Z, [x14]
        ld1w { z17.s }, p3/Z, [x14, #1, MUL VL]
        addvl x14, x14, #2
        st1w { z10.s }, p0, [x10]
        st1w { z11.s }, p0, [x10, #1, MUL VL]
        st1w { z12.s }, p0, [x10, #2, MUL VL]
        st1w { z13.s }, p0, [x10, #3, MUL VL]
        st1w { z14.s }, p0, [x10, #4, MUL VL]
        st1w { z15.s }, p0, [x10, #5, MUL VL]
        st1w { z16.s }, p0, [x10, #6, MUL VL]
        st1w { z17.s }, p0, [x10, #7, MUL VL]
        add x10, x10, x5
        blt Store4xRows
        cmp x2, fpack_rhs_head_rows
        addvl fpack_output, fpack_output, #8
        bge Process_4xRows
        cbz x2, end
        #End of loop for processing 4 rows
        SkipProcess4xRows:
        #Tail loop to process last few rows of input
        Process1xRows:
        mov x11, fpack_input
        mov x10, fpack_output
        mov x14, #0
        add fpack_input, x11, x3
        sub x2, x2, fpack_rhs_tail_rows
        #End of tail loop
        Store1xRows:
        whilelt p2.s, x14, x6
        incw x14
        whilelt p3.s, x14, x6
        incw x14
        cmp x14, x6
        ld1w { z10.s }, p2/Z, [x11]
        ld1w { z11.s }, p3/Z, [x11, #1, MUL VL]
        addvl x11, x11, #2
        st1w { z10.s }, p0, [x10]
        st1w { z11.s }, p0, [x10, #1, MUL VL]
        add x10, x10, x5
        blt Store1xRows
        cmp x2, fpack_rhs_tail_rows
        addvl fpack_output, fpack_output, #2
        bge Process1xRows
        #End
        end:
        SMSTOP
        ldp x19, x20, [sp, #48]
        ldp x21, x22, [sp, #32]
        ldp x23, x24, [sp, #16]
        ldp x25, x26, [sp, #64]
        ldp x27, x28, [sp, #80]
        ldp x29, x30, [sp], #96
        ret
END_FUNCTION xnn_pf32_pack_rhs__asm_aarch64_neonsme

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
