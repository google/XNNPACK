// Auto-generated file. Do not edit!
//   Template: src/x32-packw/x8-aarch32-neon.S.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

.syntax unified

// void xnn_x32_packw_gemm_goi_ukernel_x8__asm_aarch32_neon(
//     size_t g,                       r0
//     size_t nc,                      r1
//     size_t kc,                      r2
//     size_t nr,                      r3
//     size_t kr,               (sp + 372) ignored
//     size_t sr,                sp + 376
//     const uint32_t* k,        sp + 380 -> r5
//     const uint32_t* b,        sp + 384 -> r2
//     uint32_t* packed_weights, sp + 388 -> sl
//     size_t extra_bytes,      (sp + 392) ignored
//     const void* params);     (sp + 396)

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// K0  r9  d16, d17
// K1  r5  d18, d19
// K2  r1  d20, d21
// K3  r8  d22, d23
// K4  r6  d24, d25
// K5  r3  d26, d27
// K6  r7  d28, d29
// K7  r0  d30, d31

// main loop register usage
// r5     bias
// r9,r10 k
// r4     kc (stride)
// r12,r2 out
// r2     kloop

BEGIN_FUNCTION xnn_x32_packw_gemm_goi_ukernel_x8__asm_aarch32_neon
        .arm
#ifndef __APPLE__
        .arch   armv7-a
        .fpu    neon
#endif
        # Push 368 bytes
        PUSH     {r4, r5, r6, r7, r8, r9, r10, r11, lr} // 36
        SUB      sp, sp, #4                             //  +4 = 40
        VPUSH    {d8-d15}                               // +64 = 104
        SUB      sp, sp, #264                           // +264 = 368

        MOV      lr, r2
        MUL      r2, r2, r1
        VMOV.I32 q6, #0x0
        LDR      r10, [sp, #388]
        LSL      r4, lr, #2
        VORR     q12, q6, q6
        LDR      r5, [sp, #380]
        STR      r2, [sp, #220]
        LSL      r2, lr, #4
        STR      r2, [sp, #256]
        LSL      r2, lr, #3
        STR      r2, [sp, #252]
        RSB      r2, lr, lr, lsl #3
        VORR     q8, q6, q6
        LDR      r9, [sp, #376]
        LSL      r2, r2, #2
        VORR     q13, q6, q6
        STR      r2, [sp, #248]
        ADD      r2, lr, lr, lsl #2
        VORR     q9, q6, q6
        VLDR     s3, .LCPI0_1
        LSL      r2, r2, #2
        VORR     q14, q6, q6
        STR      r2, [sp, #244]
        ADD      r2, lr, lr, lsl #1
        VORR     q10, q6, q6
        STR      r1, [sp, #224]
        LSL      r3, r2, #3
        LSL      r2, r2, #2
        STR      r3, [sp, #240]
        ADD      r3, sp, #136
        STR      r2, [sp, #236]
        VORR     q15, q6, q6
        LDR      r2, [sp, #384]
        VORR     q11, q6, q6
        VSTMIA   r3, {d0, d1}
        ADD      r3, sp, #152
        VMOV.I32 q2, #0x0
        VSTMIA   r3, {d0, d1, d2, d3, d4, d5, d6, d7}
.LBB0_1:

        CMP      r1, #8
        STR      r0, [sp, #232]
        STR      r9, [sp, #228]
        BLO      .LBB0_18

        # Copy 8 bias values
.LBB0_2:
        CMP      r5, #0            // bias = NULL?
        BEQ      .LBB0_4

        MOV      r3, r2
        VLD1.32  {d2, d3}, [r5]!
        VLD1.32  {d4, d5}, [r5]!
        VST1.32  {d2, d3}, [r3]!
        VST1.32  {d4, d5}, [r3]
.LBB0_4:
        ADD      r12, r2, #32
        SUBS     r2, lr, #4         // Is there K of 4?
        STR      r5, [sp, #260]
        BLO      .LBB0_11           // No, handle remainder

        ADD      r10, r9, r4        // 2nd row

        # Main loop - 8x4 transposed to 4x8
.LBB0_6:
        VLD4.32  {d16[0], d18[0], d20[0], d22[0]}, [r9]!
        VLD4.32  {d16[1], d18[1], d20[1], d22[1]}, [r10], r4
        VLD4.32  {d17[0], d19[0], d21[0], d23[0]}, [r10], r4
        VLD4.32  {d17[1], d19[1], d21[1], d23[1]}, [r10], r4

        VLD4.32  {d24[0], d26[0], d28[0], d30[0]}, [r10], r4
        VLD4.32  {d24[1], d26[1], d28[1], d30[1]}, [r10], r4
        VLD4.32  {d25[0], d27[0], d29[0], d31[0]}, [r10], r4
        VLD4.32  {d25[1], d27[1], d29[1], d31[1]}, [r10]


        SUBS     r2, r2, #4

        VST1.32  {d16, d17}, [r12]!
        VST1.32  {d24, d25}, [r12]!
        VST1.32  {d18, d19}, [r12]!
        VST1.32  {d26, d27}, [r12]!
        ADD      r10, r9, r4        // 2nd row
        VST1.32  {d20, d21}, [r12]!
        VST1.32  {d28, d29}, [r12]!
        VST1.32  {d22, d23}, [r12]!
        VST1.32  {d30, d31}, [r12]!

        BHS      .LBB0_6

        LDR      r5, [sp, #256]
        LDR      r3, [sp, #240]
        ADD      r6, r9, r5
        LDR      r5, [sp, #236]
        LDR      r0, [sp, #248]
        ADD      r7, r9, r3
        LDR      r3, [sp, #244]
        ADD      r8, r9, r5
        LDR      r5, [sp, #252]
        ADD      r0, r9, r0
        ADD      r3, r9, r3
        ADD      r11, r9, r5
        ADD      r5, r9, r4
        LDR      r10, [sp, #388]
        ADDS     r2, r2, #4    // Remainder 0?
        BEQ      .LBB0_12
.LBB0_8:
        CMP      r2, #3        // Remainder of 3?
        BEQ      .LBB0_15

        CMP      r2, #1        // Remainder of 1?
        BEQ      .LBB0_16

        ADD      r2, sp, #152
        VLDMIA   r2, {d0, d1, d2, d3, d4, d5, d6, d7}
        ADD      r2, sp, #72
        VORR     q3, q2, q2
        VSTMIA   r2, {d16, d17, d18, d19, d20, d21, d22, d23}
        ADD      r2, sp, #152
        VORR     q8, q2, q2
        VORR     q9, q3, q3
        VSTMIA   r2, {d0, d1, d2, d3, d4, d5, d6, d7}
        VORR     q1, q2, q2
        MOV      r2, r12
        VORR     q2, q3, q3
        VLD2.32  {d16[0], d18[0]}, [r9]
        VLD2.32  {d2[0], d4[0]}, [r6]
        ADD      r6, sp, #72
        VLD2.32  {d16[1], d18[1]}, [r5]
        MOV      r5, #12
        VLD2.32  {d2[1], d4[1]}, [r3]
        MOV      r3, #16
        VLD2.32  {d17[0], d19[0]}, [r11]
        VORR     q11, q9, q9
        VLD2.32  {d3[0], d5[0]}, [r7]
        VORR     q10, q8, q8
        VORR     q9, q2, q2
        VORR     q8, q1, q1
        VLD2.32  {d21[1], d23[1]}, [r8]
        VORR     q1, q10, q10
        VLD2.32  {d17[1], d19[1]}, [r0]!
        VORR     q5, q11, q11
        VORR     q10, q8, q8
        VST1.32  {d2, d3}, [r2]!
        VORR     q1, q9, q9
        VST1.32  {d20, d21}, [r2]!
        VLDMIA   r6, {d16, d17, d18, d19, d20, d21, d22, d23}
        B        .LBB0_17
.LBB0_11:
        ADD      r5, r9, lr, lsl #2
        MOV      r2, lr
        ADD      r11, r5, lr, lsl #2
        ADD      r8, r11, lr, lsl #2
        ADD      r6, r8, lr, lsl #2
        ADD      r3, r6, lr, lsl #2
        ADD      r7, r3, lr, lsl #2
        ADD      r0, r7, lr, lsl #2
        CMP      r2, #0
        BNE      .LBB0_8
.LBB0_12:
        MOV      r9, r0
.LBB0_13:
        LDR      r5, [sp, #260]
        SUB      r1, r1, #8
        ADD      r2, r12, r10
        CMP      r1, #7
        BHI      .LBB0_2
        B        .LBB0_18
.LBB0_15:
        ADD      r2, sp, #152
        VLDMIA   r2, {d0, d1, d2, d3, d4, d5, d6, d7}
        ADD      r2, sp, #72
        VORR     q0, q2, q2
        VORR     q1, q2, q2
        VSTMIA   r2, {d16, d17, d18, d19, d20, d21, d22, d23}
        VORR     q8, q0, q0
        ADD      r2, sp, #8
        VORR     q9, q1, q1
        VORR     q10, q2, q2
        VORR     q11, q3, q3
        VLD3.32  {d16[0], d18[0], d20[0]}, [r6]
        VLD3.32  {d16[1], d18[1], d20[1]}, [r3]
        ADD      r3, sp, #8
        VLD3.32  {d17[0], d19[0], d21[0]}, [r7]
        VORR     q6, q10, q10
        VORR     q5, q9, q9
        VORR     q4, q8, q8
        VLD3.32  {d9[1], d11[1], d13[1]}, [r0]!
        VSTMIA   r2, {d8, d9, d10, d11, d12, d13, d14, d15}
        ADD      r2, sp, #152
        VSTMIA   r2, {d0, d1, d2, d3, d4, d5, d6, d7}
        MOV      r2, r12
        VLD3.32  {d0[0], d2[0], d4[0]}, [r9]
        VORR     q10, q2, q2
        VORR     q9, q1, q1
        VORR     q8, q0, q0
        VLD3.32  {d16[1], d18[1], d20[1]}, [r5]
        MOV      r5, #20
        VORR     q2, q10, q10
        VORR     q1, q9, q9
        VORR     q0, q8, q8
        VLD3.32  {d1[0], d3[0], d5[0]}, [r11]
        VORR     q10, q2, q2
        VORR     q9, q1, q1
        VORR     q8, q0, q0
        VLD3.32  {d17[1], d19[1], d21[1]}, [r8]
        VORR     q0, q8, q8
        VORR     q5, q10, q10
        VORR     q8, q9, q9
        VST1.32  {d0, d1}, [r2]!
        VLDMIA   r3, {d0, d1, d2, d3, d4, d5, d6, d7}
        VORR     q9, q1, q1
        ADD      r3, sp, #72
        VORR     q1, q2, q2
        VORR     q10, q0, q0
        VST1.32  {d20, d21}, [r2]!
        VST1.32  {d16, d17}, [r2]!
        VST1.32  {d18, d19}, [r2]!
        VLDMIA   r3, {d16, d17, d18, d19, d20, d21, d22, d23}
        MOV      r3, #24
        B        .LBB0_17

        # KC Remainder of 1
.LBB0_16:
        VLD1.32  {d10[0]}, [r9:32]
        VLD1.32  {d10[1]}, [r5:32]
        VLD1.32  {d11[0]}, [r11:32]
        VLD1.32  {d11[1]}, [r8:32]
        VLD1.32  {d2[0]}, [r6:32]
        VLD1.32  {d2[1]}, [r3:32]
        VLD1.32  {d3[0]}, [r7:32]
        VLD1.32  {d3[1]}, [r0:32]!
        MOV      r5, #4
        MOV      r3, #8
        MOV      r2, r12
.LBB0_17:
        VST1.32  {d10, d11}, [r2]
        ADD      r2, r12, r5, lsl #2
        ADD      r12, r12, r3, lsl #2
        MOV      r9, r0
        VST1.32  {d2, d3}, [r2]
        B        .LBB0_13
.LBB0_18:
        MOV      r3, r5
        CMP      r1, #0
        BNE      .LBB0_22

        MOV      r5, r3
        LDR      r1, [sp, #224]
        LDR      r0, [sp, #232]
.LBB0_20:
        LDR      r3, [sp, #220]
        SUBS     r0, r0, #1
        LDR      r9, [sp, #228]
        ADD      r9, r9, r3, lsl #2
        BNE      .LBB0_1
        B        .LBB0_45
        .P2ALIGN 2

.LCPI0_1 :
        .LONG    0x00000000
        .P2ALIGN 2
.LBB0_22:
        CMP      r3, #0
        BEQ      .LBB0_38

        CMP      r1, #4
        BLO      .LBB0_28

        ADD      r7, r3, r1, lsl #2
        CMP      r2, r7
        ADDLO    r7, r2, r1, lsl #2
        CMPLO    r3, r7
        BLO      .LBB0_28

        BIC      r12, r1, #3
        VORR     q0, q8, q8
        VORR     q1, q9, q9
        AND      r6, r1, #3
        VORR     q2, q10, q10
        ADD      r5, r3, r12, lsl #2
        VORR     q3, q11, q11
        ADD      r7, r2, r12, lsl #2
        MOV      r0, r12
.LBB0_26:

        VLD1.32  {d16, d17}, [r3]!
        SUBS     r0, r0, #4
        VST1.32  {d16, d17}, [r2]!
        BNE      .LBB0_26

        VORR     q8, q0, q0
        CMP      r1, r12
        VORR     q9, q1, q1
        VORR     q10, q2, q2
        VORR     q11, q3, q3
        BNE      .LBB0_29
        B        .LBB0_30
.LBB0_28:
        MOV      r6, r1
        MOV      r7, r2
        MOV      r5, r3
.LBB0_29:

        LDR      r2, [r5], #4
        SUBS     r6, r6, #1
        STR      r2, [r7], #4
        BNE      .LBB0_29
.LBB0_30:
        RSB      r2, r1, #8
        ADD      r12, r7, r2, lsl #2
.LBB0_31:
        CMP      r1, #3
        MOV      r2, lr
        MOVWLO   r2, #0
        CMP      r1, #2
        MOV      r3, lr
        MOVWLO   r3, #0
        CMP      r1, #4
        ADD      r0, r9, r3, lsl #2
        ADD      r3, r0, r2, lsl #2
        MOV      r2, lr
        MOVWLO   r2, #0
        STR      r3, [sp, #72]
        ADD      r3, r3, r2, lsl #2
        CMP      r1, #5
        MOV      r2, lr
        MOVWLO   r2, #0
        CMP      r1, #6
        ADD      r7, r3, r2, lsl #2
        MOV      r2, lr
        MOVWLO   r2, #0
        CMP      r1, #7
        MOV      r1, lr
        ADD      r6, r7, r2, lsl #2
        MOVWLO   r1, #0
        MOV      r8, r7
        CMP      lr, #4
        ADD      r11, r6, r1, lsl #2
        STR      r11, [sp, #260]
        BLO      .LBB0_35

        VORR     q4, q12, q12
        MOV      r11, r0
        VORR     q5, q13, q13
        LDR      r0, [sp, #72]
        VORR     q6, q14, q14
        MOV      r1, #0
        VORR     q7, q15, q15
        MOV      r2, lr
        MOV      r7, r3
.LBB0_33:

        ADD      r3, r9, r1
        SUB      r2, r2, #4
        CMP      r2, #3
        VLD4.32  {d16[0], d18[0], d20[0], d22[0]}, [r3]
        ADD      r3, r11, r1
        VLD4.32  {d16[1], d18[1], d20[1], d22[1]}, [r3]
        ADD      r3, r0, r1
        VORR     q12, q8, q8
        VORR     q13, q9, q9
        VORR     q14, q10, q10
        VORR     q15, q11, q11
        VLD4.32  {d17[0], d19[0], d21[0], d23[0]}, [r3]
        ADD      r3, r7, r1
        VORR     q3, q11, q11
        VORR     q2, q10, q10
        VORR     q1, q9, q9
        VORR     q0, q8, q8
        VLD4.32  {d1[1], d3[1], d5[1], d7[1]}, [r3]
        ADD      r3, r8, r1
        VORR     q15, q3, q3
        VLD4.32  {d8[0], d10[0], d12[0], d14[0]}, [r3]
        ADD      r3, r6, r1
        VORR     q14, q2, q2
        VORR     q13, q1, q1
        VORR     q12, q0, q0
        VORR     q11, q7, q7
        VORR     q10, q6, q6
        VORR     q9, q5, q5
        VORR     q8, q4, q4
        VLD4.32  {d16[1], d18[1], d20[1], d22[1]}, [r3]
        VORR     q3, q11, q11
        VORR     q2, q10, q10
        LDR      r3, [sp, #260]
        VORR     q1, q9, q9
        VORR     q0, q8, q8
        ADD      r3, r3, r1
        VORR     q8, q12, q12
        ADD      r1, r1, #16
        VORR     q9, q13, q13
        VLD4.32  {d1[0], d3[0], d5[0], d7[0]}, [r3]
        VORR     q10, q14, q14
        VORR     q11, q15, q15
        VORR     q7, q3, q3
        VORR     q6, q2, q2
        VST1.32  {d16, d17}, [r12]!
        VORR     q5, q1, q1
        VORR     q4, q0, q0
        VST1.32  {d8, d9}, [r12]!
        VST1.32  {d18, d19}, [r12]!
        VST1.32  {d10, d11}, [r12]!
        VST1.32  {d20, d21}, [r12]!
        VST1.32  {d12, d13}, [r12]!
        VST1.32  {d22, d23}, [r12]!
        VST1.32  {d14, d15}, [r12]!
        BHI      .LBB0_33

        LDR      r3, [sp, #260]
        VORR     q12, q4, q4
        VORR     q13, q5, q5
        ADD      r8, r8, r1
        VORR     q14, q6, q6
        ADD      r3, r3, r1
        STR      r3, [sp, #260]
        MOV      r3, r7
        VORR     q15, q7, q7
        ADD      r3, r7, r1
        ADD      r7, r0, r1
        ADD      r9, r9, r1
        ADD      r6, r6, r1
        ADD      r11, r11, r1
        STR      r6, [sp, #4]
        STR      r11, [sp, #8]
        B        .LBB0_36
.LBB0_35:
        STR      r6, [sp, #4]
        MOV      r2, lr
        STR      r0, [sp, #8]
        LDR      r7, [sp, #72]
.LBB0_36:
        LDR      r1, [sp, #224]
        CMP      r2, #0
        LDR      r0, [sp, #232]
        BNE      .LBB0_39
.LBB0_37:
        ADD      r2, r12, r10
        B        .LBB0_20
.LBB0_38:
        ADD      r12, r2, #32
        MOV      r5, #0
        B        .LBB0_31
.LBB0_39:
        MOV      r1, r3
        ADD      r3, sp, #72
        MOV      r6, r7
        MOV      r10, r0
        CMP      r2, #3
        VSTMIA   r3, {d16, d17, d18, d19, d20, d21, d22, d23}
        BEQ      .LBB0_42

        CMP      r2, #2
        BNE      .LBB0_43

        ADD      r0, sp, #152
        VLDMIA   r0, {d0, d1, d2, d3, d4, d5, d6, d7}
        VORR     q8, q2, q2
        VORR     q9, q2, q2
        VORR     q10, q8, q8
        LDR      r0, [sp, #8]
        VORR     q11, q9, q9
        VLD2.32  {d16[0], d18[0]}, [r8]
        VLD2.32  {d20[0], d22[0]}, [r9]
        VLD2.32  {d20[1], d22[1]}, [r0]
        VORR     q1, q11, q11
        LDR      r0, [sp, #4]
        VORR     q0, q10, q10
        VLD2.32  {d16[1], d18[1]}, [r0]
        VORR     q11, q9, q9
        VLD2.32  {d1[0], d3[0]}, [r6]
        VORR     q10, q8, q8
        VORR     q9, q1, q1
        VORR     q8, q0, q0
        LDR      r0, [sp, #260]
        VLD2.32  {d17[1], d19[1]}, [r1]
        MOV      r1, #12
        VORR     q1, q8, q8
        VLD2.32  {d21[0], d23[0]}, [r0]
        MOV      r0, r12
        VORR     q8, q9, q9
        VORR     q0, q10, q10
        VST1.32  {d2, d3}, [r0]!
        VORR     q1, q11, q11
        VST1.32  {d0, d1}, [r0]!
        VST1.32  {d16, d17}, [r0]
        MOV      r0, #16
        B        .LBB0_44
.LBB0_42:
        ADD      r0, sp, #152
        VLDMIA   r0, {d16, d17, d18, d19, d20, d21, d22, d23}
        VORR     q4, q10, q10
        VORR     q5, q10, q10
        VORR     q6, q10, q10
        LDR      r0, [sp, #8]
        VORR     q8, q4, q4
        VORR     q9, q5, q5
        VORR     q10, q6, q6
        VORR     q11, q7, q7
        VLD3.32  {d8[0], d10[0], d12[0]}, [r8]
        VLD3.32  {d16[0], d18[0], d20[0]}, [r9]
        VLD3.32  {d16[1], d18[1], d20[1]}, [r0]
        VLD3.32  {d17[0], d19[0], d21[0]}, [r6]
        VORR     q2, q10, q10
        VORR     q1, q9, q9
        LDR      r0, [sp, #4]
        VORR     q0, q8, q8
        VORR     q10, q6, q6
        VORR     q9, q5, q5
        VORR     q8, q4, q4
        VLD3.32  {d1[1], d3[1], d5[1]}, [r1]
        MOV      r1, #20
        VLD3.32  {d16[1], d18[1], d20[1]}, [r0]
        VORR     q6, q10, q10
        VORR     q5, q9, q9
        LDR      r0, [sp, #260]
        VORR     q4, q8, q8
        VORR     q10, q0, q0
        VORR     q9, q1, q1
        VLD3.32  {d9[0], d11[0], d13[0]}, [r0]
        MOV      r0, r12
        VORR     q8, q2, q2
        VORR     q0, q4, q4
        VORR     q11, q5, q5
        VST1.32  {d20, d21}, [r0]!
        VORR     q1, q6, q6
        VST1.32  {d0, d1}, [r0]!
        VST1.32  {d18, d19}, [r0]!
        VST1.32  {d22, d23}, [r0]!
        VST1.32  {d16, d17}, [r0]
        MOV      r0, #24
        B        .LBB0_44
.LBB0_43:
        ADD      r0, sp, #136
        VLD1.32  {d17[0]}, [r6:32]
        ADD      r2, sp, #136
        VLD1.32  {d16[0]}, [r9:32]
        VLD1.32  {d17[1]}, [r1:32]
        MOV      r1, #4
        VLDMIA   r0, {d2, d3}
        LDR      r0, [sp, #8]
        VLD1.32  {d2[0]}, [r8:32]
        VLD1.32  {d16[1]}, [r0:32]
        LDR      r0, [sp, #260]
        VLDR     s6, [r0]
        LDR      r0, [sp, #4]
        VLD1.32  {d2[1]}, [r0:32]
        MOV      r0, #8
        VST1.32  {d16, d17}, [r12]
        VSTMIA   r2, {d2, d3}
.LBB0_44:
        ADD      r2, sp, #72
        ADD      r1, r12, r1, lsl #2
        ADD      r12, r12, r0, lsl #2
        MOV      r0, r10
        VLDMIA   r2, {d16, d17, d18, d19, d20, d21, d22, d23}
        VST1.32  {d2, d3}, [r1]
        LDR      r1, [sp, #224]
        LDR      r10, [sp, #388]
        B        .LBB0_37
.LBB0_45 :
        ADD      sp, sp, #264
        VPOP     {d8, d9, d10, d11, d12, d13, d14, d15}
        ADD      sp, sp, #4
        POP      {r4, r5, r6, r7, r8, r9, r10, r11, pc}


END_FUNCTION xnn_x32_packw_gemm_goi_ukernel_x8__asm_aarch32_neon

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif

