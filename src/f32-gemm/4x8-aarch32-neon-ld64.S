// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

.syntax unified

// void xnn_f32_gemm_ukernel_4x8__aarch32_neon_ld64(
//     size_t mr,                            r0
//     size_t nc,                            r1
//     size_t kc,                            r2
//     const uint8_t*restrict a,             r3
//     size_t a_stride,          sp + 136
//     const void*restrict w,    sp + 140 -> r9
//     uint8_t*restrict c,       sp + 144 -> (r14)
//     size_t cm_stride,         sp + 148
//     size_t cn_stride,         sp + 152
//     const union xnn_f32_output_params params[restrict static 1])  sp + 156


// inner loop registers

// A0      r8 d0-d1
// A1  r5/r12 d4-d5
// A2     r10 d6-d7
// A3  r3/r11 d8-d9

// B    r9  d8,  d9, d10, d11
// B       d12, d13, d14, d15

// C0   r0 d16-d17  q8  d18-d19  q9
// C1   r4 d20-d21 q10  d22-d23 q11
// C2   r3 d24-d25 q12  d26-d27 q13
// C3  r14 d28-d29 q14  d30-d31 q15


BEGIN_FUNCTION xnn_f32_gemm_ukernel_4x8__aarch32_neon_ld64
        .arm
#ifndef __APPLE__
        .arch armv7-a
        .fpu neon
#endif
        // Push 136 bytes
        PUSH   {r4, r5, r6, r7, r8, r9, r10, r11, r14}
        SUB    sp, sp, 4   // to maintain alignment
        VPUSH  {d8-d15}
        SUB    sp, sp, 48

        // Clamp A and C pointers
        MOV    r7, r2                        // kc -> r7, sp + 12
        LDR    r2, [sp, 152]                 // a_stride
        CMP    r0, 2
        MOV    r10, r3
        ADDCS    r10, r10, r2
        CMP    r0, 3
        LDR    r14, [sp, 160]                // c
        MOV    r11, r3
        MOV    r12, r10
        LDR    r6, [sp, 164]                 // cm_stride
        ADDCS    r12, r12, r2
        CMP    r0, 4
        MOV    r3, r14
        MOV    r5, 24
        MOV    r8, r12
        LDR    r9, [sp, 156]                 // W
        ADDEQ    r8, r8, r2
        CMP    r0, 2
        ADDCS    r3, r3, r6
        CMP    r0, 3
        SUB    r2, r7, 8
        STR    r7, [sp, 12]
        MOV    r4, r3
        ADDCS    r4, r4, r6
        CMP    r0, 4
        MOV    r0, r4
        ADDEQ    r0, r0, r6

        LSR    r6, r2, 3   // kc / 8
        ADD    r5, r5, r6, lsl #4
        STR    r5, [sp, 8]
        MOV    r5, 2
        AND    r2, r2, 7           // kc & 7
        ADD    r5, r5, r6, lsl #1  // kc with low bits removed
        STR    r5, [sp, 40]        // kc with low bits removed
        STR    r2, [sp, 4]         // kc & 7

1:
        # Load initial bias from w into accumulators
        MOV         r2, r9
        VLDM        r9,  {d30-d31}           // VLD1.32    {d30-d31}, [r9], r6
        ADD         r9, r9, 32
        CMP         r7, 8
        STR         r2, [sp, 44]
        ADD         r2, r2, 16
        VLDM        r2,  {d26-d27}           // VLD1.32    {d26-d27}, [r2]
        VMOV         q8, q13
        VMOV         q9, q15
        VMOV        q10, q13
        VMOV        q11, q15
        VMOV        q14, q15
        VMOV        q12, q13
        BLO         6f                       // less than 2 channels

        LDR         r2, [sp, 40]
        STR         r3, [sp, 28]
        STR         r1, [sp, 32]
        ADD         r1, r8, r2, lsl #2
        MOV         r3, r11
        MOV         r5, r12
        STR         r1, [sp, 20]
        ADD         r1, r10, r2, lsl #2
        STR         r4, [sp, 24]
        STR         r14, [sp, 36]
        STR         r1, [sp, 16]
        SUB         r7, r7, 8

        // Main loop - 2 floats of A (8 bytes)
2:
        VLDM        r8!, {d2}        // A0
        VLDM        r9!, {d8-d11}    // B0
        VLDM        r5!, {d0}        // A1
        VLDM       r10!, {d4}        // A2
        VLDM        r3!, {d6}        // A3
        VMLA.F32    q14, q4, d2[0]
        VMLA.F32     q8, q5, d2[0]
        VMLA.F32     q9, q4, d0[0]
        VMLA.F32    q10, q5, d0[0]
        VLDM        r9!, {d12-d15}   // B1
        VMLA.F32    q11, q4, d4[0]
        VMLA.F32    q12, q5, d4[0]
        VMLA.F32    q15, q4, d6[0]
        VMLA.F32    q13, q5, d6[0]
        VMLA.F32    q14, q6, d2[1]
        VMLA.F32     q8, q7, d2[1]
        VMLA.F32     q9, q6, d0[1]
        VMLA.F32    q10, q7, d0[1]
        VMLA.F32    q11, q6, d4[1]
        VMLA.F32    q12, q7, d4[1]
        SUBS         r7, r7, 8
        VMLA.F32    q15, q6, d6[1]
        VMLA.F32    q13, q7, d6[1]
        BHS    2b

        LDR    r1, [sp, 8]
        LDR    r9, [sp, 44]
        LDR    r2, [sp, 4]
        ADD    r9, r9, r1, lsl #2
        LDR    r1, [sp, 40]
        LDR    r14, [sp, 36]
        LDR    r3, [sp, 28]
        ADD    r12, r12, r1, lsl #2
        ADD    r11, r11, r1, lsl #2
        LDR    r1, [sp, 32]
        LDR    r4, [sp, 24]
        LDR    r7, [sp, 12]
        LDR    r8, [sp, 20]
        LDR    r10, [sp, 16]
        TST    r7, 4
        BNE    7f

        # Clamp
3:
        # Load params pointer
        LDR    r2, [sp, 172]

        # Load clamping_params values
        VLD1.32    {d0[]-d1[]}, [r2]!
        VLD1.32    {d8[]-d9[]}, [r2]

        VMIN.F32    q8, q8, q0
        VMIN.F32    q2, q9, q0
        VMIN.F32    q10, q10, q0
        VMIN.F32    q3, q11, q0
        VMIN.F32    q12, q12, q0
        VMIN.F32    q1, q13, q0
        VMIN.F32    q14, q14, q0
        VMIN.F32    q0, q15, q0

        VMAX.F32    q9, q8, q4
        VMAX.F32    q11, q10, q4
        VMAX.F32    q13, q12, q4
        VMAX.F32    q8, q14, q4
        VMAX.F32    q15, q1, q4
        VMAX.F32    q10, q2, q4
        VMAX.F32    q12, q3, q4
        VMAX.F32    q14, q0, q4

        # Store full 4 x 8

        CMP         r1, 8
        BLO         8f

        LDR         r5, [sp, 168]           // cn_stride
        SUBS        r1, r1, 8               // Loop counter
        VST1.32     {d16-d19}, [r0], r5
        SUB         r11, r11, r7
        VST1.32     {d20-d23}, [r4], r5
        SUB         r10, r10, r7
        VST1.32     {d24-d27}, [r3], r5
        SUB         r12, r12, r7
        VST1.32     {d28-d31}, [r14], r5
        SUB         r8, r8, r7
        BNE         1b

4:
        ADD         sp, sp, 48
        VPOP        {d8-d15}
        ADD         sp, sp, 4
        POP         {r4, r5, r6, r7, r8, r9, r10, r11, pc}
6:
        TST         r7, 4
        BEQ         3b

// Remainder
7:
        VLD1.32     {d0[]-d1[]}, [r8]!
        VLD1.32     {d4[]-d5[]}, [r12]!
        VLD1.32     {d6[]-d7[]}, [r10]!
        VLD1.32     {d8[]-d9[]}, [r11]!
        VLDM        r9!, {d2-d3}                // VLD1.32     {d2-d3}, [r9]!
        VMLA.F32    q14, q0, q1
        VMLA.F32     q9, q2, q1
        VMLA.F32    q11, q3, q1
        VMLA.F32    q15, q4, q1
        VLDM        r9!, {d2-d3}                // VLD1.32     {d2-d3}, [r9]!
        VMLA.F32     q8, q0, q1
        VMLA.F32    q10, q2, q1
        VMLA.F32    q12, q3, q1
        VMLA.F32    q13, q4, q1
        B           3b

// Partial stores
8:
        TST         r1, 4
        BNE         9f
        VMOV         q9, q8
        VMOV        q11, q10
        VMOV        q13, q12
        VMOV        q15, q14
        TST         r1, 2
        BEQ         10f
        B           11f

9:
        VST1.32    {d16-d17}, [r0]!
        VST1.32    {d20-d21}, [r4]!
        VST1.32    {d24-d25}, [r3]!
        VST1.32    {d28-d29}, [r14]!
        TST        r1, 2
        BNE        11f

10:
        VMOV       d19, d18
        VMOV       d23, d22
        VMOV       d27, d26
        VMOV       d31, d30
        TST        r1, 1
        BEQ        4b
        B          12f

11:
        VST1.32    {d18}, [r0]!
        VST1.32    {d22}, [r4]!
        VST1.32    {d26}, [r3]!
        VST1.32    {d30}, [r14]!
        TST        r1, 1
        BEQ        4b

12:
        VST1.32    {d19[0]}, [r0]
        VST1.32    {d23[0]}, [r4]
        VST1.32    {d27[0]}, [r3]
        VST1.32    {d31[0]}, [r14]
        B          4b

END_FUNCTION xnn_f32_gemm_ukernel_4x8__aarch32_neon_ld64

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif



