// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

.syntax unified

// void xnn_f32_gemm_ukernel_4x8__aarch32_neon_ld64(
//     size_t mr,
//     size_t nc,
//     size_t kc,
//     const uint8_t*restrict a,
//     size_t a_stride,
//     const void*restrict w,
//     uint8_t*restrict c,
//     size_t cm_stride,
//     size_t cn_stride,
//     const union xnn_f32_output_params params[restrict static 1])


// inner loop registers

// A0      r8 d0-d1
// A1  r5/r12 d4-d5
// A2     r10 d6-d7
// A3  r3/r11 d8-d9

// B    r9 d2-d3

// C0   r0 d30-d31   d22-d23
// C1   r4 d28-d29   d20-d21
// C2   r3 d26-d27   d18-d19
// C3  r14 d24-d25   d16-d17

BEGIN_FUNCTION xnn_f32_gemm_ukernel_4x8__aarch32_neon_ld64
        .arm
#ifndef __APPLE__
        .arch armv7-a
        .fpu neon
#endif
        PUSH    {r4, r5, r6, r7, r8, r9, r10, r11, r14}
        SUB    sp, sp, 4
        VPUSH    {d8-d13}
        SUB    sp, sp, 48
        MOV    r7, r2
        LDR    r2, [sp, 136]
        CMP    r0, 2
        MOV    r10, r3
        ADDCS    r10, r10, r2
        CMP    r0, 3
        LDR    r14, [sp, 144]
        MOV    r11, r3
        MOV    r12, r10
        LDR    r6, [sp, 148]
        ADDCS    r12, r12, r2
        CMP    r0, 4
        MOV    r3, r14
        MOV    r5, 24
        MOV    r8, r12
        LDR    r9, [sp, 140]
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
        LSR    r6, r2, 3
        ADD    r5, r5, r6, lsl #4
        STR    r5, [sp, 8]
        MOV    r5, 2
        AND    r2, r2, 7
        ADD    r5, r5, r6, lsl #1
        MOV    r6, 32
        STR    r5, [sp, 40]
        STR    r2, [sp, 4]

1:
        MOV         r2, r9
        VLDM        r9,  {d16-d17}           // VLD1.32    {d16-d17}, [r9], r6
        ADD         r9, r9, 32
        CMP         r7, 8
        STR         r2, [sp, 44]
        ADD         r2, r2, 16
        VLDM        r2,  {d18-d19}           // VLD1.32    {d18-d19}, [r2]
        BCC         6f                       // less than 2 channels

        LDR         r2, [sp, 40]
        VMOV        q10, q8
        VMOV        q13, q9
        STR         r3, [sp, 28]
        VMOV        q11, q8
        STR         r1, [sp, 32]
        VMOV        q14, q9
        ADD         r1, r8, r2, lsl #2
        VMOV        q12, q8
        MOV         r3, r11
        VMOV        q15, q9
        MOV         r5, r12
        STR         r1, [sp, 20]
        ADD         r1, r10, r2, lsl #2
        STR         r4, [sp, 24]
        STR         r14, [sp, 36]
        STR         r1, [sp, 16]

// Main loop 2 channels

2:
        VLD1.32    {d2}, [r8]!       // A0
        VLDM        r9!, {d8-d9}     // B0
        VLD1.32    {d0}, [r5]!       // A1
        VLD1.32    {d4}, [r10]!      // A2
        VLD1.32    {d6}, [r3]!       // A3
        VLDM        r9!, {d10-d11}   // B1
        VMLA.f32    q11, q4, d0[0]
        VMLA.f32    q12, q4, d2[0]
        VMLA.f32    q10, q4, d4[0]
        VMLA.f32     q8, q4, d6[0]
        VMLA.f32    q15, q5, d2[0]
        VLDM        r9!, {d8-d9}     // B2
        VMLA.f32    q14, q5, d0[0]
        VMLA.f32    q13, q5, d4[0]
        VMLA.f32     q9, q5, d6[0]
        VMLA.f32    q12, q4, d2[1]
        VLDM        r9!, {d10-d11}   // B3
        VMLA.f32    q11, q4, d0[1]
        SUB         r7, r7, 8
        VMLA.f32    q10, q4, d4[1]
        VMLA.f32     q8, q4, d6[1]
        VMLA.f32    q15, q5, d2[1]
        CMP         r7, 7
        VMLA.f32    q14, q5, d0[1]
        VMLA.f32    q13, q5, d4[1]
        VMLA.f32     q9, q5, d6[1]

        BHI    2b


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
        CMP    r2, 0
        BNE    7f

// clamp
3:
        LDR    r2, [sp, 156]
        CMP    r1, 7
        VLD1.32    {d0[]-d1[]}, [r2 :32]!
        VMIN.f32    q15, q15, q0
        VLD1.32    {d8[]-d9[]}, [r2 :32]
        VMIN.f32    q2, q11, q0
        VMIN.f32    q14, q14, q0
        VMIN.f32    q3, q10, q0
        VMIN.f32    q13, q13, q0
        VMIN.f32    q1, q9, q0
        VMIN.f32    q12, q12, q0
        VMIN.f32    q0, q8, q0
        VMAX.f32    q11, q15, q4
        VMAX.f32    q10, q14, q4
        VMAX.f32    q9, q13, q4
        VMAX.f32    q15, q12, q4
        VMAX.f32    q8, q1, q4
        VMAX.f32    q14, q2, q4
        VMAX.f32    q13, q3, q4
        VMAX.f32    q12, q0, q4
        BLS    8f

// full store
        LDR         r5, [sp, 152]
        ADD         r2, r0, 16
        SUBS        r1, r1, 8               // Loop counter
        VST1.32     {d30-d31}, [r0], r5
        SUB         r11, r11, r7
        VST1.32     {d22-d23}, [r2]
        ADD         r2, r4, 16
        VST1.32     {d28-d29}, [r4], r5
        SUB         r10, r10, r7
        VST1.32     {d20-d21}, [r2]
        ADD         r2, r3, 16
        VST1.32     {d26-d27}, [r3], r5
        SUB         r12, r12, r7
        VST1.32     {d18-d19}, [r2]
        ADD         r2, r14, 16
        VST1.32     {d24-d25}, [r14], r5
        SUB         r8, r8, r7
        VST1.32     {d16-d17}, [r2]
        BNE         1b

4:
        ADD     sp, sp, 48
        VPOP    {d8-d13}
        ADD     sp, sp, 4
        POP     {r4, r5, r6, r7, r8, r9, r10, r11, pc}
6:
        MOV     r2, r7
        VMOV    q15, q9
        VMOV    q12, q8
        VMOV    q14, q9
        VMOV    q11, q8
        VMOV    q13, q9
        VMOV    q10, q8
        CMP     r2, 0
        BEQ     3b

// Remainder
7:
        VLD1.32     {d0[]-d1[]}, [r8 :32]!
        VLD1.32     {d4[]-d5[]}, [r12 :32]!
        VLD1.32     {d6[]-d7[]}, [r10 :32]!
        VLD1.32     {d8[]-d9[]}, [r11 :32]!
        VLDM        r9!, {d2-d3}                // VLD1.32     {d2-d3}, [r9]!
        VMLA.f32    q12, q0, q1
        VMLA.f32    q11, q2, q1
        VMLA.f32    q10, q3, q1
        VMLA.f32     q8, q4, q1
        VLDM        r9!, {d2-d3}                // VLD1.32     {d2-d3}, [r9]!
        VMLA.f32    q15, q0, q1
        VMLA.f32    q14, q2, q1
        VMLA.f32    q13, q3, q1
        VMLA.f32     q9, q4, q1
        B    3b

// Partial stores
8:
        TST    r1, 4
        BNE    9f
        VMOV   q11, q15
        VMOV   q10, q14
        VMOV    q9, q13
        VMOV    q8, q12
        TST    r1, 2
        BEQ    10f
        B    11f

9:
        VST1.32    {d30-d31}, [r0]!
        VST1.32    {d28-d29}, [r4]!
        VST1.32    {d26-d27}, [r3]!
        VST1.32    {d24-d25}, [r14]!
        TST    r1, 2
        BNE    11f

10:
        VMOV   d23, d22
        VMOV   d21, d20
        VMOV   d19, d18
        VMOV   d17, d16
        TST    r1, 1
        BEQ    4b
        B    12f

11:
        VST1.32    {d22}, [r0]!
        VST1.32    {d20}, [r4]!
        VST1.32    {d18}, [r3]!
        VST1.32    {d16}, [r14]!
        TST    r1, 1
        BEQ    4b

12:
        VST1.32    {d23[0]}, [r0 :32]
        VST1.32    {d21[0]}, [r4 :32]
        VST1.32    {d19[0]}, [r3 :32]
        VST1.32    {d17[0]}, [r14 :32]
        B    4b

END_FUNCTION xnn_f32_gemm_ukernel_4x8__aarch32_neon_ld64

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif



