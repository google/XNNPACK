// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

.syntax unified

// void xnn_f32_gemm_ukernel_4x8__aarch32_neon_ld64(
//     size_t mr,                            r0
//     size_t nc,                            r1
//     size_t kc,                            r2 -> r7
//     const uint8_t*restrict a,             r3
//     size_t a_stride,          sp + 96  -> (r5)
//     const void*restrict w,    sp + 100 -> r9
//     uint8_t*restrict c,       sp + 104 -> r14
//     size_t cm_stride,         sp + 108 -> (r6)
//     size_t cn_stride,         sp + 112 -> r5
//     const union xnn_f32_output_params params[restrict static 1])  sp + 116 -> (r5)


// inner loop registers

// A0   r3  d0
// A1  r12  d1
// A2  r10  d2
// A3   r0  d3

// B    r9  d8,  d9, d10, d11
// B       d12, d13, d14, d15

// C0  r14 d16-d17  q8  d18-d19  q9
// C1   r4 d20-d21 q10  d22-d23 q11
// C2   r8 d24-d25 q12  d26-d27 q13
// C3   r6 d28-d29 q14  d30-d31 q15

// Clamp (r5) d4 d5 d6 d7
// Unused r11

BEGIN_FUNCTION xnn_f32_gemm_ukernel_4x8__aarch32_neon_ld64
        .arm
#ifndef __APPLE__
        .arch armv7-a
        .fpu neon
#endif
        // Push 96 bytes
        PUSH   {r4, r5, r6, r7, r8, r9, r10, r14}  // 32
        VPUSH  {d8-d15}                            // +64 = 96

        // Clamp A and C pointers

        LDR    r5, [sp, 96]          // a_stride
        LDR    r14, [sp, 104]        // c
        LDR    r6, [sp, 108]         // cm_stride

        CMP    r0, 2                 // if mr >= 2
        ADDHS  r12, r3, r5           //   a1 = a0 + a_stride
        ADDHS  r4, r14, r6           //   c1 = c0 + cm_stride
        MOVLO  r12, r3               // a1
        MOVLO  r4, r14               // c1
                                     // if mr > 2
        ADDHI  r10, r12, r5          //   a2 = a1 + a_stride
        ADDHI  r8, r4, r6            //   c2 = c1 + cm_stride
        MOVLS  r10, r12              // a2
        MOVLS  r8, r4                // c2

        CMP    r0, 4                 // if mr >=4
        ADDHS  r0, r10, r5           //   a3 = a2 + a_stride
        ADDHS  r6, r8, r6            //   c3 = c2 + cm_stride
        MOVLO  r0, r10               // a3
        MOVLO  r6, r8                // c3

        // Load params pointer
        LDR    r5, [sp, 116]         // clamping_params
        LDR    r9, [sp, 100]         // W

        // Load clamping_params values
        VLD1.32     {d4[]-d5[]}, [r5]!
        VLD1.32     {d6[]-d7[]}, [r5]

        LDR    r5, [sp, 112]         // cn_stride

1:
        # Load initial bias from w into accumulators
        VLDM        r9!, {d16-d19}   // Bias
        SUBS        r7, r2, 8
        VMOV        q10, q8
        VMOV        q11, q9
        VMOV        q12, q8
        VMOV        q13, q9
        VMOV        q14, q8
        VMOV        q15, q9
        BLO         3f               // less than 2 channels?

        // Main loop - 2 floats of A (8 bytes)
2:
        VLD1.32    {d0}, [r3]!       // A0
        VLDM        r9!, {d8-d11}    // B0
        VLD1.32    {d1}, [r12]!      // A1
        VLD1.32    {d2}, [r10]!      // A2
        VLD1.32    {d3}, [ r0]!      // A3

        VMLA.F32     q8, q4, d0[0]
        VMLA.F32     q9, q5, d0[0]
        VMLA.F32    q10, q4, d1[0]
        VMLA.F32    q11, q5, d1[0]
        VLDM        r9!, {d12-d15}   // B1
        VMLA.F32    q12, q4, d2[0]
        VMLA.F32    q13, q5, d2[0]
        VMLA.F32    q14, q4, d3[0]
        VMLA.F32    q15, q5, d3[0]
        VMLA.F32     q8, q6, d0[1]
        VMLA.F32     q9, q7, d0[1]
        VMLA.F32    q10, q6, d1[1]
        VMLA.F32    q11, q7, d1[1]
        VMLA.F32    q12, q6, d2[1]
        VMLA.F32    q13, q7, d2[1]
        SUBS         r7, r7, 8
        VMLA.F32    q14, q6, d3[1]
        VMLA.F32    q15, q7, d3[1]
        BHS         2b

3:
        // Is there a remainder?- 1 floats of A (4 bytes)
        TST         r7, 4
        BNE         7f

4:
        // Clamp
        VMIN.F32     q8,  q8, q2
        VMIN.F32     q9,  q9, q2
        VMIN.F32    q10, q10, q2
        VMIN.F32    q11, q11, q2
        VMIN.F32    q12, q12, q2
        VMIN.F32    q13, q13, q2
        VMIN.F32    q14, q14, q2
        VMIN.F32    q15, q15, q2
        VMAX.F32     q8,  q8, q3
        VMAX.F32     q9,  q9, q3
        VMAX.F32    q10, q10, q3
        VMAX.F32    q11, q11, q3
        VMAX.F32    q12, q12, q3
        VMAX.F32    q13, q13, q3
        VMAX.F32    q14, q14, q3
        VMAX.F32    q15, q15, q3

        // Store full 4 x 8

        CMP         r1, 8
        BLO         8f

        SUBS        r1, r1, 8               // Loop counter
        VST1.32     {d16-d19}, [r14], r5
        SUB         r0, r0, r2
        VST1.32     {d20-d23}, [r4], r5
        SUB         r10, r10, r2
        VST1.32     {d24-d27}, [r8], r5
        SUB         r12, r12, r2
        VST1.32     {d28-d31}, [r6], r5
        SUB         r3, r3, r2
        BNE         1b

6:
        VPOP        {d8-d15}
        POP         {r4, r5, r6, r7, r8, r9, r10, pc}

7:
        // Remainder- 1 floats of A (4 bytes)
        VLDM        r3!,  {s0}       // A0
        VLDM        r9!, {d8-d11}    // B0
        VLDM        r12!, {s2}       // A1
        VLDM        r10!, {s4}       // A2
        VLDM         r0!, {s6}       // A3
        VMLA.F32     q8, q4, d0[0]
        VMLA.F32     q9, q5, d0[0]
        VMLA.F32    q10, q4, d1[0]
        VMLA.F32    q11, q5, d1[0]
        VMLA.F32    q12, q4, d2[0]
        VMLA.F32    q13, q5, d2[0]
        VMLA.F32    q14, q4, d3[0]
        VMLA.F32    q15, q5, d3[0]
        B           4b

// Store odd width

9:
        VST1.32    {d16-d17}, [r14]!
        VST1.32    {d20-d21}, [r4]!
        VST1.32    {d24-d25}, [r8]!
        VST1.32    {d28-d29}, [r6]!
        TST        r1, 2
        BNE        11f

10:
        VMOV       d19, d18
        VMOV       d23, d22
        VMOV       d27, d26
        VMOV       d31, d30
        TST        r1, 1
        BEQ        6b
        B          12f

8:
        TST         r1, 4
        BNE         9b
        VMOV         q9, q8
        VMOV        q11, q10
        VMOV        q13, q12
        VMOV        q15, q14
        TST         r1, 2
        BEQ         10b

11:
        VST1.32    {d18}, [r14]!
        VST1.32    {d22}, [r4]!
        VST1.32    {d26}, [r8]!
        VST1.32    {d30}, [r6]!
        TST        r1, 1
        BEQ        6b

12:
        VST1.32    {d19[0]}, [r14]
        VST1.32    {d23[0]}, [r4]
        VST1.32    {d27[0]}, [r8]
        VST1.32    {d31[0]}, [r6]
        B          6b

END_FUNCTION xnn_f32_gemm_ukernel_4x8__aarch32_neon_ld64

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif



