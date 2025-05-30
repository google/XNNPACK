// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/4x8c4-aarch32-neondot-cortex-a55.S.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "src/xnnpack/assembly.h"

.syntax unified

// void xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55(
//     size_t mr,                            r0
//     size_t nc,                            r1
//     size_t kc,                            r2 -> r5
//     const uint8_t* restrict a,             r3
//     size_t a_stride,           sp + 80 -> (r7)
//     const void* restrict w,     sp + 84 -> r9
//     uint8_t* restrict c,        sp + 88 -> r11
//     size_t cm_stride,          sp + 92 -> (r6)
//     size_t cn_stride,          sp + 96 -> r7
//     xnn_f16_minmax_params params,  sp + 100 -> (r5)
//     const struct xnn_qd8_quantization_params *quantization_params) [sp + 104] -> (r5)

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// A0   r3  d0
// A1  r12  d1
// A2  r10  d2
// A3   r0  d3
// B    r9  q2 q3 q4 q5
// C0  r11 d16-d17  q8  d18-d19  q9
// C1   r4 d20-d21 q10  d22-d23 q11
// C2   r8 d24-d25 q12  d26-d27 q13
// C3   r6 d28-d29 q14  d30-d31 q15
// r5 params, zero point & scale
// d6, d7, d14, d15 zero point and scale
// q6, q7 zero point and scale.
// params structure is 8 bytes
// struct {
//   float min;
//   float max;
// } scalar;

// iOS does not support 32 bit ARM with Neon DotProduct.
#ifndef __APPLE__

BEGIN_FUNCTION xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55
        # Push 96 bytes
        PUSH        {r4, r5, r6, r7, r8, r9, r10, r11}      // 32
        VPUSH       {d8-d15}                                    // +64 = 96

        LDR         r7, [sp, 96]            // a_stride
        ADD         r2, r2, 3                           // kc = (kc + 3) & ~3
        LDR         r11, [sp, 104]       // c

        LDR         r6, [sp, 108]       // cm_stride
        LDR         r9, [sp, 100]        // w
        BIC         r2, r2, 3

        # Clamp A and C pointers
        CMP         r0, 2                   // if mr >= 2
        ADD         r12, r3, r7             //   a1 = a0 + a_stride
        ADD         r4, r11, r6             //   c1 = c0 + cm_stride
        MOVLO       r12, r3                 // a1
        MOVLO       r4, r11                 // c1
                                        // if mr > 2
        ADD         r10, r12, r7            //   a2 = a1 + a_stride
        ADD         r8, r4, r6              //   c2 = c1 + cm_stride
        MOVLS       r10, r12                // a2
        MOVLS       r8, r4                  // c2

        CMP         r0, 4                   // if mr >=4
        ADD         r0, r10, r7             //   a3 = a2 + a_stride
        ADD         r6, r8, r6              //   c3 = c2 + cm_stride
        MOVLO       r0, r10                 // a3
        MOVLO       r6, r8                  // c3

        LDR         r7, [sp, 112]               // cn_stride

        LDR         r5, [sp, 120]   // &quantization_params[0].zero_point
        VLD1.8      {q6, q7},  [r5]

        .p2align    3
0:
        # Load initial bias from w into accumulators
        VLDM        r9!, {d16-d19}          // Bias
        SUBS        r5, r2, 8               // k = kc - 8

        # Prologue + Bias
        // ksum * zero_point
        VLD1.8      {d4},  [r9]!            // B0
        VMUL.S32    q10, q8, d13[0]
        VLD1.8      {d0},  [r3]!            // A0
        VMUL.S32    q12, q8, d14[0]
        VLD1.8      {d5},  [r9]!            // B1
        VMUL.S32    q14, q8, d15[0]
        VLD1.8      {d6},  [r9]!            // B2
        VMUL.S32    q8, q8, d12[0]
        VLD1.8      {d1}, [r12]!            // A1
        VMUL.S32    q11, q9, d13[0]
        VLD1.8      {d7},  [r9]!            // B3
        VMUL.S32    q13, q9, d14[0]
        VMUL.S32    q15, q9, d15[0]
        VMUL.S32    q9, q9, d12[0]
        BLO         5f                      // less than 8 channels?

        SUBS        r5, r5, 8               // k = k - 8
        BLO         2f                      // less than 16 channels - skip mainloop

        # Main loop - 8 bytes of A.
        # 16 SDOT, 12 LD64
        .p2align    3
1:
        VSDOT.S8    q8, q2, d0[0]
        VLD1.8      {d2}, [r10]!            // A2
        VSDOT.S8    q9, q3, d0[0]
        VLD1.8      {d3},  [r0]!            // A3
        VSDOT.S8    q10, q2, d1[0]
        VLD1.8      {d8},  [r9]!            // B4
        VSDOT.S8    q11, q3, d1[0]
        VLD1.8      {d9},  [r9]!            // B5
        VSDOT.S8    q12, q2, d2[0]
        VLD1.8      {d10},  [r9]!           // B6
        VSDOT.S8    q13, q3, d2[0]
        VLD1.8      {d11},  [r9]!           // B7
        VSDOT.S8    q14, q2, d3[0]
        VSDOT.S8    q15, q3, d3[0]
        SUBS        r5, r5, 8

        VSDOT.S8    q8, q4, d0[1]
        VLD1.8      {d4},  [r9]!            // B0
        VSDOT.S8    q9, q5, d0[1]
        VLD1.8      {d5},  [r9]!            // B1
        VSDOT.S8    q10, q4, d1[1]
        VLD1.8      {d6},  [r9]!            // B2
        VSDOT.S8    q11, q5, d1[1]
        VLD1.8      {d7},  [r9]!            // B3
        VSDOT.S8    q12, q4, d2[1]
        VLD1.8      {d0},  [r3]!            // A0
        VSDOT.S8    q13, q5, d2[1]
        VLD1.8      {d1}, [r12]!            // A1
        VSDOT.S8    q14, q4, d3[1]
        VSDOT.S8    q15, q5, d3[1]
        BHS         1b

        # Epilogue
        .p2align    3
2:
        VSDOT.S8    q8, q2, d0[0]
        VLD1.8      {d2}, [r10]!            // A2
        VSDOT.S8    q9, q3, d0[0]
        VLD1.8      {d3},  [r0]!            // A3
        VSDOT.S8    q10, q2, d1[0]
        VLD1.8      {d8},  [r9]!            // B4
        VSDOT.S8    q11, q3, d1[0]
        VLD1.8      {d9},  [r9]!            // B5
        VSDOT.S8    q12, q2, d2[0]
        VLD1.8      {d10},  [r9]!           // B6
        VSDOT.S8    q13, q3, d2[0]
        VLD1.8      {d11},  [r9]!           // B7
        VSDOT.S8    q14, q2, d3[0]
        VSDOT.S8    q15, q3, d3[0]
        TST         r5, 7

        VSDOT.S8    q8, q4, d0[1]
        VSDOT.S8    q9, q5, d0[1]
        VSDOT.S8    q10, q4, d1[1]
        VSDOT.S8    q11, q5, d1[1]
        VSDOT.S8    q12, q4, d2[1]
        VSDOT.S8    q13, q5, d2[1]
        VSDOT.S8    q14, q4, d3[1]
        VSDOT.S8    q15, q5, d3[1]

        # Is there a remainder?- 4 bytes of A
        BNE         4f

3:
        LDR         r5, [sp, 116]                   // params
        VCVT.F32.S32    q8,  q8
        VCVT.F32.S32    q9,  q9
        VCVT.F32.S32    q10, q10
        VCVT.F32.S32    q11, q11
        VCVT.F32.S32    q12, q12
        VCVT.F32.S32    q13, q13
        VCVT.F32.S32    q14, q14
        VCVT.F32.S32    q15, q15

        // Load bias
        VLD1.8      {q0-q1},  [r9]!

        VMUL.F32    q2, q0, d12[1]
        VMUL.F32    q3, q1, d12[1]
        VMUL.F32    q4, q0, d13[1]
        VMUL.F32    q5, q1, d13[1]
        VMUL.F32    q8,  q8, q2
        VMUL.F32    q9,  q9, q3
        VMUL.F32    q10, q10, q4
        VMUL.F32    q11, q11, q5
        VMUL.F32    q2, q0, d14[1]
        VMUL.F32    q3, q1, d14[1]
        VMUL.F32    q4, q0, d15[1]
        VMUL.F32    q5, q1, d15[1]
        VMUL.F32    q12, q12, q2
        VMUL.F32    q13, q13, q3
        VMUL.F32    q14, q14, q4
        VMUL.F32    q15, q15, q5

        // Load bias
        VLD1.8      {q0-q1},  [r9]!

        VLD1.32     {d5[0]}, [r5]                   // params.min/max
        VADD.F32    q8,  q8, q0
        VADD.F32    q10, q10, q0
        VADD.F32    q12, q12, q0
        VADD.F32    q14, q14, q0
        VDUP.16     q4, d5[0]
        VADD.F32    q9,  q9, q1
        VADD.F32    q11, q11, q1
        VADD.F32    q13, q13, q1
        VADD.F32    q15, q15, q1

        VCVT.F16.F32    d16,  q8
        VCVT.F16.F32    d17,  q9
        VCVT.F16.F32    d20,  q10
        VCVT.F16.F32    d21,  q11
        VCVT.F16.F32    d24,  q12
        VCVT.F16.F32    d25,  q13
        VCVT.F16.F32    d28,  q14
        VCVT.F16.F32    d29,  q15
        VMAX.F16    q8, q8, q4
        VMAX.F16    q10, q10, q4
        VDUP.16     q5, d5[1]
        VMAX.F16    q12, q12, q4
        VMAX.F16    q14, q14, q4

        VMIN.F16    q8, q8, q5
        VMIN.F16    q10, q10, q5
        VMIN.F16    q12, q12, q5
        VMIN.F16    q14, q14, q5

        SUBS        r1, r1, 8

        # Store full 4 x 8
        BLO         10f
        VST1.32     {q14}, [r6], r7
        SUB         r0, r0, r2
        VST1.32     {q12}, [r8], r7
        SUB         r10, r10, r2
        VST1.32     {q10}, [r4], r7
        SUB         r12, r12, r2
        VST1.32     {q8}, [r11], r7
        SUB         r3, r3, r2

        BHI         0b

        VPOP        {d8-d15}
        POP         {r4, r5, r6, r7, r8, r9, r10, r11}
        BX          lr

        # Remainder prologue
        .p2align    3
4:
        VLD1.8      {d4},  [r9]!            // B0
        VLD1.8      {d0},  [r3]!            // A0
        VLD1.8      {d5},  [r9]!            // B1
        VLD1.8      {d6},  [r9]!            // B2
        VLD1.8      {d1}, [r12]!            // A1
        VLD1.8      {d7},  [r9]!            // B3

        # Remainder- 4 bytes of A
5:
        VSDOT.S8    q8, q2, d0[0]
        VLD1.32     {d2[0]}, [r10]!         // A2
        VSDOT.S8    q9, q3, d0[0]
        VLD1.32     {d3[0]},  [r0]!         // A3
        VSDOT.S8    q10, q2, d1[0]
        SUB         r3, r3, 4               // Rewind A0
        VSDOT.S8    q11, q3, d1[0]
        SUB         r12, r12, 4             // Rewind A1
        VSDOT.S8    q12, q2, d2[0]
        VSDOT.S8    q13, q3, d2[0]
        VSDOT.S8    q14, q2, d3[0]
        VSDOT.S8    q15, q3, d3[0]
        B           3b

        # Store odd width
        .p2align    3
10:
        TST         r1, 4
        BEQ         11f
        VST1.16     {d28}, [r6]!
        VMOV        d28, d29
        VST1.16     {d24}, [r8]!
        VMOV        d24, d25
        VST1.16     {d20}, [r4]!
        VMOV        d20, d21
        VST1.16     {d16}, [r11]!
        VMOV        d16, d17

11:
        TST         r1, 2
        BEQ         12f
        VST1.32     {d28[0]}, [r6]!
        VEXT.8      d28, d28, d29, 4
        VST1.32     {d24[0]}, [r8]!
        VEXT.8      d24, d24, d25, 4
        VST1.32     {d20[0]}, [r4]!
        VEXT.8      d20, d20, d21, 4
        VST1.32     {d16[0]}, [r11]!
        VEXT.8      d16, d16, d17, 4
12:
        TST         r1, 1
        BEQ         13f
        VST1.16     {d28[0]}, [r6]
        VST1.16     {d24[0]}, [r8]
        VST1.16     {d20[0]}, [r4]
        VST1.16     {d16[0]}, [r11]

13:
        VPOP        {d8-d15}
        POP         {r4, r5, r6, r7, r8, r9, r10, r11}
        BX          lr


END_FUNCTION xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55
#endif  // __APPLE__

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif

