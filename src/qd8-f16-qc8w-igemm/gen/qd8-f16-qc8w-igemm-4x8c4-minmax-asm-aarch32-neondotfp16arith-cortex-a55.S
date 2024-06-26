// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/4x8c4-aarch32-neondot-cortex-a55.S.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include "xnnpack/assembly.h"

.syntax unified

// void xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55(
//     size_t mr,                                      r0
//     size_t nc,                                      r1
//     size_t kc,                                      r2 -> r5 -> sp + 52
//     size_t ks,                                      r3 -> sp + 56 -> r14
//     const int8_t** restrict a,           sp + 96  -> r2
//     const void* restrict w,              sp + 100 -> r9
//     int8_t* restrict c,                  sp + 104 -> r11
//     size_t cm_stride,                   sp + 108 -> (r6)
//     size_t cn_stride,                   sp + 112 -> (r7)
//     size_t a_offset,                    sp + 116 -> (r5)
//     const int8_t* zero,                 sp + 120 -> (r7)
//     const int8_t* zero_data,               sp + 124 -> (r4)
//     xnn_f16_minmax_params *params,  sp + 128 -> (r5)
//     const struct xnn_qd8_quantization_params *quantization_params) [sp + 132] -> (r5)

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
// unused q7

// params structure is 8 bytes
// struct {
//   float min;
//   float max;
// } scalar;

// iOS does not support 32 bit ARM with Neon DotProduct.
#ifndef __APPLE__

BEGIN_FUNCTION xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55
        ADD         r2, r2, 3               // kc = (kc + 3) & ~3
        BIC         r2, r2, 3
        # Push 96 bytes
        # r2 will be reloaded in outer loop.  r3 is ks
        PUSH        {r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, lr}      // +44
        SUB         sp, sp, 4                                           // 4
        VPUSH       {d8-d13}                                            // +48 = 96

        LDR         r11, [sp, 104]          // c
        LDR         r6, [sp, 108]           // cm_stride
        LDR         r2, [sp, 96]            // a
        LDR         r9, [sp, 100]           // w

        MOV         r14, r3                 // p = ks

        # Clamp C pointers
        CMP         r0, 2                   // if mr >= 2
        ADD         r4, r11, r6             //   c1 = c0 + cm_stride
        MOVLO       r4, r11                 // c1
                                        // if mr > 2
        ADD         r8, r4, r6              //   c2 = c1 + cm_stride
        MOVLS       r8, r4                  // c2
        CMP         r0, 4                   // if mr >=4
        ADD         r6, r8, r6              //   c3 = c2 + cm_stride
        MOVLO       r6, r8                  // c3


        LDR         r5, [sp, 132]       // &quantization_params[0].zero_point
        VLD1.8      {q6, q7},  [r5]

0:
        # Load initial bias from w into accumulators
        VLDM        r9!, {d16-d19}          // ksum
        // ksum * zero_point
        VMUL.S32    q8, q8, d12[0]
        VMUL.S32    q9, q9, d12[0]
        VMOV        q10, q8
        VMOV        q11, q9
        LDR         r7, [sp, 120]           // zero
        VMOV        q12, q8
        VMOV        q13, q9
        VMOV        q14, q8
        VMOV        q15, q9

1:
        # Load next 4 A pointers + Add a_offset + Prologue
        # - Load next 4 A pointers to GPR
        # - Adjust A pointers by a_offset if not zero
        # - Load prologue
        # - Load k = kc from stack
        LDR         r3, [r2,  0]            // A0
        LDR         r5, [sp, 116]           // a_offset
        PUSH        {r4}
        LDR         r4, [sp, 128]           // zero_data
        CMP         r3,  r7                 // if a0 == zero
        LDR         r12, [r2,  4]           // A1
        ADD         r3,  r3, r5             // a0 += a_offset
        LDR         r10, [r2,  8]           // A2
        MOVEQ       r3,  r4                 // a0 = zero_data, else += a0 + a_offset
        LDR         r0, [r2, 12]            // A3
        CMP         r12,  r7                // if a1 == zero
        VLD1.8      {d4},  [r9]!            // B0
        ADD         r12, r12, r5            // a1 += a_offset
        VLD1.8      {d5},  [r9]!            // B1
        MOVEQ       r12,  r4                // a1 = zero_data, else += a1 + a_offset
        VLD1.8      {d6},  [r9]!            // B2
        CMP         r10,  r7                // if a2 == zero
        VLD1.8      {d7},  [r9]!            // B3
        ADD         r10, r10, r5            // a2 += a_offset
        VLD1.8      {d0},  [r3]!            // A0
        MOVEQ       r10,  r4                // a2 = zero_data, else += a2 + a_offset
        VLD1.8      {d1}, [r12]!            // A1
        CMP         r0,  r7                 // if a3 == zero
        ADD         r0,  r0, r5             // a3 += a_offset
        MOVEQ       r0,  r4                 // a3 = zero_data, else += a3 + a_offset
        ADD         r2, r2, 16
        POP         {r4}
        LDR         r5, [sp, 52]            // k = kc
        SUBS        r5, r5, 8               // k = k - 8

        BLO         6f                      // less than 8 channels?

        SUBS        r5, r5, 8               // k = k - 8
        BLO         3f                      // less than 8 channels?

        # Main loop - 8 bytes of A.
        # 16 SDOT, 12 LD64
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
        BHS         2b

        # Epilogue
        .p2align    3
3:
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
        TST         r5, 5

        VSDOT.S8    q8, q4, d0[1]
        VSDOT.S8    q9, q5, d0[1]
        VSDOT.S8    q10, q4, d1[1]
        VSDOT.S8    q11, q5, d1[1]
        VSDOT.S8    q12, q4, d2[1]
        VSDOT.S8    q13, q5, d2[1]
        VSDOT.S8    q14, q4, d3[1]
        VSDOT.S8    q15, q5, d3[1]
        # Is there a remainder?- 4 bytes of A
        BNE         5f

4:
        # ks loop
        SUBS        r14, r14, 16            // ks -= MR * sizeof(void*)
        BHI         1b

        LDR         r7, [sp, 112]           // cn_stride
        LDR         r14, [sp, 56]           // p = ks

        LDR         r5, [sp, 128]       // params
        VCVT.F32.S32    q8,  q8
        VCVT.F32.S32    q9,  q9
        VCVT.F32.S32    q10, q10
        VCVT.F32.S32    q11, q11
        VCVT.F32.S32    q12, q12
        VCVT.F32.S32    q13, q13
        VCVT.F32.S32    q14, q14
        VCVT.F32.S32    q15, q15

        // Load scale
        VLD1.8      {q0-q1},  [r9]!

        VMUL.F32    q2, q0, d12[1]
        VMUL.F32    q3, q1, d12[1]
        VMUL.F32    q4, q0, d12[1]
        VMUL.F32    q5, q1, d12[1]
        VMUL.F32    q8,  q8, q2
        VMUL.F32    q9,  q9, q3
        VMUL.F32    q10, q10, q4
        VMUL.F32    q11, q11, q5
        VMUL.F32    q2, q0, d12[1]
        VMUL.F32    q3, q1, d12[1]
        VMUL.F32    q4, q0, d12[1]
        VMUL.F32    q5, q1, d12[1]
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
        SUBS        r1, r1, 8               // nc -= 8

        # Store full 4 x 8
        BLO         11f
        VST1.16     {q14}, [r6], r7
        VST1.16     {q12}, [r8], r7
        VST1.16     {q10}, [r4], r7
        VST1.16     {q8}, [r11], r7

        SUB         r2, r2, r14             // a -= ks

        BHI         0b

        VPOP        {d8-d13}
        ADD         sp, sp, 12              // skip pad, r2, r3
        POP         {r4, r5, r6, r7, r8, r9, r10, r11, pc}

        # Remainder prologue
        .p2align    3
5:
        VLD1.8      {d4},  [r9]!            // B0
        VLD1.8      {d0},  [r3]!            // A0
        VLD1.8      {d5},  [r9]!            // B1
        VLD1.8      {d6},  [r9]!            // B2
        VLD1.8      {d1}, [r12]!            // A1
        VLD1.8      {d7},  [r9]!            // B3

        # Remainder- 4 bytes of A
6:
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
        B           4b

        # Store odd width
        .p2align    3
11:
        TST         r1, 4
        BEQ         12f
        VST1.16     {d28}, [r6]!
        VMOV        d28, d29
        VST1.16     {d24}, [r8]!
        VMOV        d24, d25
        VST1.16     {d20}, [r4]!
        VMOV        d20, d21
        VST1.16     {d16}, [r11]!
        VMOV        d16, d17

12:
        TST         r1, 2
        BEQ         13f
        VST1.32     {d28[0]}, [r6]!
        VEXT.8      d28, d28, d29, 4
        VST1.32     {d24[0]}, [r8]!
        VEXT.8      d24, d24, d25, 4
        VST1.32     {d20[0]}, [r4]!
        VEXT.8      d20, d20, d21, 4
        VST1.32     {d16[0]}, [r11]!
        VEXT.8      d16, d16, d17, 4
13:
        TST         r1, 1
        BEQ         14f
        VST1.16     {d28[0]}, [r6]
        VST1.16     {d24[0]}, [r8]
        VST1.16     {d20[0]}, [r4]
        VST1.16     {d16[0]}, [r11]

14:
        VPOP        {d8-d13}
        ADD         sp, sp, 12              // skip pad, r2, r3
        POP         {r4, r5, r6, r7, r8, r9, r10, r11, pc}

END_FUNCTION xnn_qd8_f16_qc8w_igemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55
#endif  // __APPLE__

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
