// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

.syntax unified

// void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__asm_aarch32_neonv8_mla8_cortex_a35(
//   size_t channels,                          r0, r11
//   size_t output_width,                      r1
//   const int8_t** input,                     r2
//   const void* weights,                      r3
//   int8_t* output,                           r10, [sp, 88]
//   intptr_t input_stride,                    r6,  [sp, 92]
//   size_t output_increment,                  r12, [sp, 96]
//   size_t input_offset,                     (r11),[sp, 100]
//   const int8_t* zero,                       r4,  [sp, 104]
//   const union xnn_qs8_minmax_params params  r5,  [sp, 108]

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// A0   r5  q4
// A1   r6  q5
// A2   r8  q6
// B    r7/r3/lr  q12 q13 q14
// C0  r10 q12 q13 q14 q15
// Prod q0 q1 q2 q3

// params structure is 4 bytes
//   struct {
//     int16_t output_zero_point;  d20[0] q10
//     int8_t output_min;          d20[2] q9
//     int8_t output_max;          d20[3] q11
//   } xnn_qs8_minmax_params.neonv8;

// r7 temp B
// r9 B post increment 80 or 16
// unused q7

BEGIN_FUNCTION xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__asm_aarch32_neonv8_mla8_cortex_a35
        // 88 bytes of stack
        PUSH        {r4, r5, r6, r7, r8, r9, r10, r11, lr}  // 40
        SUB         sp, sp, 4
        VPUSH       {d8, d9, d10, d11, d12, d13}            // 48

        LDR         r5, [sp, 108]           // params
        LDR         r10, [sp, 88]           // output
        LDR         r12, [sp, 96]           // output_increment
        LDR         r4, [sp, 104]           // zero

        VLD1.32     {d20[]}, [r5]               // QC8 params
        VDUP.8      q9 , d20[2]                 // output_min
        VDUP.8      q11, d20[3]                 // output_max
        VDUP.16     q10, d20[0]                 // output_zero_point

        .p2align    3
0:
        LDR         r11, [sp, 100]              // input_offset
        LDMIB       r2, {r5, r6}                // i0, i1
        LDR         r8, [r2]                    // i2
        CMP         r5, r4                      // i0 == zero?
        ADDNE       r5, r5, r11                 // i0 += input_offset
        CMP         r6, r4                      // i1 == zero?
        ADDNE       r6, r6, r11                 // i1 += input_offset
        CMP         r8, r4                      // i2 == zero?
        ADDNE       r8, r8, r11                 // i2 += input_offset

        MOV         lr, r3
        MOV         r9, 80

        // Is there at least 16 channels for main loop?
        SUBS        r11, r0, 16
        BLO         2f

// Main loop - 16 channels
// lr weights.  r3 reset
// r0/r11  loop counter.
// r5 i0
// r6 i1
// r8 i2
// q12 q13 q14 q15   accumulators

        .p2align    3
1:
        ADD         r7, lr, 64              // skip over bias to get weights
        VLD1.8      {q4}, [r8]!             // i2
        VLD1.8      {q12}, [r7]!            // w0
        VLD1.8      {q5}, [r5]!             // i0
        VLD1.8      {q13}, [r7]!            // w1
        VLD1.8      {q6}, [r6]!             // i1
        VLD1.8      {q14}, [r7]             // w2

        VMULL.S8    q1, d8,  d24            // i2 * w0
        VMULL.S8    q2, d9,  d25
        VMLAL.S8    q1, d10, d26            // i0 * w1
        VMLAL.S8    q2, d11, d27
        VMULL.S8    q0, d12, d28            // i1 * w2
        VLD1.8      {q12, q13}, [lr]!       // load bias
        VMULL.S8    q3, d13, d29
        VLD1.8      {q14, q15}, [lr], r9

        VADDW.S16   q12, q12, d0
        VADDW.S16   q13, q13, d1
        VADDW.S16   q14, q14, d4
        VADDW.S16   q15, q15, d5
        VADDW.S16   q12, q12, d2
        VADDW.S16   q13, q13, d3
        VADDW.S16   q14, q14, d6
        VLD1.32     {q0, q1}, [lr]!         // quant per channel scale values
        VADDW.S16   q15, q15, d7
        VLD1.32     {q2, q3}, [lr]!

        // QC8 FP32 quantization

        VCVT.F32.S32    q12, q12
        VCVT.F32.S32    q13, q13
        VCVT.F32.S32    q14, q14
        VCVT.F32.S32    q15, q15

        VMUL.F32    q12, q0, q12
        VMUL.F32    q13, q1, q13
        VMUL.F32    q14, q2, q14
        VMUL.F32    q15, q3, q15

        VCVTN.S32.F32   q12, q12
        VCVTN.S32.F32   q13, q13
        VCVTN.S32.F32   q14, q14
        VCVTN.S32.F32   q15, q15

        VQMOVN.S32  d24, q12
        VQMOVN.S32  d25, q13
        VQMOVN.S32  d28, q14
        VQMOVN.S32  d29, q15

        VQADD.S16   q12, q12, q10
        VQADD.S16   q14, q14, q10
        VQMOVN.S16  d24, q12
        VQMOVN.S16  d25, q14
        VMIN.S8     q12, q12, q11
        VMAX.S8     q12, q12, q9
        SUBS        r11, r11, 16
        VST1.8      {q12}, [r10]!
        BHS         1b

2:
        // Is there a remainder channels?  1-15
        ANDS        r11, r11, 15
        BNE         4f

3:
        LDR         r6, [sp, 92]            // input_stride
        SUBS        r1, r1, 1               // output_width
        ADD         r10, r10, r12           // output += output_increment
        ADD         r2, r2, r6              // input += input_stride
        BNE         0b

        VPOP        {d8, d9, d10, d11, d12, d13}
        ADD         sp, sp, 4               // pad
        POP         {r4, r5, r6, r7, r8, r9, r10, r11, pc}

// Small Remainder - 1-8 channels
4:
        CMP         r11, 9                  // handle 9 or more
        ADD         r7, lr, 64              // skip over bias to get weights
        BHS         5f

        MOV         r9, 16

        VLD1.8      {d8}, [r8]              // i2
        VLD1.8      {d24}, [r7], r9         // w0
        VLD1.8      {d10}, [r5]             // i0
        VLD1.8      {d26}, [r7], r9         // w1
        VLD1.8      {d12}, [r6]             // i1
        VLD1.8      {d28}, [r7]             // w2

        VMULL.S8    q1, d8,  d24            // i2 * w0
        VMLAL.S8    q1, d10, d26            // i0 * w1
        VMULL.S8    q0, d12, d28            // i1 * w2
        VLD1.8      {q12, q13}, [lr]        // load bias
        ADD         lr, lr, 112

        VADDW.S16   q12, q12, d0
        VADDW.S16   q13, q13, d1
        VADDW.S16   q12, q12, d2
        VADDW.S16   q13, q13, d3
        VLD1.32     {q0, q1}, [lr]          // quant per channel scale values

        // QC8 FP32 quantization

        VCVT.F32.S32    q12, q12
        VCVT.F32.S32    q13, q13

        VMUL.F32    q12, q0, q12
        VMUL.F32    q13, q1, q13

        VCVTN.S32.F32   q12, q12
        VCVTN.S32.F32   q13, q13

        VQMOVN.S32  d24, q12
        VQMOVN.S32  d25, q13

        VQADD.S16   q12, q12, q10
        VQMOVN.S16  d24, q12
        VMIN.S8     d24, d24, d22
        VMAX.S8     d24, d24, d18

        //  Store 8
        TST         r11, 8
        BEQ         6f
        VST1.8      {d24}, [r10]!
        B           3b

        .p2align    3
// Large Remainder - 9-15 channels
// Process 16 same as main loop, but conditional store

5:
        VLD1.8      {q4}, [r8]!             // i2
        VLD1.8      {q12}, [r7]!            // w0
        VLD1.8      {q5}, [r5]!             // i0
        VLD1.8      {q13}, [r7]!            // w1
        VLD1.8      {q6}, [r6]!             // i1
        VLD1.8      {q14}, [r7]             // w2

        VMULL.S8    q1, d8,  d24            // i2 * w0
        VMULL.S8    q2, d9,  d25
        VMLAL.S8    q1, d10, d26            // i0 * w1
        VMLAL.S8    q2, d11, d27
        VMULL.S8    q0, d12, d28            // i1 * w2
        VLD1.8      {q12, q13}, [lr]!       // load bias
        VMULL.S8    q3, d13, d29
        VLD1.8      {q14, q15}, [lr], r9

        VADDW.S16   q12, q12, d0
        VADDW.S16   q13, q13, d1
        VADDW.S16   q14, q14, d4
        VADDW.S16   q15, q15, d5
        VADDW.S16   q12, q12, d2
        VADDW.S16   q13, q13, d3
        VADDW.S16   q14, q14, d6
        VLD1.32     {q0, q1}, [lr]!         // quant per channel scale values
        VADDW.S16   q15, q15, d7
        VLD1.32     {q2, q3}, [lr]

        // QC8 FP32 quantization

        VCVT.F32.S32    q12, q12
        VCVT.F32.S32    q13, q13
        VCVT.F32.S32    q14, q14
        VCVT.F32.S32    q15, q15

        VMUL.F32    q12, q0, q12
        VMUL.F32    q13, q1, q13
        VMUL.F32    q14, q2, q14
        VMUL.F32    q15, q3, q15

        VCVTN.S32.F32   q12, q12
        VCVTN.S32.F32   q13, q13
        VCVTN.S32.F32   q14, q14
        VCVTN.S32.F32   q15, q15

        VQMOVN.S32  d24, q12
        VQMOVN.S32  d25, q13
        VQMOVN.S32  d28, q14
        VQMOVN.S32  d29, q15

        VQADD.S16   q12, q12, q10
        VQADD.S16   q14, q14, q10
        VQMOVN.S16  d24, q12
        VQMOVN.S16  d25, q14
        VMIN.S8     q12, q12, q11
        VMAX.S8     q12, q12, q9

        // Store 8
        VST1.8      {d24}, [r10]!
        VMOV        d24, d25

        // Store 4
6:
        TST         r11, 4
        BEQ         7f
        VST1.32     {d24[0]}, [r10]!
        VEXT.8      d24, d24, d24, 4

        // Store 2
7:
        TST         r11, 2
        BEQ         8f
        VST1.16     {d24[0]}, [r10]!
        VEXT.8      d24, d24, d24, 2

        // Store 1
8:
        TST         r11, 1
        BEQ         3b
        VST1.8      {d24[0]}, [r10]!
        B           3b


END_FUNCTION xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p16c__asm_aarch32_neonv8_mla8_cortex_a35
#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
