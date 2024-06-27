// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

.syntax unified

// void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p8c__asm_aarch32_neonv8_mla8_cortex_a35(
//   size_t channels,                          r0, r11
//   size_t output_width,                      r1
//   const int8_t** input,                     r2
//   const void* weights,                      r3
//   int8_t* output,                           r10, [sp, 40]
//   intptr_t input_stride,                    r9,  [sp, 44]
//   size_t output_increment,                  r12, [sp, 48]
//   size_t input_offset,                      r7,  [sp, 52]
//   const int8_t* zero,                       r4,  [sp, 56]
//   const union xnn_qs8_minmax_params params  r5,  [sp, 60]

// d8-d15, r4-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// A0   r5  d4
// A1   r6  d5
// A2   r8  d6
// B    r3/lr  d7 d16 d17
// C0  r10 q12 q13 q14 q15
// Prod q0 q1

// params structure is 4 bytes
//   struct {
//     int16_t output_zero_point;  d20[0] q10
//     int8_t output_min;          d20[2] d18 q9
//     int8_t output_max;          d20[3] d19
//   } xnn_qs8_minmax_params.neonv8;
// unused q4 q5 q6 q7 q11

BEGIN_FUNCTION xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p8c__asm_aarch32_neonv8_mla8_cortex_a35
        // 40 bytes of stack.  36 + 4 pad
        PUSH        {r4, r5, r6, r7, r8, r9, r10, r11, lr}  // 40
        SUB         sp, sp, 4
        LDR         r5,  [sp, 60]               // params
        LDR         r10, [sp, 40]               // output
        LDR         r9,  [sp, 44]               // input_stride
        LDR         r12, [sp, 48]               // output_increment
        LDR         r7,  [sp, 52]               // input_offset
        LDR         r4,  [sp, 56]               // zero

        VLD1.32     {d20[]}, [r5]               // QC8 params
        VDUP.8      d18, d20[2]                 // output_min
        VDUP.8      d19, d20[3]                 // output_max
        VDUP.16     q10, d20[0]                 // output_zero_point

        .p2align    3
0:
        LDMIB       r2, {r5, r6}                // i0, i1
        LDR         r8, [r2]                    // i2
        CMP         r5, r4                      // i0 == zero?
        ADDNE       r5, r5, r7                  // i0 += input_offset
        CMP         r6, r4                      // i1 == zero?
        ADDNE       r6, r6, r7                  // i1 += input_offset
        CMP         r8, r4                      // i2 == zero?
        ADDNE       r8, r8, r7                  // i2 += input_offset

        MOV         lr, r3
        MOV         r11, r0                 // channel count as is, fall into loop

// Main loop - 8 channels
// lr weights.  r3 reset
// r0/r11  loop counter.
// r5 i0
// r6 i1
// r8 i2
// q12 q13 q14 q15   accumulators
// Weights are:
//   32 bias - 8 int
//   24 weights - 3 * 8 byte
//   32 quant scale - 8 int
//   88 bytes total

        .p2align    3
1:
        VLD1.8      {q12, q13}, [lr]!       // load bias

        VLD1.8      {d4}, [r8]!             // i2
        VLD1.8      {d7}, [lr]!             // w0
        VLD1.8      {d5}, [r5]!             // i0
        VLD1.8      {d16}, [lr]!            // w1
        VLD1.8      {d6}, [r6]!             // i1
        VLD1.8      {d17}, [lr]!            // w2

        VMULL.S8    q1, d4, d7              // i2 * w0
        VMLAL.S8    q1, d5, d16             // i0 * w1
        VMULL.S8    q0, d6, d17             // i1 * w2


        VADDW.S16   q12, q12, d0
        VADDW.S16   q13, q13, d1
        VADDW.S16   q12, q12, d2
        VADDW.S16   q13, q13, d3

        VLD1.32     {q0, q1}, [lr]!         // quant per channel scale values

        // QC8 FP32 quantization

        VCVT.F32.S32    q12, q12
        VCVT.F32.S32    q13, q13

        VMUL.F32    q12, q0, q12
        VMUL.F32    q13, q1, q13

        VCVTN.S32.F32   q12, q12
        VCVTN.S32.F32   q13, q13

        VQMOVN.S32  d24, q12
        VQMOVN.S32  d25, q13
        SUBS        r11, r11, 8             // 8 channels per loop

        VQADD.S16   q12, q12, q10
        VQMOVN.S16  d24, q12
        VMIN.S8     d24, d24, d19
        VMAX.S8     d24, d24, d18

        BLO         3f                      // less than 8?

        VST1.8      {d24}, [r10]!
        BHI         1b                      // at least 1, continue loop

2:
        SUBS        r1, r1, 1               // output_width
        ADD         r10, r10, r12           // output += output_increment
        ADD         r2, r2, r9              // input += input_stride
        BNE         0b

        ADD         sp, sp, 4
        POP         {r4, r5, r6, r7, r8, r9, r10, r11, pc}

        .p2align    3
        // Store 4
3:
        TST         r11, 4
        BEQ         4f
        VST1.32     {d24[0]}, [r10]!
        VEXT.8      d24, d24, d24, 4

        // Store 2
4:
        TST         r11, 2
        BEQ         5f
        VST1.16     {d24[0]}, [r10]!
        VEXT.8      d24, d24, d24, 2

        // Store 1
5:
        TST         r11, 1
        BEQ         2b
        VST1.8      {d24[0]}, [r10]!
        B           2b


END_FUNCTION xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_3p8c__asm_aarch32_neonv8_mla8_cortex_a35
#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
