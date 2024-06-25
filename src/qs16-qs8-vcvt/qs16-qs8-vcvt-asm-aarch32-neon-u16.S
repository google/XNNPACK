// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

.syntax unified

// void xnn_qs16_qs8_vcvt_ukernel__asm_aarch32_neon_u16(
//     size_t batch,                         r0
//     const int16_t* input,                 r1
//     int8_t* output,                       r2
//     xnn_qs16_qs8_cvt_neon_params params   r3

// d8-d15, r12-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// vin        r1 d24 d25 d26 d27
// vacc          q8 q9 q10 q11
// vout       r2 d4 d5
// multiplier r3 d0 d1
// zero point    q1

BEGIN_FUNCTION xnn_qs16_qs8_vcvt_ukernel__asm_aarch32_neon_u16
        .arm
#ifndef __APPLE__
        .arch       armv7-a
        .fpu        neon
#endif

        VLD1.32     {d0[],d1[]}, [r3]!      // vmultiplier
        SUBS        r0, r0, 32              // batch of 32 bytes
        VLD1.16     {d2[],d3[]}, [r3]       // zero point
        BLO         1f

        // Main loop 16 bytes output
0:
        VLD1.16     {d24,d25,d26,d27}, [r1]!    // load 16 int16_t
        SUBS        r0, r0, 32
        VSHLL.S16   q8, d24, 15
        VSHLL.S16   q9, d25, 15
        VSHLL.S16   q10, d26, 15
        VSHLL.S16   q11, d27, 15
        VQRDMULH.S32    q8, q8, q0
        VQRDMULH.S32    q9, q9, q0
        VQRDMULH.S32    q10, q10, q0
        VQRDMULH.S32    q11, q11, q0
        VQMOVN.S32  d24, q8
        VQMOVN.S32  d25, q9
        VQMOVN.S32  d26, q10
        VQMOVN.S32  d27, q11
        VQADD.S16   q12, q12, q1
        VQADD.S16   q13, q13, q1
        VQMOVN.S16  d4, q12
        VQMOVN.S16  d5, q13
        VST1.8      {d4,d5}, [r2]!          // store 16 int8_t
        BHS         0b

        TST         r0, 31                  // Is there a remainder?
        BXEQ        lr

        // Remainder 1 to 15 bytes of output
1:
        VLD1.16     {d24,d25,d26,d27}, [r1]!    // load 16 int16_t
        VSHLL.S16   q8, d24, 15
        VSHLL.S16   q9, d25, 15
        VQRDMULH.S32    q8, q8, q0
        VQRDMULH.S32    q9, q9, q0
        VQMOVN.S32  d24, q8
        VQMOVN.S32  d25, q9
        VQADD.S16   q12, q12, q1
        VQMOVN.S16  d4, q12
        TST         r0, 16
        BEQ         2f

        VST1.8      {d4}, [r2]!             // store 8 int8_t
        VSHLL.S16   q10, d26, 15
        VSHLL.S16   q11, d27, 15
        VQRDMULH.S32    q10, q10, q0
        VQRDMULH.S32    q11, q11, q0
        VQMOVN.S32  d26, q10
        VQMOVN.S32  d27, q11
        VQADD.S16   q13, q13, q1
        VQMOVN.S16  d4, q13
2:
        TST         r0, 8
        BEQ         3f
        VST1.32     {d4[0]}, [r2]!          // store 4 int8_t
        VEXT.8      d4, d4, d4, #4
3:
        TST         r0, 4
        BEQ         4f

        VST1.16     {d4[0]}, [r2]!          // store 2 int8_t
        VEXT.8      d4, d4, d4, #2

4:
        TST         r0, 2
        BXEQ        lr
        VST1.8      {d4[0]}, [r2]!          // store 1 int8_t
        BX          lr

END_FUNCTION xnn_qs16_qs8_vcvt_ukernel__asm_aarch32_neon_u16

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
