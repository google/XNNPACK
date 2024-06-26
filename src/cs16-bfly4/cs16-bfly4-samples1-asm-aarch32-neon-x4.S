// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

.syntax unified

// void xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x4(
//     size_t batch,                         r0
//     size_t samples,                       (unused)
//     int16_t* data,                        r2
//     const int16_t* twiddle,               (unused)
//     size_t stride)                        (unused)

// d8-d15, r12-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// vout0 r2 q0
// vout1    q1
// vout2    q2
// vout3    q3
// vtmp3    q8
// vtmp4    q9
// vtmp5    q10
// vtmp0    q11

// vdiv4    q12
// vnegr    q13

BEGIN_FUNCTION xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x4
        .arm
#ifndef __APPLE__
        .arch       armv7-a
        .fpu        neon
#endif
        SUBS        r0, r0, 4               // batch
        VMVN.U16    q12, 57344              // 8191
        VMOV.I32    q13, 0x0001ffff         // vnegr
        BLO         1f

        MOV         r3, r2                  // output = input for post inc

        // batch of 4 main loop
0:
        VLD4.32     {d0,d2,d4,d6}, [r2]!    // input first 2 batch
        VLD4.32     {d1,d3,d5,d7}, [r2]!    // input second 2 batch
        SUBS        r0, r0, 4               // batch
        VQRDMULH.S16    q1, q1, q12         // vout1 /= 4
        VQRDMULH.S16    q3, q3, q12         // vout3 /= 4
        VQRDMULH.S16    q0, q0, q12         // vout0 /= 4
        VQRDMULH.S16    q2, q2, q12         // vout2 /= 4

        VSUB.I16    q9, q1, q3              // vtmp4 = vout1 - vout3
        VADD.I16    q8, q1, q3              // vtmp3 = vout1 + vout3

        VMUL.S16    q9, q9, q13             // vrev4 = vtmp4 -r, i
        VADD.I16    q11, q0, q2             // vtmp0 = vout0 + vout2
        VSUB.I16    q10, q0, q2             // vtmp5 = vout0 - vout2

        VADD.I16    q0, q11, q8             // vout0 = vtmp0 + vtmp3
        VSUB.I16    q2, q11, q8             // vout2 = vtmp0 - vtmp3

        VREV32.16   q9, q9                  // vrev4 = vtmp4 i, -r
        VADD.I16    q1, q10, q9             // vout1 = vtmp5 + vrev4
        VSUB.I16    q3, q10, q9             // vout3 = vtmp5 - vrev4

        VST4.32     {d0,d2,d4,d6}, [r3]!    // output first 2 batch
        VST4.32     {d1,d3,d5,d7}, [r3]!    // output second 2 batch

        BHS         0b

1:
        ANDS        r0, r0, 3               // batch remainder?
        BXEQ        lr

        // Remainder batch of 1 to 3
2:
        VLD4.32     {d0[0],d1[0],d2[0],d3[0]}, [r2]     // input 1 batch
        SUBS        r0, r0, 1               // batch
        VQRDMULH.S16    d1, d1, d24         // vout1 /= 4
        VQRDMULH.S16    d3, d3, d24         // vout3 /= 4
        VQRDMULH.S16    d0, d0, d24         // vout0 /= 4
        VQRDMULH.S16    d2, d2, d24         // vout2 /= 4

        VSUB.I16    d5, d1, d3              // vtmp4 = vout1 - vout3
        VADD.I16    d4, d1, d3              // vtmp3 = vout1 + vout3
        VMUL.S16    d5, d5, d26             // vrev4 = vtmp4 -r, i

        VADD.I16    d7, d0, d2              // vtmp0 = vout0 + vout2
        VSUB.I16    d6, d0, d2              // vtmp5 = vout0 - vout2

        VADD.I16    d0, d7, d4              // vout0 = vtmp0 + vtmp3
        VSUB.I16    d2, d7, d4              // vout2 = vtmp0 - vtmp3

        VREV32.16   d5, d5                  // vrev4 = vtmp4 i, -r
        VADD.I16    d1, d6, d5              // vout1 = vtmp5 + vrev4
        VSUB.I16    d3, d6, d5              // vout3 = vtmp5 - vrev4

        VST4.32     {d0[0],d1[0],d2[0],d3[0]}, [r2]!    // output 1 batch
        BHI         2b

        BX          lr

END_FUNCTION xnn_cs16_bfly4_samples1_ukernel__asm_aarch32_neon_x4

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
