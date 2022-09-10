// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

.syntax unified

// void xnn_cs16_bfly4_samples1_ukernel__aarch32_neon_x4(
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

// div4     q8
// vtmp3    q9
// vtmp4    q10
// vtmp5    q11
// maski    q12

BEGIN_FUNCTION xnn_cs16_bfly4_samples1_ukernel__aarch32_neon_x4
        .arm
#ifndef __APPLE__
        .arch   armv7-a
        .fpu    neon
#endif
        SUBS    r0, r0, 4               // batch
        VMVN.U16 q8, 57344              // 8191
        VMOV.I64 q12, 0xffff0000ffff0000 // mask 0xffff0000
        BLO     1f

        MOV     r3, r2                  // output = input for post inc

        // batch of 4 main loop
0:
        VLD4.32 {d0,d2,d4,d6}, [r2]!    // input first 2 batch
        VLD4.32 {d1,d3,d5,d7}, [r2]!    // input second 2 batch
        SUBS    r0, r0, 4               // batch
        VQRDMULH.S16 q1, q1, q8         // vout1 /= 4
        VQRDMULH.S16 q3, q3, q8         // vout3 /= 4
        VQRDMULH.S16 q0, q0, q8         // vout0 /= 4
        VQRDMULH.S16 q2, q2, q8         // vout2 /= 4

        VSUB.I16 q10, q1, q3            // vtmp4 = vout1 - vout3
        VADD.I16 q9, q1, q3             // vtmp3 = vout1 + vout3

        VREV32.16 q10, q10              // vrev4 = vtmp4 r and i swapped
        VSUB.I16 q11, q0, q2            // vtmp5 = vout0 - vout2
        VADD.I16 q0, q0, q2             // vout0 = vout0 + vout2

        VADD.I16 q1, q11, q10           // vout1 = vtmp5 + vrev4
        VSUB.I16 q3, q11, q10           // vout3 = vtmp5 - vrev4
        VMOV.I16 q11, q1                // vout1 copy

        VSUB.I16 q2, q0, q9             // vout2 = vout0 - vtmp3
        VBIT.I16 q1, q3, q12            // vout1 = 1r 3i x 3r 1i -> 1r 1i
        VADD.I16 q0, q0, q9             // vout0 = vout0 + vtmp3
        VBIT.I16 q3, q11, q12           // vout3 = 3r 1i x 1r 3i -> 3r 3i

        VST4.32 {d0,d2,d4,d6}, [r3]!    // output first 2 batch
        VST4.32 {d1,d3,d5,d7}, [r3]!    // output second 2 batch

        BHS     0b

1:
        ANDS    r0, r0, 3               // batch remainder?
        BXEQ    lr

        // Remainder batch of 1 to 3
2:
        VLD4.32 {d0[0],d1[0],d2[0],d3[0]}, [r2]  // input 1 batch
        SUBS    r0, r0, 1               // batch
        VQRDMULH.S16 d1, d1, d16        // vout1 /= 4
        VQRDMULH.S16 d3, d3, d16        // vout3 /= 4
        VQRDMULH.S16 d0, d0, d16        // vout0 /= 4
        VQRDMULH.S16 d2, d2, d16        // vout2 /= 4

        VSUB.I16 d6, d1, d3             // vtmp4 = vout1 - vout3
        VADD.I16 d5, d1, d3             // vtmp3 = vout1 + vout3

        VREV32.16 d6, d6                // vrev4 = vtmp4 r and i swapped
        VSUB.I16 d7, d0, d2             // vtmp5 = vout0 - vout2
        VADD.I16 d0, d0, d2             // vout0 = vout0 + vout2

        VADD.I16 d1, d7, d6             // vout1 = vtmp5 + vrev4
        VSUB.I16 d3, d7, d6             // vout3 = vtmp5 - vrev4
        VMOV.I16 d6, d1                 // vout1 copy

        VSUB.I16 d2, d0, d5             // vout2 = vout0 - vtmp3
        VBIT.I16 d1, d3, d24            // vout1 = 1r 3i x 3r 1i -> 1r 1i
        VADD.I16 d0, d0, d5             // vout0 = vout0 + vtmp3
        VBIT.I16 d3, d6, d24            // vout3 = 3r 1i x 1r 3i -> 3r 3i

        VST4.32 {d0[0],d1[0],d2[0],d3[0]}, [r2]! // output 1 batch
        BHI     2b

        BX      lr

END_FUNCTION xnn_cs16_bfly4_samples1_ukernel__aarch32_neon_x4

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
