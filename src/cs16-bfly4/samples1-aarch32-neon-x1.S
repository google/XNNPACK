// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

.syntax unified

// void xnn_cs16_bfly4_samples1_ukernel__aarch32_neon_x1(
//     size_t batch,                         r0
//     size_t samples,                       (unused)
//     int16_t* data,                        r2
//     const int16_t* twiddle,               (unused)
//     size_t stride)                        (unused)

// d8-d15, r12-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// vout0 r2 d0
// vout1    d1
// vout2    d2
// vout3    d3

// div4     d4
// vtmp3    d5
// vtmp4    d6
// vtmp5    d7

BEGIN_FUNCTION xnn_cs16_bfly4_samples1_ukernel__aarch32_neon_x1
        .arm
#ifndef __APPLE__
        .arch   armv7-a
        .fpu    neon
#endif
        VMVN.U16 d4, 57344              // 8191

        // Remainder batch of 1
0:
        VLD4.32 {d0[0],d1[0],d2[0],d3[0]}, [r2]   // input 1 batch
        SUBS    r0, r0, 1               // batch
        VQRDMULH.S16 d1, d1, d4         // vout1 /= 4
        VQRDMULH.S16 d3, d3, d4         // vout3 /= 4
        VQRDMULH.S16 d0, d0, d4         // vout0 /= 4
        VQRDMULH.S16 d2, d2, d4         // vout2 /= 4

        VSUB.I16 d6, d1, d3             // vtmp4 = vout1 - vout3
        VADD.I16 d5, d1, d3             // vtmp3 = vout1 + vout3

        VREV32.16 d6, d6                // vrev4 = vtmp4 r and i swapped
        VSUB.I16 d7, d0, d2             // vtmp5 = vout0 - vout2
        VADD.I16 d0, d0, d2             // vout0 = vout0 + vout2

        VSUB.I16 d3, d7, d6             // vout3 = vtmp5 - vrev4
        VADD.I16 d1, d7, d6             // vout1 = vtmp5 + vrev4

        VREV32.16 d3, d3                // vout3 = 3r 1i -> 1i 3r
        VSUB.I16 d2, d0, d5             // vout2 = vout0 - vtmp3
        VTRN.I16 d1, d3                 // vout1 = 1r 3i x 1i 3r -> 1r 1i x 3i 3r
        VADD.I16 d0, d0, d5             // vout0 = vout0 + vtmp3
        VREV32.16 d3, d3                // vout3 = 3i 3r -> 3r 3i

        VST4.32 {d0[0],d1[0],d2[0],d3[0]}, [r2]! // output 1 batch
        BHI     0b

        BX      lr

END_FUNCTION xnn_cs16_bfly4_samples1_ukernel__aarch32_neon_x1

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
