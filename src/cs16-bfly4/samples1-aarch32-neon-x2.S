// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

.syntax unified

// void xnn_cs16_bfly4_samples1_ukernel__aarch32_neon_x2(
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
// maski    d16

BEGIN_FUNCTION xnn_cs16_bfly4_samples1_ukernel__aarch32_neon_x2
        .arm
#ifndef __APPLE__
        .arch   armv7-a
        .fpu    neon
#endif
        SUBS    r0, r0, 1               // batch
        VMVN.U16 d4, 57344              // 8191
        VMOV.I64 d16, 0xffff0000ffff0000 // mask 0xffff0000
        BLS     1f

        // Batch of 2 main loop
0:
        VLD4.32 {d0,d1,d2,d3}, [r2]     // input 2 batches
        SUBS    r0, r0, 2               // batch
        VQRDMULH.S16 d1, d1, d4         // vout1 /= 4
        VQRDMULH.S16 d3, d3, d4         // vout3 /= 4
        VQRDMULH.S16 d0, d0, d4         // vout0 /= 4
        VQRDMULH.S16 d2, d2, d4         // vout2 /= 4

        VSUB.I16 d6, d1, d3             // vtmp4 = vout1 - vout3
        VADD.I16 d5, d1, d3             // vtmp3 = vout1 + vout3

        VREV32.16 d6, d6                // vrev4 = vtmp4 r and i swapped
        VSUB.I16 d7, d0, d2             // vtmp5 = vout0 - vout2
        VADD.I16 d0, d0, d2             // vout0 = vout0 + vout2

        VADD.I16 d1, d7, d6             // vout1 = vtmp5 + vrev4
        VSUB.I16 d3, d7, d6             // vout3 = vtmp5 - vrev4

        VMOV.I16 d6, d1                 // vout1 copy
        VSUB.I16 d2, d0, d5             // vout2 = vout0 - vtmp3
        VBIT.I16 d1, d3, d16            // vout1 = 1r 3i x 3r 1i -> 1r 1i
        VADD.I16 d0, d0, d5             // vout0 = vout0 + vtmp3
        VBIT.I16 d3, d6, d16            // vout3 = 3r 1i x 1r 3i -> 3r 3i

        VST4.32 {d0,d1,d2,d3}, [r2]!    // output 2 batches

        BHI     0b

        BXLO    lr                      // no remainder?  early return

        // Remainder batch of 1
1:
        VLD4.32 {d0[0],d1[0],d2[0],d3[0]}, [r2]   // input 1 batch
        VQRDMULH.S16 d1, d1, d4         // vout1 /= 4
        VQRDMULH.S16 d3, d3, d4         // vout3 /= 4
        VQRDMULH.S16 d0, d0, d4         // vout0 /= 4
        VQRDMULH.S16 d2, d2, d4         // vout2 /= 4

        VSUB.I16 d6, d1, d3             // vtmp4 = vout1 - vout3
        VADD.I16 d5, d1, d3             // vtmp3 = vout1 + vout3

        VREV32.16 d6, d6                // vrev4 = vtmp4 r and i swapped
        VSUB.I16 d7, d0, d2             // vtmp5 = vout0 - vout2
        VADD.I16 d0, d0, d2             // vout0 = vout0 + vout2

        VADD.I16 d1, d7, d6             // vout1 = vtmp5 + vrev4
        VSUB.I16 d3, d7, d6             // vout3 = vtmp5 - vrev4

        VMOV.I16 d6, d1                 // vout1 copy
        VSUB.I16 d2, d0, d5             // vout2 = vout0 - vtmp3
        VBIT.I16 d1, d3, d16            // vout1 = 1r 3i x 3r 1i -> 1r 1i
        VADD.I16 d0, d0, d5             // vout0 = vout0 + vtmp3
        VBIT.I16 d3, d6, d16            // vout3 = 3r 1i x 1r 3i -> 3r 3i

        VST4.32 {d0[0],d1[0],d2[0],d3[0]}, [r2] // output 1 batch
        BX      lr

END_FUNCTION xnn_cs16_bfly4_samples1_ukernel__aarch32_neon_x2

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
