// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

.syntax unified

// void xnn_cs16_bfly4_samples1_ukernel__aarch32_neon(
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

BEGIN_FUNCTION xnn_cs16_bfly4_samples1_ukernel__aarch32_neon
        .arm
#ifndef __APPLE__
        .arch   armv7-a
        .fpu    neon
#endif
        SUBS          r0, r0, 4          // batch
        VMVN.U16      q8, 57344          // 8191
        MOV           r3, r2             // output = input for post inc
        BLO           1f

        // batch of 4 main loop
0:
        VLD4.32       {d0,d2,d4,d6}, [r2]!   // input first 2 batch
        VLD4.32       {d1,d3,d5,d7}, [r2]!   // input second 2 batch
        SUBS          r0, r0, 4          // batch
        VQRDMULH.S16  q0, q0, q8         // vout0 /= 4
        VQRDMULH.S16  q1, q1, q8         // vout1 /= 4
        VQRDMULH.S16  q2, q2, q8         // vout2 /= 4
        VQRDMULH.S16  q3, q3, q8         // vout3 /= 4

        VSUB.I16      q11, q0, q2        // vtmp5 = vout0 - vout2
        VADD.I16      q0, q0, q2         // vout0 = vout0 + vout2

        VADD.I16      q9, q1, q3         // vtmp3 = vout1 + vout3
        VSUB.I16      q10, q1, q3        // vtmp4 = vout1 - vout3

        VSUB.I16      q2, q0, q9         // vout2 = vout0 - vtmp3
        VREV32.16     q10, q10           // vrev4 = vtmp4 r and i swapped
        VADD.I16      q0, q0, q9         // vout0 = vout0 + vtmp3

        VSUB.I16      q3, q11, q10       // vout3 = vtmp5 - vrev4
        VADD.I16      q1, q11, q10       // vout1 = vtmp5 + vrev4

        VREV32.16     q3, q3             // vout3 = 3r 1i -> 1i 3r
        VTRN.I16      q1, q3             // vout1 = 1r 3i x 1i 3r -> 1r 1i x 3i 3r
        VREV32.16     q3, q3             // vout3 = 3i 3r -> 3r 3i

        VST4.32       {d0,d2,d4,d6}, [r3]!  // output first 2 batch
        VST4.32       {d1,d3,d5,d7}, [r3]!  // output second 2 batch

        BHS           0b

1:
        ADDS          r0, r0, 3          // batch remainder?
        BLO           3f

        // Remainder batch of 1 to 3
2:
        VLD1.16       {d18-d19}, [r2]    // input
        SUBS          r0, r0, 1          // batch
        VQRDMULH.S16  q9, q9, q8         // DIV4
        VADD.I16      d20, d19, d18
        VSUB.I16      d18, d18, d19
        VEXT.8        d19, d20, d20, 4
        VREV64.16     d21, d18
        VADD.I16      d22, d19, d20
        VADD.I16      d23, d21, d18
        VST1.32       {d22[0]}, [r2]!    // output
        VSUB.I16      d18, d18, d21
        VST1.16       {d23[0]}, [r2]!
        VSUB.I16      d19, d20, d19
        VST1.16       {d18[1]}, [r2]!
        VST1.32       {d19[0]}, [r2]!
        VST1.16       {d18[0]}, [r2]!
        VST1.16       {d23[1]}, [r2]!
        BHS           2b

3:
        BX            lr

END_FUNCTION xnn_cs16_bfly4_samples1_ukernel__aarch32_neon

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif