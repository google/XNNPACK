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

BEGIN_FUNCTION xnn_cs16_bfly4_samples1_ukernel__aarch32_neon_x1
        .arm
#ifndef __APPLE__
        .arch   armv7-a
        .fpu    neon
#endif
        VMVN.U16      q8, 57344          // 8191
0:
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
        BNE           0b
        BX            lr

END_FUNCTION xnn_cs16_bfly4_samples1_ukernel__aarch32_neon_x1

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif