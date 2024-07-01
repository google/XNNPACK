// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "xnnpack/assembly.h"

.syntax unified

// void xnn_cs16_fftr_ukernel__asm_aarch32_neon_x4(
//     size_t samples,                       r0 (256)
//     int16_t* data,                        r1
//     const int16_t* twiddle)               r2

// d8-d15, r12-r11,r14(lr) need to be preserved if used. r13(sp),r15(pc) are reserved.

// Register usage
// vilr   r1 d0
// vili      d1
// virr   r3 d2
// viri      d3
// vdiv2     d4
// vtwr   r2 d6
// vtwi      d7

// vacc1r    q8 = vilr + virr;
// vacc1i    q9 = vili + viri;
// vacc2r    d0 = vilr - virr;
// vacc2i    d1 = vili - viri;

// vaccr    q10 (d20/d21)
// vacci    q11 (d22/d23)
// voutlr   q12 (vaccr + vacc1r) / 2
// voutli   q13 (vacci + vacc1i) / 2
// voutrr   q14 (vacc1r - vaccr) / 2
// voutri   q15 (vacci - vacc1i) / 2
// unused d5, d8-d15

BEGIN_FUNCTION xnn_cs16_fftr_ukernel__asm_aarch32_neon_x4
        .arm
#ifndef __APPLE__
        .arch       armv7-a
        .fpu        neon
#endif
        ADD         r3, r1, r0, lsl #2      // dr = data + samples * 4
        VMOV.U16    q0, 0
        VMVN.U16    d4, 49152               // 16383
        VLD2.16     {d0[0],d1[0]}, [r1]     // first value
        VQRDMULH.S16    q0, q0, d4[0]       // vilr /= 2
        VADD.I16    d16, d0, d1             // dl[0] = vicr + vici;
        VSUB.I16    d18, d0, d1             // dr[0] = vicr - vici;
        VST1.32     {d16[0]}, [r1]!
        VST1.32     {d18[0]}, [r3]

        // Main loop of 4 cs16 value at a time
0:
        SUB         r3, r3, 16              // dr -= 16
        VLD2.16     {d0,d1}, [r1]           // load left r and i
        VLD2.16     {d2,d3}, [r3]           // load right r and i
        VLD2.16     {d6,d7}, [r2]!          // load twiddle values vtwr, vtwi
        VREV64.16   q1, q1                  // reverse right side

        VQRDMULH.S16    q0, q0, d4[0]       // vilr /= 2
        VQRDMULH.S16    q1, q1, d4[0]       // virr /= 2
        VADDL.S16   q8, d0, d2              // vacc1r = vilr + virr;
        VSUBL.S16   q9, d1, d3              // vacc1i = vili - viri;

        VSUB.I16    d0, d0, d2              // vacc2r = vilr - virr;
        VADD.I16    d1, d1, d3              // vacc2i = vili + viri;

        VMULL.S16   q10, d0, d6             // vaccr = vacc2r * vtwr
        VMULL.S16   q11, d0, d7             // vacci = vacc2r * vtwi
        VMLSL.S16   q10, d1, d7             // vaccr -= vacc2i * vtwi
        VMLAL.S16   q11, d1, d6             // vacci += vacc2i * vtwr
        VRSHR.S32   q10, q10, 15            // (vaccr + 16384) >> 15
        VRSHR.S32   q11, q11, 15            // (vacci + 16384) >> 15

        VHADD.S32   q12, q10, q8            // (vaccr + vacc1r) / 2
        VHADD.S32   q13, q11, q9            // (vacci + vacc1i) / 2
        VHSUB.S32   q14, q8, q10            // (vacc1r - vaccr) / 2
        VHSUB.S32   q15, q11, q9            // (vacci - vacc1i) / 2

        VMOVN.S32   d0, q12
        VMOVN.S32   d1, q13
        VMOVN.S32   d2, q14
        VMOVN.S32   d3, q15

        SUBS        r0, r0, 8               // 8 samples (left and right) per loop
        VREV64.16   q1, q1                  // reverse right side

        VST2.16     {d0,d1}, [r1]!          // store left r and i
        VST2.16     {d2,d3}, [r3]           // store right r and i
        BHI         0b

        BX          lr

END_FUNCTION xnn_cs16_fftr_ukernel__asm_aarch32_neon_x4

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
