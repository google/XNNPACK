// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32(
#     size_t mr,                         x0
#     size_t nc,                         x1
#     size_t kc,                         x2 / x0
#     size_t ks,                         x3 / x9
#     const void** restrict a,            x4
#     const void* restrict w,             x5
#     void* restrict c,                   x6
#     size_t cm_stride,                  (x7) - unused
#     size_t cn_stride,                  [sp] -> x10
#     size_t a_offset,                   [sp + 8] -> x11
#     const void* zero,                  [sp + 16] -> x12
#     const xnn_f16_minmax_params params [sp + 24] -> (x8)

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

// Register usage
// A0  x8 v0
// B   x5 v20 v21 v22 v23
// C0  x6 v24 v25
// clamp  v4, v5

BEGIN_FUNCTION xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32
        # Load cn_stride, a_offset
        LDP         x10, x11, [sp]

        # Load zero, params pointer
        LDP         x12, x8, [sp, 16]

        # Load params values
        LD2R        {v4.8h, v5.8h}, [x8]

0:
        # Load initial bias from w into accumulators
        LDR         q24, [x5], 16
        LDR         q25, [x5], 16
        MOVI        v26.8h, 0               // second set of C for pipelining FMLA
        MOVI        v27.8h, 0

        MOV         x9, x3                  // p = ks

1:
        # Load next A pointer
        LDR         x8, [x4], 8

        CMP         x8, x12                 // if a0 == zero
        ADD         x8, x8, x11             // a0 += a_offset
        CSEL        x8, x12, x8, EQ         //   a0 = zero, else += a0 + a_offset

        # Is there at least 2 halffloats (4 bytes)?
        SUBS        x0, x2, 4               // k = kc - 4
        B.LO        4f

       .p2align 3
        # Main loop - 2 halffloats of A (4 bytes)
2:
        LDR         s0,  [x8], 4
        LDR         q20, [x5, 0]
        LDR         q21, [x5, 16]
        LDR         q22, [x5, 32]
        LDR         q23, [x5, 48]
        SUBS        x0, x0, 4
        FMLA        v24.8h, v20.8h, v0.h[0]
        FMLA        v25.8h, v21.8h, v0.h[0]
        FMLA        v26.8h, v22.8h, v0.h[1]
        FMLA        v27.8h, v23.8h, v0.h[1]
        ADD         x5, x5, 64
        B.HS        2b

        # Is there a remainder?- 1 halffloat of A (2 bytes)
        TBNZ        x0, 1, 4f

3:
        # ks loop
        SUBS        x9, x9, 8               // ks -= MR * sizeof(void*)
        B.HI        1b

        FADD        v24.8h, v24.8h, v26.8h
        FADD        v25.8h, v25.8h, v27.8h

        # Clamp
        FMAX        v24.8h, v24.8h, v4.8h
        FMAX        v25.8h, v25.8h, v4.8h
        FMIN        v24.8h, v24.8h, v5.8h
        FMIN        v25.8h, v25.8h, v5.8h

        # Store full 1 x 16
        SUBS        x1, x1, 16
        B.LO        5f

        STP         q24, q25,  [x6]
        ADD         x6,  x6, x10

        SUB         x4, x4, x3              // a -= ks

        # nc loop
        B.HI        0b
        RET

        # Remainder- 1 halffloat of A
4:
        LDR         h0, [x8], 2
        LDR         q20, [x5], 16
        LDR         q21, [x5], 16
        FMLA        v24.8h, v20.8h, v0.h[0]
        FMLA        v25.8h, v21.8h, v0.h[0]
        B           3b

        # Store odd width
5:
        TBZ         x1, 3, 6f
        STR         q24, [x6], 16
        MOV         v24.16b, v25.16b
6:
        TBZ         x1, 2, 7f
        STR         d24, [x6], 8
        DUP         d24, v24.d[1]
7:
        TBZ         x1, 1, 8f
        STR         s24,  [x6], 4
        DUP         s24, v24.s[1]
8:
        TBZ         x1, 0, 9f
        STR         h24,  [x6]
9:
        RET

END_FUNCTION xnn_f16_igemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
