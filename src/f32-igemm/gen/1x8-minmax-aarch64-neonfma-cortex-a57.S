// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/1x8-aarch64-neonfma-cortex-a75.S.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f32_igemm_minmax_ukernel_4x8__aarch64_neonfma_cortex_a57(
#     size_t mr,                         (x0) - unused.  mr = 1
#     size_t nc,                         x1
#     size_t kc,                         x2 / x0
#     size_t ks,                         x3 / x9
#     const float**restrict a,           x4
#     const float*restrict w,            x5
#     float*restrict c,                  x6
#     size_t cm_stride,                  (x7) - unused
#     size_t cn_stride,                  [sp] -> x10
#     size_t a_offset,                   [sp + 8] -> x11
#     const float* zero,                 [sp + 16] -> x12
#     const xnn_f32_minmax_params params [sp + 24] -> x8

# d8-d15 need to be preserved if used.
# x19-30 need to be preserved if used.

# A pointer
# x8  a0

# C pointer
# x6  c0

BEGIN_FUNCTION xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a57

        # Load cn_stride, a_offset
        LDP x10, x11, [sp]

        # Load zero, clamping params pointer
        LDP x12, x8, [sp, 16]

        # Load min/max values
        LD2R {v30.4s, v31.4s}, [x8]

0:
        # Load initial bias from w into accumulators
        LDP q16, q17, [x5], 32
        MOVI v18.4s, 0  // second set of C for pipelining FMLA
        MOVI v19.4s, 0

        MOV x9, x3  // p = ks

1:
        # Load next A pointer
        LDR x8, [x4], 8

        CMP x8, x12           // if a0 == zero
        ADD x8, x8, x11       // a0 += a_offset
        CSEL x8, x12, x8, EQ  //   a0 = zero, else += a0 + a_offset

        # Is there at least 8 floats (32 bytes) for prologue + epilogue?
        SUBS x0, x2, 32  // k = kc - 32    // k = kc
        B.LO 4f

        # 16 prologue
        # Read first block of A and B.
        LDP q20, q21, [x5], 32
        LDP q22, q23, [x5], 32
        LDP q24, q25, [x5], 32
        LDP q26, q27, [x5], 32
        LDR q0, [x8], 16

        # Is there at least 8.  yes do main loop
        SUBS x0, x0, 32
        B.LO 3f

        # Main loop - 8 floats of A (32 bytes)
2:
        # First block of 4.  FMA for first 4, loads for 2nd block of 4.
        FMLA v16.4s, v20.4s, v0.s[0]
        LDR q1, [x8], 16
        FMLA v17.4s, v21.4s, v0.s[0]
        LDP q20, q21, [x5], 32
        FMLA v18.4s, v22.4s, v0.s[1]
        FMLA v19.4s, v23.4s, v0.s[1]
        LDP q22, q23, [x5], 32
        FMLA v16.4s, v24.4s, v0.s[2]
        FMLA v17.4s, v25.4s, v0.s[2]
        LDP q24, q25, [x5], 32
        FMLA v18.4s, v26.4s, v0.s[3]
        FMLA v19.4s, v27.4s, v0.s[3]
        LDP q26, q27, [x5], 32

        # Second block of 4.  FMA for second 4, loads for 1st block of 4.
        FMLA v16.4s, v20.4s, v1.s[0]
        LDR q0, [x8], 16
        FMLA v17.4s, v21.4s, v1.s[0]
        LDP q20, q21, [x5], 32
        FMLA v18.4s, v22.4s, v1.s[1]
        FMLA v19.4s, v23.4s, v1.s[1]
        LDP q22, q23, [x5], 32
        FMLA v16.4s, v24.4s, v1.s[2]
        FMLA v17.4s, v25.4s, v1.s[2]
        LDP q24, q25, [x5], 32
        FMLA v18.4s, v26.4s, v1.s[3]
        FMLA v19.4s, v27.4s, v1.s[3]
        SUBS x0, x0, 32
        LDP q26, q27, [x5], 32
        B.HS 2b

3:
        # Epilogue

        # First block of 4.  FMA for first 4, loads for 2nd block of 4.
        FMLA v16.4s, v20.4s, v0.s[0]
        LDR q1, [x8], 16
        FMLA v17.4s, v21.4s, v0.s[0]
        LDP q20, q21, [x5], 32
        FMLA v18.4s, v22.4s, v0.s[1]
        FMLA v19.4s, v23.4s, v0.s[1]
        LDP q22, q23, [x5], 32
        FMLA v16.4s, v24.4s, v0.s[2]
        FMLA v17.4s, v25.4s, v0.s[2]
        LDP q24, q25, [x5], 32
        FMLA v18.4s, v26.4s, v0.s[3]
        FMLA v19.4s, v27.4s, v0.s[3]
        LDP q26, q27, [x5], 32

        # Second block of 4.  no loads
        FMLA v16.4s, v20.4s, v1.s[0]
        FMLA v17.4s, v21.4s, v1.s[0]
        FMLA v18.4s, v22.4s, v1.s[1]
        FMLA v19.4s, v23.4s, v1.s[1]
        FMLA v16.4s, v24.4s, v1.s[2]
        FMLA v17.4s, v25.4s, v1.s[2]
        FMLA v18.4s, v26.4s, v1.s[3]
        FMLA v19.4s, v27.4s, v1.s[3]

4:
        # Is there a remainder?- 4 floats of A (16 bytes)
        TBNZ x0, 4, 6f
        # Is there a remainder?- 2 floats of A (8 bytes)
        TBNZ x0, 3, 7f
        # Is there a remainder?- 1 floats of A (4 bytes)
        TBNZ x0, 2, 9f

5:
        # ks loop
        SUBS x9, x9, 8  // ks -= MR * sizeof(void*)
        B.HI 1b

        FADD v16.4s, v16.4s, v18.4s
        FADD v17.4s, v17.4s, v19.4s

        # Clamp
        FMAX v16.4s, v16.4s, v30.4s
        FMAX v17.4s, v17.4s, v30.4s
        FMIN v16.4s, v16.4s, v31.4s
        FMIN v17.4s, v17.4s, v31.4s

        # Store full 1 x 8
        SUBS x1, x1, 8
        B.LO 10f

        STP q16, q17, [x6]
        ADD x6, x6, x10

        SUB x4, x4, x3  // a -= ks

        # nc loop
        B.HI 0b

        RET

6:
        # Remainder- 4 floats of A (16 bytes)
        LDP q20, q21, [x5], 32
        LDR q0, [x8], 16
        FMLA v16.4s, v20.4s, v0.s[0]
        FMLA v17.4s, v21.4s, v0.s[0]
        LDP q22, q23, [x5], 32
        LDP q24, q25, [x5], 32
        LDP q26, q27, [x5], 32
        FMLA v18.4s, v22.4s, v0.s[1]
        FMLA v19.4s, v23.4s, v0.s[1]
        FMLA v16.4s, v24.4s, v0.s[2]
        FMLA v17.4s, v25.4s, v0.s[2]
        FMLA v18.4s, v26.4s, v0.s[3]
        FMLA v19.4s, v27.4s, v0.s[3]

        TBZ x0, 3, 8f
7:
        # Remainder- 2 floats of A (8 bytes)
        LDP q20, q21, [x5], 32
        LDR d0, [x8], 8
        FMLA v16.4s, v20.4s, v0.s[0]
        FMLA v17.4s, v21.4s, v0.s[0]
        LDP q22, q23, [x5], 32
        FMLA v18.4s, v22.4s, v0.s[1]
        FMLA v19.4s, v23.4s, v0.s[1]
8:
        TBZ x0, 2, 5b
9:
        # Remainder- 1 float of A (4 bytes)
        LDP q20, q21, [x5], 32
        LDR s0, [x8], 4
        FMLA v16.4s, v20.4s, v0.s[0]
        FMLA v17.4s, v21.4s, v0.s[0]
        B 5b

10:
        # Store odd channels
        TBZ x1, 2, 11f
        STR q16, [x6], 16
        MOV v16.16b, v17.16b

11:
        TBZ x1, 1, 12f
        STR d16, [x6], 8
        DUP d16, v16.d[1]

12:
        TBZ x1, 0, 13f
        STR s16, [x6], 4
13:
        RET

END_FUNCTION xnn_f32_igemm_minmax_ukernel_1x8__aarch64_neonfma_cortex_a57

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
