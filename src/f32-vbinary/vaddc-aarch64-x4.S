// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

# void xnn_f32_vaddc_ukernel__neon_x4(
#    size_t n,       x0
#    const float* a, x1
#    const float* b, x2
#    float* y,       x3

# d8-d15, x19-x30 need to be preserved if used. x18 is reserved by the OS.

# A pointer
# x1  a0 (v3)

# B pointer
# x2  b0 (v6)

# C pointer
# x3  c0

# Register usage:
# A0/C0  v1(q1)
# B      v6(q6)

BEGIN_FUNCTION xnn_f32_vaddc_ukernel__aarch64_x4
    # Load constant b.
    LD1R    { v6.4s }, [x2]

    // Not enough floats to enter prologue.
    SUBS    x0, x0, 16  // n -= 16
    B.LO    3f          // Go to remainder.

    // Prologue and epilogue handles 4 floats (16 bytes).
    LDR     q1, [x1], #16
    FADD    v1.4s, v6.4s, v1.4s

    // Not enough bytes for main loop, go to epilogue.
    SUBS    x0, x0, 16
    B.LO    2f

1:
    // Main loop. Handle 4 floats (16 bytes).
    STR    q1, [x3], #16
    LDR    q1, [x1], #16
    FADD   v1.4s, v6.4s, v1.4s
    SUBS   x0, x0, 16
    B.HS   1b

2:
    // Epilogue
    STR    q1, [x3], #16

3:
    //  Remainder code (< 16 bytes).
    ADDS   w0, w0, #16  // Make it positive again so the bit tests below work.
    B.EQ   7f           // No remainder, return.

5:
    // Remainder 8 bytes (2 floats).
    TBZ     w0, 3, 6f
    LDR     d1, [x1], #8
    FADD    v1.4s, v6.4s, v1.4s
    TBZ     w0, 3, 6f
    STR     d1, [x3], 8
6:
    // Remainder 4 bytes (1 float).
    TBZ     w0, 2, 7f
    LDR     s1, [x1]
    FADD    v1.4s, v6.4s, v1.4s
    STR     s1, [x3]
    RET
7:
    RET
