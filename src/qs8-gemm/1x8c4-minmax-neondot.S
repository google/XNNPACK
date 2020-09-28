// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_gemm_minmax_ukernel_1x8c4__aarch64_neondot
        ldp     x9, x8, [sp]
        sub     x12, x2, 8
        lsl     x10, x12, 3
        and     x13, x12, 0xfffffffffffffff8
        and     x11, x10, 0xffffffffffffffc0
        add     x10, x13, 8
        add     x11, x11, 64
        sub     x12, x12, x13
1:
        ldp     q28, q29, [x5], 32
        cmp     x2, 8
        b.cc    3f
        add     x13, x3, x10
        mov     x14, x5
        mov     x15, x2
2:
        ldr     d30, [x3], 8
        ldp     q31, q27, [x14]
        ldp     q26, q25, [x14, 32]
        sub     x15, x15, 8
        cmp     x15, 7
        sdot    v28.4s, v31.16b, v30.4b[0]
        sdot    v29.4s, v27.16b, v30.4b[0]
        sdot    v28.4s, v26.16b, v30.4b[1]
        sdot    v29.4s, v25.16b, v30.4b[1]
        add     x14, x14, 64
        b.hi    2b
        add     x5, x5, x11
        mov     x14, x12
        cbz     x14, 4f
        b       5f
3:
        mov     x14, x2
        mov     x13, x3
        cbnz    x14, 5f
4:
        mov     x14, x8
        ld1r    {v30.4s}, [x14], 4
        cmp     x1, 7
        ld1r    {v31.4s}, [x14]
        sqrdmulh        v27.4s, v28.4s, v30.4s
        add     x14, x8, 8
        sqrdmulh        v30.4s, v29.4s, v30.4s
        cmeq    v26.4s, v31.4s, 0
        mvn     v26.16b, v26.16b
        and     v28.16b, v28.16b, v26.16b
        and     v29.16b, v29.16b, v26.16b
        ssra    v27.4s, v28.4s, 31
        ld1r    {v26.8h}, [x14]
        ssra    v30.4s, v29.4s, 31
        srshl   v28.4s, v27.4s, v31.4s
        add     x14, x8, 10
        srshl   v29.4s, v30.4s, v31.4s
        sqxtn   v28.4h, v28.4s
        sqxtn2  v28.8h, v29.4s
        ld1r    {v29.8b}, [x14]
        add     x14, x8, 11
        ld1r    {v30.8b}, [x14]
        sqadd   v28.8h, v28.8h, v26.8h
        sqxtn   v28.8b, v28.8h
        smax    v28.8b, v28.8b, v29.8b
        smin    v28.8b, v28.8b, v30.8b
        b.ls    7f
        sub     x3, x13, x2
        str     d28, [x6]
        subs    x1, x1, 8
        add     x6, x6, x9
        b.ne    1b
        b       9f
5:
        ldr     d30, [x13]
        ldp     q31, q27, [x5]
        add     x13, x13, x14
        cmp     x14, 5
        sdot    v28.4s, v31.16b, v30.4b[0]
        sdot    v29.4s, v27.16b, v30.4b[0]
        b.cc    6f
        ldp     q31, q27, [x5, 32]
        add     x5, x5, 64
        sdot    v28.4s, v31.16b, v30.4b[1]
        sdot    v29.4s, v27.16b, v30.4b[1]
        b       4b
6:
        add     x5, x5, 32
        b       4b
7:
        tbnz    w1, 2, 10f
        tbnz    w1, 1, 11f
8:
        tbnz    w1, 0, 12f
9:
        ret
10:
        st1     {v28.s}[0], [x6], 4
        ext     v28.8b, v28.8b, v28.8b, 4
        tbz     w1, 1, 8b
11:
        st1     {v28.h}[0], [x6], 2
        ext     v28.8b, v28.8b, v28.8b, 2
        tbz     w1, 0, 9b
12:
        st1     {v28.b}[0], [x6]
        ret

END_FUNCTION xnn_qs8_gemm_minmax_ukernel_1x8c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
