// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_igemm_minmax_ukernel_1x8c4__aarch64_neondot
        ldp     x9, x8, [sp, 16]
        ldp     x11, x10, [sp]
        sub     x14, x2, 8
        lsl     x12, x14, 3
        and     x15, x14, 0xfffffffffffffff8
        and     x13, x12, 0xffffffffffffffc0
        add     x12, x15, 8
        add     x13, x13, 64
        sub     x14, x14, x15
1:
        ldp     q28, q29, [x5], 32
        mov     x15, x3
2:
        ldr     x16, [x4], 8
        add     x17, x16, x10
        cmp     x16, x9
        csel    x17, x16, x17, eq
        cmp     x2, 8
        b.cc    5f
        add     x16, x17, x12
        mov     x0, x5
        mov     x7, x2
3:
        ldr     d30, [x17], 8
        ldp     q31, q27, [x0]
        ldp     q26, q25, [x0, 32]
        sub     x7, x7, 8
        cmp     x7, 7
        sdot    v28.4s, v31.16b, v30.4b[0]
        sdot    v29.4s, v27.16b, v30.4b[0]
        sdot    v28.4s, v26.16b, v30.4b[1]
        sdot    v29.4s, v25.16b, v30.4b[1]
        add     x0, x0, 64
        b.hi    3b
        add     x5, x5, x13
        mov     x0, x14
        cbnz    x0, 6f
4:
        subs    x15, x15, 8
        b.ne    2b
        b       8f
5:
        mov     x0, x2
        mov     x16, x17
        cbz     x0, 4b
6:
        ldr     d30, [x16]
        ldp     q31, q27, [x5]
        cmp     x0, 5
        sdot    v28.4s, v31.16b, v30.4b[0]
        sdot    v29.4s, v27.16b, v30.4b[0]
        b.cc    7f
        ldp     q31, q27, [x5, 32]
        add     x5, x5, 64
        sdot    v28.4s, v31.16b, v30.4b[1]
        sdot    v29.4s, v27.16b, v30.4b[1]
        subs    x15, x15, 8
        b.ne    2b
        b       8f
7:
        add     x5, x5, 32
        subs    x15, x15, 8
        b.ne    2b
8:
        mov     x15, x8
        ld1r    {v30.4s}, [x15], 4
        cmp     x1, 7
        ld1r    {v31.4s}, [x15]
        sqrdmulh        v28.4s, v28.4s, v30.4s
        sqrdmulh        v29.4s, v29.4s, v30.4s
        add     x15, x8, 8
        cmeq    v30.4s, v31.4s, 0
        mvn     v30.16b, v30.16b
        and     v27.16b, v28.16b, v30.16b
        ssra    v28.4s, v27.4s, 31
        and     v30.16b, v29.16b, v30.16b
        ld1r    {v27.8h}, [x15]
        ssra    v29.4s, v30.4s, 31
        srshl   v28.4s, v28.4s, v31.4s
        add     x15, x8, 10
        srshl   v29.4s, v29.4s, v31.4s
        sqxtn   v28.4h, v28.4s
        sqxtn2  v28.8h, v29.4s
        ld1r    {v29.8b}, [x15]
        add     x15, x8, 11
        ld1r    {v30.8b}, [x15]
        sqadd   v28.8h, v28.8h, v27.8h
        sqxtn   v28.8b, v28.8h
        smax    v28.8b, v28.8b, v29.8b
        smin    v28.8b, v28.8b, v30.8b
        b.ls    9f
        str     d28, [x6]
        add     x6, x6, x11
        subs    x1, x1, 8
        sub     x4, x4, x3
        b.ne    1b
        b       11f
9:
        tbnz    w1, 2, 12f
        tbnz    w1, 1, 13f
10:
        tbnz    w1, 0, 14f
11:
        ret
12:
        st1     {v28.s}[0], [x6], 4
        ext     v28.8b, v28.8b, v28.8b, 4
        tbz     w1, 1, 10b
13:
        st1     {v28.h}[0], [x6], 2
        ext     v28.8b, v28.8b, v28.8b, 2
        tbz     w1, 0, 11b
14:
        st1     {v28.b}[0], [x6]
        ret

END_FUNCTION xnn_qs8_igemm_minmax_ukernel_1x8c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif

