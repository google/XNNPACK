// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_igemm_minmax_ukernel_1x16c4__aarch64_neondot
        ldp     x9, x8, [sp, 16]
        ldp     x11, x10, [sp]
        sub     x14, x2, 8
        lsl     x12, x14, 4
        and     x15, x14, 0xfffffffffffffff8
        and     x13, x12, 0xffffffffffffff80
        add     x12, x15, 8
        add     x13, x13, 0x80
        sub     x14, x14, x15
1:
        ldp     q28, q29, [x5]
        ldp     q30, q31, [x5, 32]
        add     x5, x5, 64
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
        ldr     d27, [x17], 8
        ldp     q26, q25, [x0]
        ldp     q24, q23, [x0, 32]
        ldp     q22, q21, [x0, 64]
        ldp     q20, q19, [x0, 96]
        sub     x7, x7, 8
        sdot    v28.4s, v26.16b, v27.4b[0]
        sdot    v29.4s, v25.16b, v27.4b[0]
        sdot    v30.4s, v24.16b, v27.4b[0]
        sdot    v31.4s, v23.16b, v27.4b[0]
        cmp     x7, 7
        sdot    v28.4s, v22.16b, v27.4b[1]
        sdot    v29.4s, v21.16b, v27.4b[1]
        sdot    v30.4s, v20.16b, v27.4b[1]
        sdot    v31.4s, v19.16b, v27.4b[1]
        add     x0, x0, 0x80
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
        ldr     d27, [x16]
        ldp     q26, q25, [x5]
        ldp     q24, q23, [x5, 32]
        cmp     x0, 5
        sdot    v28.4s, v26.16b, v27.4b[0]
        sdot    v29.4s, v25.16b, v27.4b[0]
        sdot    v30.4s, v24.16b, v27.4b[0]
        sdot    v31.4s, v23.16b, v27.4b[0]
        b.cc    7f
        ldp     q26, q25, [x5, 64]
        ldp     q24, q23, [x5, 96]
        add     x5, x5, 0x80
        sdot    v28.4s, v26.16b, v27.4b[1]
        sdot    v29.4s, v25.16b, v27.4b[1]
        sdot    v30.4s, v24.16b, v27.4b[1]
        sdot    v31.4s, v23.16b, v27.4b[1]
        subs    x15, x15, 8
        b.ne    2b
        b       8f
7:
        add     x5, x5, 64
        subs    x15, x15, 8
        b.ne    2b
8:
        mov     x15, x8
        ld1r    {v27.4s}, [x15], 4
        cmp     x1, 0xf
        ld1r    {v26.4s}, [x15]
        sqrdmulh        v28.4s, v28.4s, v27.4s
        sqrdmulh        v29.4s, v29.4s, v27.4s
        sqrdmulh        v30.4s, v30.4s, v27.4s
        sqrdmulh        v31.4s, v31.4s, v27.4s
        cmeq    v27.4s, v26.4s, 0
        mvn     v27.16b, v27.16b
        and     v25.16b, v28.16b, v27.16b
        ssra    v28.4s, v25.4s, 31
        and     v25.16b, v29.16b, v27.16b
        add     x15, x8, 8
        ssra    v29.4s, v25.4s, 31
        and     v25.16b, v30.16b, v27.16b
        ssra    v30.4s, v25.4s, 31
        ld1r    {v25.8h}, [x15]
        and     v27.16b, v31.16b, v27.16b
        srshl   v28.4s, v28.4s, v26.4s
        ssra    v31.4s, v27.4s, 31
        srshl   v29.4s, v29.4s, v26.4s
        srshl   v30.4s, v30.4s, v26.4s
        sqxtn   v28.4h, v28.4s
        srshl   v31.4s, v31.4s, v26.4s
        sqxtn   v30.4h, v30.4s
        sqxtn2  v28.8h, v29.4s
        sqxtn2  v30.8h, v31.4s
        sqadd   v28.8h, v28.8h, v25.8h
        add     x15, x8, 10
        sqadd   v29.8h, v30.8h, v25.8h
        sqxtn   v28.8b, v28.8h
        sqxtn2  v28.16b, v29.8h
        ld1r    {v29.16b}, [x15]
        add     x15, x8, 11
        ld1r    {v30.16b}, [x15]
        smax    v28.16b, v28.16b, v29.16b
        smin    v28.16b, v28.16b, v30.16b
        b.ls    9f
        str     q28, [x6]
        add     x6, x6, x11
        subs    x1, x1, 0x10
        sub     x4, x4, x3
        b.ne    1b
        b       12f
9:
        tbnz    w1, 3, 13f
        tbnz    w1, 2, 14f
10:
        tbnz    w1, 1, 15f
11:
        tbnz    w1, 0, 16f
12:
        ret
13:
        str     d28, [x6], 8
        ext     v28.16b, v28.16b, v28.16b, 8
        tbz     w1, 2, 10b
14:
        st1     {v28.s}[0], [x6], 4
        ext     v28.8b, v28.8b, v28.8b, 4
        tbz     w1, 1, 11b
15:
        st1     {v28.h}[0], [x6], 2
        ext     v28.8b, v28.8b, v28.8b, 2
        tbz     w1, 0, 12b
16:
        st1     {v28.b}[0], [x6]
        ret

END_FUNCTION xnn_qs8_igemm_minmax_ukernel_1x16c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
