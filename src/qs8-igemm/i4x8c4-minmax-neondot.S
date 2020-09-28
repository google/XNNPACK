// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_igemm_minmax_ukernel_4x8c4__aarch64_neondot
        str     x27, [sp, -80]!
        stp     x26, x25, [sp, 16]
        stp     x24, x23, [sp, 32]
        stp     x22, x21, [sp, 48]
        stp     x20, x19, [sp, 64]
        add     x8, x6, x7
        cmp     x0, 2
        sub     x17, x2, 8
        csel    x8, x6, x8, lo
        ldp     x12, x11, [sp, 96]
        ldp     x14, x13, [sp, 80]
        cmp     x0, 3
        lsl     x9, x17, 3
        add     x10, x8, x7
        and     x16, x9, 0xffffffffffffffc0
        csel    x9, x8, x10, lo
        and     x19, x17, 0xfffffffffffffff8
        cmp     x0, 4
        add     x10, x9, x7
        add     x15, x19, 8
        add     x16, x16, 64
        csel    x10, x10, x9, eq
        sub     x17, x17, x19
1:
        ldp     q28, q29, [x5], 32
        mov     x0, x3
        mov     v30.16b, v29.16b
        mov     v31.16b, v28.16b
        mov     v27.16b, v29.16b
        mov     v26.16b, v28.16b
        mov     v25.16b, v29.16b
        mov     v24.16b, v28.16b
2:
        ldp     x7, x20, [x4]
        ldp     x21, x23, [x4, 16]
        add     x19, x7, x13
        cmp     x7, x12
        csel    x19, x7, x19, eq
        add     x7, x20, x13
        cmp     x20, x12
        csel    x22, x20, x7, eq
        add     x7, x21, x13
        cmp     x21, x12
        csel    x7, x21, x7, eq
        add     x20, x23, x13
        cmp     x23, x12
        csel    x23, x23, x20, eq
        cmp     x2, 8
        b.cc    5f
        add     x21, x22, x15
        add     x20, x23, x15
        mov     x24, x5
        mov     x25, x19
        mov     x26, x7
        mov     x27, x2
3:
        ldr     d23, [x25], 8
        ldr     d22, [x22], 8
        ldr     d21, [x26], 8
        ldr     d20, [x23], 8
        ldp     q19, q18, [x24]
        ldp     q17, q16, [x24, 32]
        sub     x27, x27, 8
        cmp     x27, 7
        sdot    v24.4s, v19.16b, v23.4b[0]
        sdot    v25.4s, v18.16b, v23.4b[0]
        sdot    v26.4s, v19.16b, v22.4b[0]
        sdot    v27.4s, v18.16b, v22.4b[0]
        sdot    v31.4s, v19.16b, v21.4b[0]
        sdot    v30.4s, v18.16b, v21.4b[0]
        sdot    v28.4s, v19.16b, v20.4b[0]
        sdot    v29.4s, v18.16b, v20.4b[0]
        sdot    v24.4s, v17.16b, v23.4b[1]
        sdot    v25.4s, v16.16b, v23.4b[1]
        sdot    v26.4s, v17.16b, v22.4b[1]
        sdot    v27.4s, v16.16b, v22.4b[1]
        sdot    v31.4s, v17.16b, v21.4b[1]
        sdot    v30.4s, v16.16b, v21.4b[1]
        sdot    v28.4s, v17.16b, v20.4b[1]
        sdot    v29.4s, v16.16b, v20.4b[1]
        add     x24, x24, 64
        b.hi    3b
        add     x19, x19, x15
        add     x7, x7, x15
        add     x5, x5, x16
        mov     x24, x17
        add     x4, x4, 32
        cbnz    x24, 6f
4:
        subs    x0, x0, 32
        b.ne    2b
        b       8f
5:
        mov     x24, x2
        mov     x20, x23
        mov     x21, x22
        add     x4, x4, 32
        cbz     x24, 4b
6:
        ldr     d20, [x19]
        ldr     d21, [x21]
        ldr     d23, [x7]
        ldp     q19, q18, [x5]
        ldr     d22, [x20]
        cmp     x24, 5
        sdot    v24.4s, v19.16b, v20.4b[0]
        sdot    v25.4s, v18.16b, v20.4b[0]
        sdot    v26.4s, v19.16b, v21.4b[0]
        sdot    v27.4s, v18.16b, v21.4b[0]
        sdot    v31.4s, v19.16b, v23.4b[0]
        sdot    v30.4s, v18.16b, v23.4b[0]
        sdot    v28.4s, v19.16b, v22.4b[0]
        sdot    v29.4s, v18.16b, v22.4b[0]
        b.cc    7f
        ldp     q19, q18, [x5, 32]
        add     x5, x5, 64
        sdot    v24.4s, v19.16b, v20.4b[1]
        sdot    v25.4s, v18.16b, v20.4b[1]
        sdot    v26.4s, v19.16b, v21.4b[1]
        sdot    v27.4s, v18.16b, v21.4b[1]
        sdot    v31.4s, v19.16b, v23.4b[1]
        sdot    v30.4s, v18.16b, v23.4b[1]
        sdot    v28.4s, v19.16b, v22.4b[1]
        sdot    v29.4s, v18.16b, v22.4b[1]
        subs    x0, x0, 32
        b.ne    2b
        b       8f
7:
        add     x5, x5, 32
        subs    x0, x0, 32
        b.ne    2b
8:
        mov     x0, x11
        ld1r    {v22.4s}, [x0], 4
        cmp     x1, 7
        ld1r    {v23.4s}, [x0]
        sqrdmulh        v24.4s, v24.4s, v22.4s
        sqrdmulh        v25.4s, v25.4s, v22.4s
        sqrdmulh        v26.4s, v26.4s, v22.4s
        sqrdmulh        v27.4s, v27.4s, v22.4s
        sqrdmulh        v31.4s, v31.4s, v22.4s
        sqrdmulh        v30.4s, v30.4s, v22.4s
        sqrdmulh        v28.4s, v28.4s, v22.4s
        sqrdmulh        v29.4s, v29.4s, v22.4s
        cmeq    v22.4s, v23.4s, 0
        mvn     v22.16b, v22.16b
        and     v21.16b, v24.16b, v22.16b
        ssra    v24.4s, v21.4s, 31
        and     v21.16b, v25.16b, v22.16b
        ssra    v25.4s, v21.4s, 31
        and     v21.16b, v26.16b, v22.16b
        ssra    v26.4s, v21.4s, 31
        and     v21.16b, v27.16b, v22.16b
        ssra    v27.4s, v21.4s, 31
        and     v21.16b, v31.16b, v22.16b
        ssra    v31.4s, v21.4s, 31
        and     v21.16b, v30.16b, v22.16b
        add     x0, x11, 8
        ssra    v30.4s, v21.4s, 31
        and     v21.16b, v28.16b, v22.16b
        ssra    v28.4s, v21.4s, 31
        ld1r    {v21.8h}, [x0]
        and     v22.16b, v29.16b, v22.16b
        srshl   v24.4s, v24.4s, v23.4s
        srshl   v26.4s, v26.4s, v23.4s
        srshl   v31.4s, v31.4s, v23.4s
        ssra    v29.4s, v22.4s, 31
        srshl   v25.4s, v25.4s, v23.4s
        srshl   v27.4s, v27.4s, v23.4s
        srshl   v30.4s, v30.4s, v23.4s
        srshl   v28.4s, v28.4s, v23.4s
        sqxtn   v24.4h, v24.4s
        sqxtn   v26.4h, v26.4s
        sqxtn   v31.4h, v31.4s
        srshl   v29.4s, v29.4s, v23.4s
        sqxtn   v28.4h, v28.4s
        sqxtn2  v24.8h, v25.4s
        sqxtn2  v26.8h, v27.4s
        sqxtn2  v31.8h, v30.4s
        sqxtn2  v28.8h, v29.4s
        sqadd   v29.8h, v24.8h, v21.8h
        sqadd   v31.8h, v31.8h, v21.8h
        add     x0, x11, 10
        sqadd   v30.8h, v26.8h, v21.8h
        sqadd   v28.8h, v28.8h, v21.8h
        sqxtn   v29.8b, v29.8h
        sqxtn   v31.8b, v31.8h
        sqxtn2  v29.16b, v30.8h
        sqxtn2  v31.16b, v28.8h
        ld1r    {v28.16b}, [x0]
        add     x0, x11, 11
        ld1r    {v30.16b}, [x0]
        smax    v31.16b, v31.16b, v28.16b
        smax    v28.16b, v29.16b, v28.16b
        smin    v29.16b, v31.16b, v30.16b
        smin    v28.16b, v28.16b, v30.16b
        b.ls    9f
        ext     v30.16b, v29.16b, v29.16b, 8
        ext     v31.16b, v28.16b, v28.16b, 8
        subs    x1, x1, 8
        str     d30, [x10]
        str     d29, [x9]
        str     d31, [x8]
        str     d28, [x6]
        add     x10, x10, x14
        add     x9, x9, x14
        add     x8, x8, x14
        add     x6, x6, x14
        sub     x4, x4, x3
        b.ne    1b
        b       12f
9:
        tbnz    w1, 2, 13f
        tbnz    w1, 1, 14f
10:
        tbz     w1, 0, 12f
11:
        st1     {v29.b}[8], [x10]
        st1     {v29.b}[0], [x9]
        st1     {v28.b}[8], [x8]
        st1     {v28.b}[0], [x6]
12:
        ldp     x20, x19, [sp, 64]
        ldp     x22, x21, [sp, 48]
        ldp     x24, x23, [sp, 32]
        ldp     x26, x25, [sp, 16]
        ldr     x27, [sp], 80
        ret
13:
        st1     {v29.s}[2], [x10], 4
        st1     {v29.s}[0], [x9], 4
        st1     {v28.s}[2], [x8], 4
        st1     {v28.s}[0], [x6], 4
        ext     v29.16b, v29.16b, v29.16b, 4
        ext     v28.16b, v28.16b, v28.16b, 4
        tbz     w1, 1, 10b
14:
        st1     {v29.h}[4], [x10], 2
        st1     {v29.h}[0], [x9], 2
        st1     {v28.h}[4], [x8], 2
        st1     {v28.h}[0], [x6], 2
        ext     v29.16b, v29.16b, v29.16b, 2
        ext     v28.16b, v28.16b, v28.16b, 2
        tbnz    w1, 0, 11b
        b       12b

END_FUNCTION xnn_qs8_igemm_minmax_ukernel_4x8c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
