// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_igemm_minmax_ukernel_6x8c4__aarch64_neondot
        sub     sp, sp, 0x80
        stp     x28, x27, [sp, 32]
        stp     x26, x25, [sp, 48]
        stp     x24, x23, [sp, 64]
        stp     x22, x21, [sp, 80]
        stp     x20, x19, [sp, 96]
        stp     x29, x30, [sp, 112]
        ldp     x14, x8, [sp, 144]
        cmp     x0, 2
        sub     x13, x2, 8
        lsl     x9, x13, 3
        str     x8, [sp, 24]
        ldp     x8, x15, [sp, 128]
        and     x12, x9, 0xffffffffffffffc0
        mov     x16, x2
        and     x2, x13, 0xfffffffffffffff8
        str     x8, [sp, 8]
        add     x8, x6, x7
        csel    x8, x6, x8, lo
        cmp     x0, 3
        add     x10, x8, x7
        csel    x9, x8, x10, lo
        cmp     x0, 4
        add     x10, x9, x7
        csel    x10, x9, x10, lo
        cmp     x0, 5
        add     x11, x10, x7
        csel    x11, x10, x11, lo
        cmp     x0, 6
        add     x0, x12, 64
        add     x12, x11, x7
        add     x17, x2, 8
        csel    x12, x12, x11, eq
        sub     x7, x13, x2
        str     x3, [sp, 16]
1:
        ldp     q28, q29, [x5], 32
        mov     x19, x3
        mov     v30.16b, v29.16b
        mov     v31.16b, v28.16b
        mov     v27.16b, v29.16b
        mov     v26.16b, v28.16b
        mov     v25.16b, v29.16b
        mov     v24.16b, v28.16b
        mov     v23.16b, v29.16b
        mov     v22.16b, v28.16b
        mov     v21.16b, v29.16b
        mov     v20.16b, v28.16b
2:
        ldp     x13, x2, [x4]
        ldp     x3, x20, [x4, 16]
        ldp     x21, x23, [x4, 32]
        add     x22, x13, x15
        cmp     x13, x14
        csel    x22, x13, x22, eq
        add     x13, x2, x15
        cmp     x2, x14
        csel    x26, x2, x13, eq
        add     x13, x3, x15
        cmp     x3, x14
        csel    x13, x3, x13, eq
        add     x2, x20, x15
        cmp     x20, x14
        csel    x27, x20, x2, eq
        add     x2, x21, x15
        cmp     x21, x14
        csel    x3, x21, x2, eq
        add     x2, x23, x15
        cmp     x23, x14
        csel    x28, x23, x2, eq
        cmp     x16, 8
        b.cc    5f
        add     x25, x26, x17
        add     x24, x27, x17
        add     x23, x28, x17
        mov     x29, x5
        mov     x30, x22
        mov     x21, x13
        mov     x20, x3
        mov     x2, x16
3:
        ldr     d19, [x30], 8
        ldr     d18, [x26], 8
        ldr     d17, [x21], 8
        ldr     d16, [x27], 8
        ldr     d0, [x20], 8
        ldr     d1, [x28], 8
        ldp     q12, q13, [x29]
        ldp     q2, q3, [x29, 32]
        sub     x2, x2, 8
        cmp     x2, 7
        sdot    v20.4s, v12.16b, v19.4b[0]
        sdot    v21.4s, v13.16b, v19.4b[0]
        sdot    v22.4s, v12.16b, v18.4b[0]
        sdot    v23.4s, v13.16b, v18.4b[0]
        sdot    v24.4s, v12.16b, v17.4b[0]
        sdot    v25.4s, v13.16b, v17.4b[0]
        sdot    v26.4s, v12.16b, v16.4b[0]
        sdot    v27.4s, v13.16b, v16.4b[0]
        sdot    v31.4s, v12.16b, v0.4b[0]
        sdot    v30.4s, v13.16b, v0.4b[0]
        sdot    v28.4s, v12.16b, v1.4b[0]
        sdot    v29.4s, v13.16b, v1.4b[0]
        sdot    v20.4s, v2.16b, v19.4b[1]
        sdot    v21.4s, v3.16b, v19.4b[1]
        sdot    v22.4s, v2.16b, v18.4b[1]
        sdot    v23.4s, v3.16b, v18.4b[1]
        sdot    v24.4s, v2.16b, v17.4b[1]
        sdot    v25.4s, v3.16b, v17.4b[1]
        sdot    v26.4s, v2.16b, v16.4b[1]
        sdot    v27.4s, v3.16b, v16.4b[1]
        sdot    v31.4s, v2.16b, v0.4b[1]
        sdot    v30.4s, v3.16b, v0.4b[1]
        sdot    v28.4s, v2.16b, v1.4b[1]
        sdot    v29.4s, v3.16b, v1.4b[1]
        add     x29, x29, 64
        b.hi    3b
        add     x22, x22, x17
        add     x13, x13, x17
        add     x3, x3, x17
        add     x5, x5, x0
        mov     x2, x7
        add     x4, x4, 0x30
        cbnz    x2, 6f
4:
        subs    x19, x19, 0x30
        b.ne    2b
        b       8f
5:
        mov     x2, x16
        mov     x23, x28
        mov     x24, x27
        mov     x25, x26
        add     x4, x4, 0x30
        cbz     x2, 4b
6:
        ldr     d1, [x22]
        ldr     d0, [x25]
        ldr     d16, [x13]
        ldr     d17, [x24]
        ldr     d19, [x3]
        ldp     q2, q3, [x5]
        ldr     d18, [x23]
        cmp     x2, 5
        sdot    v20.4s, v2.16b, v1.4b[0]
        sdot    v21.4s, v3.16b, v1.4b[0]
        sdot    v22.4s, v2.16b, v0.4b[0]
        sdot    v23.4s, v3.16b, v0.4b[0]
        sdot    v24.4s, v2.16b, v16.4b[0]
        sdot    v25.4s, v3.16b, v16.4b[0]
        sdot    v26.4s, v2.16b, v17.4b[0]
        sdot    v27.4s, v3.16b, v17.4b[0]
        sdot    v31.4s, v2.16b, v19.4b[0]
        sdot    v30.4s, v3.16b, v19.4b[0]
        sdot    v28.4s, v2.16b, v18.4b[0]
        sdot    v29.4s, v3.16b, v18.4b[0]
        b.cc    7f
        ldp     q2, q3, [x5, 32]
        add     x5, x5, 64
        sdot    v20.4s, v2.16b, v1.4b[1]
        sdot    v21.4s, v3.16b, v1.4b[1]
        sdot    v22.4s, v2.16b, v0.4b[1]
        sdot    v23.4s, v3.16b, v0.4b[1]
        sdot    v24.4s, v2.16b, v16.4b[1]
        sdot    v25.4s, v3.16b, v16.4b[1]
        sdot    v26.4s, v2.16b, v17.4b[1]
        sdot    v27.4s, v3.16b, v17.4b[1]
        sdot    v31.4s, v2.16b, v19.4b[1]
        sdot    v30.4s, v3.16b, v19.4b[1]
        sdot    v28.4s, v2.16b, v18.4b[1]
        sdot    v29.4s, v3.16b, v18.4b[1]
        subs    x19, x19, 0x30
        b.ne    2b
        b       8f
7:
        add     x5, x5, 32
        subs    x19, x19, 0x30
        b.ne    2b
8:
        ldr     x2, [sp, 24]
        cmp     x1, 7
        mov     x13, x2
        ld1r    {v18.4s}, [x13], 4
        ld1r    {v19.4s}, [x13]
        sqrdmulh        v20.4s, v20.4s, v18.4s
        sqrdmulh        v21.4s, v21.4s, v18.4s
        sqrdmulh        v22.4s, v22.4s, v18.4s
        sqrdmulh        v23.4s, v23.4s, v18.4s
        sqrdmulh        v24.4s, v24.4s, v18.4s
        sqrdmulh        v25.4s, v25.4s, v18.4s
        sqrdmulh        v26.4s, v26.4s, v18.4s
        sqrdmulh        v27.4s, v27.4s, v18.4s
        sqrdmulh        v31.4s, v31.4s, v18.4s
        sqrdmulh        v30.4s, v30.4s, v18.4s
        sqrdmulh        v28.4s, v28.4s, v18.4s
        sqrdmulh        v29.4s, v29.4s, v18.4s
        cmeq    v18.4s, v19.4s, 0
        mvn     v18.16b, v18.16b
        and     v17.16b, v20.16b, v18.16b
        ssra    v20.4s, v17.4s, 31
        and     v17.16b, v21.16b, v18.16b
        ssra    v21.4s, v17.4s, 31
        and     v17.16b, v22.16b, v18.16b
        ssra    v22.4s, v17.4s, 31
        and     v17.16b, v23.16b, v18.16b
        ssra    v23.4s, v17.4s, 31
        and     v17.16b, v24.16b, v18.16b
        ssra    v24.4s, v17.4s, 31
        and     v17.16b, v25.16b, v18.16b
        ssra    v25.4s, v17.4s, 31
        and     v17.16b, v26.16b, v18.16b
        ssra    v26.4s, v17.4s, 31
        and     v17.16b, v27.16b, v18.16b
        ssra    v27.4s, v17.4s, 31
        and     v17.16b, v31.16b, v18.16b
        ssra    v31.4s, v17.4s, 31
        and     v17.16b, v30.16b, v18.16b
        add     x13, x2, 8
        ssra    v30.4s, v17.4s, 31
        and     v17.16b, v28.16b, v18.16b
        ssra    v28.4s, v17.4s, 31
        ld1r    {v17.8h}, [x13]
        and     v18.16b, v29.16b, v18.16b
        srshl   v20.4s, v20.4s, v19.4s
        srshl   v22.4s, v22.4s, v19.4s
        srshl   v24.4s, v24.4s, v19.4s
        srshl   v26.4s, v26.4s, v19.4s
        srshl   v31.4s, v31.4s, v19.4s
        ssra    v29.4s, v18.4s, 31
        srshl   v21.4s, v21.4s, v19.4s
        srshl   v23.4s, v23.4s, v19.4s
        srshl   v25.4s, v25.4s, v19.4s
        srshl   v27.4s, v27.4s, v19.4s
        srshl   v30.4s, v30.4s, v19.4s
        srshl   v28.4s, v28.4s, v19.4s
        sqxtn   v20.4h, v20.4s
        sqxtn   v22.4h, v22.4s
        sqxtn   v24.4h, v24.4s
        sqxtn   v26.4h, v26.4s
        sqxtn   v31.4h, v31.4s
        srshl   v29.4s, v29.4s, v19.4s
        sqxtn   v28.4h, v28.4s
        sqxtn2  v20.8h, v21.4s
        sqxtn2  v22.8h, v23.4s
        sqxtn2  v24.8h, v25.4s
        sqxtn2  v26.8h, v27.4s
        sqxtn2  v31.8h, v30.4s
        sqxtn2  v28.8h, v29.4s
        sqadd   v29.8h, v20.8h, v17.8h
        sqadd   v27.8h, v24.8h, v17.8h
        sqadd   v31.8h, v31.8h, v17.8h
        add     x13, x2, 10
        sqadd   v30.8h, v22.8h, v17.8h
        sqadd   v26.8h, v26.8h, v17.8h
        sqadd   v28.8h, v28.8h, v17.8h
        sqxtn   v29.8b, v29.8h
        sqxtn   v27.8b, v27.8h
        sqxtn   v31.8b, v31.8h
        sqxtn2  v29.16b, v30.8h
        sqxtn2  v27.16b, v26.8h
        sqxtn2  v31.16b, v28.8h
        ld1r    {v28.16b}, [x13]
        add     x13, x2, 11
        ld1r    {v26.16b}, [x13]
        smax    v30.16b, v31.16b, v28.16b
        smax    v31.16b, v27.16b, v28.16b
        smax    v28.16b, v29.16b, v28.16b
        smin    v30.16b, v30.16b, v26.16b
        smin    v29.16b, v31.16b, v26.16b
        smin    v28.16b, v28.16b, v26.16b
        b.ls    9f
        ext     v31.16b, v30.16b, v30.16b, 8
        str     d31, [x12]
        ext     v31.16b, v29.16b, v29.16b, 8
        str     d30, [x11]
        ext     v30.16b, v28.16b, v28.16b, 8
        str     d31, [x10]
        str     d29, [x9]
        str     d30, [x8]
        str     d28, [x6]
        ldp     x13, x3, [sp, 8]
        subs    x1, x1, 8
        add     x12, x12, x13
        add     x11, x11, x13
        add     x10, x10, x13
        add     x9, x9, x13
        add     x8, x8, x13
        add     x6, x6, x13
        sub     x4, x4, x3
        b.ne    1b
        b       12f
9:
        tbnz    w1, 2, 13f
        tbnz    w1, 1, 14f
10:
        tbz     w1, 0, 12f
11:
        st1     {v30.b}[8], [x12]
        st1     {v30.b}[0], [x11]
        st1     {v29.b}[8], [x10]
        st1     {v29.b}[0], [x9]
        st1     {v28.b}[8], [x8]
        st1     {v28.b}[0], [x6]
12:
        ldp     x29, x30, [sp, 112]
        ldp     x20, x19, [sp, 96]
        ldp     x22, x21, [sp, 80]
        ldp     x24, x23, [sp, 64]
        ldp     x26, x25, [sp, 48]
        ldp     x28, x27, [sp, 32]
        add     sp, sp, 0x80
        ret
13:
        st1     {v30.s}[2], [x12], 4
        st1     {v30.s}[0], [x11], 4
        st1     {v29.s}[2], [x10], 4
        st1     {v29.s}[0], [x9], 4
        st1     {v28.s}[2], [x8], 4
        st1     {v28.s}[0], [x6], 4
        ext     v30.16b, v30.16b, v30.16b, 4
        ext     v29.16b, v29.16b, v29.16b, 4
        ext     v28.16b, v28.16b, v28.16b, 4
        tbz     w1, 1, 10b
14:
        st1     {v30.h}[4], [x12], 2
        st1     {v30.h}[0], [x11], 2
        st1     {v29.h}[4], [x10], 2
        st1     {v29.h}[0], [x9], 2
        st1     {v28.h}[4], [x8], 2
        st1     {v28.h}[0], [x6], 2
        ext     v30.16b, v30.16b, v30.16b, 2
        ext     v29.16b, v29.16b, v29.16b, 2
        ext     v28.16b, v28.16b, v28.16b, 2
        tbnz    w1, 0, 11b
        b       12b

END_FUNCTION xnn_qs8_igemm_minmax_ukernel_6x8c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
