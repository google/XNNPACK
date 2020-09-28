// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_gemm_minmax_ukernel_4x8c4__aarch64_neondot
        stp     x22, x21, [sp, -32]!
        stp     x20, x19, [sp, 16]
        add     x9, x3, x4
        add     x8, x6, x7
        cmp     x0, 2
        sub     x16, x2, 8
        csel    x8, x6, x8, lo
        csel    x19, x3, x9, lo
        ldp     x12, x11, [sp, 32]
        cmp     x0, 3
        lsl     x9, x16, 3
        add     x10, x19, x4
        add     x13, x8, x7
        and     x15, x9, 0xffffffffffffffc0
        csel    x9, x8, x13, lo
        csel    x13, x19, x10, lo
        and     x17, x16, 0xfffffffffffffff8
        cmp     x0, 4
        add     x0, x13, x4
        add     x10, x9, x7
        add     x14, x17, 8
        add     x15, x15, 64
        csel    x10, x10, x9, eq
        csel    x4, x0, x13, eq
        sub     x16, x16, x17
1:
        ldp     q28, q29, [x5], 32
        cmp     x2, 8
        b.cc    3f
        add     x17, x19, x14
        add     x0, x4, x14
        mov     x7, x5
        mov     x20, x3
        mov     x21, x13
        mov     v24.16b, v28.16b
        mov     v25.16b, v29.16b
        mov     v26.16b, v28.16b
        mov     v27.16b, v29.16b
        mov     v31.16b, v28.16b
        mov     v30.16b, v29.16b
        mov     x22, x2
2:
        ldr     d23, [x20], 8
        ldr     d22, [x19], 8
        ldr     d21, [x21], 8
        ldr     d20, [x4], 8
        ldp     q19, q18, [x7]
        ldp     q17, q16, [x7, 32]
        sub     x22, x22, 8
        cmp     x22, 7
        sdot    v28.4s, v19.16b, v23.4b[0]
        sdot    v29.4s, v18.16b, v23.4b[0]
        sdot    v24.4s, v19.16b, v22.4b[0]
        sdot    v25.4s, v18.16b, v22.4b[0]
        sdot    v26.4s, v19.16b, v21.4b[0]
        sdot    v27.4s, v18.16b, v21.4b[0]
        sdot    v31.4s, v19.16b, v20.4b[0]
        sdot    v30.4s, v18.16b, v20.4b[0]
        sdot    v28.4s, v17.16b, v23.4b[1]
        sdot    v29.4s, v16.16b, v23.4b[1]
        sdot    v24.4s, v17.16b, v22.4b[1]
        sdot    v25.4s, v16.16b, v22.4b[1]
        sdot    v26.4s, v17.16b, v21.4b[1]
        sdot    v27.4s, v16.16b, v21.4b[1]
        sdot    v31.4s, v17.16b, v20.4b[1]
        sdot    v30.4s, v16.16b, v20.4b[1]
        add     x7, x7, 64
        b.hi    2b
        add     x3, x3, x14
        add     x13, x13, x14
        add     x5, x5, x15
        mov     x7, x16
        cbz     x7, 4f
        b       5f
3:
        mov     x7, x2
        mov     v30.16b, v29.16b
        mov     v31.16b, v28.16b
        mov     v27.16b, v29.16b
        mov     v26.16b, v28.16b
        mov     v25.16b, v29.16b
        mov     v24.16b, v28.16b
        mov     x0, x4
        mov     x17, x19
        cbnz    x7, 5f
4:
        mov     x4, x11
        ld1r    {v22.4s}, [x4], 4
        cmp     x1, 7
        ld1r    {v23.4s}, [x4]
        add     x4, x11, 8
        sqrdmulh        v21.4s, v28.4s, v22.4s
        sqrdmulh        v19.4s, v24.4s, v22.4s
        cmeq    v1.4s, v23.4s, 0
        mvn     v1.16b, v1.16b
        sqrdmulh        v17.4s, v26.4s, v22.4s
        and     v28.16b, v28.16b, v1.16b
        and     v24.16b, v24.16b, v1.16b
        and     v26.16b, v26.16b, v1.16b
        sqrdmulh        v20.4s, v29.4s, v22.4s
        sqrdmulh        v18.4s, v25.4s, v22.4s
        sqrdmulh        v16.4s, v27.4s, v22.4s
        sqrdmulh        v0.4s, v31.4s, v22.4s
        sqrdmulh        v22.4s, v30.4s, v22.4s
        and     v29.16b, v29.16b, v1.16b
        and     v25.16b, v25.16b, v1.16b
        and     v27.16b, v27.16b, v1.16b
        and     v31.16b, v31.16b, v1.16b
        and     v30.16b, v30.16b, v1.16b
        ld1r    {v1.8h}, [x4]
        ssra    v21.4s, v28.4s, 31
        ssra    v19.4s, v24.4s, 31
        ssra    v17.4s, v26.4s, 31
        ssra    v20.4s, v29.4s, 31
        ssra    v18.4s, v25.4s, 31
        ssra    v16.4s, v27.4s, 31
        ssra    v0.4s, v31.4s, 31
        ssra    v22.4s, v30.4s, 31
        srshl   v28.4s, v21.4s, v23.4s
        srshl   v30.4s, v19.4s, v23.4s
        srshl   v27.4s, v17.4s, v23.4s
        srshl   v29.4s, v20.4s, v23.4s
        srshl   v31.4s, v18.4s, v23.4s
        srshl   v26.4s, v16.4s, v23.4s
        srshl   v25.4s, v0.4s, v23.4s
        sqxtn   v28.4h, v28.4s
        sqxtn   v30.4h, v30.4s
        sqxtn   v27.4h, v27.4s
        srshl   v24.4s, v22.4s, v23.4s
        sqxtn   v25.4h, v25.4s
        sqxtn2  v28.8h, v29.4s
        sqxtn2  v30.8h, v31.4s
        sqxtn2  v27.8h, v26.4s
        sqxtn2  v25.8h, v24.4s
        sqadd   v28.8h, v28.8h, v1.8h
        sqadd   v29.8h, v30.8h, v1.8h
        sqadd   v30.8h, v27.8h, v1.8h
        add     x4, x11, 10
        sqadd   v31.8h, v25.8h, v1.8h
        sqxtn   v28.8b, v28.8h
        sqxtn   v30.8b, v30.8h
        sqxtn2  v28.16b, v29.8h
        sqxtn2  v30.16b, v31.8h
        ld1r    {v29.16b}, [x4]
        add     x4, x11, 11
        ld1r    {v31.16b}, [x4]
        smax    v28.16b, v28.16b, v29.16b
        smax    v30.16b, v30.16b, v29.16b
        smin    v29.16b, v28.16b, v31.16b
        smin    v28.16b, v30.16b, v31.16b
        b.ls    7f
        str     d29, [x6]
        ext     v29.16b, v29.16b, v29.16b, 8
        ext     v30.16b, v28.16b, v28.16b, 8
        sub     x4, x0, x2
        sub     x13, x13, x2
        sub     x19, x17, x2
        sub     x3, x3, x2
        add     x6, x6, x12
        subs    x1, x1, 8
        str     d29, [x8]
        str     d28, [x9]
        str     d30, [x10]
        add     x8, x8, x12
        add     x9, x9, x12
        add     x10, x10, x12
        b.ne    1b
        b       10f
5:
        ldr     d20, [x3]
        ldr     d21, [x17]
        ldr     d22, [x13]
        ldr     d23, [x0]
        ldp     q19, q18, [x5]
        add     x3, x3, x7
        add     x17, x17, x7
        add     x13, x13, x7
        add     x0, x0, x7
        cmp     x7, 5
        sdot    v28.4s, v19.16b, v20.4b[0]
        sdot    v29.4s, v18.16b, v20.4b[0]
        sdot    v24.4s, v19.16b, v21.4b[0]
        sdot    v25.4s, v18.16b, v21.4b[0]
        sdot    v26.4s, v19.16b, v22.4b[0]
        sdot    v27.4s, v18.16b, v22.4b[0]
        sdot    v31.4s, v19.16b, v23.4b[0]
        sdot    v30.4s, v18.16b, v23.4b[0]
        b.cc    6f
        ldp     q19, q18, [x5, 32]
        add     x5, x5, 64
        sdot    v28.4s, v19.16b, v20.4b[1]
        sdot    v29.4s, v18.16b, v20.4b[1]
        sdot    v24.4s, v19.16b, v21.4b[1]
        sdot    v25.4s, v18.16b, v21.4b[1]
        sdot    v26.4s, v19.16b, v22.4b[1]
        sdot    v27.4s, v18.16b, v22.4b[1]
        sdot    v31.4s, v19.16b, v23.4b[1]
        sdot    v30.4s, v18.16b, v23.4b[1]
        b       4b
6:
        add     x5, x5, 32
        b       4b
7:
        tbnz    w1, 2, 11f
        tbnz    w1, 1, 12f
8:
        tbz     w1, 0, 10f
9:
        st1     {v29.b}[0], [x6]
        st1     {v29.b}[8], [x8]
        st1     {v28.b}[0], [x9]
        st1     {v28.b}[8], [x10]
10:
        ldp     x20, x19, [sp, 16]
        ldp     x22, x21, [sp], 32
        ret
11:
        st1     {v29.s}[0], [x6], 4
        st1     {v29.s}[2], [x8], 4
        st1     {v28.s}[0], [x9], 4
        st1     {v28.s}[2], [x10], 4
        ext     v29.16b, v29.16b, v29.16b, 4
        ext     v28.16b, v28.16b, v28.16b, 4
        tbz     w1, 1, 8b
12:
        st1     {v29.h}[0], [x6], 2
        st1     {v29.h}[4], [x8], 2
        st1     {v28.h}[0], [x9], 2
        st1     {v28.h}[4], [x10], 2
        ext     v29.16b, v29.16b, v29.16b, 2
        ext     v28.16b, v28.16b, v28.16b, 2
        tbnz    w1, 0, 9b
        b       10b

END_FUNCTION xnn_qs8_gemm_minmax_ukernel_4x8c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
