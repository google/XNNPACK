// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_gemm_minmax_ukernel_6x8c4__aarch64_neondot
        stp     d9, d8, [sp, -96]!
        stp     x28, x27, [sp, 16]
        stp     x26, x25, [sp, 32]
        stp     x24, x23, [sp, 48]
        stp     x22, x21, [sp, 64]
        stp     x20, x19, [sp, 80]
        add     x9, x3, x4
        add     x8, x6, x7
        cmp     x0, 2
        sub     x19, x2, 8
        csel    x8, x6, x8, lo
        csel    x21, x3, x9, lo
        cmp     x0, 3
        lsl     x9, x19, 3
        add     x10, x21, x4
        add     x11, x8, x7
        and     x12, x9, 0xffffffffffffffc0
        csel    x9, x8, x11, lo
        csel    x15, x21, x10, lo
        cmp     x0, 4
        add     x11, x15, x4
        add     x10, x9, x7
        csel    x10, x9, x10, lo
        csel    x22, x15, x11, lo
        ldp     x14, x13, [sp, 96]
        cmp     x0, 5
        add     x16, x22, x4
        add     x11, x10, x7
        csel    x11, x10, x11, lo
        csel    x16, x22, x16, lo
        and     x20, x19, 0xfffffffffffffff8
        cmp     x0, 6
        add     x0, x12, 64
        add     x4, x16, x4
        add     x12, x11, x7
        add     x17, x20, 8
        csel    x12, x12, x11, eq
        csel    x23, x4, x16, eq
        sub     x4, x19, x20
1:
        ldp     q28, q29, [x5], 32
        cmp     x2, 8
        b.cc    3f
        add     x7, x21, x17
        add     x19, x22, x17
        add     x20, x23, x17
        mov     x24, x5
        mov     x25, x3
        mov     x26, x15
        mov     x27, x16
        mov     v20.16b, v28.16b
        mov     v21.16b, v29.16b
        mov     v22.16b, v28.16b
        mov     v23.16b, v29.16b
        mov     v24.16b, v28.16b
        mov     v25.16b, v29.16b
        mov     v26.16b, v28.16b
        mov     v27.16b, v29.16b
        mov     v31.16b, v28.16b
        mov     v30.16b, v29.16b
        mov     x28, x2
2:
        ldr     d19, [x25], 8
        ldr     d18, [x21], 8
        ldr     d17, [x26], 8
        ldr     d16, [x22], 8
        ldr     d0, [x27], 8
        ldr     d1, [x23], 8
        ldp     q12, q13, [x24]
        ldp     q2, q3, [x24, 32]
        sub     x28, x28, 8
        cmp     x28, 7
        sdot    v28.4s, v12.16b, v19.4b[0]
        sdot    v29.4s, v13.16b, v19.4b[0]
        sdot    v20.4s, v12.16b, v18.4b[0]
        sdot    v21.4s, v13.16b, v18.4b[0]
        sdot    v22.4s, v12.16b, v17.4b[0]
        sdot    v23.4s, v13.16b, v17.4b[0]
        sdot    v24.4s, v12.16b, v16.4b[0]
        sdot    v25.4s, v13.16b, v16.4b[0]
        sdot    v26.4s, v12.16b, v0.4b[0]
        sdot    v27.4s, v13.16b, v0.4b[0]
        sdot    v31.4s, v12.16b, v1.4b[0]
        sdot    v30.4s, v13.16b, v1.4b[0]
        sdot    v28.4s, v2.16b, v19.4b[1]
        sdot    v29.4s, v3.16b, v19.4b[1]
        sdot    v20.4s, v2.16b, v18.4b[1]
        sdot    v21.4s, v3.16b, v18.4b[1]
        sdot    v22.4s, v2.16b, v17.4b[1]
        sdot    v23.4s, v3.16b, v17.4b[1]
        sdot    v24.4s, v2.16b, v16.4b[1]
        sdot    v25.4s, v3.16b, v16.4b[1]
        sdot    v26.4s, v2.16b, v0.4b[1]
        sdot    v27.4s, v3.16b, v0.4b[1]
        sdot    v31.4s, v2.16b, v1.4b[1]
        sdot    v30.4s, v3.16b, v1.4b[1]
        add     x24, x24, 64
        b.hi    2b
        add     x3, x3, x17
        add     x15, x15, x17
        add     x16, x16, x17
        add     x5, x5, x0
        mov     x24, x4
        cbz     x24, 4f
        b       5f
3:
        mov     x24, x2
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
        mov     x20, x23
        mov     x19, x22
        mov     x7, x21
        cbnz    x24, 5f
4:
        mov     x21, x13
        ld1r    {v18.4s}, [x21], 4
        cmp     x1, 7
        ld1r    {v19.4s}, [x21]
        add     x21, x13, 8
        sqrdmulh        v17.4s, v28.4s, v18.4s
        sqrdmulh        v0.4s, v20.4s, v18.4s
        cmeq    v9.4s, v19.4s, 0
        mvn     v9.16b, v9.16b
        sqrdmulh        v2.4s, v22.4s, v18.4s
        sqrdmulh        v12.4s, v24.4s, v18.4s
        sqrdmulh        v14.4s, v26.4s, v18.4s
        and     v28.16b, v28.16b, v9.16b
        and     v20.16b, v20.16b, v9.16b
        and     v22.16b, v22.16b, v9.16b
        and     v24.16b, v24.16b, v9.16b
        and     v26.16b, v26.16b, v9.16b
        sqrdmulh        v16.4s, v29.4s, v18.4s
        sqrdmulh        v1.4s, v21.4s, v18.4s
        sqrdmulh        v3.4s, v23.4s, v18.4s
        sqrdmulh        v13.4s, v25.4s, v18.4s
        sqrdmulh        v15.4s, v27.4s, v18.4s
        sqrdmulh        v8.4s, v31.4s, v18.4s
        sqrdmulh        v18.4s, v30.4s, v18.4s
        and     v29.16b, v29.16b, v9.16b
        and     v21.16b, v21.16b, v9.16b
        and     v23.16b, v23.16b, v9.16b
        and     v25.16b, v25.16b, v9.16b
        and     v27.16b, v27.16b, v9.16b
        and     v31.16b, v31.16b, v9.16b
        and     v30.16b, v30.16b, v9.16b
        ld1r    {v9.8h}, [x21]
        ssra    v17.4s, v28.4s, 31
        ssra    v0.4s, v20.4s, 31
        ssra    v2.4s, v22.4s, 31
        ssra    v12.4s, v24.4s, 31
        ssra    v14.4s, v26.4s, 31
        ssra    v16.4s, v29.4s, 31
        ssra    v1.4s, v21.4s, 31
        ssra    v3.4s, v23.4s, 31
        ssra    v13.4s, v25.4s, 31
        ssra    v15.4s, v27.4s, 31
        ssra    v8.4s, v31.4s, 31
        ssra    v18.4s, v30.4s, 31
        srshl   v28.4s, v17.4s, v19.4s
        srshl   v30.4s, v0.4s, v19.4s
        srshl   v27.4s, v2.4s, v19.4s
        srshl   v25.4s, v12.4s, v19.4s
        srshl   v23.4s, v14.4s, v19.4s
        srshl   v29.4s, v16.4s, v19.4s
        srshl   v31.4s, v1.4s, v19.4s
        srshl   v26.4s, v3.4s, v19.4s
        srshl   v24.4s, v13.4s, v19.4s
        srshl   v22.4s, v15.4s, v19.4s
        srshl   v21.4s, v8.4s, v19.4s
        sqxtn   v28.4h, v28.4s
        sqxtn   v30.4h, v30.4s
        sqxtn   v27.4h, v27.4s
        sqxtn   v25.4h, v25.4s
        sqxtn   v23.4h, v23.4s
        srshl   v20.4s, v18.4s, v19.4s
        sqxtn   v21.4h, v21.4s
        sqxtn2  v28.8h, v29.4s
        sqxtn2  v30.8h, v31.4s
        sqxtn2  v27.8h, v26.4s
        sqxtn2  v25.8h, v24.4s
        sqxtn2  v23.8h, v22.4s
        sqxtn2  v21.8h, v20.4s
        sqadd   v28.8h, v28.8h, v9.8h
        sqadd   v29.8h, v30.8h, v9.8h
        sqadd   v30.8h, v27.8h, v9.8h
        sqadd   v27.8h, v23.8h, v9.8h
        add     x21, x13, 10
        sqadd   v31.8h, v25.8h, v9.8h
        sqadd   v26.8h, v21.8h, v9.8h
        sqxtn   v28.8b, v28.8h
        sqxtn   v30.8b, v30.8h
        sqxtn   v27.8b, v27.8h
        sqxtn2  v28.16b, v29.8h
        sqxtn2  v30.16b, v31.8h
        sqxtn2  v27.16b, v26.8h
        ld1r    {v29.16b}, [x21]
        add     x21, x13, 11
        ld1r    {v31.16b}, [x21]
        smax    v28.16b, v28.16b, v29.16b
        smax    v26.16b, v30.16b, v29.16b
        smax    v27.16b, v27.16b, v29.16b
        smin    v30.16b, v28.16b, v31.16b
        smin    v29.16b, v26.16b, v31.16b
        smin    v28.16b, v27.16b, v31.16b
        b.ls    7f
        str     d30, [x6]
        ext     v30.16b, v30.16b, v30.16b, 8
        str     d30, [x8]
        ext     v30.16b, v29.16b, v29.16b, 8
        str     d29, [x9]
        ext     v29.16b, v28.16b, v28.16b, 8
        sub     x23, x20, x2
        sub     x16, x16, x2
        sub     x22, x19, x2
        sub     x15, x15, x2
        sub     x21, x7, x2
        sub     x3, x3, x2
        add     x6, x6, x14
        subs    x1, x1, 8
        str     d30, [x10]
        str     d28, [x11]
        str     d29, [x12]
        add     x8, x8, x14
        add     x9, x9, x14
        add     x10, x10, x14
        add     x11, x11, x14
        add     x12, x12, x14
        b.ne    1b
        b       10f
5:
        ldr     d1, [x3]
        ldr     d0, [x7]
        ldr     d16, [x15]
        ldr     d17, [x19]
        ldr     d18, [x16]
        ldr     d19, [x20]
        ldp     q2, q3, [x5]
        add     x3, x3, x24
        add     x7, x7, x24
        add     x15, x15, x24
        add     x19, x19, x24
        add     x16, x16, x24
        add     x20, x20, x24
        cmp     x24, 5
        sdot    v28.4s, v2.16b, v1.4b[0]
        sdot    v29.4s, v3.16b, v1.4b[0]
        sdot    v20.4s, v2.16b, v0.4b[0]
        sdot    v21.4s, v3.16b, v0.4b[0]
        sdot    v22.4s, v2.16b, v16.4b[0]
        sdot    v23.4s, v3.16b, v16.4b[0]
        sdot    v24.4s, v2.16b, v17.4b[0]
        sdot    v25.4s, v3.16b, v17.4b[0]
        sdot    v26.4s, v2.16b, v18.4b[0]
        sdot    v27.4s, v3.16b, v18.4b[0]
        sdot    v31.4s, v2.16b, v19.4b[0]
        sdot    v30.4s, v3.16b, v19.4b[0]
        b.cc    6f
        ldp     q2, q3, [x5, 32]
        add     x5, x5, 64
        sdot    v28.4s, v2.16b, v1.4b[1]
        sdot    v29.4s, v3.16b, v1.4b[1]
        sdot    v20.4s, v2.16b, v0.4b[1]
        sdot    v21.4s, v3.16b, v0.4b[1]
        sdot    v22.4s, v2.16b, v16.4b[1]
        sdot    v23.4s, v3.16b, v16.4b[1]
        sdot    v24.4s, v2.16b, v17.4b[1]
        sdot    v25.4s, v3.16b, v17.4b[1]
        sdot    v26.4s, v2.16b, v18.4b[1]
        sdot    v27.4s, v3.16b, v18.4b[1]
        sdot    v31.4s, v2.16b, v19.4b[1]
        sdot    v30.4s, v3.16b, v19.4b[1]
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
        st1     {v30.b}[0], [x6]
        st1     {v30.b}[8], [x8]
        st1     {v29.b}[0], [x9]
        st1     {v29.b}[8], [x10]
        st1     {v28.b}[0], [x11]
        st1     {v28.b}[8], [x12]
10:
        ldp     x20, x19, [sp, 80]
        ldp     x22, x21, [sp, 64]
        ldp     x24, x23, [sp, 48]
        ldp     x26, x25, [sp, 32]
        ldp     x28, x27, [sp, 16]
        ldp     d9, d8, [sp], 96
        ret
11:
        st1     {v30.s}[0], [x6], 4
        st1     {v30.s}[2], [x8], 4
        st1     {v29.s}[0], [x9], 4
        st1     {v29.s}[2], [x10], 4
        st1     {v28.s}[0], [x11], 4
        st1     {v28.s}[2], [x12], 4
        ext     v30.16b, v30.16b, v30.16b, 4
        ext     v29.16b, v29.16b, v29.16b, 4
        ext     v28.16b, v28.16b, v28.16b, 4
        tbz     w1, 1, 8b
12:
        st1     {v30.h}[0], [x6], 2
        st1     {v30.h}[4], [x8], 2
        st1     {v29.h}[0], [x9], 2
        st1     {v29.h}[4], [x10], 2
        st1     {v28.h}[0], [x11], 2
        st1     {v28.h}[4], [x12], 2
        ext     v30.16b, v30.16b, v30.16b, 2
        ext     v29.16b, v29.16b, v29.16b, 2
        ext     v28.16b, v28.16b, v28.16b, 2
        tbnz    w1, 0, 9b
        b       10b

END_FUNCTION xnn_qs8_gemm_minmax_ukernel_6x8c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif

