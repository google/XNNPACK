// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_gemm_minmax_ukernel_8x8c4__aarch64_neondot
        sub     sp, sp, 0xd0
        stp     d7, d6, [sp, 48]
        stp     d5, d4, [sp, 64]
        stp     d11, d10, [sp, 80]
        stp     d9, d8, [sp, 96]
        stp     x28, x27, [sp, 112]
        stp     x26, x25, [sp, 128]
        stp     x24, x23, [sp, 144]
        stp     x22, x21, [sp, 160]
        stp     x20, x19, [sp, 176]
        stp     x29, x30, [sp, 192]
        add     x9, x3, x4
        add     x8, x6, x7
        cmp     x0, 2
        sub     x17, x2, 8
        csel    x8, x6, x8, lo
        csel    x25, x3, x9, lo
        cmp     x0, 3
        lsl     x9, x17, 3
        add     x10, x25, x4
        add     x11, x8, x7
        and     x14, x9, 0xffffffffffffffc0
        csel    x9, x8, x11, lo
        csel    x21, x25, x10, lo
        cmp     x0, 4
        add     x11, x21, x4
        add     x10, x9, x7
        csel    x10, x9, x10, lo
        csel    x26, x21, x11, lo
        cmp     x0, 5
        add     x12, x26, x4
        add     x11, x10, x7
        csel    x11, x10, x11, lo
        csel    x22, x26, x12, lo
        cmp     x0, 6
        add     x13, x22, x4
        add     x12, x11, x7
        csel    x12, x11, x12, lo
        csel    x27, x22, x13, lo
        ldp     x16, x15, [sp, 208]
        cmp     x0, 7
        add     x19, x27, x4
        add     x13, x12, x7
        csel    x13, x12, x13, lo
        csel    x23, x27, x19, lo
        add     x14, x14, 64
        mov     x24, x2
        and     x2, x17, 0xfffffffffffffff8
        cmp     x0, 8
        str     x14, [sp, 8]
        add     x4, x23, x4
        add     x14, x13, x7
        add     x0, x2, 8
        csel    x14, x14, x13, eq
        csel    x28, x4, x23, eq
        sub     x17, x17, x2
        str     x17, [sp]
1:
        ldp     q30, q28, [x5], 32
        cmp     x24, 8
        b.cc    3f
        add     x17, x25, x0
        str     x17, [sp, 32]
        add     x17, x26, x0
        str     x17, [sp, 16]
        add     x7, x27, x0
        add     x4, x28, x0
        add     x29, x5, 32
        mov     x30, x3
        mov     x17, x21
        mov     x19, x22
        mov     x20, x23
        mov     v16.16b, v30.16b
        mov     v17.16b, v28.16b
        mov     v18.16b, v30.16b
        mov     v19.16b, v28.16b
        mov     v20.16b, v30.16b
        mov     v21.16b, v28.16b
        mov     v22.16b, v30.16b
        mov     v23.16b, v28.16b
        mov     v24.16b, v30.16b
        mov     v25.16b, v28.16b
        mov     v26.16b, v30.16b
        mov     v27.16b, v28.16b
        mov     v31.16b, v30.16b
        mov     v29.16b, v28.16b
        mov     x2, x24
2:
        ldr     d0, [x30], 8
        ldr     d1, [x25], 8
        ldr     d2, [x17], 8
        ldr     d3, [x26], 8
        ldr     d12, [x19], 8
        ldr     d13, [x27], 8
        ldr     d14, [x20], 8
        ldr     d15, [x28], 8
        ldp     q10, q11, [x29, -32]
        ldp     q8, q9, [x29], 64
        sub     x2, x2, 8
        cmp     x2, 7
        sdot    v30.4s, v10.16b, v0.4b[0]
        sdot    v28.4s, v11.16b, v0.4b[0]
        sdot    v16.4s, v10.16b, v1.4b[0]
        sdot    v17.4s, v11.16b, v1.4b[0]
        sdot    v18.4s, v10.16b, v2.4b[0]
        sdot    v19.4s, v11.16b, v2.4b[0]
        sdot    v20.4s, v10.16b, v3.4b[0]
        sdot    v21.4s, v11.16b, v3.4b[0]
        sdot    v22.4s, v10.16b, v12.4b[0]
        sdot    v23.4s, v11.16b, v12.4b[0]
        sdot    v24.4s, v10.16b, v13.4b[0]
        sdot    v25.4s, v11.16b, v13.4b[0]
        sdot    v26.4s, v10.16b, v14.4b[0]
        sdot    v27.4s, v11.16b, v14.4b[0]
        sdot    v31.4s, v10.16b, v15.4b[0]
        sdot    v29.4s, v11.16b, v15.4b[0]
        sdot    v30.4s, v8.16b, v0.4b[1]
        sdot    v28.4s, v9.16b, v0.4b[1]
        sdot    v16.4s, v8.16b, v1.4b[1]
        sdot    v17.4s, v9.16b, v1.4b[1]
        sdot    v18.4s, v8.16b, v2.4b[1]
        sdot    v19.4s, v9.16b, v2.4b[1]
        sdot    v20.4s, v8.16b, v3.4b[1]
        sdot    v21.4s, v9.16b, v3.4b[1]
        sdot    v22.4s, v8.16b, v12.4b[1]
        sdot    v23.4s, v9.16b, v12.4b[1]
        sdot    v24.4s, v8.16b, v13.4b[1]
        sdot    v25.4s, v9.16b, v13.4b[1]
        sdot    v26.4s, v8.16b, v14.4b[1]
        sdot    v27.4s, v9.16b, v14.4b[1]
        sdot    v31.4s, v8.16b, v15.4b[1]
        sdot    v29.4s, v9.16b, v15.4b[1]
        b.hi    2b
        ldp     x17, x26, [sp, 8]
        ldr     x25, [sp, 32]
        add     x3, x3, x0
        add     x21, x21, x0
        add     x5, x5, x17
        ldr     x17, [sp]
        add     x22, x22, x0
        add     x23, x23, x0
        cbz     x17, 4f
        b       5f
3:
        mov     x17, x24
        mov     v29.16b, v28.16b
        mov     v31.16b, v30.16b
        mov     v27.16b, v28.16b
        mov     v26.16b, v30.16b
        mov     v25.16b, v28.16b
        mov     v24.16b, v30.16b
        mov     v23.16b, v28.16b
        mov     v22.16b, v30.16b
        mov     v21.16b, v28.16b
        mov     v20.16b, v30.16b
        mov     v19.16b, v28.16b
        mov     v18.16b, v30.16b
        mov     v17.16b, v28.16b
        mov     v16.16b, v30.16b
        mov     x4, x28
        mov     x7, x27
        cbnz    x17, 5f
4:
        mov     x17, x15
        ld1r    {v1.4s}, [x17], 4
        cmp     x1, 7
        ld1r    {v0.4s}, [x17]
        sqrdmulh        v2.4s, v30.4s, v1.4s
        str     q2, [sp, 32]
        sqrdmulh        v2.4s, v28.4s, v1.4s
        str     q2, [sp, 16]
        cmeq    v2.4s, v0.4s, 0
        mvn     v2.16b, v2.16b
        sqrdmulh        v3.4s, v16.4s, v1.4s
        sqrdmulh        v12.4s, v17.4s, v1.4s
        sqrdmulh        v13.4s, v18.4s, v1.4s
        sqrdmulh        v14.4s, v19.4s, v1.4s
        sqrdmulh        v15.4s, v20.4s, v1.4s
        sqrdmulh        v8.4s, v21.4s, v1.4s
        sqrdmulh        v9.4s, v22.4s, v1.4s
        sqrdmulh        v10.4s, v23.4s, v1.4s
        sqrdmulh        v11.4s, v24.4s, v1.4s
        sqrdmulh        v4.4s, v25.4s, v1.4s
        sqrdmulh        v5.4s, v26.4s, v1.4s
        sqrdmulh        v6.4s, v27.4s, v1.4s
        sqrdmulh        v7.4s, v31.4s, v1.4s
        sqrdmulh        v1.4s, v29.4s, v1.4s
        and     v30.16b, v30.16b, v2.16b
        and     v28.16b, v28.16b, v2.16b
        and     v16.16b, v16.16b, v2.16b
        and     v17.16b, v17.16b, v2.16b
        and     v18.16b, v18.16b, v2.16b
        and     v19.16b, v19.16b, v2.16b
        and     v20.16b, v20.16b, v2.16b
        and     v21.16b, v21.16b, v2.16b
        and     v22.16b, v22.16b, v2.16b
        and     v23.16b, v23.16b, v2.16b
        and     v24.16b, v24.16b, v2.16b
        and     v25.16b, v25.16b, v2.16b
        and     v26.16b, v26.16b, v2.16b
        and     v27.16b, v27.16b, v2.16b
        and     v31.16b, v31.16b, v2.16b
        and     v29.16b, v29.16b, v2.16b
        ldr     q2, [sp, 32]
        ssra    v3.4s, v16.4s, 31
        ssra    v12.4s, v17.4s, 31
        ssra    v13.4s, v18.4s, 31
        ssra    v2.4s, v30.4s, 31
        ldr     q30, [sp, 16]
        ssra    v14.4s, v19.4s, 31
        ssra    v15.4s, v20.4s, 31
        ssra    v8.4s, v21.4s, 31
        ssra    v30.4s, v28.4s, 31
        ssra    v9.4s, v22.4s, 31
        ssra    v10.4s, v23.4s, 31
        ssra    v11.4s, v24.4s, 31
        ssra    v4.4s, v25.4s, 31
        ssra    v5.4s, v26.4s, 31
        ssra    v6.4s, v27.4s, 31
        ssra    v7.4s, v31.4s, 31
        ssra    v1.4s, v29.4s, 31
        add     x17, x15, 8
        srshl   v28.4s, v2.4s, v0.4s
        srshl   v29.4s, v30.4s, v0.4s
        srshl   v30.4s, v3.4s, v0.4s
        srshl   v31.4s, v12.4s, v0.4s
        srshl   v27.4s, v13.4s, v0.4s
        srshl   v26.4s, v14.4s, v0.4s
        srshl   v25.4s, v15.4s, v0.4s
        srshl   v24.4s, v8.4s, v0.4s
        srshl   v23.4s, v9.4s, v0.4s
        srshl   v22.4s, v10.4s, v0.4s
        srshl   v21.4s, v11.4s, v0.4s
        srshl   v20.4s, v4.4s, v0.4s
        srshl   v19.4s, v5.4s, v0.4s
        srshl   v18.4s, v6.4s, v0.4s
        srshl   v17.4s, v7.4s, v0.4s
        srshl   v16.4s, v1.4s, v0.4s
        ld1r    {v0.8h}, [x17]
        sqxtn   v28.4h, v28.4s
        sqxtn2  v28.8h, v29.4s
        sqxtn   v29.4h, v30.4s
        sqxtn   v30.4h, v27.4s
        sqxtn2  v29.8h, v31.4s
        sqxtn2  v30.8h, v26.4s
        sqxtn   v31.4h, v25.4s
        sqxtn   v27.4h, v23.4s
        sqxtn   v26.4h, v21.4s
        sqxtn   v25.4h, v19.4s
        sqadd   v28.8h, v28.8h, v0.8h
        sqxtn2  v31.8h, v24.4s
        sqxtn2  v27.8h, v22.4s
        sqxtn2  v26.8h, v20.4s
        sqxtn2  v25.8h, v18.4s
        sqxtn   v24.4h, v17.4s
        sqadd   v29.8h, v29.8h, v0.8h
        sqadd   v30.8h, v30.8h, v0.8h
        sqxtn   v28.8b, v28.8h
        sqxtn2  v24.8h, v16.4s
        sqadd   v31.8h, v31.8h, v0.8h
        sqadd   v27.8h, v27.8h, v0.8h
        sqadd   v25.8h, v25.8h, v0.8h
        sqxtn2  v28.16b, v29.8h
        sqxtn   v29.8b, v30.8h
        sqadd   v26.8h, v26.8h, v0.8h
        sqadd   v24.8h, v24.8h, v0.8h
        sqxtn2  v29.16b, v31.8h
        add     x17, x15, 10
        sqxtn   v30.8b, v27.8h
        sqxtn   v31.8b, v25.8h
        sqxtn2  v30.16b, v26.8h
        sqxtn2  v31.16b, v24.8h
        ld1r    {v27.16b}, [x17]
        add     x17, x15, 11
        ld1r    {v26.16b}, [x17]
        smax    v28.16b, v28.16b, v27.16b
        smax    v29.16b, v29.16b, v27.16b
        smax    v25.16b, v30.16b, v27.16b
        smax    v27.16b, v31.16b, v27.16b
        smin    v31.16b, v28.16b, v26.16b
        smin    v30.16b, v29.16b, v26.16b
        smin    v29.16b, v25.16b, v26.16b
        smin    v28.16b, v27.16b, v26.16b
        b.ls    7f
        str     d31, [x6]
        ext     v31.16b, v31.16b, v31.16b, 8
        str     d31, [x8]
        ext     v31.16b, v30.16b, v30.16b, 8
        str     d30, [x9]
        ext     v30.16b, v29.16b, v29.16b, 8
        str     d31, [x10]
        ext     v31.16b, v28.16b, v28.16b, 8
        sub     x28, x4, x24
        sub     x23, x23, x24
        sub     x27, x7, x24
        sub     x22, x22, x24
        sub     x26, x26, x24
        sub     x21, x21, x24
        sub     x25, x25, x24
        sub     x3, x3, x24
        add     x6, x6, x16
        subs    x1, x1, 8
        str     d29, [x11]
        str     d30, [x12]
        str     d28, [x13]
        str     d31, [x14]
        add     x8, x8, x16
        add     x9, x9, x16
        add     x10, x10, x16
        add     x11, x11, x16
        add     x12, x12, x16
        add     x13, x13, x16
        add     x14, x14, x16
        b.ne    1b
        b       10f
5:
        ldr     d15, [x3]
        ldr     d14, [x25]
        ldr     d13, [x21]
        ldr     d12, [x26]
        ldr     d3, [x22]
        ldr     d2, [x7]
        ldr     d1, [x23]
        ldr     d0, [x4]
        ldp     q8, q9, [x5]
        add     x3, x3, x17
        add     x25, x25, x17
        add     x21, x21, x17
        add     x26, x26, x17
        add     x22, x22, x17
        add     x7, x7, x17
        add     x23, x23, x17
        add     x4, x4, x17
        cmp     x17, 5
        sdot    v30.4s, v8.16b, v15.4b[0]
        sdot    v28.4s, v9.16b, v15.4b[0]
        sdot    v16.4s, v8.16b, v14.4b[0]
        sdot    v17.4s, v9.16b, v14.4b[0]
        sdot    v18.4s, v8.16b, v13.4b[0]
        sdot    v19.4s, v9.16b, v13.4b[0]
        sdot    v20.4s, v8.16b, v12.4b[0]
        sdot    v21.4s, v9.16b, v12.4b[0]
        sdot    v22.4s, v8.16b, v3.4b[0]
        sdot    v23.4s, v9.16b, v3.4b[0]
        sdot    v24.4s, v8.16b, v2.4b[0]
        sdot    v25.4s, v9.16b, v2.4b[0]
        sdot    v26.4s, v8.16b, v1.4b[0]
        sdot    v27.4s, v9.16b, v1.4b[0]
        sdot    v31.4s, v8.16b, v0.4b[0]
        sdot    v29.4s, v9.16b, v0.4b[0]
        b.cc    6f
        ldp     q8, q9, [x5, 32]
        add     x5, x5, 64
        sdot    v30.4s, v8.16b, v15.4b[1]
        sdot    v28.4s, v9.16b, v15.4b[1]
        sdot    v16.4s, v8.16b, v14.4b[1]
        sdot    v17.4s, v9.16b, v14.4b[1]
        sdot    v18.4s, v8.16b, v13.4b[1]
        sdot    v19.4s, v9.16b, v13.4b[1]
        sdot    v20.4s, v8.16b, v12.4b[1]
        sdot    v21.4s, v9.16b, v12.4b[1]
        sdot    v22.4s, v8.16b, v3.4b[1]
        sdot    v23.4s, v9.16b, v3.4b[1]
        sdot    v24.4s, v8.16b, v2.4b[1]
        sdot    v25.4s, v9.16b, v2.4b[1]
        sdot    v26.4s, v8.16b, v1.4b[1]
        sdot    v27.4s, v9.16b, v1.4b[1]
        sdot    v31.4s, v8.16b, v0.4b[1]
        sdot    v29.4s, v9.16b, v0.4b[1]
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
        st1     {v31.b}[0], [x6]
        st1     {v31.b}[8], [x8]
        st1     {v30.b}[0], [x9]
        st1     {v30.b}[8], [x10]
        st1     {v29.b}[0], [x11]
        st1     {v29.b}[8], [x12]
        st1     {v28.b}[0], [x13]
        st1     {v28.b}[8], [x14]
10:
        ldp     x29, x30, [sp, 192]
        ldp     x20, x19, [sp, 176]
        ldp     x22, x21, [sp, 160]
        ldp     x24, x23, [sp, 144]
        ldp     x26, x25, [sp, 128]
        ldp     x28, x27, [sp, 112]
        ldp     d9, d8, [sp, 96]
        ldp     d11, d10, [sp, 80]
        ldp     d5, d4, [sp, 64]
        ldp     d7, d6, [sp, 48]
        add     sp, sp, 0xd0
        ret
11:
        st1     {v31.s}[0], [x6], 4
        st1     {v31.s}[2], [x8], 4
        st1     {v30.s}[0], [x9], 4
        st1     {v30.s}[2], [x10], 4
        st1     {v29.s}[0], [x11], 4
        st1     {v29.s}[2], [x12], 4
        st1     {v28.s}[0], [x13], 4
        st1     {v28.s}[2], [x14], 4
        ext     v31.16b, v31.16b, v31.16b, 4
        ext     v30.16b, v30.16b, v30.16b, 4
        ext     v29.16b, v29.16b, v29.16b, 4
        ext     v28.16b, v28.16b, v28.16b, 4
        tbz     w1, 1, 8b
12:
        st1     {v31.h}[0], [x6], 2
        st1     {v31.h}[4], [x8], 2
        st1     {v30.h}[0], [x9], 2
        st1     {v30.h}[4], [x10], 2
        st1     {v29.h}[0], [x11], 2
        st1     {v29.h}[4], [x12], 2
        st1     {v28.h}[0], [x13], 2
        st1     {v28.h}[4], [x14], 2
        ext     v31.16b, v31.16b, v31.16b, 2
        ext     v30.16b, v30.16b, v30.16b, 2
        ext     v29.16b, v29.16b, v29.16b, 2
        ext     v28.16b, v28.16b, v28.16b, 2
        tbnz    w1, 0, 9b
        b       10b

END_FUNCTION xnn_qs8_gemm_minmax_ukernel_8x8c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
