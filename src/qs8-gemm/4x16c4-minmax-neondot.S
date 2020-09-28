// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_gemm_minmax_ukernel_4x16c4__aarch64_neondot
        sub     sp, sp, 0x80
        stp     d7, d6, [sp, 32]
        stp     d5, d4, [sp, 48]
        stp     d11, d10, [sp, 64]
        stp     d9, d8, [sp, 80]
        stp     x22, x21, [sp, 96]
        stp     x20, x19, [sp, 112]
        add     x9, x3, x4
        add     x8, x6, x7
        cmp     x0, 2
        sub     x16, x2, 8
        csel    x8, x6, x8, lo
        csel    x19, x3, x9, lo
        ldp     x12, x11, [sp, 128]
        cmp     x0, 3
        lsl     x9, x16, 4
        add     x10, x19, x4
        add     x13, x8, x7
        and     x15, x9, 0xffffffffffffff80
        csel    x9, x8, x13, lo
        csel    x13, x19, x10, lo
        and     x17, x16, 0xfffffffffffffff8
        cmp     x0, 4
        add     x0, x13, x4
        add     x10, x9, x7
        add     x14, x17, 8
        add     x15, x15, 0x80
        csel    x10, x10, x9, eq
        csel    x4, x0, x13, eq
        sub     x16, x16, x17
1:
        ldp     q25, q27, [x5]
        ldp     q31, q29, [x5, 32]
        cmp     x2, 8
        add     x5, x5, 64
        b.cc    3f
        add     x17, x19, x14
        add     x0, x4, x14
        mov     x7, x5
        mov     x20, x3
        mov     x21, x13
        mov     v16.16b, v25.16b
        mov     v17.16b, v27.16b
        mov     v18.16b, v31.16b
        mov     v19.16b, v29.16b
        mov     v20.16b, v25.16b
        mov     v21.16b, v27.16b
        mov     v22.16b, v31.16b
        mov     v23.16b, v29.16b
        mov     v24.16b, v25.16b
        mov     v26.16b, v27.16b
        mov     v30.16b, v31.16b
        mov     v28.16b, v29.16b
        mov     x22, x2
2:
        ldr     d0, [x20], 8
        ldr     d1, [x19], 8
        ldr     d2, [x21], 8
        ldr     d3, [x4], 8
        ldp     q8, q9, [x7]
        ldp     q10, q11, [x7, 32]
        ldp     q12, q13, [x7, 64]
        ldp     q14, q15, [x7, 96]
        sub     x22, x22, 8
        sdot    v25.4s, v8.16b, v0.4b[0]
        sdot    v27.4s, v9.16b, v0.4b[0]
        sdot    v31.4s, v10.16b, v0.4b[0]
        sdot    v29.4s, v11.16b, v0.4b[0]
        sdot    v16.4s, v8.16b, v1.4b[0]
        sdot    v17.4s, v9.16b, v1.4b[0]
        sdot    v18.4s, v10.16b, v1.4b[0]
        sdot    v19.4s, v11.16b, v1.4b[0]
        sdot    v20.4s, v8.16b, v2.4b[0]
        sdot    v21.4s, v9.16b, v2.4b[0]
        sdot    v22.4s, v10.16b, v2.4b[0]
        sdot    v23.4s, v11.16b, v2.4b[0]
        sdot    v24.4s, v8.16b, v3.4b[0]
        sdot    v26.4s, v9.16b, v3.4b[0]
        sdot    v30.4s, v10.16b, v3.4b[0]
        sdot    v28.4s, v11.16b, v3.4b[0]
        cmp     x22, 7
        sdot    v25.4s, v12.16b, v0.4b[1]
        sdot    v27.4s, v13.16b, v0.4b[1]
        sdot    v31.4s, v14.16b, v0.4b[1]
        sdot    v29.4s, v15.16b, v0.4b[1]
        sdot    v16.4s, v12.16b, v1.4b[1]
        sdot    v17.4s, v13.16b, v1.4b[1]
        sdot    v18.4s, v14.16b, v1.4b[1]
        sdot    v19.4s, v15.16b, v1.4b[1]
        sdot    v20.4s, v12.16b, v2.4b[1]
        sdot    v21.4s, v13.16b, v2.4b[1]
        sdot    v22.4s, v14.16b, v2.4b[1]
        sdot    v23.4s, v15.16b, v2.4b[1]
        sdot    v24.4s, v12.16b, v3.4b[1]
        sdot    v26.4s, v13.16b, v3.4b[1]
        sdot    v30.4s, v14.16b, v3.4b[1]
        sdot    v28.4s, v15.16b, v3.4b[1]
        add     x7, x7, 0x80
        b.hi    2b
        add     x3, x3, x14
        add     x13, x13, x14
        add     x5, x5, x15
        mov     x7, x16
        cbz     x7, 4f
        b       5f
3:
        mov     x7, x2
        mov     v28.16b, v29.16b
        mov     v30.16b, v31.16b
        mov     v26.16b, v27.16b
        mov     v24.16b, v25.16b
        mov     v23.16b, v29.16b
        mov     v22.16b, v31.16b
        mov     v21.16b, v27.16b
        mov     v20.16b, v25.16b
        mov     v19.16b, v29.16b
        mov     v18.16b, v31.16b
        mov     v17.16b, v27.16b
        mov     v16.16b, v25.16b
        mov     x0, x4
        mov     x17, x19
        cbnz    x7, 5f
4:
        mov     x4, x11
        ld1r    {v1.4s}, [x4], 4
        cmp     x1, 0xf
        ld1r    {v0.4s}, [x4]
        sqrdmulh        v2.4s, v25.4s, v1.4s
        str     q2, [sp, 16]
        sqrdmulh        v2.4s, v27.4s, v1.4s
        str     q2, [sp]
        cmeq    v2.4s, v0.4s, 0
        mvn     v2.16b, v2.16b
        sqrdmulh        v3.4s, v31.4s, v1.4s
        sqrdmulh        v12.4s, v29.4s, v1.4s
        sqrdmulh        v13.4s, v16.4s, v1.4s
        sqrdmulh        v14.4s, v17.4s, v1.4s
        sqrdmulh        v15.4s, v18.4s, v1.4s
        sqrdmulh        v8.4s, v19.4s, v1.4s
        sqrdmulh        v9.4s, v20.4s, v1.4s
        sqrdmulh        v10.4s, v21.4s, v1.4s
        sqrdmulh        v11.4s, v22.4s, v1.4s
        sqrdmulh        v4.4s, v23.4s, v1.4s
        sqrdmulh        v5.4s, v24.4s, v1.4s
        sqrdmulh        v6.4s, v26.4s, v1.4s
        sqrdmulh        v7.4s, v30.4s, v1.4s
        sqrdmulh        v1.4s, v28.4s, v1.4s
        and     v25.16b, v25.16b, v2.16b
        and     v27.16b, v27.16b, v2.16b
        and     v31.16b, v31.16b, v2.16b
        and     v29.16b, v29.16b, v2.16b
        and     v16.16b, v16.16b, v2.16b
        and     v17.16b, v17.16b, v2.16b
        and     v18.16b, v18.16b, v2.16b
        and     v19.16b, v19.16b, v2.16b
        and     v20.16b, v20.16b, v2.16b
        and     v21.16b, v21.16b, v2.16b
        and     v22.16b, v22.16b, v2.16b
        and     v23.16b, v23.16b, v2.16b
        and     v24.16b, v24.16b, v2.16b
        and     v26.16b, v26.16b, v2.16b
        and     v30.16b, v30.16b, v2.16b
        and     v28.16b, v28.16b, v2.16b
        ldr     q2, [sp, 16]
        ssra    v3.4s, v31.4s, 31
        ssra    v12.4s, v29.4s, 31
        ssra    v13.4s, v16.4s, 31
        ssra    v2.4s, v25.4s, 31
        ldr     q25, [sp]
        ssra    v14.4s, v17.4s, 31
        ssra    v15.4s, v18.4s, 31
        ssra    v8.4s, v19.4s, 31
        ssra    v25.4s, v27.4s, 31
        ssra    v9.4s, v20.4s, 31
        ssra    v10.4s, v21.4s, 31
        ssra    v11.4s, v22.4s, 31
        ssra    v4.4s, v23.4s, 31
        ssra    v5.4s, v24.4s, 31
        ssra    v6.4s, v26.4s, 31
        ssra    v7.4s, v30.4s, 31
        ssra    v1.4s, v28.4s, 31
        add     x4, x11, 8
        srshl   v28.4s, v2.4s, v0.4s
        srshl   v29.4s, v25.4s, v0.4s
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
        ld1r    {v0.8h}, [x4]
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
        add     x4, x11, 10
        sqxtn   v30.8b, v27.8h
        sqxtn   v31.8b, v25.8h
        sqxtn2  v30.16b, v26.8h
        sqxtn2  v31.16b, v24.8h
        ld1r    {v27.16b}, [x4]
        add     x4, x11, 11
        ld1r    {v26.16b}, [x4]
        smax    v28.16b, v28.16b, v27.16b
        smax    v25.16b, v29.16b, v27.16b
        smax    v30.16b, v30.16b, v27.16b
        smax    v27.16b, v31.16b, v27.16b
        smin    v29.16b, v28.16b, v26.16b
        smin    v31.16b, v25.16b, v26.16b
        smin    v28.16b, v30.16b, v26.16b
        smin    v30.16b, v27.16b, v26.16b
        b.ls    7f
        sub     x4, x0, x2
        sub     x13, x13, x2
        sub     x19, x17, x2
        sub     x3, x3, x2
        str     q29, [x6]
        str     q31, [x8]
        str     q28, [x9]
        str     q30, [x10]
        add     x6, x6, x12
        add     x8, x8, x12
        add     x9, x9, x12
        subs    x1, x1, 0x10
        add     x10, x10, x12
        b.ne    1b
        b       11f
5:
        ldr     d3, [x3]
        ldr     d2, [x17]
        ldr     d1, [x13]
        ldr     d0, [x0]
        ldp     q12, q13, [x5]
        ldp     q14, q15, [x5, 32]
        add     x3, x3, x7
        add     x17, x17, x7
        add     x13, x13, x7
        add     x0, x0, x7
        cmp     x7, 5
        sdot    v25.4s, v12.16b, v3.4b[0]
        sdot    v27.4s, v13.16b, v3.4b[0]
        sdot    v31.4s, v14.16b, v3.4b[0]
        sdot    v29.4s, v15.16b, v3.4b[0]
        sdot    v16.4s, v12.16b, v2.4b[0]
        sdot    v17.4s, v13.16b, v2.4b[0]
        sdot    v18.4s, v14.16b, v2.4b[0]
        sdot    v19.4s, v15.16b, v2.4b[0]
        sdot    v20.4s, v12.16b, v1.4b[0]
        sdot    v21.4s, v13.16b, v1.4b[0]
        sdot    v22.4s, v14.16b, v1.4b[0]
        sdot    v23.4s, v15.16b, v1.4b[0]
        sdot    v24.4s, v12.16b, v0.4b[0]
        sdot    v26.4s, v13.16b, v0.4b[0]
        sdot    v30.4s, v14.16b, v0.4b[0]
        sdot    v28.4s, v15.16b, v0.4b[0]
        b.cc    6f
        ldp     q12, q13, [x5, 64]
        ldp     q14, q15, [x5, 96]
        add     x5, x5, 0x80
        sdot    v25.4s, v12.16b, v3.4b[1]
        sdot    v27.4s, v13.16b, v3.4b[1]
        sdot    v31.4s, v14.16b, v3.4b[1]
        sdot    v29.4s, v15.16b, v3.4b[1]
        sdot    v16.4s, v12.16b, v2.4b[1]
        sdot    v17.4s, v13.16b, v2.4b[1]
        sdot    v18.4s, v14.16b, v2.4b[1]
        sdot    v19.4s, v15.16b, v2.4b[1]
        sdot    v20.4s, v12.16b, v1.4b[1]
        sdot    v21.4s, v13.16b, v1.4b[1]
        sdot    v22.4s, v14.16b, v1.4b[1]
        sdot    v23.4s, v15.16b, v1.4b[1]
        sdot    v24.4s, v12.16b, v0.4b[1]
        sdot    v26.4s, v13.16b, v0.4b[1]
        sdot    v30.4s, v14.16b, v0.4b[1]
        sdot    v28.4s, v15.16b, v0.4b[1]
        b       4b
6:
        add     x5, x5, 64
        b       4b
7:
        tbz     w1, 3, 8f
        str     d29, [x6], 8
        str     d31, [x8], 8
        str     d28, [x9], 8
        str     d30, [x10], 8
        ext     v29.16b, v29.16b, v29.16b, 8
        ext     v31.16b, v31.16b, v31.16b, 8
        ext     v28.16b, v28.16b, v28.16b, 8
        ext     v30.16b, v30.16b, v30.16b, 8
8:
        mov     v29.d[1], v31.d[0]
        mov     v28.d[1], v30.d[0]
        tbnz    w1, 2, 12f
        tbnz    w1, 1, 13f
9:
        tbz     w1, 0, 11f
10:
        st1     {v29.b}[0], [x6]
        st1     {v29.b}[8], [x8]
        st1     {v28.b}[0], [x9]
        st1     {v28.b}[8], [x10]
11:
        ldp     x20, x19, [sp, 112]
        ldp     x22, x21, [sp, 96]
        ldp     d9, d8, [sp, 80]
        ldp     d11, d10, [sp, 64]
        ldp     d5, d4, [sp, 48]
        ldp     d7, d6, [sp, 32]
        add     sp, sp, 0x80
        ret
12:
        st1     {v29.s}[0], [x6], 4
        st1     {v29.s}[2], [x8], 4
        st1     {v28.s}[0], [x9], 4
        st1     {v28.s}[2], [x10], 4
        ext     v29.16b, v29.16b, v29.16b, 4
        ext     v28.16b, v28.16b, v28.16b, 4
        tbz     w1, 1, 9b
13:
        st1     {v29.h}[0], [x6], 2
        st1     {v29.h}[4], [x8], 2
        st1     {v28.h}[0], [x9], 2
        st1     {v28.h}[4], [x10], 2
        ext     v29.16b, v29.16b, v29.16b, 2
        ext     v28.16b, v28.16b, v28.16b, 2
        tbnz    w1, 0, 10b
        b       11b

END_FUNCTION xnn_qs8_gemm_minmax_ukernel_4x16c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif

