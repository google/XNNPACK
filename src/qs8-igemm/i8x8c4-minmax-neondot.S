// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_igemm_minmax_ukernel_8x8c4__aarch64_neondot
        sub     sp, sp, 0xd0
        stp     d11, d10, [sp, 80]
        stp     d9, d8, [sp, 96]
        stp     x28, x27, [sp, 112]
        stp     x26, x25, [sp, 128]
        stp     x24, x23, [sp, 144]
        stp     x22, x21, [sp, 160]
        stp     x20, x19, [sp, 176]
        stp     x29, x30, [sp, 192]
        ldp     x16, x8, [sp, 224]
        cmp     x0, 2
        mov     x11, x2
        sub     x9, x2, 8
        str     x8, [sp, 16]
        add     x8, x6, x7
        csel    x14, x6, x8, lo
        cmp     x0, 3
        add     x12, x14, x7
        csel    x15, x14, x12, lo
        cmp     x0, 4
        add     x12, x15, x7
        csel    x2, x15, x12, lo
        cmp     x0, 5
        add     x12, x2, x7
        csel    x21, x2, x12, lo
        cmp     x0, 6
        add     x12, x21, x7
        csel    x12, x21, x12, lo
        ldp     x19, x17, [sp, 208]
        lsl     x8, x9, 3
        cmp     x0, 7
        add     x13, x12, x7
        and     x8, x8, 0xffffffffffffffc0
        csel    x13, x12, x13, lo
        and     x10, x9, 0xfffffffffffffff8
        cmp     x0, 8
        add     x20, x8, 64
        add     x8, x13, x7
        add     x0, x10, 8
        csel    x8, x8, x13, eq
        sub     x7, x9, x10
        str     x3, [sp, 8]
1:
        ldp     q28, q29, [x5], 32
        stp     x8, x21, [sp, 24]
        mov     x21, x3
        stp     x2, x15, [sp, 40]
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
        mov     v19.16b, v29.16b
        mov     v18.16b, v28.16b
        mov     v17.16b, v29.16b
        mov     v16.16b, v28.16b
        stp     x14, x1, [sp, 56]
        str     x6, [sp, 72]
2:
        ldp     x8, x9, [x4]
        ldp     x10, x15, [x4, 16]
        ldp     x2, x3, [x4, 32]
        ldp     x6, x22, [x4, 48]
        add     x14, x8, x17
        cmp     x8, x16
        csel    x1, x8, x14, eq
        add     x8, x9, x17
        cmp     x9, x16
        csel    x30, x9, x8, eq
        add     x8, x10, x17
        cmp     x10, x16
        csel    x14, x10, x8, eq
        add     x8, x15, x17
        cmp     x15, x16
        csel    x28, x15, x8, eq
        add     x8, x2, x17
        cmp     x2, x16
        csel    x9, x2, x8, eq
        add     x8, x3, x17
        cmp     x3, x16
        csel    x27, x3, x8, eq
        add     x8, x6, x17
        cmp     x6, x16
        csel    x10, x6, x8, eq
        add     x8, x22, x17
        cmp     x22, x16
        csel    x26, x22, x8, eq
        cmp     x11, 8
        b.cc    5f
        add     x29, x30, x0
        add     x15, x28, x0
        add     x3, x27, x0
        add     x8, x26, x0
        add     x6, x5, 32
        mov     x25, x1
        mov     x24, x14
        mov     x23, x9
        mov     x22, x10
        mov     x2, x11
3:
        ldr     d0, [x25], 8
        ldr     d1, [x30], 8
        ldr     d2, [x24], 8
        ldr     d3, [x28], 8
        ldr     d12, [x23], 8
        ldr     d13, [x27], 8
        ldr     d14, [x22], 8
        ldr     d15, [x26], 8
        ldp     q10, q11, [x6, -32]
        ldp     q8, q9, [x6], 64
        sub     x2, x2, 8
        cmp     x2, 7
        sdot    v16.4s, v10.16b, v0.4b[0]
        sdot    v17.4s, v11.16b, v0.4b[0]
        sdot    v18.4s, v10.16b, v1.4b[0]
        sdot    v19.4s, v11.16b, v1.4b[0]
        sdot    v20.4s, v10.16b, v2.4b[0]
        sdot    v21.4s, v11.16b, v2.4b[0]
        sdot    v22.4s, v10.16b, v3.4b[0]
        sdot    v23.4s, v11.16b, v3.4b[0]
        sdot    v24.4s, v10.16b, v12.4b[0]
        sdot    v25.4s, v11.16b, v12.4b[0]
        sdot    v26.4s, v10.16b, v13.4b[0]
        sdot    v27.4s, v11.16b, v13.4b[0]
        sdot    v31.4s, v10.16b, v14.4b[0]
        sdot    v30.4s, v11.16b, v14.4b[0]
        sdot    v28.4s, v10.16b, v15.4b[0]
        sdot    v29.4s, v11.16b, v15.4b[0]
        sdot    v16.4s, v8.16b, v0.4b[1]
        sdot    v17.4s, v9.16b, v0.4b[1]
        sdot    v18.4s, v8.16b, v1.4b[1]
        sdot    v19.4s, v9.16b, v1.4b[1]
        sdot    v20.4s, v8.16b, v2.4b[1]
        sdot    v21.4s, v9.16b, v2.4b[1]
        sdot    v22.4s, v8.16b, v3.4b[1]
        sdot    v23.4s, v9.16b, v3.4b[1]
        sdot    v24.4s, v8.16b, v12.4b[1]
        sdot    v25.4s, v9.16b, v12.4b[1]
        sdot    v26.4s, v8.16b, v13.4b[1]
        sdot    v27.4s, v9.16b, v13.4b[1]
        sdot    v31.4s, v8.16b, v14.4b[1]
        sdot    v30.4s, v9.16b, v14.4b[1]
        sdot    v28.4s, v8.16b, v15.4b[1]
        sdot    v29.4s, v9.16b, v15.4b[1]
        b.hi    3b
        add     x1, x1, x0
        add     x14, x14, x0
        add     x9, x9, x0
        add     x10, x10, x0
        add     x5, x5, x20
        mov     x2, x7
        add     x4, x4, 64
        cbnz    x2, 6f
4:
        subs    x21, x21, 64
        b.ne    2b
        b       8f
5:
        mov     x2, x11
        mov     x8, x26
        mov     x3, x27
        mov     x15, x28
        mov     x29, x30
        add     x4, x4, 64
        cbz     x2, 4b
6:
        ldr     d15, [x1]
        ldr     d14, [x29]
        ldr     d13, [x14]
        ldr     d12, [x15]
        ldr     d3, [x9]
        ldr     d2, [x3]
        ldr     d0, [x10]
        ldp     q8, q9, [x5]
        ldr     d1, [x8]
        cmp     x2, 5
        sdot    v16.4s, v8.16b, v15.4b[0]
        sdot    v17.4s, v9.16b, v15.4b[0]
        sdot    v18.4s, v8.16b, v14.4b[0]
        sdot    v19.4s, v9.16b, v14.4b[0]
        sdot    v20.4s, v8.16b, v13.4b[0]
        sdot    v21.4s, v9.16b, v13.4b[0]
        sdot    v22.4s, v8.16b, v12.4b[0]
        sdot    v23.4s, v9.16b, v12.4b[0]
        sdot    v24.4s, v8.16b, v3.4b[0]
        sdot    v25.4s, v9.16b, v3.4b[0]
        sdot    v26.4s, v8.16b, v2.4b[0]
        sdot    v27.4s, v9.16b, v2.4b[0]
        sdot    v31.4s, v8.16b, v0.4b[0]
        sdot    v30.4s, v9.16b, v0.4b[0]
        sdot    v28.4s, v8.16b, v1.4b[0]
        sdot    v29.4s, v9.16b, v1.4b[0]
        b.cc    7f
        ldp     q8, q9, [x5, 32]
        add     x5, x5, 64
        sdot    v16.4s, v8.16b, v15.4b[1]
        sdot    v17.4s, v9.16b, v15.4b[1]
        sdot    v18.4s, v8.16b, v14.4b[1]
        sdot    v19.4s, v9.16b, v14.4b[1]
        sdot    v20.4s, v8.16b, v13.4b[1]
        sdot    v21.4s, v9.16b, v13.4b[1]
        sdot    v22.4s, v8.16b, v12.4b[1]
        sdot    v23.4s, v9.16b, v12.4b[1]
        sdot    v24.4s, v8.16b, v3.4b[1]
        sdot    v25.4s, v9.16b, v3.4b[1]
        sdot    v26.4s, v8.16b, v2.4b[1]
        sdot    v27.4s, v9.16b, v2.4b[1]
        sdot    v31.4s, v8.16b, v0.4b[1]
        sdot    v30.4s, v9.16b, v0.4b[1]
        sdot    v28.4s, v8.16b, v1.4b[1]
        sdot    v29.4s, v9.16b, v1.4b[1]
        subs    x21, x21, 64
        b.ne    2b
        b       8f
7:
        add     x5, x5, 32
        subs    x21, x21, 64
        b.ne    2b
8:
        ldr     x9, [sp, 16]
        mov     x8, x9
        ld1r    {v1.4s}, [x8], 4
        ld1r    {v0.4s}, [x8]
        sqrdmulh        v16.4s, v16.4s, v1.4s
        sqrdmulh        v17.4s, v17.4s, v1.4s
        sqrdmulh        v18.4s, v18.4s, v1.4s
        sqrdmulh        v19.4s, v19.4s, v1.4s
        sqrdmulh        v20.4s, v20.4s, v1.4s
        sqrdmulh        v21.4s, v21.4s, v1.4s
        sqrdmulh        v22.4s, v22.4s, v1.4s
        sqrdmulh        v23.4s, v23.4s, v1.4s
        sqrdmulh        v24.4s, v24.4s, v1.4s
        sqrdmulh        v25.4s, v25.4s, v1.4s
        sqrdmulh        v26.4s, v26.4s, v1.4s
        sqrdmulh        v27.4s, v27.4s, v1.4s
        sqrdmulh        v31.4s, v31.4s, v1.4s
        sqrdmulh        v30.4s, v30.4s, v1.4s
        sqrdmulh        v28.4s, v28.4s, v1.4s
        sqrdmulh        v29.4s, v29.4s, v1.4s
        cmeq    v1.4s, v0.4s, 0
        mvn     v1.16b, v1.16b
        and     v2.16b, v16.16b, v1.16b
        ssra    v16.4s, v2.4s, 31
        and     v2.16b, v17.16b, v1.16b
        ssra    v17.4s, v2.4s, 31
        and     v2.16b, v18.16b, v1.16b
        ssra    v18.4s, v2.4s, 31
        and     v2.16b, v19.16b, v1.16b
        ssra    v19.4s, v2.4s, 31
        and     v2.16b, v20.16b, v1.16b
        ssra    v20.4s, v2.4s, 31
        and     v2.16b, v21.16b, v1.16b
        ssra    v21.4s, v2.4s, 31
        and     v2.16b, v22.16b, v1.16b
        ssra    v22.4s, v2.4s, 31
        and     v2.16b, v23.16b, v1.16b
        ssra    v23.4s, v2.4s, 31
        and     v2.16b, v24.16b, v1.16b
        ssra    v24.4s, v2.4s, 31
        and     v2.16b, v25.16b, v1.16b
        ssra    v25.4s, v2.4s, 31
        and     v2.16b, v26.16b, v1.16b
        ssra    v26.4s, v2.4s, 31
        and     v2.16b, v27.16b, v1.16b
        ssra    v27.4s, v2.4s, 31
        and     v2.16b, v31.16b, v1.16b
        ssra    v31.4s, v2.4s, 31
        and     v2.16b, v30.16b, v1.16b
        add     x8, x9, 8
        ssra    v30.4s, v2.4s, 31
        and     v2.16b, v28.16b, v1.16b
        ssra    v28.4s, v2.4s, 31
        ld1r    {v2.8h}, [x8]
        and     v1.16b, v29.16b, v1.16b
        srshl   v16.4s, v16.4s, v0.4s
        srshl   v18.4s, v18.4s, v0.4s
        srshl   v20.4s, v20.4s, v0.4s
        srshl   v22.4s, v22.4s, v0.4s
        srshl   v24.4s, v24.4s, v0.4s
        srshl   v26.4s, v26.4s, v0.4s
        srshl   v31.4s, v31.4s, v0.4s
        ssra    v29.4s, v1.4s, 31
        srshl   v17.4s, v17.4s, v0.4s
        srshl   v19.4s, v19.4s, v0.4s
        srshl   v21.4s, v21.4s, v0.4s
        srshl   v23.4s, v23.4s, v0.4s
        srshl   v25.4s, v25.4s, v0.4s
        srshl   v27.4s, v27.4s, v0.4s
        srshl   v30.4s, v30.4s, v0.4s
        srshl   v28.4s, v28.4s, v0.4s
        sqxtn   v16.4h, v16.4s
        sqxtn   v18.4h, v18.4s
        sqxtn   v20.4h, v20.4s
        sqxtn   v22.4h, v22.4s
        sqxtn   v24.4h, v24.4s
        sqxtn   v26.4h, v26.4s
        sqxtn   v31.4h, v31.4s
        srshl   v29.4s, v29.4s, v0.4s
        sqxtn   v28.4h, v28.4s
        sqxtn2  v16.8h, v17.4s
        sqxtn2  v18.8h, v19.4s
        sqxtn2  v20.8h, v21.4s
        sqxtn2  v22.8h, v23.4s
        sqxtn2  v24.8h, v25.4s
        sqxtn2  v26.8h, v27.4s
        sqxtn2  v31.8h, v30.4s
        sqxtn2  v28.8h, v29.4s
        sqadd   v29.8h, v16.8h, v2.8h
        sqadd   v27.8h, v20.8h, v2.8h
        sqadd   v24.8h, v24.8h, v2.8h
        sqadd   v31.8h, v31.8h, v2.8h
        add     x8, x9, 10
        sqadd   v30.8h, v18.8h, v2.8h
        sqadd   v25.8h, v22.8h, v2.8h
        sqadd   v26.8h, v26.8h, v2.8h
        sqadd   v28.8h, v28.8h, v2.8h
        sqxtn   v29.8b, v29.8h
        sqxtn   v27.8b, v27.8h
        sqxtn   v24.8b, v24.8h
        sqxtn   v31.8b, v31.8h
        sqxtn2  v29.16b, v30.8h
        sqxtn2  v27.16b, v25.8h
        sqxtn2  v24.16b, v26.8h
        sqxtn2  v31.16b, v28.8h
        ld1r    {v28.16b}, [x8]
        add     x8, x9, 11
        ldr     x1, [sp, 64]
        ld1r    {v26.16b}, [x8]
        smax    v30.16b, v31.16b, v28.16b
        smax    v25.16b, v24.16b, v28.16b
        smax    v27.16b, v27.16b, v28.16b
        smax    v28.16b, v29.16b, v28.16b
        cmp     x1, 7
        smin    v31.16b, v30.16b, v26.16b
        smin    v30.16b, v25.16b, v26.16b
        smin    v29.16b, v27.16b, v26.16b
        smin    v28.16b, v28.16b, v26.16b
        b.ls    9f
        ldp     x8, x21, [sp, 24]
        ext     v27.16b, v31.16b, v31.16b, 8
        ldr     x14, [sp, 56]
        ldr     x6, [sp, 72]
        str     d27, [x8]
        ext     v27.16b, v30.16b, v30.16b, 8
        str     d31, [x13]
        str     d27, [x12]
        str     d30, [x21]
        ldp     x2, x15, [sp, 40]
        ldr     x3, [sp, 8]
        ext     v31.16b, v29.16b, v29.16b, 8
        ext     v27.16b, v28.16b, v28.16b, 8
        subs    x1, x1, 8
        str     d31, [x2]
        str     d29, [x15]
        str     d27, [x14]
        str     d28, [x6]
        add     x8, x8, x19
        add     x13, x13, x19
        add     x12, x12, x19
        add     x21, x21, x19
        add     x2, x2, x19
        add     x15, x15, x19
        add     x14, x14, x19
        add     x6, x6, x19
        sub     x4, x4, x3
        b.ne    1b
        b       12f
9:
        ldr     x8, [sp, 72]
        ldp     x10, x9, [sp, 48]
        ldp     x14, x11, [sp, 32]
        ldr     x15, [sp, 24]
        tbnz    w1, 2, 13f
        tbnz    w1, 1, 14f
10:
        tbz     w1, 0, 12f
11:
        st1     {v31.b}[8], [x15]
        st1     {v31.b}[0], [x13]
        st1     {v30.b}[8], [x12]
        st1     {v30.b}[0], [x14]
        st1     {v29.b}[8], [x11]
        st1     {v29.b}[0], [x10]
        st1     {v28.b}[8], [x9]
        st1     {v28.b}[0], [x8]
12:
        ldp     x29, x30, [sp, 192]
        ldp     x20, x19, [sp, 176]
        ldp     x22, x21, [sp, 160]
        ldp     x24, x23, [sp, 144]
        ldp     x26, x25, [sp, 128]
        ldp     x28, x27, [sp, 112]
        ldp     d9, d8, [sp, 96]
        ldp     d11, d10, [sp, 80]
        add     sp, sp, 0xd0
        ret
13:
        st1     {v31.s}[2], [x15], 4
        st1     {v31.s}[0], [x13], 4
        st1     {v30.s}[2], [x12], 4
        st1     {v30.s}[0], [x14], 4
        st1     {v29.s}[2], [x11], 4
        st1     {v29.s}[0], [x10], 4
        st1     {v28.s}[2], [x9], 4
        st1     {v28.s}[0], [x8], 4
        ext     v31.16b, v31.16b, v31.16b, 4
        ext     v30.16b, v30.16b, v30.16b, 4
        ext     v29.16b, v29.16b, v29.16b, 4
        ext     v28.16b, v28.16b, v28.16b, 4
        tbz     w1, 1, 10b
14:
        st1     {v31.h}[4], [x15], 2
        st1     {v31.h}[0], [x13], 2
        st1     {v30.h}[4], [x12], 2
        st1     {v30.h}[0], [x14], 2
        st1     {v29.h}[4], [x11], 2
        st1     {v29.h}[0], [x10], 2
        st1     {v28.h}[4], [x9], 2
        st1     {v28.h}[0], [x8], 2
        ext     v31.16b, v31.16b, v31.16b, 2
        ext     v30.16b, v30.16b, v30.16b, 2
        ext     v29.16b, v29.16b, v29.16b, 2
        ext     v28.16b, v28.16b, v28.16b, 2
        tbnz    w1, 0, 11b
        b       12b

END_FUNCTION xnn_qs8_igemm_minmax_ukernel_8x8c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
