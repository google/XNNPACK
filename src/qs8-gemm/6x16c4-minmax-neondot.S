// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_gemm_minmax_ukernel_6x16c4__aarch64_neondot
        stp     d7, d6, [sp, -160]!
        stp     d5, d4, [sp, 16]
        stp     d11, d10, [sp, 32]
        stp     d9, d8, [sp, 48]
        str     x28, [sp, 64]
        stp     x27, x26, [sp, 80]
        stp     x25, x24, [sp, 96]
        stp     x23, x22, [sp, 112]
        stp     x21, x20, [sp, 128]
        stp     x19, x29, [sp, 144]
        sub     sp, sp, 0x1d0
        add     x9, x3, x4
        add     x8, x6, x7
        cmp     x0, 2
        sub     x19, x2, 8
        csel    x8, x6, x8, lo
        csel    x21, x3, x9, lo
        cmp     x0, 3
        lsl     x9, x19, 4
        add     x10, x21, x4
        add     x11, x8, x7
        and     x12, x9, 0xffffffffffffff80
        csel    x9, x8, x11, lo
        csel    x15, x21, x10, lo
        cmp     x0, 4
        add     x11, x15, x4
        add     x10, x9, x7
        csel    x10, x9, x10, lo
        csel    x22, x15, x11, lo
        ldr     x13, [sp, 632]
        ldr     x14, [sp, 624]
        cmp     x0, 5
        add     x16, x22, x4
        add     x11, x10, x7
        csel    x11, x10, x11, lo
        csel    x16, x22, x16, lo
        and     x20, x19, 0xfffffffffffffff8
        cmp     x0, 6
        add     x0, x12, 0x80
        add     x4, x16, x4
        add     x12, x11, x7
        add     x17, x20, 8
        csel    x12, x12, x11, eq
        csel    x23, x4, x16, eq
        sub     x4, x19, x20
1:
        ldp     q0, q21, [x5]
        ldp     q31, q24, [x5, 32]
        cmp     x2, 8
        add     x5, x5, 64
        b.cc    3f
        stp     q24, q31, [sp, 400]
        stp     q24, q31, [sp, 368]
        mov     v17.16b, v31.16b
        mov     v27.16b, v21.16b
        mov     v18.16b, v31.16b
        mov     v25.16b, v31.16b
        mov     v31.16b, v24.16b
        add     x7, x21, x17
        add     x19, x22, x17
        add     x20, x23, x17
        mov     x24, x5
        mov     x25, x3
        mov     x26, x15
        mov     x27, x16
        stp     q21, q0, [sp, 432]
        mov     v12.16b, v0.16b
        mov     v3.16b, v21.16b
        mov     v16.16b, v0.16b
        mov     v1.16b, v21.16b
        mov     v19.16b, v24.16b
        mov     v20.16b, v0.16b
        mov     v22.16b, v21.16b
        mov     v26.16b, v24.16b
        mov     v21.16b, v0.16b
        mov     v23.16b, v27.16b
        mov     v24.16b, v17.16b
        mov     v29.16b, v31.16b
        mov     x28, x2
2:
        ldr     d5, [x25], 8
        ldr     d4, [x21], 8
        ldr     d11, [x26], 8
        ldr     d10, [x22], 8
        ldr     d9, [x27], 8
        ldr     d8, [x23], 8
        ldp     q6, q7, [x24]
        mov     v30.16b, v29.16b
        ldp     q28, q29, [x24, 32]
        ldr     q13, [sp, 416]
        sdot    v12.4s, v6.16b, v11.4b[0]
        sdot    v3.4s, v7.16b, v11.4b[0]
        sdot    v16.4s, v6.16b, v10.4b[0]
        sdot    v1.4s, v7.16b, v10.4b[0]
        stp     q1, q16, [sp, 336]
        mov     v2.16b, v3.16b
        mov     v3.16b, v12.16b
        ldp     q16, q12, [sp, 384]
        ldr     q1, [sp, 368]
        sdot    v17.4s, v28.16b, v5.4b[0]
        sdot    v31.4s, v29.16b, v5.4b[0]
        sdot    v13.4s, v28.16b, v4.4b[0]
        sdot    v12.4s, v29.16b, v4.4b[0]
        sdot    v16.4s, v28.16b, v11.4b[0]
        sdot    v1.4s, v29.16b, v11.4b[0]
        sdot    v18.4s, v28.16b, v10.4b[0]
        sdot    v19.4s, v29.16b, v10.4b[0]
        sdot    v25.4s, v28.16b, v9.4b[0]
        sdot    v26.4s, v29.16b, v9.4b[0]
        sdot    v24.4s, v28.16b, v8.4b[0]
        sdot    v30.4s, v29.16b, v8.4b[0]
        ldp     q28, q29, [x24, 96]
        ldp     q14, q15, [sp, 432]
        sdot    v0.4s, v6.16b, v5.4b[0]
        sdot    v27.4s, v7.16b, v5.4b[0]
        sdot    v16.4s, v28.16b, v11.4b[1]
        sdot    v1.4s, v29.16b, v11.4b[1]
        sdot    v15.4s, v6.16b, v4.4b[0]
        sdot    v14.4s, v7.16b, v4.4b[0]
        sdot    v20.4s, v6.16b, v9.4b[0]
        sdot    v22.4s, v7.16b, v9.4b[0]
        sdot    v21.4s, v6.16b, v8.4b[0]
        sdot    v23.4s, v7.16b, v8.4b[0]
        ldp     q6, q7, [x24, 64]
        str     q16, [sp, 384]
        str     q1, [sp, 368]
        ldp     q1, q16, [sp, 336]
        sdot    v13.4s, v28.16b, v4.4b[1]
        sdot    v12.4s, v29.16b, v4.4b[1]
        stp     q12, q13, [sp, 400]
        mov     v12.16b, v3.16b
        mov     v3.16b, v2.16b
        sdot    v30.4s, v29.16b, v8.4b[1]
        sub     x28, x28, 8
        sdot    v0.4s, v6.16b, v5.4b[1]
        sdot    v27.4s, v7.16b, v5.4b[1]
        sdot    v17.4s, v28.16b, v5.4b[1]
        sdot    v31.4s, v29.16b, v5.4b[1]
        sdot    v15.4s, v6.16b, v4.4b[1]
        sdot    v14.4s, v7.16b, v4.4b[1]
        sdot    v12.4s, v6.16b, v11.4b[1]
        sdot    v3.4s, v7.16b, v11.4b[1]
        sdot    v16.4s, v6.16b, v10.4b[1]
        sdot    v1.4s, v7.16b, v10.4b[1]
        sdot    v18.4s, v28.16b, v10.4b[1]
        sdot    v19.4s, v29.16b, v10.4b[1]
        sdot    v20.4s, v6.16b, v9.4b[1]
        sdot    v22.4s, v7.16b, v9.4b[1]
        sdot    v25.4s, v28.16b, v9.4b[1]
        sdot    v26.4s, v29.16b, v9.4b[1]
        sdot    v21.4s, v6.16b, v8.4b[1]
        sdot    v23.4s, v7.16b, v8.4b[1]
        sdot    v24.4s, v28.16b, v8.4b[1]
        mov     v29.16b, v30.16b
        cmp     x28, 7
        add     x24, x24, 0x80
        stp     q14, q15, [sp, 432]
        b.hi    2b
        mov     v8.16b, v23.16b
        mov     v23.16b, v21.16b
        mov     v7.16b, v1.16b
        mov     v21.16b, v27.16b
        ldp     q1, q2, [sp, 368]
        ldp     q14, q15, [sp, 400]
        ldp     q27, q6, [sp, 432]
        add     x3, x3, x17
        add     x15, x15, x17
        add     x16, x16, x17
        add     x5, x5, x0
        mov     x24, x4
        mov     v13.16b, v24.16b
        mov     v24.16b, v31.16b
        mov     v31.16b, v17.16b
        mov     v17.16b, v12.16b
        cbz     x24, 4f
        b       5f
3:
        mov     x24, x2
        mov     v29.16b, v24.16b
        mov     v13.16b, v31.16b
        mov     v8.16b, v21.16b
        mov     v23.16b, v0.16b
        mov     v26.16b, v24.16b
        mov     v25.16b, v31.16b
        mov     v22.16b, v21.16b
        mov     v20.16b, v0.16b
        mov     v19.16b, v24.16b
        mov     v18.16b, v31.16b
        mov     v7.16b, v21.16b
        mov     v16.16b, v0.16b
        mov     v1.16b, v24.16b
        mov     v2.16b, v31.16b
        mov     v3.16b, v21.16b
        mov     v17.16b, v0.16b
        mov     v14.16b, v24.16b
        mov     v15.16b, v31.16b
        mov     v27.16b, v21.16b
        mov     v6.16b, v0.16b
        mov     x20, x23
        mov     x19, x22
        mov     x7, x21
        cbnz    x24, 5f
4:
        str     q13, [sp, 64]
        mov     x21, x13
        ld1r    {v28.4s}, [x21], 4
        cmp     x1, 0xf
        sqrdmulh        v30.4s, v0.4s, v28.4s
        str     q30, [sp, 448]
        sqrdmulh        v30.4s, v21.4s, v28.4s
        str     q30, [sp, 432]
        sqrdmulh        v30.4s, v31.4s, v28.4s
        str     q30, [sp, 416]
        sqrdmulh        v30.4s, v24.4s, v28.4s
        str     q30, [sp, 400]
        sqrdmulh        v30.4s, v6.4s, v28.4s
        str     q30, [sp, 384]
        sqrdmulh        v30.4s, v27.4s, v28.4s
        str     q30, [sp, 368]
        sqrdmulh        v30.4s, v15.4s, v28.4s
        str     q30, [sp, 352]
        sqrdmulh        v30.4s, v14.4s, v28.4s
        str     q30, [sp, 336]
        sqrdmulh        v30.4s, v17.4s, v28.4s
        str     q30, [sp, 320]
        sqrdmulh        v30.4s, v3.4s, v28.4s
        str     q30, [sp, 304]
        sqrdmulh        v30.4s, v2.4s, v28.4s
        str     q30, [sp, 288]
        sqrdmulh        v30.4s, v1.4s, v28.4s
        str     q30, [sp, 272]
        sqrdmulh        v30.4s, v16.4s, v28.4s
        str     q30, [sp, 256]
        sqrdmulh        v30.4s, v7.4s, v28.4s
        ld1r    {v9.4s}, [x21]
        str     q30, [sp, 240]
        sqrdmulh        v30.4s, v18.4s, v28.4s
        str     q30, [sp, 224]
        sqrdmulh        v30.4s, v19.4s, v28.4s
        str     q30, [sp, 208]
        sqrdmulh        v30.4s, v20.4s, v28.4s
        sqrdmulh        v12.4s, v25.4s, v28.4s
        str     q30, [sp, 192]
        sqrdmulh        v30.4s, v22.4s, v28.4s
        str     q12, [sp, 160]
        sqrdmulh        v12.4s, v8.4s, v28.4s
        str     q30, [sp, 176]
        mov     v30.16b, v7.16b
        sqrdmulh        v7.4s, v26.4s, v28.4s
        sqrdmulh        v5.4s, v23.4s, v28.4s
        str     q12, [sp, 144]
        sqrdmulh        v12.4s, v13.4s, v28.4s
        sqrdmulh        v4.4s, v29.4s, v28.4s
        cmeq    v28.4s, v9.4s, 0
        str     q12, [sp, 128]
        mov     v12.16b, v15.16b
        mov     v15.16b, v8.16b
        mvn     v8.16b, v28.16b
        and     v28.16b, v0.16b, v8.16b
        str     q28, [sp, 112]
        and     v28.16b, v21.16b, v8.16b
        str     q28, [sp, 96]
        and     v28.16b, v31.16b, v8.16b
        str     q28, [sp, 80]
        mov     v28.16b, v25.16b
        and     v0.16b, v28.16b, v8.16b
        ldr     q28, [sp, 64]
        mov     v21.16b, v29.16b
        and     v29.16b, v24.16b, v8.16b
        str     q29, [sp, 48]
        and     v29.16b, v6.16b, v8.16b
        and     v31.16b, v14.16b, v8.16b
        and     v10.16b, v16.16b, v8.16b
        and     v16.16b, v26.16b, v8.16b
        and     v14.16b, v28.16b, v8.16b
        ldp     q26, q6, [sp, 432]
        ldr     q28, [sp, 112]
        str     q29, [sp, 32]
        and     v24.16b, v27.16b, v8.16b
        and     v29.16b, v12.16b, v8.16b
        ssra    v6.4s, v28.4s, 31
        ldr     q28, [sp, 96]
        str     q31, [sp, 16]
        and     v31.16b, v17.16b, v8.16b
        and     v27.16b, v3.16b, v8.16b
        and     v25.16b, v2.16b, v8.16b
        and     v17.16b, v1.16b, v8.16b
        and     v13.16b, v30.16b, v8.16b
        and     v18.16b, v18.16b, v8.16b
        and     v3.16b, v19.16b, v8.16b
        and     v2.16b, v20.16b, v8.16b
        and     v1.16b, v22.16b, v8.16b
        and     v11.16b, v23.16b, v8.16b
        and     v12.16b, v15.16b, v8.16b
        and     v15.16b, v21.16b, v8.16b
        ssra    v26.4s, v28.4s, 31
        ldp     q8, q23, [sp, 400]
        ldr     q28, [sp, 80]
        ldr     q21, [sp, 384]
        str     q27, [sp]
        ldr     q27, [sp]
        ssra    v23.4s, v28.4s, 31
        ldr     q28, [sp, 48]
        ldp     q19, q20, [sp, 224]
        ssra    v7.4s, v16.4s, 31
        add     x21, x13, 8
        ssra    v8.4s, v28.4s, 31
        ldp     q30, q28, [sp, 16]
        ssra    v19.4s, v18.4s, 31
        srshl   v6.4s, v6.4s, v9.4s
        ssra    v20.4s, v13.4s, 31
        ssra    v21.4s, v28.4s, 31
        ldr     q28, [sp, 368]
        ssra    v5.4s, v11.4s, 31
        ssra    v4.4s, v15.4s, 31
        srshl   v15.4s, v26.4s, v9.4s
        ssra    v28.4s, v24.4s, 31
        ldr     q24, [sp, 352]
        srshl   v26.4s, v28.4s, v9.4s
        sqxtn   v28.4h, v6.4s
        srshl   v19.4s, v19.4s, v9.4s
        ssra    v24.4s, v29.4s, 31
        ldr     q29, [sp, 336]
        srshl   v11.4s, v24.4s, v9.4s
        srshl   v5.4s, v5.4s, v9.4s
        srshl   v7.4s, v7.4s, v9.4s
        ssra    v29.4s, v30.4s, 31
        ldr     q30, [sp, 320]
        srshl   v24.4s, v29.4s, v9.4s
        ssra    v30.4s, v31.4s, 31
        ldr     q31, [sp, 304]
        ssra    v31.4s, v27.4s, 31
        ldr     q27, [sp, 288]
        srshl   v13.4s, v31.4s, v9.4s
        sqxtn   v31.4h, v11.4s
        ssra    v27.4s, v25.4s, 31
        ldp     q22, q25, [sp, 256]
        ssra    v25.4s, v17.4s, 31
        ldp     q17, q18, [sp, 192]
        ssra    v22.4s, v10.4s, 31
        srshl   v10.4s, v25.4s, v9.4s
        srshl   v25.4s, v22.4s, v9.4s
        ssra    v17.4s, v2.4s, 31
        ldr     q2, [sp, 176]
        ssra    v18.4s, v3.4s, 31
        srshl   v3.4s, v21.4s, v9.4s
        srshl   v21.4s, v27.4s, v9.4s
        ssra    v2.4s, v1.4s, 31
        ldr     q1, [sp, 160]
        srshl   v22.4s, v20.4s, v9.4s
        srshl   v20.4s, v17.4s, v9.4s
        srshl   v17.4s, v2.4s, v9.4s
        ssra    v1.4s, v0.4s, 31
        ldp     q0, q16, [sp, 128]
        srshl   v2.4s, v1.4s, v9.4s
        sqxtn   v25.4h, v25.4s
        srshl   v18.4s, v18.4s, v9.4s
        ssra    v16.4s, v12.4s, 31
        srshl   v12.4s, v8.4s, v9.4s
        ld1r    {v8.8h}, [x21]
        ssra    v0.4s, v14.4s, 31
        srshl   v14.4s, v23.4s, v9.4s
        srshl   v23.4s, v30.4s, v9.4s
        sqxtn   v29.4h, v14.4s
        sqxtn   v30.4h, v3.4s
        sqxtn2  v28.8h, v15.4s
        sqxtn2  v29.8h, v12.4s
        sqxtn2  v30.8h, v26.4s
        sqxtn   v27.4h, v23.4s
        sqxtn   v26.4h, v21.4s
        srshl   v1.4s, v16.4s, v9.4s
        srshl   v16.4s, v0.4s, v9.4s
        sqxtn2  v31.8h, v24.4s
        sqxtn2  v27.8h, v13.4s
        sqxtn2  v26.8h, v10.4s
        sqxtn2  v25.8h, v22.4s
        sqxtn   v24.4h, v19.4s
        sqxtn   v23.4h, v20.4s
        sqxtn   v22.4h, v2.4s
        sqxtn   v21.4h, v5.4s
        sqadd   v28.8h, v28.8h, v8.8h
        srshl   v9.4s, v4.4s, v9.4s
        sqxtn2  v24.8h, v18.4s
        sqxtn2  v23.8h, v17.4s
        sqxtn2  v22.8h, v7.4s
        sqxtn2  v21.8h, v1.4s
        sqxtn   v20.4h, v16.4s
        sqadd   v29.8h, v29.8h, v8.8h
        sqadd   v30.8h, v30.8h, v8.8h
        sqadd   v27.8h, v27.8h, v8.8h
        sqxtn   v28.8b, v28.8h
        sqxtn2  v20.8h, v9.4s
        sqadd   v31.8h, v31.8h, v8.8h
        sqadd   v26.8h, v26.8h, v8.8h
        sqadd   v25.8h, v25.8h, v8.8h
        sqadd   v23.8h, v23.8h, v8.8h
        sqadd   v21.8h, v21.8h, v8.8h
        sqxtn2  v28.16b, v29.8h
        sqxtn   v29.8b, v30.8h
        sqxtn   v30.8b, v27.8h
        sqadd   v24.8h, v24.8h, v8.8h
        sqadd   v22.8h, v22.8h, v8.8h
        sqadd   v20.8h, v20.8h, v8.8h
        sqxtn2  v29.16b, v31.8h
        sqxtn2  v30.16b, v26.8h
        sqxtn   v31.8b, v25.8h
        add     x21, x13, 10
        sqxtn   v27.8b, v23.8h
        sqxtn   v26.8b, v21.8h
        sqxtn2  v31.16b, v24.8h
        sqxtn2  v27.16b, v22.8h
        sqxtn2  v26.16b, v20.8h
        ld1r    {v25.16b}, [x21]
        add     x21, x13, 11
        ld1r    {v24.16b}, [x21]
        smax    v28.16b, v28.16b, v25.16b
        smax    v29.16b, v29.16b, v25.16b
        smax    v23.16b, v30.16b, v25.16b
        smax    v31.16b, v31.16b, v25.16b
        smax    v22.16b, v27.16b, v25.16b
        smax    v25.16b, v26.16b, v25.16b
        smin    v30.16b, v28.16b, v24.16b
        smin    v26.16b, v29.16b, v24.16b
        smin    v29.16b, v23.16b, v24.16b
        smin    v27.16b, v31.16b, v24.16b
        smin    v28.16b, v22.16b, v24.16b
        smin    v31.16b, v25.16b, v24.16b
        b.ls    8f
        sub     x23, x20, x2
        sub     x16, x16, x2
        sub     x22, x19, x2
        sub     x15, x15, x2
        sub     x21, x7, x2
        sub     x3, x3, x2
        str     q30, [x6]
        str     q26, [x8]
        str     q29, [x9]
        str     q27, [x10]
        str     q28, [x11]
        str     q31, [x12]
        add     x6, x6, x14
        add     x8, x8, x14
        add     x9, x9, x14
        add     x10, x10, x14
        add     x11, x11, x14
        subs    x1, x1, 0x10
        add     x12, x12, x14
        b.ne    1b
        b       12f
5:
        ldr     d5, [x3]
        ldr     d4, [x7]
        ldr     d11, [x15]
        ldr     d9, [x19]
        str     q29, [sp, 432]
        ldp     q28, q29, [x5]
        mov     v12.16b, v31.16b
        mov     v31.16b, v8.16b
        ldr     d10, [x16]
        ldr     d8, [x20]
        sdot    v0.4s, v28.16b, v5.4b[0]
        sdot    v21.4s, v29.16b, v5.4b[0]
        sdot    v6.4s, v28.16b, v4.4b[0]
        sdot    v27.4s, v29.16b, v4.4b[0]
        sdot    v17.4s, v28.16b, v11.4b[0]
        sdot    v3.4s, v29.16b, v11.4b[0]
        sdot    v16.4s, v28.16b, v9.4b[0]
        sdot    v7.4s, v29.16b, v9.4b[0]
        sdot    v20.4s, v28.16b, v10.4b[0]
        sdot    v23.4s, v28.16b, v8.4b[0]
        sdot    v22.4s, v29.16b, v10.4b[0]
        sdot    v31.4s, v29.16b, v8.4b[0]
        ldp     q28, q29, [x5, 32]
        mov     v30.16b, v13.16b
        add     x3, x3, x24
        add     x7, x7, x24
        sdot    v12.4s, v28.16b, v5.4b[0]
        sdot    v15.4s, v28.16b, v4.4b[0]
        sdot    v2.4s, v28.16b, v11.4b[0]
        sdot    v18.4s, v28.16b, v9.4b[0]
        sdot    v25.4s, v28.16b, v10.4b[0]
        sdot    v30.4s, v28.16b, v8.4b[0]
        ldr     q28, [sp, 432]
        sdot    v24.4s, v29.16b, v5.4b[0]
        sdot    v14.4s, v29.16b, v4.4b[0]
        sdot    v1.4s, v29.16b, v11.4b[0]
        sdot    v28.4s, v29.16b, v8.4b[0]
        str     q28, [sp, 432]
        sdot    v19.4s, v29.16b, v9.4b[0]
        sdot    v26.4s, v29.16b, v10.4b[0]
        ldr     q29, [sp, 432]
        add     x15, x15, x24
        add     x19, x19, x24
        add     x16, x16, x24
        cmp     x24, 5
        add     x20, x20, x24
        b.cc    6f
        ldp     q28, q29, [x5, 64]
        str     q6, [sp, 448]
        mov     v13.16b, v7.16b
        ldp     q6, q7, [x5, 96]
        sdot    v0.4s, v28.16b, v5.4b[1]
        sdot    v21.4s, v29.16b, v5.4b[1]
        sdot    v27.4s, v29.16b, v4.4b[1]
        sdot    v12.4s, v6.16b, v5.4b[1]
        sdot    v24.4s, v7.16b, v5.4b[1]
        ldr     q5, [sp, 448]
        sdot    v15.4s, v6.16b, v4.4b[1]
        sdot    v3.4s, v29.16b, v11.4b[1]
        sdot    v2.4s, v6.16b, v11.4b[1]
        sdot    v5.4s, v28.16b, v4.4b[1]
        str     q5, [sp, 448]
        sdot    v13.4s, v29.16b, v9.4b[1]
        sdot    v18.4s, v6.16b, v9.4b[1]
        sdot    v22.4s, v29.16b, v10.4b[1]
        sdot    v25.4s, v6.16b, v10.4b[1]
        sdot    v31.4s, v29.16b, v8.4b[1]
        sdot    v30.4s, v6.16b, v8.4b[1]
        ldp     q29, q6, [sp, 432]
        sdot    v14.4s, v7.16b, v4.4b[1]
        sdot    v17.4s, v28.16b, v11.4b[1]
        sdot    v1.4s, v7.16b, v11.4b[1]
        sdot    v16.4s, v28.16b, v9.4b[1]
        sdot    v19.4s, v7.16b, v9.4b[1]
        sdot    v20.4s, v28.16b, v10.4b[1]
        sdot    v26.4s, v7.16b, v10.4b[1]
        sdot    v23.4s, v28.16b, v8.4b[1]
        sdot    v29.4s, v7.16b, v8.4b[1]
        mov     v7.16b, v13.16b
        add     x5, x5, 0x80
        b       7f
6:
        add     x5, x5, 64
7:
        mov     v13.16b, v30.16b
        mov     v8.16b, v31.16b
        mov     v31.16b, v12.16b
        b       4b
8:
        tbz     w1, 3, 9f
        str     d30, [x6], 8
        str     d26, [x8], 8
        str     d29, [x9], 8
        str     d27, [x10], 8
        str     d28, [x11], 8
        str     d31, [x12], 8
        ext     v30.16b, v30.16b, v30.16b, 8
        ext     v26.16b, v26.16b, v26.16b, 8
        ext     v29.16b, v29.16b, v29.16b, 8
        ext     v27.16b, v27.16b, v27.16b, 8
        ext     v28.16b, v28.16b, v28.16b, 8
        ext     v31.16b, v31.16b, v31.16b, 8
9:
        mov     v30.d[1], v26.d[0]
        mov     v29.d[1], v27.d[0]
        mov     v28.d[1], v31.d[0]
        tbnz    w1, 2, 13f
        tbnz    w1, 1, 14f
10:
        tbz     w1, 0, 12f
11:
        st1     {v30.b}[0], [x6]
        st1     {v30.b}[8], [x8]
        st1     {v29.b}[0], [x9]
        st1     {v29.b}[8], [x10]
        st1     {v28.b}[0], [x11]
        st1     {v28.b}[8], [x12]
12:
        add     sp, sp, 0x1d0
        ldp     x19, x29, [sp, 144]
        ldp     x21, x20, [sp, 128]
        ldp     x23, x22, [sp, 112]
        ldp     x25, x24, [sp, 96]
        ldp     x27, x26, [sp, 80]
        ldr     x28, [sp, 64]
        ldp     d9, d8, [sp, 48]
        ldp     d11, d10, [sp, 32]
        ldp     d5, d4, [sp, 16]
        ldp     d7, d6, [sp], 160
        ret
13:
        st1     {v30.s}[0], [x6], 4
        st1     {v30.s}[2], [x8], 4
        st1     {v29.s}[0], [x9], 4
        st1     {v29.s}[2], [x10], 4
        st1     {v28.s}[0], [x11], 4
        st1     {v28.s}[2], [x12], 4
        ext     v30.16b, v30.16b, v30.16b, 4
        ext     v29.16b, v29.16b, v29.16b, 4
        ext     v28.16b, v28.16b, v28.16b, 4
        tbz     w1, 1, 10b
14:
        st1     {v30.h}[0], [x6], 2
        st1     {v30.h}[4], [x8], 2
        st1     {v29.h}[0], [x9], 2
        st1     {v29.h}[4], [x10], 2
        st1     {v28.h}[0], [x11], 2
        st1     {v28.h}[4], [x12], 2
        ext     v30.16b, v30.16b, v30.16b, 2
        ext     v29.16b, v29.16b, v29.16b, 2
        ext     v28.16b, v28.16b, v28.16b, 2
        tbnz    w1, 0, 11b
        b       12b

END_FUNCTION xnn_qs8_gemm_minmax_ukernel_6x16c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif
