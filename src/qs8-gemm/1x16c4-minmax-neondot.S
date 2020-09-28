// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/assembly.h>

BEGIN_FUNCTION xnn_qs8_gemm_minmax_ukernel_1x16c4__aarch64_neondot
        ldp     x9, x8, [sp]
        sub     x12, x2, 8
        lsl     x10, x12, 4
        and     x13, x12, 0xfffffffffffffff8
        and     x11, x10, 0xffffffffffffff80
        add     x10, x13, 8
        add     x11, x11, 0x80
        sub     x12, x12, x13
1:
        ldp     q28, q29, [x5]
        ldp     q30, q31, [x5, 32]
        cmp     x2, 8
        add     x5, x5, 64
        b.cc    3f
        add     x13, x3, x10
        mov     x14, x5
        mov     x15, x2
2:
        ldr     d27, [x3], 8
        ldp     q26, q25, [x14]
        ldp     q24, q23, [x14, 32]
        ldp     q22, q21, [x14, 64]
        ldp     q20, q19, [x14, 96]
        sub     x15, x15, 8
        sdot    v28.4s, v26.16b, v27.4b[0]
        sdot    v29.4s, v25.16b, v27.4b[0]
        sdot    v30.4s, v24.16b, v27.4b[0]
        sdot    v31.4s, v23.16b, v27.4b[0]
        cmp     x15, 7
        sdot    v28.4s, v22.16b, v27.4b[1]
        sdot    v29.4s, v21.16b, v27.4b[1]
        sdot    v30.4s, v20.16b, v27.4b[1]
        sdot    v31.4s, v19.16b, v27.4b[1]
        add     x14, x14, 0x80
        b.hi    2b
        add     x5, x5, x11
        mov     x14, x12
        cbz     x14, 4f
        b       5f
3:
        mov     x14, x2
        mov     x13, x3
        cbnz    x14, 5f
4:
        mov     x14, x8
        ld1r    {v27.4s}, [x14], 4
        cmp     x1, 0xf
        ld1r    {v26.4s}, [x14]
        add     x14, x8, 8
        sqrdmulh        v25.4s, v28.4s, v27.4s
        sqrdmulh        v24.4s, v29.4s, v27.4s
        cmeq    v22.4s, v26.4s, 0
        mvn     v22.16b, v22.16b
        and     v28.16b, v28.16b, v22.16b
        sqrdmulh        v23.4s, v30.4s, v27.4s
        sqrdmulh        v27.4s, v31.4s, v27.4s
        and     v29.16b, v29.16b, v22.16b
        and     v30.16b, v30.16b, v22.16b
        and     v31.16b, v31.16b, v22.16b
        ld1r    {v22.8h}, [x14]
        ssra    v25.4s, v28.4s, 31
        ssra    v24.4s, v29.4s, 31
        ssra    v23.4s, v30.4s, 31
        srshl   v28.4s, v25.4s, v26.4s
        ssra    v27.4s, v31.4s, 31
        srshl   v29.4s, v24.4s, v26.4s
        srshl   v30.4s, v23.4s, v26.4s
        sqxtn   v28.4h, v28.4s
        srshl   v31.4s, v27.4s, v26.4s
        sqxtn   v30.4h, v30.4s
        sqxtn2  v28.8h, v29.4s
        sqxtn2  v30.8h, v31.4s
        sqadd   v28.8h, v28.8h, v22.8h
        add     x14, x8, 10
        sqadd   v29.8h, v30.8h, v22.8h
        sqxtn   v28.8b, v28.8h
        sqxtn2  v28.16b, v29.8h
        ld1r    {v29.16b}, [x14]
        add     x14, x8, 11
        ld1r    {v30.16b}, [x14]
        smax    v28.16b, v28.16b, v29.16b
        smin    v28.16b, v28.16b, v30.16b
        b.ls    7f
        sub     x3, x13, x2
        str     q28, [x6]
        subs    x1, x1, 0x10
        add     x6, x6, x9
        b.ne    1b
        b       10f
5:
        ldr     d27, [x13]
        ldp     q26, q25, [x5]
        ldp     q24, q23, [x5, 32]
        add     x13, x13, x14
        cmp     x14, 5
        sdot    v28.4s, v26.16b, v27.4b[0]
        sdot    v29.4s, v25.16b, v27.4b[0]
        sdot    v30.4s, v24.16b, v27.4b[0]
        sdot    v31.4s, v23.16b, v27.4b[0]
        b.cc    6f
        ldp     q26, q25, [x5, 64]
        ldp     q24, q23, [x5, 96]
        add     x5, x5, 0x80
        sdot    v28.4s, v26.16b, v27.4b[1]
        sdot    v29.4s, v25.16b, v27.4b[1]
        sdot    v30.4s, v24.16b, v27.4b[1]
        sdot    v31.4s, v23.16b, v27.4b[1]
        b       4b
6:
        add     x5, x5, 64
        b       4b
7:
        tbnz    w1, 3, 11f
        tbnz    w1, 2, 12f
8:
        tbnz    w1, 1, 13f
9:
        tbnz    w1, 0, 14f
10:
        ret
11:
        str     d28, [x6], 8
        ext     v28.16b, v28.16b, v28.16b, 8
        tbz     w1, 2, 8b
12:
        st1     {v28.s}[0], [x6], 4
        ext     v28.8b, v28.8b, v28.8b, 4
        tbz     w1, 1, 9b
13:
        st1     {v28.h}[0], [x6], 2
        ext     v28.8b, v28.8b, v28.8b, 2
        tbz     w1, 0, 10b
14:
        st1     {v28.b}[0], [x6]
        ret

END_FUNCTION xnn_qs8_gemm_minmax_ukernel_1x16c4__aarch64_neondot

#ifdef __ELF__
.section ".note.GNU-stack","",%progbits
#endif

