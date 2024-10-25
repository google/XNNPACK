"""
Microkernel filenames lists for ssse3.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""

PROD_SSSE3_MICROKERNEL_SRCS = [
    "src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-2x4-acc2.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-ssse3-madd.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-ssse3-madd.c",
    "src/qs8-rsum/gen/qs8-rsum-ssse3-u32-acc2.c",
    "src/qs8-vcvt/gen/qs8-vcvt-ssse3-u32.c",
    "src/qs8-vlrelu/gen/qs8-vlrelu-ssse3-u32.c",
    "src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-ssse3-u16.c",
    "src/qu8-rdsum/gen/qu8-rdsum-7p7x-ssse3-c64.c",
    "src/qu8-vcvt/gen/qu8-vcvt-ssse3-u32.c",
    "src/qu8-vlrelu/gen/qu8-vlrelu-ssse3-u32.c",
    "src/x24-transposec/x24-transposec-4x4-ssse3.c",
]

NON_PROD_SSSE3_MICROKERNEL_SRCS = [
    "src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-1x4-acc2.c",
    "src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-1x4-acc3.c",
    "src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-1x4-acc4.c",
    "src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-1x4.c",
    "src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-2x4.c",
    "src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-3x4.c",
    "src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-4x4.c",
    "src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-5x4.c",
    "src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-6x4.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x4c8-minmax-ssse3-madd-prfm.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-ssse3-madd-prfm.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x4c8-minmax-ssse3-madd.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-ssse3-madd-prfm.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x4c8-minmax-ssse3-madd.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x4c8-minmax-ssse3-madd-prfm.c",
    "src/qs8-rsum/gen/qs8-rsum-ssse3-u16.c",
    "src/qs8-rsum/gen/qs8-rsum-ssse3-u64-acc2.c",
    "src/qs8-rsum/gen/qs8-rsum-ssse3-u64-acc4.c",
    "src/qs8-vcvt/gen/qs8-vcvt-ssse3-u16.c",
    "src/qs8-vlrelu/gen/qs8-vlrelu-ssse3-u16.c",
    "src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-ssse3-u4.c",
    "src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-ssse3-u8.c",
    "src/qu8-rdsum/gen/qu8-rdsum-7p7x-ssse3-c16.c",
    "src/qu8-rdsum/gen/qu8-rdsum-7p7x-ssse3-c32.c",
    "src/qu8-vcvt/gen/qu8-vcvt-ssse3-u16.c",
    "src/qu8-vlrelu/gen/qu8-vlrelu-ssse3-u16.c",
    "src/x8-lut/gen/x8-lut-ssse3-u16.c",
    "src/x8-lut/gen/x8-lut-ssse3-u32.c",
]

ALL_SSSE3_MICROKERNEL_SRCS = PROD_SSSE3_MICROKERNEL_SRCS + NON_PROD_SSSE3_MICROKERNEL_SRCS
