#
# Microkernel filenames lists for avx512bf16.
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py
#

PROD_AVX512BF16_MICROKERNEL_SRCS = [
    "src/bf16-vbinary/gen/bf16-vadd-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vaddc-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vdiv-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vdivc-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vmax-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vmaxc-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vmin-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vminc-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vmul-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vmulc-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vprelu-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vpreluc-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vrdivc-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vrpreluc-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vrsubc-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vsqrdiff-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vsqrdiffc-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vsub-avx512bf16-u32.c",
    "src/bf16-vbinary/gen/bf16-vsubc-avx512bf16-u32.c",
    "src/f32-bf16-vcvt/gen/f32-bf16-vcvt-avx512bf16-u16.c",
]

NON_PROD_AVX512BF16_MICROKERNEL_SRCS = [
    "src/bf16-vbinary/gen/bf16-vadd-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vaddc-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vdiv-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vdivc-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vmax-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vmaxc-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vmin-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vminc-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vmul-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vmulc-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vprelu-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vpreluc-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vrdivc-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vrpreluc-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vrsubc-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vsqrdiff-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vsqrdiffc-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vsub-avx512bf16-u16.c",
    "src/bf16-vbinary/gen/bf16-vsubc-avx512bf16-u16.c",
    "src/f32-bf16-vcvt/gen/f32-bf16-vcvt-avx512bf16-u32.c",
]

ALL_AVX512BF16_MICROKERNEL_SRCS = PROD_AVX512BF16_MICROKERNEL_SRCS + NON_PROD_AVX512BF16_MICROKERNEL_SRCS
