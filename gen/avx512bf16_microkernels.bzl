#
# Microkernel filenames lists for avx512bf16.
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py
#

PROD_AVX512BF16_MICROKERNEL_SRCS = [
    "src/f32-bf16-vcvt/gen/f32-bf16-vcvt-avx512bf16-u16.c",
]

NON_PROD_AVX512BF16_MICROKERNEL_SRCS = [
    "src/f32-bf16-vcvt/gen/f32-bf16-vcvt-avx512bf16-u32.c",
]

ALL_AVX512BF16_MICROKERNEL_SRCS = PROD_AVX512BF16_MICROKERNEL_SRCS + NON_PROD_AVX512BF16_MICROKERNEL_SRCS
