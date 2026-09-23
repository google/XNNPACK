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
    "src/bf16-f32-igemm/gen/bf16-f32-igemm-1x16c2-minmax-avx512bf16-broadcast.c",
    "src/bf16-f32-igemm/gen/bf16-f32-igemm-1x32c2-minmax-avx512bf16-broadcast.c",
    "src/bf16-f32-igemm/gen/bf16-f32-igemm-4x16c2-minmax-avx512bf16-broadcast.c",
    "src/bf16-f32-igemm/gen/bf16-f32-igemm-4x32c2-minmax-avx512bf16-broadcast.c",
    "src/bf16-f32-igemm/gen/bf16-f32-igemm-5x16c2-minmax-avx512bf16-broadcast.c",
    "src/bf16-f32-igemm/gen/bf16-f32-igemm-5x32c2-minmax-avx512bf16-broadcast.c",
    "src/bf16-f32-igemm/gen/bf16-f32-igemm-6x16c2-minmax-avx512bf16-broadcast.c",
    "src/bf16-f32-igemm/gen/bf16-f32-igemm-6x32c2-minmax-avx512bf16-broadcast.c",
    "src/bf16-f32-igemm/gen/bf16-f32-igemm-7x16c2-minmax-avx512bf16-broadcast.c",
    "src/bf16-f32-igemm/gen/bf16-f32-igemm-7x32c2-minmax-avx512bf16-broadcast.c",
    "src/bf16-f32-igemm/gen/bf16-f32-igemm-8x16c2-minmax-avx512bf16-broadcast.c",
    "src/bf16-f32-igemm/gen/bf16-f32-igemm-8x32c2-minmax-avx512bf16-broadcast.c",
    "src/f32-bf16-vcvt/gen/f32-bf16-vcvt-avx512bf16-u32.c",
]

ALL_AVX512BF16_MICROKERNEL_SRCS = PROD_AVX512BF16_MICROKERNEL_SRCS + NON_PROD_AVX512BF16_MICROKERNEL_SRCS
