#
# Microkernel filenames lists for sve2.
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py
#

PROD_SVE2_MICROKERNEL_SRCS = [
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-sve2-lane-ld128.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-8x8-minmax-sve2-lane-ld128.c",
]

NON_PROD_SVE2_MICROKERNEL_SRCS = [
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x16-minmax-sve2-lane-ld128.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-minmax-sve2-lane-ld128.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x16-minmax-sve2-lane-ld128.c",
    "src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-minmax-sve2-lane-ld128.c",
]

ALL_SVE2_MICROKERNEL_SRCS = PROD_SVE2_MICROKERNEL_SRCS + NON_PROD_SVE2_MICROKERNEL_SRCS
