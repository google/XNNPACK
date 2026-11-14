#
# Microkernel filenames lists for hmx.
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py
#

"""Microkernel filenames lists for hmx."""

PROD_HMX_MICROKERNEL_SRCS = [
    "src/f16-gemm/gen/f16-gemm-32x64c2-minmax-hmx.c",
    "src/pqs8-f32-qc8w-igemm/pqs8-f32-qc8w-igemm-32x64c4-minmax-hmx.c",
    "src/pqs8-qc4w-gemm/gen/pqs8-qc4w-gemm-32x64c4-minmax-fp32-hmx.c",
    "src/pqs8-qc8w-gemm/gen/pqs8-qc8w-gemm-32x64c4-minmax-hmx.c",
    "src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-32x64c4-minmax-hmx.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-32x64c4-minmax-hmx.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-32x64c4-minmax-hmx.c",
    "src/qs8-qc4w-gemm/gen/qs8-qc4w-gemm-32x64c4-minmax-fp32-hmx.c",
    "src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-32x64c4-minmax-fp32-hmx.c",
    "src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-32x64c4-minmax-fp32-hmx.c",
]

NON_PROD_HMX_MICROKERNEL_SRCS = [
]

ALL_HMX_MICROKERNEL_SRCS = PROD_HMX_MICROKERNEL_SRCS + NON_PROD_HMX_MICROKERNEL_SRCS
