#
# Microkernel filenames lists for hmx.
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py
#

PROD_HMX_MICROKERNEL_SRCS = [
    "src/f16-gemm/gen/f16-gemm-32x64c2-minmax-hmx-pingpong.c",
]

NON_PROD_HMX_MICROKERNEL_SRCS = [
    "src/f16-gemm/gen/f16-gemm-32x64c2-minmax-hmx.c",
]

ALL_HMX_MICROKERNEL_SRCS = PROD_HMX_MICROKERNEL_SRCS + NON_PROD_HMX_MICROKERNEL_SRCS
