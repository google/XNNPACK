#
# Microkernel filenames lists for sme2.
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py
#

PROD_SME2_MICROKERNEL_SRCS = [
    "src/qp8-f32-qc4w-gemm/qp8-f32-qc4w-gemm-1x32c4-minmax-sme2.c",
    "src/qp8-f32-qc4w-gemm/qp8-f32-qc4w-gemm-32x32c4-minmax-sme2.c",
    "src/qp8-f32-qc8w-gemm/qp8-f32-qc8w-gemm-1x32c4-minmax-sme2.c",
    "src/qp8-f32-qc8w-gemm/qp8-f32-qc8w-gemm-32x32c4-minmax-sme2.c",
    "src/x8-packq/x8-packq-f32qp8-sme2.c",
]

NON_PROD_SME2_MICROKERNEL_SRCS = [
]

ALL_SME2_MICROKERNEL_SRCS = PROD_SME2_MICROKERNEL_SRCS + NON_PROD_SME2_MICROKERNEL_SRCS
