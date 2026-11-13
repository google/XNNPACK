#
# Microkernel filenames lists for sme2.
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py
#

# buildifier: disable=module-docstring
PROD_SME2_MICROKERNEL_SRCS = [
    "src/x8-packq/x8-packq-f32qp8-sme2.c",
]

NON_PROD_SME2_MICROKERNEL_SRCS = [
    "src/pf16-gemm/pf16-gemm-1x32c2-minmax-sme2.c",
    "src/pf16-gemm/pf16-gemm-32x32c2-minmax-sme2.c",
    "src/pf32-gemm/pf32-gemm-1x32-minmax-sme2.c",
    "src/pf32-gemm/pf32-gemm-32x32-minmax-sme2.c",
    "src/pf32-qc4w-gemm/pf32-qc4w-gemm-1x32-minmax-sme2.c",
    "src/pf32-qc4w-gemm/pf32-qc4w-gemm-32x32-minmax-sme2.c",
    "src/pf32-qc8w-gemm/pf32-qc8w-gemm-1x32-minmax-sme2.c",
    "src/pf32-qc8w-gemm/pf32-qc8w-gemm-32x32-minmax-sme2.c",
    "src/qp8-f32-qc4w-gemm/qp8-f32-qc4w-gemm-1x32c4-minmax-sme2.c",
    "src/qp8-f32-qc4w-gemm/qp8-f32-qc4w-gemm-32x32c4-minmax-sme2.c",
    "src/qp8-f32-qc8w-gemm/qp8-f32-qc8w-gemm-1x32c4-minmax-sme2.c",
    "src/qp8-f32-qc8w-gemm/qp8-f32-qc8w-gemm-32x32c4-minmax-sme2.c",
    "src/x16-pack-lh/x16-packlh-sme2.c",
    "src/x32-pack-lh/x32-packlh-sme2.c",
]

ALL_SME2_MICROKERNEL_SRCS = PROD_SME2_MICROKERNEL_SRCS + NON_PROD_SME2_MICROKERNEL_SRCS
