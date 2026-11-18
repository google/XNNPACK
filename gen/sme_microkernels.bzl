#
# Microkernel filenames lists for sme.
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py
#

# buildifier: disable=module-docstring
PROD_SME_MICROKERNEL_SRCS = [
]

NON_PROD_SME_MICROKERNEL_SRCS = [
    "src/f32-dwconv/f32-dwconv-9p1vc-minmax-sme.c",
    "src/f32-dwconv/f32-dwconv-9p4xvc-minmax-sme.c",
    "src/pf32-qc4w-gemm/pf32-qc4w-gemm-1x32-minmax-sme.c",
    "src/pf32-qc4w-gemm/pf32-qc4w-gemm-32x32-minmax-sme.c",
    "src/pf32-qc8w-gemm/pf32-qc8w-gemm-1x32-minmax-sme.c",
    "src/pf32-qc8w-gemm/pf32-qc8w-gemm-32x32-minmax-sme.c",
    "src/x8-packq/x8-packq-f32qp8-sme.c",
    "src/x32-pack-lh/x32-packlh-sme.c",
]

ALL_SME_MICROKERNEL_SRCS = PROD_SME_MICROKERNEL_SRCS + NON_PROD_SME_MICROKERNEL_SRCS
