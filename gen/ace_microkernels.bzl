#
# Microkernel filenames lists for ace.
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py
#

# buildifier: disable=module-docstring
PROD_ACE_MICROKERNEL_SRCS = [
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x32c4-minmax-ace.c",
    "src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-16x32c4-minmax-ace.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x32c4-minmax-ace.c",
    "src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-16x32c4-minmax-ace.c",
    "src/qp8-f32-qc8w-gemm/qp8-f32-qc8w-gemm-minmax-1x32c4-ace.c",
    "src/qp8-f32-qc8w-gemm/qp8-f32-qc8w-gemm-minmax-16x32c4-ace.c",
    "src/x8-packq/x8-packq-ace-f32qp8-u16.c",
]

NON_PROD_ACE_MICROKERNEL_SRCS = [
]

ALL_ACE_MICROKERNEL_SRCS = PROD_ACE_MICROKERNEL_SRCS + NON_PROD_ACE_MICROKERNEL_SRCS
