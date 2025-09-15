"""
Microkernel filenames lists for neonsme.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""

PROD_NEONSME_MICROKERNEL_SRCS = [
    "src/pf32-gemm/pf32-gemm-1x32-minmax-neonsme.c",
    "src/pf32-gemm/pf32-gemm-32x32-minmax-neonsme.c",
    "src/pqs8-f32-qc8w-igemm/pqs8-f32-qc8w-igemm-32x32c4-minmax-neonsme.c",
    "src/pqs8-qc8w-gemm/pqs8-qc8w-gemm-32x32c4-minmax-neonsme.c",
    "src/x32-pack-lh/x32-packlh-neonsme.c",
    "src/x8-pack-lh/x8-packlh-igemm-neonsme.c",
    "src/x8-pack-lh/x8-packlh-neonsme.c",
]

NON_PROD_NEONSME_MICROKERNEL_SRCS = [
]

ALL_NEONSME_MICROKERNEL_SRCS = PROD_NEONSME_MICROKERNEL_SRCS + NON_PROD_NEONSME_MICROKERNEL_SRCS
