"""
Microkernel filenames lists for neonsme.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""

PROD_NEONSME_MICROKERNEL_SRCS = [
    "src/pf32-gemm/pf32-gemm-1x32-minmax-neonsme.c",
    "src/pf32-gemm/pf32-gemm-32x32-minmax-neonsme.c",
    "src/x16-pack-lh/x16-packlh-igemm-neonsme.c",
    "src/x16-pack-lh/x16-packlh-neonsme.c",
    "src/x32-pack-lh/x32-packlh-neonsme.c",
]

NON_PROD_NEONSME_MICROKERNEL_SRCS = [
    "src/pf16-f16-f16-igemm/pf16-f16-f16-igemm-32x32c2-minmax-neonsme.c",
]

ALL_NEONSME_MICROKERNEL_SRCS = PROD_NEONSME_MICROKERNEL_SRCS + NON_PROD_NEONSME_MICROKERNEL_SRCS
