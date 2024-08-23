"""
Microkernel filenames lists for avx512vbmi.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""

PROD_AVX512VBMI_MICROKERNEL_SRCS = [
    "src/x8-lut/gen/x8-lut-avx512vbmi-vpermx2b-u128.c",
]

NON_PROD_AVX512VBMI_MICROKERNEL_SRCS = [
    "src/x8-lut/gen/x8-lut-avx512vbmi-vpermx2b-u64.c",
    "src/x8-lut/gen/x8-lut-avx512vbmi-vpermx2b-u192.c",
    "src/x8-lut/gen/x8-lut-avx512vbmi-vpermx2b-u256.c",
]

ALL_AVX512VBMI_MICROKERNEL_SRCS = PROD_AVX512VBMI_MICROKERNEL_SRCS + NON_PROD_AVX512VBMI_MICROKERNEL_SRCS
