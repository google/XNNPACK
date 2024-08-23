"""
Microkernel filenames lists for neonfp16.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""

PROD_NEONFP16_MICROKERNEL_SRCS = [
    "src/f16-f32-vcvt/gen/f16-f32-vcvt-neonfp16-u16.c",
    "src/f32-f16-vcvt/gen/f32-f16-vcvt-neonfp16-u16.c",
]

NON_PROD_NEONFP16_MICROKERNEL_SRCS = [
    "src/f16-f32-vcvt/gen/f16-f32-vcvt-neonfp16-u8.c",
    "src/f32-f16-vcvt/gen/f32-f16-vcvt-neonfp16-u8.c",
]

ALL_NEONFP16_MICROKERNEL_SRCS = PROD_NEONFP16_MICROKERNEL_SRCS + NON_PROD_NEONFP16_MICROKERNEL_SRCS
