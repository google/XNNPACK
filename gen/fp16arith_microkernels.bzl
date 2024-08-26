"""
Microkernel filenames lists for fp16arith.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""

PROD_FP16ARITH_MICROKERNEL_SRCS = [
    "src/f16-vbinary/gen/f16-vdiv-minmax-fp16arith-u2.c",
    "src/f16-vbinary/gen/f16-vdivc-minmax-fp16arith-u2.c",
    "src/f16-vbinary/gen/f16-vrdivc-minmax-fp16arith-u2.c",
]

NON_PROD_FP16ARITH_MICROKERNEL_SRCS = [
    "src/f16-vbinary/gen/f16-vadd-minmax-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vadd-minmax-fp16arith-u2.c",
    "src/f16-vbinary/gen/f16-vadd-minmax-fp16arith-u4.c",
    "src/f16-vbinary/gen/f16-vaddc-minmax-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vaddc-minmax-fp16arith-u2.c",
    "src/f16-vbinary/gen/f16-vaddc-minmax-fp16arith-u4.c",
    "src/f16-vbinary/gen/f16-vdiv-minmax-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vdiv-minmax-fp16arith-u4.c",
    "src/f16-vbinary/gen/f16-vdivc-minmax-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vdivc-minmax-fp16arith-u4.c",
    "src/f16-vbinary/gen/f16-vmax-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vmax-fp16arith-u2.c",
    "src/f16-vbinary/gen/f16-vmax-fp16arith-u4.c",
    "src/f16-vbinary/gen/f16-vmaxc-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vmaxc-fp16arith-u2.c",
    "src/f16-vbinary/gen/f16-vmaxc-fp16arith-u4.c",
    "src/f16-vbinary/gen/f16-vmin-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vmin-fp16arith-u2.c",
    "src/f16-vbinary/gen/f16-vmin-fp16arith-u4.c",
    "src/f16-vbinary/gen/f16-vminc-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vminc-fp16arith-u2.c",
    "src/f16-vbinary/gen/f16-vminc-fp16arith-u4.c",
    "src/f16-vbinary/gen/f16-vmul-minmax-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vmul-minmax-fp16arith-u2.c",
    "src/f16-vbinary/gen/f16-vmul-minmax-fp16arith-u4.c",
    "src/f16-vbinary/gen/f16-vmulc-minmax-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vmulc-minmax-fp16arith-u2.c",
    "src/f16-vbinary/gen/f16-vmulc-minmax-fp16arith-u4.c",
    "src/f16-vbinary/gen/f16-vrdivc-minmax-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vrdivc-minmax-fp16arith-u4.c",
    "src/f16-vbinary/gen/f16-vrsubc-minmax-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vrsubc-minmax-fp16arith-u2.c",
    "src/f16-vbinary/gen/f16-vrsubc-minmax-fp16arith-u4.c",
    "src/f16-vbinary/gen/f16-vsqrdiff-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vsqrdiff-fp16arith-u2.c",
    "src/f16-vbinary/gen/f16-vsqrdiff-fp16arith-u4.c",
    "src/f16-vbinary/gen/f16-vsqrdiffc-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vsqrdiffc-fp16arith-u2.c",
    "src/f16-vbinary/gen/f16-vsqrdiffc-fp16arith-u4.c",
    "src/f16-vbinary/gen/f16-vsub-minmax-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vsub-minmax-fp16arith-u2.c",
    "src/f16-vbinary/gen/f16-vsub-minmax-fp16arith-u4.c",
    "src/f16-vbinary/gen/f16-vsubc-minmax-fp16arith-u1.c",
    "src/f16-vbinary/gen/f16-vsubc-minmax-fp16arith-u2.c",
    "src/f16-vbinary/gen/f16-vsubc-minmax-fp16arith-u4.c",
    "src/f16-vsqrt/gen/f16-vsqrt-fp16arith-sqrt-u1.c",
    "src/f16-vsqrt/gen/f16-vsqrt-fp16arith-sqrt-u2.c",
    "src/f16-vsqrt/gen/f16-vsqrt-fp16arith-sqrt-u4.c",
]

ALL_FP16ARITH_MICROKERNEL_SRCS = PROD_FP16ARITH_MICROKERNEL_SRCS + NON_PROD_FP16ARITH_MICROKERNEL_SRCS
