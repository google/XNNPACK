"""
Microkernel filenames lists for sse2fma.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""

PROD_SSE2FMA_MICROKERNEL_SRCS = [
    "src/f32-vapproxgelu/gen/f32-vapproxgelu-sse2fma-rational-12-10-div.c",
    "src/f32-vcos/gen/f32-vcos-sse2fma-rational-5-4-div.c",
    "src/f32-vgelu/gen/f32-vgelu-sse2fma-rational-12-10-div.c",
    "src/f32-vsin/gen/f32-vsin-sse2fma-rational-5-4-div.c",
]

NON_PROD_SSE2FMA_MICROKERNEL_SRCS = [
    "src/f32-vexp/gen/f32-vexp-sse2fma-rational-3-2-div.c",
]

ALL_SSE2FMA_MICROKERNEL_SRCS = PROD_SSE2FMA_MICROKERNEL_SRCS + NON_PROD_SSE2FMA_MICROKERNEL_SRCS
