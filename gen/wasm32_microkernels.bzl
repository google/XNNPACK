"""
Microkernel filenames lists for wasm32.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""

PROD_WASM32_ASM_MICROKERNEL_SRCS = [
]

NON_PROD_WASM32_ASM_MICROKERNEL_SRCS = [
    "src/f32-vrelu/f32-vrelu-asm-wasm32-shr-u1.S",
    "src/f32-vrelu/f32-vrelu-asm-wasm32-shr-u2.S",
    "src/f32-vrelu/f32-vrelu-asm-wasm32-shr-u4.S",
]

WASM32_ASM_MICROKERNEL_SRCS = PROD_WASM32_ASM_MICROKERNEL_SRCS + NON_PROD_WASM32_ASM_MICROKERNEL_SRCS
