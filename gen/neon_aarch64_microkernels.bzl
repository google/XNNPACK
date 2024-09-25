"""
Microkernel filenames lists for neon_aarch64.

Auto-generated file. Do not edit!
  Generator: tools/update-microkernels.py
"""

PROD_NEON_AARCH64_MICROKERNEL_SRCS = [
    "src/f32-vbinary/gen/f32-vdiv-aarch64-neon-u8.c",
    "src/f32-vbinary/gen/f32-vdivc-aarch64-neon-u8.c",
    "src/f32-vbinary/gen/f32-vrdivc-aarch64-neon-u8.c",
    "src/f32-vsqrt/gen/f32-vsqrt-aarch64-neon-sqrt-u4.c",
    "src/x8-lut/gen/x8-lut-aarch64-neon-tbx128x4-u64.c",
    "src/x8-packq/x8-packq-aarch64-neon-f32qp8-u2.c",
    "src/x24-transposec/x24-transposec-4x4-aarch64-neon-tbl128.c",
    "src/x32-transposec/x32-transposec-4x4-aarch64-neon-tbl128.c",
]

NON_PROD_NEON_AARCH64_MICROKERNEL_SRCS = [
    "src/f32-vbinary/gen/f32-vdiv-aarch64-neon-u4.c",
    "src/f32-vbinary/gen/f32-vdivc-aarch64-neon-u4.c",
    "src/f32-vbinary/gen/f32-vrdivc-aarch64-neon-u4.c",
    "src/f32-vsqrt/gen/f32-vsqrt-aarch64-neon-sqrt-u8.c",
    "src/f32-vsqrt/gen/f32-vsqrt-aarch64-neon-sqrt-u16.c",
    "src/x8-lut/gen/x8-lut-aarch64-neon-tbx128x4-u16.c",
    "src/x8-lut/gen/x8-lut-aarch64-neon-tbx128x4-u32.c",
    "src/x8-lut/gen/x8-lut-aarch64-neon-tbx128x4-u48.c",
]

ALL_NEON_AARCH64_MICROKERNEL_SRCS = PROD_NEON_AARCH64_MICROKERNEL_SRCS + NON_PROD_NEON_AARCH64_MICROKERNEL_SRCS
