# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description:
#   XNNPACK - optimized floating-point neural network operators library

licenses(["notice"])

exports_files(["LICENSE"])

load(":build_defs.bzl", "xnnpack_aggregate_library", "xnnpack_benchmark", "xnnpack_binary", "xnnpack_cc_library", "xnnpack_min_size_copts", "xnnpack_optional_armcl_copts", "xnnpack_optional_armcl_deps", "xnnpack_optional_gemmlowp_copts", "xnnpack_optional_gemmlowp_deps", "xnnpack_optional_ruy_copts", "xnnpack_optional_ruy_deps", "xnnpack_optional_tflite_copts", "xnnpack_optional_tflite_deps", "xnnpack_std_copts", "xnnpack_unit_test", "xnnpack_visibility")

OPERATOR_BENCHMARK_DEPS = [
    ":XNNPACK",
    ":bench_utils",
    "@cpuinfo",
    "@pthreadpool",
]

MICROKERNEL_BENCHMARK_DEPS = [
    ":ukernels",
    ":bench_utils",
    ":enable_assembly",
    "@cpuinfo",
    "@FP16",
    "@pthreadpool",
]

ACCURACY_EVAL_DEPS = [
    ":XNNPACK",
    ":ukernels",
    "@FP16",
    "@pthreadpool",
]

MICROKERNEL_TEST_DEPS = [
    ":ukernels",
    ":enable_assembly",
    "@cpuinfo",
    "@FP16",
    "@pthreadpool",
]

OPERATOR_TEST_DEPS = [
    ":XNNPACK",
    "@pthreadpool",
    "@FP16",
]

OPERATOR_SRCS = [
    "src/add.c",
    "src/argmax-pooling.c",
    "src/average-pooling.c",
    "src/channel-pad.c",
    "src/channel-shuffle.c",
    "src/clamp.c",
    "src/convolution-spnchw.c",
    "src/convolution.c",
    "src/deconvolution.c",
    "src/fully-connected.c",
    "src/global-average-pooling-spnchw.c",
    "src/global-average-pooling.c",
    "src/hardswish.c",
    "src/leaky-relu.c",
    "src/max-pooling.c",
    "src/prelu.c",
    "src/sigmoid.c",
    "src/softargmax.c",
    "src/unpooling.c",
]

SCALAR_UKERNELS = [
    "src/f32-argmaxpool/mp9p8q-scalar.c",
    "src/f32-argmaxpool/up4-scalar.c",
    "src/f32-argmaxpool/up9-scalar.c",
    "src/f32-avgpool/mp9p8q-scalar.c",
    "src/f32-avgpool/up9-scalar.c",
    "src/f32-clamp/scalar.c",
    "src/f32-igemm/1x4-scalar.c",
    "src/f32-igemm/2x4-scalar.c",
    "src/f32-igemm/4x2-scalar.c",
    "src/f32-igemm/4x4-scalar.c",
    "src/f32-dwconv/up1x25-scalar.c",
    "src/f32-dwconv/up1x4-scalar.c",
    "src/f32-dwconv/up1x9-scalar.c",
    "src/f32-dwconv-spchw/3x3p1-scalar.c",
    "src/f32-dwconv-spchw/3x3s2p1-scalar.c",
    "src/f32-gavgpool-spchw/scalar-x1.c",
    "src/f32-gavgpool/mp7p7q-scalar.c",
    "src/f32-gavgpool/up7-scalar.c",
    "src/f32-gemm/1x4-scalar.c",
    "src/f32-gemm/2x4-scalar.c",
    "src/f32-gemm/4x2-scalar.c",
    "src/f32-gemm/4x4-scalar.c",
    "src/f32-gemminc/1x4-scalar.c",
    "src/f32-gemminc/2x4-scalar.c",
    "src/f32-gemminc/4x4-scalar.c",
    "src/f32-hswish/scalar.c",
    "src/f32-maxpool/9p8q-scalar.c",
    "src/f32-pavgpool/mp9p8q-scalar.c",
    "src/f32-pavgpool/up9-scalar.c",
    "src/f32-ppmm/2x4-scalar.c",
    "src/f32-ppmm/3x3-scalar.c",
    "src/f32-ppmm/4x2-scalar.c",
    "src/f32-ppmm/4x4-scalar.c",
    "src/f32-prelu/x4-scalar.c",
    "src/f32-rmax/scalar.c",
    "src/f32-spmm/1x1-scalar-pipelined.c",
    "src/f32-spmm/1x1-scalar-unroll2.c",
    "src/f32-spmm/1x1-scalar.c",
    "src/f32-spmm/2x1-scalar-pipelined.c",
    "src/f32-spmm/2x1-scalar-unroll2.c",
    "src/f32-spmm/2x1-scalar.c",
    "src/f32-spmm/4x1-scalar-pipelined.c",
    "src/f32-spmm/4x1-scalar-unroll2.c",
    "src/f32-spmm/4x1-scalar.c",
    "src/f32-spmm/8x1-scalar-pipelined.c",
    "src/f32-spmm/8x1-scalar-unroll2.c",
    "src/f32-spmm/8x1-scalar.c",
    "src/f32-vadd/scalar.c",
    "src/f32-vmul/scalar.c",
    "src/f32-vmulcaddc/c1-scalar-x2.c",
    "src/f32-vsub/scalar.c",
    "src/q8-avgpool/mp9p8q-scalar.c",
    "src/q8-avgpool/up9-scalar.c",
    "src/q8-igemm/2x2-scalar.c",
    "src/q8-dwconv/up1x9-scalar.c",
    "src/q8-gavgpool/mp7p7q-scalar.c",
    "src/q8-gavgpool/up7-scalar.c",
    "src/q8-gemm/2x2-scalar.c",
    "src/q8-vadd/scalar.c",
    "src/u8-clamp/scalar.c",
    "src/u8-lut32norm/scalar.c",
    "src/u8-maxpool/9p8q-scalar.c",
    "src/u8-rmax/scalar.c",
    "src/x32-packx/x2-scalar.c",
    "src/x32-packx/x3-scalar.c",
    "src/x32-packx/x4-scalar.c",
    "src/x32-pad/x2-scalar.c",
    "src/x32-unpool/scalar.c",
    "src/x32-zip/x2-scalar.c",
    "src/x32-zip/x3-scalar.c",
    "src/x32-zip/x4-scalar.c",
    "src/x32-zip/xm-scalar.c",
    "src/x8-lut/scalar.c",
    "src/x8-zip/x2-scalar.c",
    "src/x8-zip/x3-scalar.c",
    "src/x8-zip/x4-scalar.c",
    "src/x8-zip/xm-scalar.c",
]

PSIMD_UKERNELS = [
    "src/f32-argmaxpool/mp9p8q-psimd.c",
    "src/f32-argmaxpool/up4-psimd.c",
    "src/f32-argmaxpool/up9-psimd.c",
    "src/f32-avgpool/mp9p8q-psimd.c",
    "src/f32-avgpool/up9-psimd.c",
    "src/f32-clamp/psimd.c",
    "src/f32-igemm/1x8-psimd-loadsplat.c",
    "src/f32-igemm/1x8-psimd-splat.c",
    "src/f32-igemm/1x8s4-psimd.c",
    "src/f32-igemm/4x2c4-psimd.c",
    "src/f32-igemm/4x8-psimd-loadsplat.c",
    "src/f32-igemm/4x8-psimd-splat.c",
    "src/f32-igemm/4x8s4-psimd.c",
    "src/f32-igemm/6x8-psimd-loadsplat.c",
    "src/f32-igemm/6x8-psimd-splat.c",
    "src/f32-igemm/6x8s4-psimd.c",
    "src/f32-dwconv/up4x25-psimd.c",
    "src/f32-dwconv/up4x4-psimd.c",
    "src/f32-dwconv/up4x9-psimd.c",
    "src/f32-gavgpool/mp7p7q-psimd.c",
    "src/f32-gavgpool/up7-psimd.c",
    "src/f32-gemm/1x8-psimd-loadsplat.c",
    "src/f32-gemm/1x8-psimd-splat.c",
    "src/f32-gemm/1x8s4-psimd.c",
    "src/f32-gemm/4x8-psimd-loadsplat.c",
    "src/f32-gemm/4x8-psimd-splat.c",
    "src/f32-gemm/4x8s4-psimd.c",
    "src/f32-gemm/6x8-psimd-loadsplat.c",
    "src/f32-gemm/6x8-psimd-splat.c",
    "src/f32-gemm/6x8s4-psimd.c",
    "src/f32-gemminc/1x8-psimd-loadsplat.c",
    "src/f32-gemminc/1x8-psimd-splat.c",
    "src/f32-gemminc/1x8s4-psimd.c",
    "src/f32-gemminc/4x8-psimd-loadsplat.c",
    "src/f32-gemminc/4x8-psimd-splat.c",
    "src/f32-gemminc/4x8s4-psimd.c",
    "src/f32-gemminc/6x8-psimd-loadsplat.c",
    "src/f32-gemminc/6x8-psimd-splat.c",
    "src/f32-gemminc/6x8s4-psimd.c",
    "src/f32-hswish/psimd.c",
    "src/f32-maxpool/9p8q-psimd.c",
    "src/f32-pavgpool/mp9p8q-psimd.c",
    "src/f32-pavgpool/up9-psimd.c",
    "src/f32-ppmm/4x8-psimd.c",
    "src/f32-prelu/x4-psimd.c",
    "src/f32-vadd/psimd.c",
    "src/f32-vmul/psimd.c",
    "src/f32-vmulcaddc/c4-psimd-x2.c",
    "src/f32-vsub/psimd.c",
    "src/x32-packx/x4-psimd.c",
    "src/x32-pad/x2-psimd.c",
    "src/x32-unpool/psimd.c",
    "src/x32-zip/x2-psimd.c",
    "src/x32-zip/x3-psimd.c",
    "src/x32-zip/x4-psimd.c",
    "src/x32-zip/xm-psimd.c",
]

# ISA-specific micro-kernels
NEON_UKERNELS = [
    "src/f32-avgpool/mp9p8q-neon.c",
    "src/f32-avgpool/up9-neon.c",
    "src/f32-clamp/neon.c",
    "src/f32-igemm/1x8-neon-ld64.c",
    "src/f32-igemm/4x2-neon-ld64.c",
    "src/f32-igemm/4x4-neon-ld64.c",
    "src/f32-igemm/4x8-neon-ld128.c",
    "src/f32-igemm/4x8-neon-ld64.c",
    "src/f32-igemm/6x8-neon-ld64.c",
    "src/f32-dwconv/up4x9-neon.c",
    "src/f32-gavgpool-spchw/neon-x4.c",
    "src/f32-gavgpool/mp7p7q-neon.c",
    "src/f32-gavgpool/up7-neon.c",
    "src/f32-gemm/1x8-neon-ld64.c",
    "src/f32-gemm/4x2-neon-ld64.c",
    "src/f32-gemm/4x8-neon-ld128.c",
    "src/f32-gemm/4x8-neon-ld64.c",
    "src/f32-gemm/5x8-neon-ld64.c",
    "src/f32-gemm/6x8-neon-ld64.c",
    "src/f32-gemminc/1x8-neon-ld64.c",
    "src/f32-gemminc/4x8-neon-ld128.c",
    "src/f32-gemminc/4x8-neon-ld64.c",
    "src/f32-gemminc/5x8-neon-ld64.c",
    "src/f32-gemminc/6x8-neon-ld64.c",
    "src/f32-hswish/neon.c",
    "src/f32-pavgpool/mp9p8q-neon.c",
    "src/f32-pavgpool/up9-neon.c",
    "src/f32-ppmm/4x8-neon.c",
    "src/f32-ppmm/8x8-neon.c",
    "src/f32-rmax/neon.c",
    "src/f32-vmulcaddc/c4-neon-x2.c",
    "src/q8-avgpool/mp9p8q-neon.c",
    "src/q8-avgpool/up9-neon.c",
    "src/q8-igemm/4x8-neon.c",
    "src/q8-igemm/8x8-neon.c",
    "src/q8-dwconv/up8x9-neon.c",
    "src/q8-gavgpool/mp7p7q-neon.c",
    "src/q8-gavgpool/up7-neon.c",
    "src/q8-gemm/4x8-neon.c",
    "src/q8-gemm/8x8-neon.c",
    "src/q8-vadd/neon.c",
    "src/u8-clamp/neon.c",
    "src/u8-maxpool/9p8q-neon.c",
    "src/u8-rmax/neon.c",
    "src/x32-packx/x4-neon-st4.c",
    "src/x32-pad/x2-neon.c",
    "src/x32-zip/x2-neon.c",
    "src/x32-zip/x3-neon.c",
    "src/x32-zip/x4-neon.c",
    "src/x32-zip/xm-neon.c",
    "src/x8-zip/x2-neon.c",
    "src/x8-zip/x3-neon.c",
    "src/x8-zip/x4-neon.c",
    "src/x8-zip/xm-neon.c",
]

NEONFMA_UKERNELS = [
    "src/f32-igemm/1x8-neonfma-ld64.c",
    "src/f32-igemm/4x2-neonfma-ld64.c",
    "src/f32-igemm/4x4-neonfma-ld64.c",
    "src/f32-igemm/4x8-neonfma-ld128.c",
    "src/f32-igemm/4x8-neonfma-ld64.c",
    "src/f32-igemm/6x8-neonfma-ld64.c",
    "src/f32-dwconv/up4x9-neonfma.c",
    "src/f32-dwconv/up8x9-neonfma.c",
    "src/f32-gemm/1x8-neonfma-ld64.c",
    "src/f32-gemm/4x2-neonfma-ld64.c",
    "src/f32-gemm/4x8-neonfma-ld128.c",
    "src/f32-gemm/4x8-neonfma-ld64.c",
    "src/f32-gemm/5x8-neonfma-ld64.c",
    "src/f32-gemm/6x8-neonfma-ld64.c",
    "src/f32-gemminc/1x8-neonfma-ld64.c",
    "src/f32-gemminc/4x8-neonfma-ld128.c",
    "src/f32-gemminc/4x8-neonfma-ld64.c",
    "src/f32-gemminc/5x8-neonfma-ld64.c",
    "src/f32-gemminc/6x8-neonfma-ld64.c",
    "src/f32-hswish/neonfma.c",
    "src/f32-ppmm/4x8-neonfma.c",
    "src/f32-ppmm/8x8-neonfma.c",
    "src/f32-vmulcaddc/c4-neonfma-x2.c",
]

AARCH64_NEONFMA_UKERNELS = [
    "src/f32-conv-hwc/3x3s2p1c3x4-neonfma-2x2.c",
    "src/f32-conv-hwc/3x3s2p1c3x8-neonfma-2x2.c",
    "src/f32-conv-hwc2spchw/3x3s2p1c3x4-neonfma-2x2.c",
    "src/f32-dwconv-spchw/3x3p1-neonfma.c",
    "src/f32-dwconv-spchw/5x5p2-neonfma.c",
    "src/f32-dwconv-spchw/3x3s2p1-neonfma.c",
    "src/f32-dwconv-spchw/5x5s2p2-neonfma.c",
    "src/f32-spmm/12x1-neonfma.c",
    "src/f32-spmm/12x2-neonfma.c",
    "src/f32-spmm/12x4-neonfma.c",
    "src/f32-spmm/16x1-neonfma-pipelined.c",
    "src/f32-spmm/16x1-neonfma-unroll2.c",
    "src/f32-spmm/16x1-neonfma.c",
    "src/f32-spmm/16x2-neonfma.c",
    "src/f32-spmm/16x4-neonfma.c",
    "src/f32-spmm/4x1-neonfma-pipelined.c",
    "src/f32-spmm/4x1-neonfma-unroll2.c",
    "src/f32-spmm/4x1-neonfma.c",
    "src/f32-spmm/4x2-neonfma.c",
    "src/f32-spmm/4x4-neonfma.c",
    "src/f32-spmm/8x1-neonfma-pipelined.c",
    "src/f32-spmm/8x1-neonfma-unroll2.c",
    "src/f32-spmm/8x1-neonfma.c",
    "src/f32-spmm/8x2-neonfma.c",
    "src/f32-spmm/8x4-neonfma.c",
]

AARCH64_NEONFP16ARITH_UKERNELS = [
    "src/f16-gemm/4x8-neonfp16arith-ld64.c",
    "src/f16-gemm/6x8-neonfp16arith-ld64.c",
    "src/f16-gemm/8x8-neonfp16arith-ld64.c",
]

SSE_UKERNELS = [
    "src/f32-avgpool/mp9p8q-sse.c",
    "src/f32-avgpool/up9-sse.c",
    "src/f32-clamp/sse.c",
    "src/f32-igemm/1x8-sse-dup.c",
    "src/f32-igemm/1x8-sse-load1.c",
    "src/f32-igemm/1x8s4-sse.c",
    "src/f32-igemm/4x2c4-sse.c",
    "src/f32-igemm/4x8-sse-dup.c",
    "src/f32-igemm/4x8-sse-load1.c",
    "src/f32-igemm/4x8s4-sse.c",
    "src/f32-dwconv/up4x25-sse.c",
    "src/f32-dwconv/up4x4-sse.c",
    "src/f32-dwconv/up4x9-sse.c",
    "src/f32-gavgpool-spchw/sse-x4.c",
    "src/f32-gavgpool/mp7p7q-sse.c",
    "src/f32-gavgpool/up7-sse.c",
    "src/f32-gemm/1x8-sse-dup.c",
    "src/f32-gemm/1x8-sse-load1.c",
    "src/f32-gemm/1x8s4-sse.c",
    "src/f32-gemm/4x8-sse-dup.c",
    "src/f32-gemm/4x8-sse-load1.c",
    "src/f32-gemm/4x8s4-sse.c",
    "src/f32-gemminc/1x8-sse-dup.c",
    "src/f32-gemminc/1x8-sse-load1.c",
    "src/f32-gemminc/1x8s4-sse.c",
    "src/f32-gemminc/4x8-sse-dup.c",
    "src/f32-gemminc/4x8-sse-load1.c",
    "src/f32-gemminc/4x8s4-sse.c",
    "src/f32-hswish/sse.c",
    "src/f32-maxpool/9p8q-sse.c",
    "src/f32-pavgpool/mp9p8q-sse.c",
    "src/f32-pavgpool/up9-sse.c",
    "src/f32-dwconv-spchw/3x3p1-sse.c",
    "src/f32-dwconv-spchw/3x3s2p1-sse.c",
    "src/f32-ppmm/4x8-sse.c",
    "src/f32-prelu/x4-sse.c",
    "src/f32-rmax/sse.c",
    "src/f32-spmm/4x1-sse.c",
    "src/f32-spmm/8x1-sse.c",
    "src/f32-vadd/sse.c",
    "src/f32-vmul/sse.c",
    "src/f32-vmulcaddc/c4-sse-x2.c",
    "src/f32-vsub/sse.c",
    "src/x32-packx/x4-sse.c",
]

SSE2_UKERNELS = [
    "src/f32-argmaxpool/mp9p8q-sse2.c",
    "src/f32-argmaxpool/up4-sse2.c",
    "src/f32-argmaxpool/up9-sse2.c",
    "src/q8-avgpool/mp9p8q-sse2.c",
    "src/q8-avgpool/up9-sse2.c",
    "src/q8-igemm/4x4c2-sse2.c",
    "src/q8-dwconv/up8x9-sse2.c",
    "src/q8-gavgpool/mp7p7q-sse2.c",
    "src/q8-gavgpool/up7-sse2.c",
    "src/q8-gemm/2x4c8-sse2.c",
    "src/q8-gemm/4x4c2-sse2.c",
    "src/q8-vadd/sse2.c",
    "src/u8-clamp/sse2.c",
    "src/u8-maxpool/9p8q-sse2.c",
    "src/u8-rmax/sse2.c",
    "src/x32-pad/x2-sse2.c",
    "src/x32-zip/x2-sse2.c",
    "src/x32-zip/x3-sse2.c",
    "src/x32-zip/x4-sse2.c",
    "src/x32-zip/xm-sse2.c",
    "src/x8-zip/x2-sse2.c",
    "src/x8-zip/x3-sse2.c",
    "src/x8-zip/x4-sse2.c",
    "src/x8-zip/xm-sse2.c",
]

AVX_UKERNELS = [
    "src/f32-rmax/avx.c",
]

AVX2_UKERNELS = [
    "src/math/exp-avx2-p5.c",
    "src/math/exp-avx2-perm-p3.c",
    "src/math/exp-avx2-perm-p4.c",
]

AVX512F_UKERNELS = [
    "src/f32-rmax/avx512f.c",
    "src/math/exp-avx512f-p5-scalef.c",
    "src/math/exp-avx512f-p5.c",
    "src/math/exp-avx512f-perm-p3.c",
]

AARCH32_ASM_UKERNELS = [
    "src/q8-dwconv/up8x9-aarch32-neon.S",
]

AARCH64_ASM_UKERNELS = [
    "src/f32-dwconv/up4x9-aarch64-neonfma-cortex-a55.S",
    "src/f32-dwconv/up4x9-aarch64-neonfma.S",
    "src/f32-gemm/1x12-aarch64-neonfma-cortex-a53.S",
    "src/f32-gemm/1x8-aarch64-neonfma-cortex-a53.S",
    "src/f32-gemm/1x8-aarch64-neonfma-cortex-a57.S",
    "src/f32-gemm/1x8-aarch64-neonfma-cortex-a75.S",
    "src/f32-gemm/4x12-aarch64-neonfma-cortex-a53.S",
    "src/f32-gemm/4x8-aarch64-neonfma-cortex-a57.S",
    "src/f32-gemm/4x8-aarch64-neonfma-cortex-a75.S",
    "src/f32-gemm/4x8-aarch64-neonfma-ld128.S",
    "src/f32-gemm/4x8-aarch64-neonfma-ld64.S",
    "src/f32-gemm/5x8-aarch64-neonfma-cortex-a75.S",
    "src/f32-gemm/6x8-aarch64-neonfma-cortex-a57.S",
    "src/f32-gemm/6x8-aarch64-neonfma-cortex-a73.S",
    "src/f32-gemm/6x8-aarch64-neonfma-cortex-a75.S",
    "src/f32-gemm/6x8-aarch64-neonfma-ld128.S",
    "src/f32-gemm/6x8-aarch64-neonfma-ld64.S",
    "src/f32-gemminc/1x12-aarch64-neonfma-cortex-a53.S",
    "src/f32-gemminc/1x8-aarch64-neonfma-cortex-a53.S",
    "src/f32-gemminc/1x8-aarch64-neonfma-cortex-a57.S",
    "src/f32-gemminc/1x8-aarch64-neonfma-cortex-a75.S",
    "src/f32-gemminc/4x12-aarch64-neonfma-cortex-a53.S",
    "src/f32-gemminc/4x8-aarch64-neonfma-cortex-a57.S",
    "src/f32-gemminc/4x8-aarch64-neonfma-cortex-a75.S",
    "src/f32-gemminc/4x8-aarch64-neonfma-ld128.S",
    "src/f32-gemminc/4x8-aarch64-neonfma-ld64.S",
    "src/f32-gemminc/5x8-aarch64-neonfma-cortex-a75.S",
    "src/f32-gemminc/6x8-aarch64-neonfma-cortex-a57.S",
    "src/f32-gemminc/6x8-aarch64-neonfma-cortex-a73.S",
    "src/f32-gemminc/6x8-aarch64-neonfma-cortex-a75.S",
    "src/f32-gemminc/6x8-aarch64-neonfma-ld128.S",
    "src/f32-gemminc/6x8-aarch64-neonfma-ld64.S",
    "src/f32-igemm/1x12-aarch64-neonfma-cortex-a53.S",
    "src/f32-igemm/1x8-aarch64-neonfma-cortex-a53.S",
    "src/f32-igemm/1x8-aarch64-neonfma-cortex-a57.S",
    "src/f32-igemm/1x8-aarch64-neonfma-cortex-a75.S",
    "src/f32-igemm/4x12-aarch64-neonfma-cortex-a53.S",
    "src/f32-igemm/4x8-aarch64-neonfma-cortex-a75.S",
    "src/f32-igemm/5x8-aarch64-neonfma-cortex-a75.S",
    "src/f32-igemm/6x8-aarch64-neonfma-cortex-a57.S",
    "src/f32-igemm/6x8-aarch64-neonfma-cortex-a73.S",
    "src/f32-igemm/6x8-aarch64-neonfma-cortex-a75.S",
]

INTERNAL_MICROKERNEL_HDRS = [
    "src/xnnpack/argmaxpool.h",
    "src/xnnpack/avgpool.h",
    "src/xnnpack/clamp.h",
    "src/xnnpack/common.h",
    "src/xnnpack/conv.h",
    "src/xnnpack/dwconv.h",
    "src/xnnpack/gavgpool.h",
    "src/xnnpack/gemm.h",
    "src/xnnpack/hswish.h",
    "src/xnnpack/igemm.h",
    "src/xnnpack/lut.h",
    "src/xnnpack/math.h",
    "src/xnnpack/maxpool.h",
    "src/xnnpack/packx.h",
    "src/xnnpack/pad.h",
    "src/xnnpack/params.h",
    "src/xnnpack/pavgpool.h",
    "src/xnnpack/ppmm.h",
    "src/xnnpack/prelu.h",
    "src/xnnpack/rmax.h",
    "src/xnnpack/scalar-utils.h",
    "src/xnnpack/spmm.h",
    "src/xnnpack/unpool.h",
    "src/xnnpack/vadd.h",
    "src/xnnpack/vmul.h",
    "src/xnnpack/vmulcaddc.h",
    "src/xnnpack/vsub.h",
    "src/xnnpack/zip.h",
]

INTERNAL_HDRS = INTERNAL_MICROKERNEL_HDRS + [
    "include/xnnpack.h",
    "src/xnnpack/allocator.h",
    "src/xnnpack/compute.h",
    "src/xnnpack/im2col.h",
    "src/xnnpack/indirection.h",
    "src/xnnpack/math-stubs.h",
    "src/xnnpack/operator.h",
    "src/xnnpack/pack.h",
    "src/xnnpack/requantization-stubs.h",
    "src/xnnpack/requantization.h",
]

ACCURACY_EVAL_HDRS = INTERNAL_MICROKERNEL_HDRS + [
    "src/xnnpack/math-stubs.h",
]

MICROKERNEL_BENCHMARK_HDRS = INTERNAL_MICROKERNEL_HDRS + [
    "src/xnnpack/requantization.h",
    "include/xnnpack.h",
]

MICROKERNEL_TEST_HDRS = INTERNAL_MICROKERNEL_HDRS + [
    "src/xnnpack/isa-checks.h",
    "src/xnnpack/requantization.h",
    "include/xnnpack.h",
]

OPERATOR_TEST_PARAMS_HDRS = [
    "src/xnnpack/params.h",
    "src/xnnpack/common.h",
]

WEIGHTS_PACK_HDRS = [
    "src/xnnpack/pack.h",
    "src/xnnpack/operator.h",
    "src/xnnpack/compute.h",
]

LOGGING_COPTS = select({
    # No logging in optimized mode
    ":optimized_build": ["-DXNN_LOG_LEVEL=0"],
    # Full logging in debug mode
    ":debug_build": ["-DXNN_LOG_LEVEL=5"],
    # Error-only logging in default (fastbuild) mode
    "//conditions:default": ["-DXNN_LOG_LEVEL=2"],
})

LOGGING_HDRS = [
    "src/xnnpack/log.h",
]

xnnpack_cc_library(
    name = "scalar_ukernels",
    srcs = SCALAR_UKERNELS,
    hdrs = INTERNAL_HDRS,
    aarch32_copts = ["-marm"],
    copts = xnnpack_std_copts(),
    deps = [
        "@FP16",
        "@FXdiv",
    ],
)

xnnpack_cc_library(
    name = "psimd_ukernels",
    srcs = PSIMD_UKERNELS,
    hdrs = INTERNAL_HDRS,
    aarch32_copts = [
        "-marm",
        "-mfpu=neon",
    ],
    copts = xnnpack_std_copts(),
    optimized_copts = [
        "-O3",
        "-ffast-math",
    ],
    deps = [
        "@FP16",
        "@psimd",
    ],
)

xnnpack_cc_library(
    name = "neon_ukernels",
    hdrs = INTERNAL_HDRS,
    aarch32_copts = [
        "-marm",
        "-mfpu=neon",
    ],
    aarch32_srcs = NEON_UKERNELS,
    aarch64_srcs = NEON_UKERNELS,
    copts = xnnpack_std_copts(),
    deps = ["@FP16"],
)

xnnpack_cc_library(
    name = "neonfma_ukernels",
    hdrs = INTERNAL_HDRS,
    aarch32_copts = [
        "-marm",
        "-mfpu=neon-vfpv4",
    ],
    aarch32_srcs = NEONFMA_UKERNELS,
    aarch64_srcs = NEONFMA_UKERNELS + AARCH64_NEONFMA_UKERNELS,
    copts = xnnpack_std_copts(),
    deps = ["@FP16"],
)

xnnpack_cc_library(
    name = "neonfp16arith_ukernels",
    hdrs = INTERNAL_HDRS,
    aarch64_copts = ["-march=armv8.2-a+fp16"],
    aarch64_srcs = AARCH64_NEONFP16ARITH_UKERNELS,
    copts = xnnpack_std_copts(),
    deps = ["@FP16"],
)

xnnpack_cc_library(
    name = "sse2_ukernels",
    hdrs = INTERNAL_HDRS,
    copts = xnnpack_std_copts(),
    x86_copts = ["-msse2"],
    x86_srcs = SSE_UKERNELS + SSE2_UKERNELS,
    deps = ["@FP16"],
)

xnnpack_cc_library(
    name = "avx_ukernels",
    hdrs = INTERNAL_HDRS,
    copts = xnnpack_std_copts(),
    x86_copts = ["-mavx"],
    x86_srcs = AVX_UKERNELS,
    deps = ["@FP16"],
)

xnnpack_cc_library(
    name = "avx2_ukernels",
    hdrs = INTERNAL_HDRS,
    copts = xnnpack_std_copts(),
    x86_copts = [
        "-mfma",
        "-mavx2",
    ],
    x86_srcs = AVX2_UKERNELS,
    deps = ["@FP16"],
)

xnnpack_cc_library(
    name = "avx512f_ukernels",
    hdrs = INTERNAL_HDRS,
    copts = xnnpack_std_copts(),
    x86_copts = ["-mavx512f"],
    x86_srcs = AVX512F_UKERNELS,
    deps = ["@FP16"],
)

xnnpack_cc_library(
    name = "asm_ukernels",
    hdrs = ["src/xnnpack/assembly.h"],
    aarch32_srcs = AARCH32_ASM_UKERNELS,
    aarch64_srcs = AARCH64_ASM_UKERNELS,
)

xnnpack_aggregate_library(
    name = "ukernels",
    aarch32_deps = [
        ":psimd_ukernels",
        ":neon_ukernels",
        ":neonfma_ukernels",
        ":asm_ukernels",
    ],
    aarch64_deps = [
        ":psimd_ukernels",
        ":neon_ukernels",
        ":neonfma_ukernels",
        ":neonfp16arith_ukernels",
        ":asm_ukernels",
    ],
    generic_deps = [":scalar_ukernels"],
    wasmsimd_deps = [
        ":psimd_ukernels",
    ],
    x86_deps = [
        ":psimd_ukernels",
        ":sse2_ukernels",
        ":avx_ukernels",
        ":avx2_ukernels",
        ":avx512f_ukernels",
    ],
)

xnnpack_cc_library(
    name = "im2col",
    srcs = ["src/im2col.c"],
    hdrs = [
        "src/xnnpack/common.h",
        "src/xnnpack/im2col.h",
    ],
    copts = xnnpack_std_copts(),
)

xnnpack_cc_library(
    name = "indirection",
    srcs = ["src/indirection.c"],
    hdrs = INTERNAL_HDRS,
    copts = xnnpack_std_copts(),
    deps = [
        "@FP16",
        "@FXdiv",
        "@pthreadpool",
    ],
)

xnnpack_cc_library(
    name = "operator_run",
    srcs = ["src/operator-run.c"],
    hdrs = INTERNAL_HDRS + LOGGING_HDRS,
    copts = xnnpack_std_copts() + LOGGING_COPTS + [
        # Wrappers for multi-pass microkernels use VLAs for temporary buffers.
        "-Wno-vla",
    ],
    deps = [
        "@FP16",
        "@FXdiv",
        "@clog",
        "@pthreadpool",
    ],
)

cc_library(
    name = "enable_assembly",
    defines = select({
        ":xnn_enable_assembly_explicit_true": ["XNN_ENABLE_ASSEMBLY=1"],
        ":xnn_enable_assembly_explicit_false": ["XNN_ENABLE_ASSEMBLY=0"],
        "//conditions:default": ["XNN_ENABLE_ASSEMBLY=1"],
    }),
)

xnnpack_cc_library(
    name = "operators",
    srcs = OPERATOR_SRCS + [
        "src/operator-delete.c",
    ],
    hdrs = INTERNAL_HDRS + LOGGING_HDRS,
    copts = xnnpack_std_copts() + LOGGING_COPTS + [
        "-Isrc",
        "-Iinclude",
    ] + select({
        ":debug_build": [],
        "//conditions:default": xnnpack_min_size_copts(),
    }),
    wasm_srcs = ["src/wasm-stubs.c"],
    wasmsimd_srcs = ["src/wasm-stubs.c"],
    deps = [
        ":indirection",
        "@FP16",
        "@FXdiv",
        "@clog",
        "@pthreadpool",
    ],
)

cc_library(
    name = "XNNPACK",
    srcs = [
        "src/init.c",
    ],
    copts = xnnpack_std_copts() + LOGGING_COPTS + [
        "-Isrc",
        "-Iinclude",
    ] + select({
        ":debug_build": [],
        "//conditions:default": xnnpack_min_size_copts(),
    }),
    includes = ["include"],
    linkstatic = True,
    textual_hdrs = ["include/xnnpack.h"],
    visibility = xnnpack_visibility(),
    deps = [
        ":enable_assembly",
        ":ukernels",
        ":operator_run",
        ":operators",
        "@clog",
        "@pthreadpool",
    ] + select({
        ":emscripten": [],
        "//conditions:default": ["@cpuinfo"],
    }),
)

cc_library(
    name = "xnnpack_operators_nhwc_f32",
    srcs = [
        "src/init.c",
    ],
    copts = xnnpack_std_copts() + LOGGING_COPTS + [
        "-Isrc",
        "-Iinclude",
    ] + select({
        ":debug_build": [],
        "//conditions:default": xnnpack_min_size_copts(),
    }),
    defines = [
        "XNN_NO_Q8_OPERATORS",
        "XNN_NO_U8_OPERATORS",
        "XNN_NO_X8_OPERATORS",
        "XNN_NO_SPNCHW_OPERATORS",
    ],
    includes = ["include"],
    linkstatic = True,
    textual_hdrs = ["include/xnnpack.h"],
    visibility = xnnpack_visibility(),
    deps = [
        ":enable_assembly",
        ":ukernels",
        ":operator_run",
        ":operators",
        "@clog",
        "@pthreadpool",
    ] + select({
        ":emscripten": [],
        "//conditions:default": ["@cpuinfo"],
    }),
)

xnnpack_cc_library(
    name = "bench_utils",
    srcs = ["bench/utils.cc"],
    hdrs = ["bench/utils.h"],
    copts = ["-Wno-unused-result"],
    deps = ["@cpuinfo"],
)

######################### Benchmarks for micro-kernels #########################

xnnpack_benchmark(
    name = "q8_gemm_bench",
    srcs = [
        "bench/gemm.h",
        "bench/q8-gemm.cc",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_BENCHMARK_HDRS,
    copts = ["-Wno-unused-function"] + xnnpack_optional_ruy_copts() + xnnpack_optional_gemmlowp_copts(),
    deps = MICROKERNEL_BENCHMARK_DEPS + xnnpack_optional_ruy_deps() + xnnpack_optional_gemmlowp_deps(),
)

xnnpack_benchmark(
    name = "f16_gemm_bench",
    srcs = [
        "bench/f16-gemm.cc",
        "bench/gemm.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_BENCHMARK_HDRS,
    copts = ["-Wno-unused-function"],
    deps = MICROKERNEL_BENCHMARK_DEPS,
)

xnnpack_benchmark(
    name = "f32_igemm_bench",
    srcs = [
        "bench/f32-igemm.cc",
        "bench/conv.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + MICROKERNEL_BENCHMARK_HDRS,
    deps = MICROKERNEL_BENCHMARK_DEPS + [":indirection"],
)

xnnpack_benchmark(
    name = "f32_conv_hwc_bench",
    srcs = [
        "bench/f32-conv-hwc.cc",
        "bench/dconv.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_BENCHMARK_HDRS,
    copts = ["-Wno-unused-function"],
    deps = MICROKERNEL_BENCHMARK_DEPS,
)

xnnpack_benchmark(
    name = "f32_dwconv_bench",
    srcs = [
        "bench/f32-dwconv.cc",
        "bench/dwconv.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + MICROKERNEL_BENCHMARK_HDRS,
    deps = MICROKERNEL_BENCHMARK_DEPS + [":indirection"],
)

xnnpack_benchmark(
    name = "f32_dwconv_spchw_bench",
    srcs = [
        "bench/f32-dwconv-spchw.cc",
        "bench/dwconv.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_BENCHMARK_HDRS,
    deps = MICROKERNEL_BENCHMARK_DEPS + [":indirection"],
)

xnnpack_benchmark(
    name = "f32_gemm_bench",
    srcs = [
        "bench/f32-gemm.cc",
        "bench/gemm.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_BENCHMARK_HDRS,
    copts = ["-Wno-unused-function"] + xnnpack_optional_ruy_copts(),
    deps = MICROKERNEL_BENCHMARK_DEPS + xnnpack_optional_ruy_deps(),
)

xnnpack_benchmark(
    name = "f32_rmax_bench",
    srcs = [
        "bench/f32-rmax.cc",
        "src/xnnpack/AlignedAllocator.h",
    ] + MICROKERNEL_BENCHMARK_HDRS,
    deps = MICROKERNEL_BENCHMARK_DEPS,
)

xnnpack_benchmark(
    name = "f32_spmm_bench",
    srcs = [
        "bench/f32-spmm.cc",
        "bench/gemm.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + MICROKERNEL_BENCHMARK_HDRS,
    copts = ["-Wno-unused-function"],
    deps = MICROKERNEL_BENCHMARK_DEPS,
)

xnnpack_benchmark(
    name = "f32_im2col_gemm_bench",
    srcs = [
        "bench/f32-im2col-gemm.cc",
        "bench/conv.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_BENCHMARK_HDRS,
    deps = MICROKERNEL_BENCHMARK_DEPS + [":im2col"],
)

########################### Benchmarks for operators ###########################

xnnpack_benchmark(
    name = "add_bench",
    srcs = ["bench/add.cc"],
    deps = OPERATOR_BENCHMARK_DEPS,
)

xnnpack_benchmark(
    name = "average_pooling_bench",
    srcs = ["bench/average-pooling.cc"],
    deps = OPERATOR_BENCHMARK_DEPS,
)

xnnpack_benchmark(
    name = "channel_shuffle_bench",
    srcs = ["bench/channel-shuffle.cc"],
    deps = OPERATOR_BENCHMARK_DEPS,
)

xnnpack_benchmark(
    name = "convolution_bench",
    srcs = ["bench/convolution.cc"],
    copts = xnnpack_optional_tflite_copts() + xnnpack_optional_armcl_copts(),
    deps = OPERATOR_BENCHMARK_DEPS + xnnpack_optional_tflite_deps() + xnnpack_optional_armcl_deps(),
)

xnnpack_benchmark(
    name = "deconvolution_bench",
    srcs = ["bench/deconvolution.cc"],
    copts = xnnpack_optional_tflite_copts(),
    deps = OPERATOR_BENCHMARK_DEPS + xnnpack_optional_tflite_deps(),
)

xnnpack_benchmark(
    name = "global_average_pooling_bench",
    srcs = ["bench/global-average-pooling.cc"],
    deps = OPERATOR_BENCHMARK_DEPS,
)

xnnpack_benchmark(
    name = "max_pooling_bench",
    srcs = ["bench/max-pooling.cc"],
    deps = OPERATOR_BENCHMARK_DEPS,
)

xnnpack_benchmark(
    name = "sigmoid_bench",
    srcs = ["bench/sigmoid.cc"],
    deps = OPERATOR_BENCHMARK_DEPS,
)

xnnpack_benchmark(
    name = "softargmax_bench",
    srcs = ["bench/softargmax.cc"],
    deps = OPERATOR_BENCHMARK_DEPS,
)

############################# End-to-end benchmarks ############################

cc_library(
    name = "mobilenet_v1",
    srcs = ["models/mobilenet-v1.cc"],
    hdrs = ["models/models.h"],
    linkstatic = True,
    deps = [
        ":XNNPACK",
        "@pthreadpool",
    ],
)

cc_library(
    name = "mobilenet_v2",
    srcs = ["models/mobilenet-v2.cc"],
    hdrs = ["models/models.h"],
    linkstatic = True,
    deps = [
        ":XNNPACK",
        "@pthreadpool",
    ],
)

xnnpack_benchmark(
    name = "end2end_bench",
    srcs = ["bench/end2end.cc"],
    deps = [
        ":XNNPACK",
        ":mobilenet_v1",
        ":mobilenet_v2",
        "@pthreadpool",
    ],
)

#################### Accuracy evaluation for math functions ####################

xnnpack_benchmark(
    name = "f32_exp_eval",
    srcs = [
        "eval/f32-exp.cc",
        "src/xnnpack/AlignedAllocator.h",
    ] + ACCURACY_EVAL_HDRS,
    deps = ACCURACY_EVAL_DEPS,
)

######################### Unit tests for micro-kernels #########################

xnnpack_unit_test(
    name = "f16_gemm_test",
    srcs = [
        "test/f16-gemm.cc",
        "test/gemm-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_argmaxpool_test",
    srcs = [
        "test/f32-argmaxpool.cc",
        "test/argmaxpool-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_avgpool_test",
    srcs = [
        "test/f32-avgpool.cc",
        "test/avgpool-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_clamp_test",
    srcs = [
        "test/f32-clamp.cc",
        "test/clamp-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_igemm_test",
    srcs = [
        "test/f32-igemm.cc",
        "test/gemm-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_conv_hwc_test",
    srcs = [
        "test/f32-conv-hwc.cc",
        "test/conv-hwc-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_conv_hwc2spchw_test",
    srcs = [
        "test/f32-conv-hwc2spchw.cc",
        "test/conv-hwc2spchw-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_dwconv_test",
    srcs = [
        "test/f32-dwconv.cc",
        "test/dwconv-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_dwconv_spchw_test",
    srcs = [
        "test/f32-dwconv-spchw.cc",
        "test/dwconv-spchw-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_gavgpool_test",
    srcs = [
        "test/f32-gavgpool.cc",
        "test/gavgpool-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_gavgpool_spchw_test",
    srcs = [
        "test/f32-gavgpool-spchw.cc",
        "test/gavgpool-spchw-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_gemm_test",
    srcs = [
        "test/f32-gemm.cc",
        "test/gemm-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_gemminc_test",
    srcs = [
        "test/f32-gemminc.cc",
        "test/gemm-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_hswish_test",
    srcs = [
        "test/f32-hswish.cc",
        "test/hswish-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_maxpool_test",
    srcs = [
        "test/f32-maxpool.cc",
        "test/maxpool-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_pavgpool_test",
    srcs = [
        "test/f32-pavgpool.cc",
        "test/avgpool-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_ppmm_test",
    srcs = [
        "test/f32-ppmm.cc",
        "test/gemm-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_prelu_test",
    srcs = [
        "test/f32-prelu.cc",
        "test/prelu-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_rmax_test",
    srcs = [
        "test/f32-rmax.cc",
        "test/rmax-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_spmm_test",
    srcs = [
        "test/f32-spmm.cc",
        "test/spmm-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_vadd_test",
    srcs = [
        "test/f32-vadd.cc",
        "test/vadd-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_vsub_test",
    srcs = [
        "test/f32-vsub.cc",
        "test/vsub-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_vmul_test",
    srcs = [
        "test/f32-vmul.cc",
        "test/vmul-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "f32_vmulcaddc_test",
    srcs = [
        "test/f32-vmulcaddc.cc",
        "test/vmulcaddc-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "q8_avgpool_test",
    srcs = [
        "test/q8-avgpool.cc",
        "test/avgpool-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "q8_igemm_test",
    srcs = [
        "test/q8-igemm.cc",
        "test/gemm-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "q8_dwconv_test",
    srcs = [
        "test/q8-dwconv.cc",
        "test/dwconv-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "q8_gavgpool_test",
    srcs = [
        "test/q8-gavgpool.cc",
        "test/gavgpool-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "q8_gemm_test",
    srcs = [
        "test/q8-gemm.cc",
        "test/gemm-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + WEIGHTS_PACK_HDRS + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "q8_vadd_test",
    srcs = [
        "test/q8-vadd.cc",
        "test/vadd-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "u8_clamp_test",
    srcs = [
        "test/u8-clamp.cc",
        "test/clamp-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "u8_lut32norm_test",
    srcs = [
        "test/u8-lut32norm.cc",
        "test/lut-norm-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "u8_maxpool_test",
    srcs = [
        "test/u8-maxpool.cc",
        "test/maxpool-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "u8_rmax_test",
    srcs = [
        "test/u8-rmax.cc",
        "test/rmax-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "x32_packx_test",
    srcs = [
        "test/x32-packx.cc",
        "test/pack-microkernel-tester.h",
        "src/xnnpack/AlignedAllocator.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "x32_pad_test",
    srcs = [
        "test/x32-pad.cc",
        "test/pad-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "x32_unpool_test",
    srcs = [
        "test/x32-unpool.cc",
        "test/unpool-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "x32_zip_test",
    srcs = [
        "test/x32-zip.cc",
        "test/zip-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "x8_lut_test",
    srcs = [
        "test/x8-lut.cc",
        "test/lut-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

xnnpack_unit_test(
    name = "x8_zip_test",
    srcs = [
        "test/x8-zip.cc",
        "test/zip-microkernel-tester.h",
    ] + MICROKERNEL_TEST_HDRS,
    deps = MICROKERNEL_TEST_DEPS,
)

########################### Size test for the library ##########################

xnnpack_binary(
    name = "size_test",
    srcs = ["test/size.c"],
    deps = [":xnnpack_operators_nhwc_f32"],
)

########################### Unit tests for operators ###########################

xnnpack_unit_test(
    name = "add_test",
    srcs = [
        "test/add.cc",
        "test/add-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "argmax_pooling_test",
    srcs = [
        "test/argmax-pooling.cc",
        "test/argmax-pooling-operator-tester.h",
    ] + OPERATOR_TEST_PARAMS_HDRS,
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "average_pooling_test",
    srcs = [
        "test/average-pooling.cc",
        "test/average-pooling-operator-tester.h",
    ] + OPERATOR_TEST_PARAMS_HDRS,
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "channel_pad_test",
    srcs = [
        "test/channel-pad.cc",
        "test/channel-pad-operator-tester.h",
    ] + OPERATOR_TEST_PARAMS_HDRS,
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "channel_shuffle_test",
    srcs = [
        "test/channel-shuffle.cc",
        "test/channel-shuffle-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "clamp_test",
    srcs = [
        "test/clamp.cc",
        "test/clamp-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "convolution_test",
    srcs = [
        "test/convolution.cc",
        "test/convolution-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "convolution_spnchw_test",
    srcs = [
        "test/convolution-spnchw.cc",
        "test/convolution-spnchw-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "deconvolution_test",
    srcs = [
        "test/deconvolution.cc",
        "test/deconvolution-operator-tester.h",
    ] + OPERATOR_TEST_PARAMS_HDRS,
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "fully_connected_test",
    srcs = [
        "test/fully-connected.cc",
        "test/fully-connected-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "global_average_pooling_test",
    srcs = [
        "test/global-average-pooling.cc",
        "test/global-average-pooling-operator-tester.h",
    ] + OPERATOR_TEST_PARAMS_HDRS,
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "global_average_pooling_spnchw_test",
    srcs = [
        "test/global-average-pooling-spnchw.cc",
        "test/global-average-pooling-spnchw-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "hardswish_test",
    srcs = [
        "test/hardswish.cc",
        "test/hardswish-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "leaky_relu_test",
    srcs = [
        "test/leaky-relu.cc",
        "test/leaky-relu-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "max_pooling_test",
    srcs = [
        "test/max-pooling.cc",
        "test/max-pooling-operator-tester.h",
    ] + OPERATOR_TEST_PARAMS_HDRS,
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "prelu_test",
    srcs = [
        "test/prelu.cc",
        "test/prelu-operator-tester.h",
    ] + OPERATOR_TEST_PARAMS_HDRS,
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "sigmoid_test",
    srcs = [
        "test/sigmoid.cc",
        "test/sigmoid-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "softargmax_test",
    srcs = [
        "test/softargmax.cc",
        "test/softargmax-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

xnnpack_unit_test(
    name = "unpooling_test",
    srcs = [
        "test/unpooling.cc",
        "test/unpooling-operator-tester.h",
    ],
    deps = OPERATOR_TEST_DEPS,
)

############################# Build configurations #############################

config_setting(
    name = "linux_k8",
    values = {
        "cpu": "k8",
    },
)

config_setting(
    name = "linux_aarch64",
    values = {
        "cpu": "aarch64",
    },
)

config_setting(
    name = "android",
    values = {"crosstool_top": "//external:android/crosstool"},
)

config_setting(
    name = "android_armv7",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "armeabi-v7a",
    },
)

config_setting(
    name = "android_arm64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "arm64-v8a",
    },
)

config_setting(
    name = "android_x86",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "x86",
    },
)

config_setting(
    name = "android_x86_64",
    values = {
        "crosstool_top": "//external:android/crosstool",
        "cpu": "x86_64",
    },
)

config_setting(
    name = "macos_x86_64",
    values = {
        "apple_platform_type": "macos",
        "cpu": "darwin",
    },
)

config_setting(
    name = "emscripten",
    values = {"crosstool_top": "//toolchain:emscripten"},
)

config_setting(
    name = "emscripten_wasm",
    values = {
        "crosstool_top": "//toolchain:emscripten",
        "cpu": "wasm",
    },
)

config_setting(
    name = "emscripten_wasmsimd",
    values = {
        "crosstool_top": "//toolchain:emscripten",
        "cpu": "wasm",
        "features": "wasmsimd",
    },
)

config_setting(
    name = "emscripten_asmjs",
    values = {
        "crosstool_top": "//toolchain:emscripten",
        "cpu": "asmjs",
    },
)

# Builds with -c dbg
config_setting(
    name = "debug_build",
    values = {
        "compilation_mode": "dbg",
    },
)

# Builds with -c opt
config_setting(
    name = "optimized_build",
    values = {
        "compilation_mode": "opt",
    },
)

# Enables usage of assembly kernels.
config_setting(
    name = "xnn_enable_assembly_explicit_true",
    define_values = {"xnn_enable_assembly": "true"},
)

# Disables usage of assembly kernels.
config_setting(
    name = "xnn_enable_assembly_explicit_false",
    define_values = {"xnn_enable_assembly": "false"},
)
