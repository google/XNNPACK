// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_PAD_UKERNEL(xnn_arch_arm_neon, xnn_xx_pad_ukernel_p16__neon_u16, 16)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_PAD_UKERNEL(0, xnn_xx_pad_ukernel_p16__sse2_u16, 16)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_PAD_UKERNEL(0, xnn_xx_pad_ukernel_p16__wasmsimd_u16, 16)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_PAD_UKERNEL(0, xnn_xx_pad_ukernel_p4__scalar_u16, 4)
