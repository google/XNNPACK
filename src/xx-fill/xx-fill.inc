// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_FILL_UKERNEL(xnn_arch_arm_neon, xnn_xx_fill_ukernel__neon_u64)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
XNN_FILL_UKERNEL(0, xnn_xx_fill_ukernel__sse2_u64)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
XNN_FILL_UKERNEL(0, xnn_xx_fill_ukernel__wasmsimd_u64)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

XNN_FILL_UKERNEL(0, xnn_xx_fill_ukernel__scalar_u16)
