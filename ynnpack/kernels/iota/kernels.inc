// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifdef YNN_ARCH_X86_AVX512
YNN_IOTA_KERNEL(arch_flag::avx512, ynn_iota_fp32_avx512, float)
YNN_IOTA_KERNEL(arch_flag::avx512, ynn_iota_int32_avx512, int32_t)
#endif
#ifdef YNN_ARCH_X86_AVX2
YNN_IOTA_KERNEL(arch_flag::avx2, ynn_iota_int32_avx2, int32_t)
#endif
#ifdef YNN_ARCH_X86_AVX
YNN_IOTA_KERNEL(arch_flag::avx, ynn_iota_fp32_avx, float)
#endif
#ifdef YNN_ARCH_X86_SSE41
YNN_IOTA_KERNEL(arch_flag::sse41, ynn_iota_int32_sse41, int32_t)
#endif
#ifdef YNN_ARCH_X86_SSE2
YNN_IOTA_KERNEL(arch_flag::sse2, ynn_iota_fp32_sse2, float)
#endif

#ifdef YNN_ARCH_ARM_NEON
YNN_IOTA_KERNEL(arch_flag::neon, ynn_iota_fp32_neon, float)
YNN_IOTA_KERNEL(arch_flag::neon, ynn_iota_int32_neon, int32_t)
#endif

YNN_IOTA_KERNEL(arch_flag::none, ynn_iota_int32, int32_t)
YNN_IOTA_KERNEL(arch_flag::none, ynn_iota_fp32, float)
