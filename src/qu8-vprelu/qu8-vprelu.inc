// clang-format off
// Copyright (C) 2024 Intel Corporation
//  
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//  
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//  
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//  
//  
// SPDX-License-Identifier: BSD-3-Clause


#if XNN_ENABLE_AVX2 && (XNN_ARCH_X86_64 || XNN_ARCH_X86)
XNN_UKERNEL(xnn_arch_x86_avx2, xnn_qu8_vprelu_ukernel__avx2_u16, 16, false, uint8_t, union xnn_qs8_vprelu_scalar_params, xnn_init_qu8_vprelu_scalar_params)
#endif  // XNN_ENABLE_AVX2 && (XNN_ARCH_X86 || XNN_ARCH_X86_64)

XNN_UKERNEL(xnn_arch_none, xnn_qu8_vprelu_ukernel__scalar_u1, 1, false, uint8_t, union xnn_qs8_vprelu_scalar_params, xnn_init_qu8_vprelu_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_vprelu_ukernel__scalar_u2, 2, false, uint8_t, union xnn_qs8_vprelu_scalar_params, xnn_init_qu8_vprelu_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_vprelu_ukernel__scalar_u4, 4, false, uint8_t, union xnn_qs8_vprelu_scalar_params, xnn_init_qu8_vprelu_scalar_params)
XNN_UKERNEL(xnn_arch_none, xnn_qu8_vprelu_ukernel__scalar_u8, 8, false, uint8_t, union xnn_qs8_vprelu_scalar_params, xnn_init_qu8_vprelu_scalar_params)


