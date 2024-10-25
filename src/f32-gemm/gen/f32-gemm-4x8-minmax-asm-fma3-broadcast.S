// void xnn_f32_gemm_minmax_ukernel_4x8__asm_fma3_broadcast2(
//     size_t mr,                                rdi
//     size_t nc,                                rsi
//     size_t kc,                                rdx
//     const float* restrict a,                  rcx
//     size_t a_stride,                          r8
//     const float* restrict w,                  r9
//     float* restrict c,             sp + 56 -> r10
//     size_t cm_stride,              sp + 64 -> r11
//     size_t cn_stride,              sp + 72 -> r12
//     xnn_f32_minmax_params params)  sp + 80 -> r13

// GP registers
// a0 rcx
// a1 r13
// a2 r14
// a3 r15
// c0 r10
// c1 rax
// c2 rbx
// c3 r0
// w r9
// k r11
// cm_stride r12
// cn_stride r13
//  SIMD registers
//  min xmm0
//  max xmm1
//  va0 xmm2
//  va1 xmm3
//  va2 xmm4
//  va3 xmm5
//  acc0 xmm6
//  acc1 xmm7
//  acc2 xmm8
//  acc3 xmm9
//  w xmm10

        .globl  xnn_f32_gemm_minmax_ukernel_asm_4x8__fma3_broadcast

        .text
xnn_f32_gemm_minmax_ukernel_asm_4x8__fma3_broadcast:
	# free up GP registers
	pushq %rbx
	pushq %rbp
	pushq %r15
	pushq %r14
	pushq %r13
	pushq %r12

	# load cn_stride
	movq 72(%rsp), %r12 # cn_stride
	# load params to free up a GP registers
	movq 80(%rsp), %r13 # params
	vbroadcastss (%r13), %ymm0
	vbroadcastss 4(%r13), %ymm1
	# a1 = a0 + a_stride
	movq %rcx, %r13
	addq %r8, %r13
	# a2 = a1 + a_stride
	movq %r13, %r14
	addq %r8, %r14
	# a3 = a2 + a_stride
	movq %r14, %r15
	addq %r8, %r15
	# r8 can now be used
	# rax, rbx, r8
	movq 56(%rsp), %r10 # c
	movq 64(%rsp), %r11 # cm_stride
	# c1 = c0 + cm_stride
	movq %r10, %rax
	addq %r11, %rax
	# c2 = c1 + cm_stride
	movq %rax, %rbx
	addq %r11, %rbx
	# c3 = c2 + cm_stride
	movq %rbx, %r8
	addq %r11, %r8
	cmp $2, %rdi
	jl 1f
	je 2f
	cmp  $4, %rdi
	jl 3f
	je 4f
1: # mr < 2
	movq %rcx, %r13
	movq %r10, %rax
2: # mr == 2
	movq %r13, %r14
	movq %rax, %rbx
3: # mr == 3
	movq %r14, %r15
	movq %rbx, %r8
4: # outer loop
	# load bias
	vmovups  (%r9), %ymm6
	vmovups  %ymm6, %ymm7
	vmovups  %ymm6, %ymm8
	vmovups  %ymm6, %ymm9
	addq $32, %r9
	# k = kc, cm_stride is no longer needed
	movq %rdx, %r11
5: # inner loop
	# broadcast a0, a1, a2, a3
	vbroadcastss  (%rcx), %ymm2
	vbroadcastss  (%r13), %ymm3
	vbroadcastss  (%r14), %ymm4
	vbroadcastss  (%r15), %ymm5
	addq $4, %rcx
	addq $4, %r13
	addq $4, %r14
	addq $4, %r15
	# load w
	vmovups  (%r9), %ymm10
	addq $32, %r9
	vfmadd231ps  %ymm2, %ymm10, %ymm6
	vfmadd231ps  %ymm3, %ymm10, %ymm7
	vfmadd231ps  %ymm4, %ymm10, %ymm8
	vfmadd231ps  %ymm5, %ymm10, %ymm9
	subq $4, %r11
	jne 5b
	# clamp
	vminps %ymm1, %ymm6, %ymm6
	vminps %ymm1, %ymm7, %ymm7
	vminps %ymm1, %ymm8, %ymm8
	vminps %ymm1, %ymm9, %ymm9
	vmaxps %ymm0, %ymm6, %ymm6
	vmaxps %ymm0, %ymm7, %ymm7
	vmaxps %ymm0, %ymm8, %ymm8
	vmaxps %ymm0, %ymm9, %ymm9

	# store
	cmpq $8, %rsi
	jl 6f
	vmovups  %ymm6, (%r10)
	vmovups  %ymm7, (%rax)
	vmovups  %ymm8, (%rbx)
	vmovups  %ymm9, (%r8)
	# c0 = c0 + cn_stride
	addq %r12, %r10
	addq %r12, %rax
	addq %r12, %rbx
	addq %r12, %r8
	# a = a0 - kc
	subq %rdx, %rcx
	subq %rdx, %r13
	subq %rdx, %r14
	subq %rdx, %r15
	subq $8, %rsi
	jne 4b
	jmp 9f
6: # store tail
	testb $4, %sil
	jz 7f
	vmovups  %xmm6, (%r10)
	vmovups  %xmm7, (%rax)
	vmovups  %xmm8, (%rbx)
	vmovups  %xmm9, (%r8)
	addq $16, %r10
	addq $16, %rax
	addq $16, %rbx
	addq $16, %r8
	vextractf128 $1, %ymm6, %xmm6
	vextractf128 $1, %ymm7, %xmm7
	vextractf128 $1, %ymm8, %xmm8
	vextractf128 $1, %ymm9, %xmm9
7:
	testb $2, %sil
	jz 8f
	vmovlps  %xmm6, (%r10)
	vmovlps  %xmm7, (%rax)
	vmovlps  %xmm8, (%rbx)
	vmovlps  %xmm9, (%r8)
	addq $8, %r10
	addq $8, %rax
	addq $8, %rbx
	addq $8, %r8
	vmovhlps %xmm6, %xmm6, %xmm6
	vmovhlps %xmm7, %xmm7, %xmm7
	vmovhlps %xmm8, %xmm8, %xmm8
	vmovhlps %xmm9, %xmm9, %xmm9
8:
	testb $1, %sil
	jz 9f
	vmovss  %xmm6, (%r10)
	vmovss  %xmm7, (%rax)
	vmovss  %xmm8, (%rbx)
	vmovss  %xmm9, (%r8)

9: # end
	# restore the callee saved registers
 	popq %r12
 	popq %r13
 	popq %r14
 	popq %r15
	popq %rbp
	popq %rbx
        ret

