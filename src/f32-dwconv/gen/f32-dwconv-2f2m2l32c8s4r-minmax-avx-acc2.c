// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/multipass-avx.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f32_dwconv_minmax_ukernel_2f2m2l32c8s4r__avx_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    size_t kernel_size,
    float* buffer,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 2);

  const __m256 vmax = _mm256_load_ps(params->avx.max);
  const __m256 vmin = _mm256_load_ps(params->avx.min);
  do {
    const float* w = weights;

    // First pass to process 2 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      input += 2;

      // Process c channels and write to buffer.
      size_t c = round_up_po2(channels, 4);
      for (; c >= 32; c -= 32) {
        __m256 vacc01234567p0 = _mm256_load_ps(w);
        __m256 vacc89ABCDEFp0 = _mm256_load_ps(w + 8);
        __m256 vaccGHIJKLMNp0 = _mm256_load_ps(w + 16);
        __m256 vaccOPQRSTUVp0 = _mm256_load_ps(w + 24);


        const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
        const __m256 vi0x89ABCDEF = _mm256_loadu_ps(i0 + 8);
        const __m256 vi0xGHIJKLMN = _mm256_loadu_ps(i0 + 16);
        const __m256 vi0xOPQRSTUV = _mm256_loadu_ps(i0 + 24);
        i0 += 32;

        const __m256 vk0x01234567 = _mm256_load_ps(w + 32);
        const __m256 vk0x89ABCDEF = _mm256_load_ps(w + 40);
        const __m256 vk0xGHIJKLMN = _mm256_load_ps(w + 48);
        const __m256 vk0xOPQRSTUV = _mm256_load_ps(w + 56);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi0x89ABCDEF, vk0x89ABCDEF));
        vaccGHIJKLMNp0 = _mm256_add_ps(vaccGHIJKLMNp0, _mm256_mul_ps(vi0xGHIJKLMN, vk0xGHIJKLMN));
        vaccOPQRSTUVp0 = _mm256_add_ps(vaccOPQRSTUVp0, _mm256_mul_ps(vi0xOPQRSTUV, vk0xOPQRSTUV));

        const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
        const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
        const __m256 vi1xGHIJKLMN = _mm256_loadu_ps(i1 + 16);
        const __m256 vi1xOPQRSTUV = _mm256_loadu_ps(i1 + 24);
        i1 += 32;

        const __m256 vk1x01234567 = _mm256_load_ps(w + 64);
        const __m256 vk1x89ABCDEF = _mm256_load_ps(w + 72);
        const __m256 vk1xGHIJKLMN = _mm256_load_ps(w + 80);
        const __m256 vk1xOPQRSTUV = _mm256_load_ps(w + 88);
        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);
        __m256 vacc89ABCDEFp1 = _mm256_mul_ps(vi1x89ABCDEF, vk1x89ABCDEF);
        __m256 vaccGHIJKLMNp1 = _mm256_mul_ps(vi1xGHIJKLMN, vk1xGHIJKLMN);
        __m256 vaccOPQRSTUVp1 = _mm256_mul_ps(vi1xOPQRSTUV, vk1xOPQRSTUV);

        w += 96;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, vacc89ABCDEFp1);
        vaccGHIJKLMNp0 = _mm256_add_ps(vaccGHIJKLMNp0, vaccGHIJKLMNp1);
        vaccOPQRSTUVp0 = _mm256_add_ps(vaccOPQRSTUVp0, vaccOPQRSTUVp1);

        _mm256_store_ps(b, vacc01234567p0);
        _mm256_store_ps(b + 8, vacc89ABCDEFp0);
        _mm256_store_ps(b + 16, vaccGHIJKLMNp0);
        _mm256_store_ps(b + 24, vaccOPQRSTUVp0);
        b += 32;
      }

      for (; c >= 8; c -= 8) {
        __m256 vacc01234567p0 = _mm256_load_ps(w);


        const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
        i0 += 8;

        const __m256 vk0x01234567 = _mm256_load_ps(w + 8);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

        const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
        i1 += 8;

        const __m256 vk1x01234567 = _mm256_load_ps(w + 16);
        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);

        w += 24;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);

        _mm256_store_ps(b, vacc01234567p0);
        b += 8;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 7);
        const __m256i vmask = _mm256_loadu_si256((const __m256i*) &params->avx.mask_table[7 - c]);
        __m256 vacc01234567p0 = _mm256_load_ps(w);


        const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);

        const __m256 vk0x01234567 = _mm256_load_ps(w + 8);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

        const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);

        const __m256 vk1x01234567 = _mm256_load_ps(w + 16);
        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);

        w += 24;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);

        _mm256_store_ps(b, vacc01234567p0);
      }
    }

    // Middle pass to process 2 inputs in each iteration.
    for (size_t ks = kernel_size - 2; ks > 2; ks -= 2) {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      input += 2;

      size_t c = round_up_po2(channels, 4);
      for (; c >= 32; c -= 32) {
        __m256 vacc01234567p0 = _mm256_load_ps(b);
        __m256 vacc89ABCDEFp0 = _mm256_load_ps(b + 8);
        __m256 vaccGHIJKLMNp0 = _mm256_load_ps(b + 16);
        __m256 vaccOPQRSTUVp0 = _mm256_load_ps(b + 24);


        const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
        const __m256 vi0x89ABCDEF = _mm256_loadu_ps(i0 + 8);
        const __m256 vi0xGHIJKLMN = _mm256_loadu_ps(i0 + 16);
        const __m256 vi0xOPQRSTUV = _mm256_loadu_ps(i0 + 24);
        i0 += 32;

        const __m256 vk0x01234567 = _mm256_load_ps(w);
        const __m256 vk0x89ABCDEF = _mm256_load_ps(w + 8);
        const __m256 vk0xGHIJKLMN = _mm256_load_ps(w + 16);
        const __m256 vk0xOPQRSTUV = _mm256_load_ps(w + 24);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi0x89ABCDEF, vk0x89ABCDEF));
        vaccGHIJKLMNp0 = _mm256_add_ps(vaccGHIJKLMNp0, _mm256_mul_ps(vi0xGHIJKLMN, vk0xGHIJKLMN));
        vaccOPQRSTUVp0 = _mm256_add_ps(vaccOPQRSTUVp0, _mm256_mul_ps(vi0xOPQRSTUV, vk0xOPQRSTUV));

        const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
        const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
        const __m256 vi1xGHIJKLMN = _mm256_loadu_ps(i1 + 16);
        const __m256 vi1xOPQRSTUV = _mm256_loadu_ps(i1 + 24);
        i1 += 32;

        const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
        const __m256 vk1x89ABCDEF = _mm256_load_ps(w + 40);
        const __m256 vk1xGHIJKLMN = _mm256_load_ps(w + 48);
        const __m256 vk1xOPQRSTUV = _mm256_load_ps(w + 56);
        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);
        __m256 vacc89ABCDEFp1 = _mm256_mul_ps(vi1x89ABCDEF, vk1x89ABCDEF);
        __m256 vaccGHIJKLMNp1 = _mm256_mul_ps(vi1xGHIJKLMN, vk1xGHIJKLMN);
        __m256 vaccOPQRSTUVp1 = _mm256_mul_ps(vi1xOPQRSTUV, vk1xOPQRSTUV);

        w += 64;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, vacc89ABCDEFp1);
        vaccGHIJKLMNp0 = _mm256_add_ps(vaccGHIJKLMNp0, vaccGHIJKLMNp1);
        vaccOPQRSTUVp0 = _mm256_add_ps(vaccOPQRSTUVp0, vaccOPQRSTUVp1);

        _mm256_store_ps(b, vacc01234567p0);
        _mm256_store_ps(b + 8, vacc89ABCDEFp0);
        _mm256_store_ps(b + 16, vaccGHIJKLMNp0);
        _mm256_store_ps(b + 24, vaccOPQRSTUVp0);
        b += 32;
      }

      for (; c >= 8; c -= 8) {
        __m256 vacc01234567p0 = _mm256_load_ps(b);


        const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
        i0 += 8;

        const __m256 vk0x01234567 = _mm256_load_ps(w);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

        const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
        i1 += 8;

        const __m256 vk1x01234567 = _mm256_load_ps(w + 8);
        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);

        w += 16;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);

        _mm256_store_ps(b, vacc01234567p0);
        b += 8;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 7);
        const __m256i vmask = _mm256_loadu_si256((const __m256i*) &params->avx.mask_table[7 - c]);
        __m256 vacc01234567p0 = _mm256_load_ps(b);


        const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);

        const __m256 vk0x01234567 = _mm256_load_ps(w);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

        const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);

        const __m256 vk1x01234567 = _mm256_load_ps(w + 8);
        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);

        w += 16;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);

        _mm256_store_ps(b, vacc01234567p0);
      }
    }

    // Last pass to process up to 2 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }

      size_t c = channels;
      for (; c >= 32; c -= 32) {
        __m256 vacc01234567p0 = _mm256_load_ps(b);
        __m256 vacc89ABCDEFp0 = _mm256_load_ps(b + 8);
        __m256 vaccGHIJKLMNp0 = _mm256_load_ps(b + 16);
        __m256 vaccOPQRSTUVp0 = _mm256_load_ps(b + 24);
        b += 32;


        const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
        const __m256 vi0x89ABCDEF = _mm256_loadu_ps(i0 + 8);
        const __m256 vi0xGHIJKLMN = _mm256_loadu_ps(i0 + 16);
        const __m256 vi0xOPQRSTUV = _mm256_loadu_ps(i0 + 24);
        i0 += 32;

        __m256 vk0x01234567 = _mm256_load_ps(w);
        __m256 vk0x89ABCDEF = _mm256_load_ps(w + 8);
        __m256 vk0xGHIJKLMN = _mm256_load_ps(w + 16);
        __m256 vk0xOPQRSTUV = _mm256_load_ps(w + 24);

        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, _mm256_mul_ps(vi0x89ABCDEF, vk0x89ABCDEF));
        vaccGHIJKLMNp0 = _mm256_add_ps(vaccGHIJKLMNp0, _mm256_mul_ps(vi0xGHIJKLMN, vk0xGHIJKLMN));
        vaccOPQRSTUVp0 = _mm256_add_ps(vaccOPQRSTUVp0, _mm256_mul_ps(vi0xOPQRSTUV, vk0xOPQRSTUV));

        const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
        const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
        const __m256 vi1xGHIJKLMN = _mm256_loadu_ps(i1 + 16);
        const __m256 vi1xOPQRSTUV = _mm256_loadu_ps(i1 + 24);
        i1 += 32;

        __m256 vk1x01234567 = _mm256_load_ps(w + 32);
        __m256 vk1x89ABCDEF = _mm256_load_ps(w + 40);
        __m256 vk1xGHIJKLMN = _mm256_load_ps(w + 48);
        __m256 vk1xOPQRSTUV = _mm256_load_ps(w + 56);

        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);
        __m256 vacc89ABCDEFp1 = _mm256_mul_ps(vi1x89ABCDEF, vk1x89ABCDEF);
        __m256 vaccGHIJKLMNp1 = _mm256_mul_ps(vi1xGHIJKLMN, vk1xGHIJKLMN);
        __m256 vaccOPQRSTUVp1 = _mm256_mul_ps(vi1xOPQRSTUV, vk1xOPQRSTUV);

        w += 64;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);
        vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, vacc89ABCDEFp1);
        vaccGHIJKLMNp0 = _mm256_add_ps(vaccGHIJKLMNp0, vaccGHIJKLMNp1);
        vaccOPQRSTUVp0 = _mm256_add_ps(vaccOPQRSTUVp0, vaccOPQRSTUVp1);

        __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
        __m256 vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEFp0, vmin);
        __m256 vaccGHIJKLMN = _mm256_max_ps(vaccGHIJKLMNp0, vmin);
        __m256 vaccOPQRSTUV = _mm256_max_ps(vaccOPQRSTUVp0, vmin);

        vacc01234567 = _mm256_min_ps(vacc01234567, vmax);
        vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vmax);
        vaccGHIJKLMN = _mm256_min_ps(vaccGHIJKLMN, vmax);
        vaccOPQRSTUV = _mm256_min_ps(vaccOPQRSTUV, vmax);

        _mm256_storeu_ps(output, vacc01234567);
        _mm256_storeu_ps(output + 8, vacc89ABCDEF);
        _mm256_storeu_ps(output + 16, vaccGHIJKLMN);
        _mm256_storeu_ps(output + 24, vaccOPQRSTUV);
        output += 32;
      }


      for (; c >= 8; c -= 8) {
        __m256 vacc01234567p0 = _mm256_load_ps(b);
        b += 8;


        const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
        i0 += 8;

        __m256 vk0x01234567 = _mm256_load_ps(w);

        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

        const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
        i1 += 8;

        __m256 vk1x01234567 = _mm256_load_ps(w + 8);

        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);

        w += 16;


        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);

        __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);

        vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

        _mm256_storeu_ps(output, vacc01234567);
        output += 8;
      }

      if XNN_UNLIKELY(c != 0) {
        assert(c >= 1);
        assert(c <= 7);
        __m256 vacc01234567p0 = _mm256_load_ps(b);
        const __m256i vmask = _mm256_loadu_si256((const __m256i*) &params->avx.mask_table[7 - c]);

        const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);
        __m256 vk0x01234567 = _mm256_load_ps(w);
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, _mm256_mul_ps(vi0x01234567, vk0x01234567));

        const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
        __m256 vk1x01234567 = _mm256_load_ps(w + 8);
        __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);

        __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
        vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

        __m128 vacc0123 = _mm256_castps256_ps128(vacc01234567);
        if (c & 4) {
          _mm_storeu_ps(output, vacc0123);
          vacc0123 = _mm256_extractf128_ps(vacc01234567, 1);
          output += 4;
        }
        if (c & 2) {
          _mm_storel_pi((__m64*) output, vacc0123);
          vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
          output += 2;
        }
        if (c & 1) {
          _mm_store_ss(output, vacc0123);
          output += 1;
        }
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
