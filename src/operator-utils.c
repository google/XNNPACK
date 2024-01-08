// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack.h>           // For xnn_operator_t.
#include <xnnpack/common.h>    // For XNN_ALLOCATION_ALIGNMENT.
#include <xnnpack/cache.h>     // For xnn_code_cache.
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>  // For xnn_operator definition.
#include <xnnpack/operator-utils.h>

#if XNN_PLATFORM_JIT
// Generate code for a single set of parameters.
// Code is generated into the code cache, and the offset of the generated code is returned.
// If code already exists in code cache, the offset of the existing code is returned.
// Stores the value XNN_CACHE_NOT_FOUND in `offset` field when no code is generated.
static enum xnn_status get_generated_gemm(
    xnn_jit_gemm_code_generator_fn generator,
    const struct jit_gemm_params *jit_gemm_params,
    size_t mr,
    size_t group_output_channels,
    size_t nr,
    size_t group_input_channels_in_bytes,
    struct xnn_code_cache* code_cache,
    struct xnn_generated_code_chunk* code_chunk)
{
  assert(code_cache != NULL);
  size_t offset = XNN_CACHE_NOT_FOUND;
  enum xnn_status status = xnn_status_success;
  if (generator == NULL) {
    status = xnn_status_uninitialized;
    goto error;
  }

  status = xnn_reserve_code_memory(&code_cache->cache.code, XNN_DEFAULT_MICROKERNEL_SIZE);
  if (xnn_status_success != status) {
    xnn_log_error("failed to ensure sufficient space in the code buffer for a microkernel");
    goto error;
  }

  const size_t old_size = code_cache->cache.code.size;
  void* old_code = (uint8_t*) code_cache->cache.code.start + old_size;
  status = generator(&code_cache->cache.code, mr, group_output_channels % nr,
                     group_input_channels_in_bytes, jit_gemm_params);

  if (xnn_status_success != status) {
    xnn_log_error("failed to generate GEMM microkernel");
    goto error;
  }

  const size_t new_size = code_cache->cache.code.size;
  const size_t code_size = new_size - old_size;
  offset = xnn_get_or_insert_code_cache(code_cache, old_code, code_size);
  *code_chunk = (struct xnn_generated_code_chunk) {offset, offset + code_size};
  return xnn_status_success;

error:
  *code_chunk = (struct xnn_generated_code_chunk) {offset, offset};
  return status;
}


void xnn_generate_gemms_up_to_max_mr(
  size_t max_mr,
  struct gemm_codegens generators,
  const struct jit_gemm_params *jit_gemm_params,
  size_t group_output_channels,
  size_t nr,
  size_t group_input_channels_in_bytes,
  xnn_operator_t op)
{
  assert(XNN_MAX_MR >= max_mr);
  if (op->code_cache == NULL || !xnn_code_cache_valid(op->code_cache)) {
    return;
  }
  for (size_t mr = 1; mr <= max_mr; mr++) {
    // Get smallest generator that is >= mr.
    size_t smallest_mr = mr;
    while (generators.gemm[smallest_mr - 1].function[XNN_UARCH_DEFAULT] == NULL && smallest_mr < max_mr) {
      smallest_mr++;
    }

    for (size_t i = 0; i < XNN_MAX_UARCH_TYPES; i++) {
      xnn_log_debug("using generator for mr %zu to generate gemm of mr %zu and uarch %zu", smallest_mr, mr, i);
      get_generated_gemm(generators.gemm[smallest_mr - 1].function[i],
                           jit_gemm_params, mr, group_output_channels, nr, group_input_channels_in_bytes, op->code_cache,
                           &op->ukernel.gemm.gemm_cases[mr - 1].generated_code_chunk[i]);
    }
  }
}

static enum xnn_status get_generated_igemm(
  xnn_jit_igemm_code_generator_fn generator,
  const struct jit_gemm_params *jit_gemm_params,
  size_t group_output_channels,
  size_t nr,
  size_t group_input_channels_in_bytes,
  size_t kernel_size,
  size_t mr,
  struct xnn_code_cache* code_cache,
  struct xnn_generated_code_chunk* code_chunk)
{
  size_t offset = XNN_CACHE_NOT_FOUND;
  enum xnn_status status = xnn_status_success;
  if (generator == NULL) {
    status = xnn_status_uninitialized;
    goto error;
  }

  status = xnn_reserve_code_memory(&code_cache->cache.code, XNN_DEFAULT_MICROKERNEL_SIZE);
  if (xnn_status_success != status) {
    xnn_log_error("failed to ensure sufficient space in code buffer for microkernel");
    goto error;
  }

  const size_t old_size = code_cache->cache.code.size;
  void* old_code = (uint8_t*) code_cache->cache.code.start + old_size;
  status = generator(&code_cache->cache.code, mr, group_output_channels % nr,
                     group_input_channels_in_bytes,
                     kernel_size * mr * sizeof(void*), jit_gemm_params);
  if (status != xnn_status_success) {
    xnn_log_error("failed to generate IGEMM microkernel");
    goto error;
  }

  const size_t new_size = code_cache->cache.code.size;
  const size_t code_size = new_size - old_size;
  offset = xnn_get_or_insert_code_cache(code_cache, old_code, code_size);
  *code_chunk = (struct xnn_generated_code_chunk) {offset, offset + code_size};
  return xnn_status_success;

error:
  *code_chunk = (struct xnn_generated_code_chunk) {offset, offset};
  return status;
}

void xnn_generate_igemms_up_to_max_mr(
  size_t max_mr,
  struct gemm_codegens generators,
  const struct jit_gemm_params *jit_gemm_params,
  size_t group_output_channels,
  size_t nr,
  size_t group_input_channels_in_bytes,
  size_t kernel_size,
  xnn_operator_t op)
{
  assert(XNN_MAX_MR >= max_mr);
  if (op->code_cache == NULL || !xnn_code_cache_valid(op->code_cache)) {
    return;
  }
  for (size_t mr = 1; mr <= max_mr; mr++) {
    // Get smallest generator that is >= mr.
    size_t smallest_mr = mr;
    while (generators.igemm[smallest_mr - 1].function[XNN_UARCH_DEFAULT] == NULL && smallest_mr < max_mr) {
      smallest_mr++;
    }

    for (size_t i = 0; i < XNN_MAX_UARCH_TYPES; i++) {
      xnn_log_debug("using generator for mr %zu to generate igemm of mr %zu and uarch %zu", smallest_mr, mr, i);
        get_generated_igemm(generators.igemm[smallest_mr - 1].function[i], jit_gemm_params,
                            group_output_channels, nr, group_input_channels_in_bytes, kernel_size, mr,
                            op->code_cache, &op->ukernel.igemm.igemm_cases[mr - 1].generated_code_chunk[i]);
    }
  }
}

static inline uintptr_t cached_code_at_offset(xnn_operator_t op, size_t offset)
{
  return (uintptr_t)op->code_cache->cache.code.start + offset;
}

void xnn_overwrite_gemm_cases_with_generated_code(
  xnn_operator_t op,
  struct xnn_hmp_gemm_ukernel *gemm_cases,
  size_t mr)
{
  if (op->code_cache == NULL) {
    return;
  }

  for (size_t i = 0; i < XNN_MAX_UARCH_TYPES; i++) {
    const struct xnn_generated_code_chunk chunk = gemm_cases[mr - 1].generated_code_chunk[i];
    if (chunk.offset != XNN_CACHE_NOT_FOUND) {
      const uintptr_t gemm_kernel = xnn_first_function_in_chunk_ptr(&op->code_cache->cache.code, chunk.offset, chunk.offset_end);
      if (gemm_kernel == (uintptr_t) XNN_INVALID_FUNCTION_INDEX) {
        xnn_log_warning("failed to finalize gemm kernel code");
        continue;
      }
      gemm_cases[mr - 1].function[i] = (xnn_gemm_ukernel_fn) gemm_kernel;
    }
  }
}

void xnn_overwrite_igemm_cases_with_generated_code(
  xnn_operator_t op,
  struct xnn_hmp_igemm_ukernel *igemm_cases,
  size_t mr)
{
  if (op->code_cache == NULL) {
    return;
  }

  for (size_t i = 0; i < XNN_MAX_UARCH_TYPES; i++) {
    const struct xnn_generated_code_chunk chunk = igemm_cases[mr - 1].generated_code_chunk[i];
    const uintptr_t gemm_kernel = xnn_first_function_in_chunk_ptr(&op->code_cache->cache.code, chunk.offset, chunk.offset_end);
    if (gemm_kernel == (uintptr_t) XNN_INVALID_FUNCTION_INDEX) {
      xnn_log_warning("failed to finalize igemm kernel code");
      continue;
    }
    igemm_cases[mr - 1].function[i] = (xnn_igemm_ukernel_fn) gemm_kernel;
  }
}
#endif  // XNN_PLATFORM_JIT

void* xnn_get_pointer_to_write_weights(
  xnn_operator_t op,
  size_t aligned_weights_size,
  int padding_byte)
{
  assert(aligned_weights_size % XNN_ALLOCATION_ALIGNMENT == 0);
  void* weights_ptr = NULL;
  if (use_weights_cache(op)) {
    weights_ptr = op->weights_cache->reserve_space(op->weights_cache->context, aligned_weights_size);
    if (weights_ptr == NULL) {
      return NULL;
    }
  } else {
    op->packed_weights.pointer = xnn_allocate_simd_memory(aligned_weights_size);
    if (op->packed_weights.pointer == NULL) {
      return NULL;
    }
    weights_ptr = op->packed_weights.pointer;
  }
  memset(weights_ptr, padding_byte, aligned_weights_size);
  return weights_ptr;
}

size_t xnn_compute_convolution_output_dimension(
  size_t padded_input_dimension,
  size_t kernel_dimension,
  size_t dilation_dimension,
  size_t subsampling_dimension)
{
  const size_t effective_kernel_dimension = (kernel_dimension - 1) * dilation_dimension + 1;
  return doz(padded_input_dimension, effective_kernel_dimension) / subsampling_dimension + 1;
}

size_t xnn_compute_deconvolution_output_dimension(
  size_t input_dimension,
  size_t output_padding_dimension,
  size_t adjustment_dimension,
  size_t kernel_dimension,
  size_t dilation_dimension,
  size_t stride_dimension)
{
  const size_t effective_kernel_dimension = (kernel_dimension - 1) * dilation_dimension + 1;
  return doz(
    stride_dimension * (input_dimension - 1) + adjustment_dimension + effective_kernel_dimension,
    output_padding_dimension);
}

size_t xnn_compute_unpooling_output_dimension(
    size_t input_dimension,
    size_t input_padding_dimension,
    size_t kernel_dimension)
{
  return xnn_compute_deconvolution_output_dimension(
      input_dimension, input_padding_dimension, /*adjustment_dimension=*/0,
      kernel_dimension, /*dilation_dimension=*/1, /*stride_dimension=*/kernel_dimension);
}

// Calculate how much work a microkernel does.
// A MxN microkernel does M+N (scalar) loads and M*N (scalar) FMAs.
// So, given batch_size, the microkernel does:
//   divide_round_up(batch_size, mr) * (mr + nr) loads, and
//   divide_round_up(batch_size, mr) * (mr * nr) FMAs.
// The total cost is then a linear combination of these 2 operations. From experimental data, use a multiplier of 3 for
// loads, to prefer higher tile sizes which have better computation intensity.
static size_t calculate_microkernel_cost(size_t batch_size, uint32_t mr, uint32_t nr)
{
  return divide_round_up(batch_size, mr) * (3 * (mr + nr) + mr * nr);
}

static bool mr_is_available_gemm(size_t mr, struct xnn_hmp_gemm_ukernel *gemm_cases, bool code_cache_available)
{
  #if XNN_PLATFORM_JIT
    if (code_cache_available) {
      return gemm_cases[mr-1].generated_code_chunk[XNN_UARCH_DEFAULT].offset != XNN_CACHE_NOT_FOUND ||
          gemm_cases[mr-1].function[XNN_UARCH_DEFAULT] != NULL;
    }
  #endif
  return gemm_cases[mr-1].function[XNN_UARCH_DEFAULT] != NULL;
}

uint32_t xnn_get_heuristic_mr_gemm(
  size_t batch_size, uint32_t max_mr, uint32_t nr, struct xnn_hmp_gemm_ukernel *gemm_cases, bool code_cache_available)
{
  if (batch_size <= max_mr && mr_is_available_gemm(batch_size, gemm_cases, code_cache_available)) {
    // We have a microkernel with MR that is the exact match with batch_size.
    return batch_size;
  }

  // Try to find the best fitting mr.
  // - use a cost heuristic to calculate how much work is done by the microkernel (see calculate_microkernel_cost)
  // - smaller cost is better
  uint32_t best_mr = max_mr;
  size_t best_cost = SIZE_MAX;
  for (uint32_t mr = 1; mr <= max_mr; mr++) {
    if (!mr_is_available_gemm(mr, gemm_cases, code_cache_available)){
      continue;
    }
    const size_t current_cost = calculate_microkernel_cost(batch_size, mr, nr);
    if (current_cost <= best_cost) {
      best_mr = mr;
      best_cost = current_cost;
    }
  }
  return best_mr;
}

static bool mr_is_available_igemm(size_t mr, struct xnn_hmp_igemm_ukernel *igemm_cases, bool code_cache_available)
{
  #if XNN_PLATFORM_JIT
    if (code_cache_available) {
      return igemm_cases[mr-1].generated_code_chunk[XNN_UARCH_DEFAULT].offset != XNN_CACHE_NOT_FOUND ||
          igemm_cases[mr-1].function[XNN_UARCH_DEFAULT] != NULL;
    }
  #endif
  return igemm_cases[mr-1].function[XNN_UARCH_DEFAULT] != NULL;
}

uint32_t xnn_get_heuristic_mr_igemm(
  size_t batch_size, uint32_t max_mr, uint32_t nr, struct xnn_hmp_igemm_ukernel *igemm_cases,
  bool code_cache_available)
{
  if (batch_size <= max_mr && mr_is_available_igemm(batch_size, igemm_cases, code_cache_available)) {
    // We have a microkernel with MR that is the exact match with batch_size.
    return batch_size;
  }

  // Try to find the best fitting mr.
  // - use a cost heuristic to calculate how much work is done by the microkernel (see calculate_microkernel_cost)
  // - smaller cost is better
  uint32_t best_mr = max_mr;
  size_t best_cost = SIZE_MAX;
  for (uint32_t mr = 1; mr <= max_mr; mr++) {
    if (!mr_is_available_igemm(mr, igemm_cases, code_cache_available)){
      continue;
    }
    const size_t current_cost = calculate_microkernel_cost(batch_size, mr, nr);
    if (current_cost <= best_cost) {
      best_mr = mr;
      best_cost = current_cost;
    }
  }
  return best_mr;
}
