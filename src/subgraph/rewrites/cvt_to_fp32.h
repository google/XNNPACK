#ifndef XNNPACK_SRC_SUBGRAPH_REWRITES_CVT_TO_FP32_H_
#define XNNPACK_SRC_SUBGRAPH_REWRITES_CVT_TO_FP32_H_

#include <stdbool.h>

#include "include/xnnpack.h"

#ifdef __cplusplus
extern "C" {
#endif

// Rewrites unsupported fp16 operations to fp32.
//
// Inserts fp16-fp32 converts for inputs and fp32-fp16 converts for outputs.
// This tries to minimise the number of converts by avoiding chains of fp32 ->
// fp16 -> fp32 converts.
enum xnn_status xnn_subgraph_fallback_from_fp16_to_fp32(xnn_subgraph_t subgraph,
                                                        int optimization_flags);

// Rewrites unsupported bf16 operations to fp32.
//
// Inserts bf16-fp32 converts for inputs and fp32-bf16 converts for outputs.
// Native bf16 kernels (e.g. the bf16 GEMM and reductions) are left in place;
// everything else falls back to fp32. This tries to minimise the number of
// converts by avoiding chains of fp32 -> bf16 -> fp32 converts.
enum xnn_status xnn_subgraph_fallback_from_bf16_to_fp32(xnn_subgraph_t subgraph,
                                                        int optimization_flags);

// Updates the weight cache with data aliases for static values that were
// converted during a previous call to
// `xnn_subgraph_fallback_from_fp16_to_fp32` or
// `xnn_subgraph_fallback_from_bf16_to_fp32`.
enum xnn_status xnn_subgraph_alias_fp32_fallback_data(xnn_subgraph_t subgraph,
                                                      xnn_weights_cache_t cache);

#ifdef __cplusplus
}
#endif

#endif  // XNNPACK_SRC_SUBGRAPH_REWRITES_CVT_TO_FP32_H_
