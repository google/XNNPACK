#ifndef XNNPACK_SRC_SUBGRAPH_REWRITES_FP16_TO_FP32_H_
#define XNNPACK_SRC_SUBGRAPH_REWRITES_FP16_TO_FP32_H_

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

// Updates the weight cache with data aliases for static values that were
// converted during a previous call to
// `xnn_subgraph_fallback_from_fp16_to_fp32`.
enum xnn_status xnn_subgraph_alias_fp16_fp32_fallback_data(
    xnn_subgraph_t subgraph, xnn_weights_cache_t cache);

#ifdef __cplusplus
}
#endif

#endif  // XNNPACK_SRC_SUBGRAPH_REWRITES_FP16_TO_FP32_H_
