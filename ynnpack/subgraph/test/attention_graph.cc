// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/test/attention_graph.h"

#include <cstddef>
#include <cstdint>

#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

ynn_status define_attention(ynn_subgraph_t subgraph, uint32_t query_id,
                            uint32_t key_id, uint32_t value_id, float scale,
                            uint32_t& output_id, bool transpose_io) {
  // With sequence-major inputs, swap the sequence and head axes to get the
  // head-major layout the body below works in: [b, {t|s}, n, h] -> [b, n, .,
  // h].
  const int32_t io_perm[] = {0, 2, 1, 3};
  uint32_t q_id = query_id, k_id = key_id, v_id = value_id;
  if (transpose_io) {
    q_id = k_id = v_id = YNN_INVALID_VALUE_ID;
    YNN_RETURN_IF_ERROR(
        ynn_define_static_transpose(subgraph, 4, io_perm, query_id, &q_id, 0));
    YNN_RETURN_IF_ERROR(
        ynn_define_static_transpose(subgraph, 4, io_perm, key_id, &k_id, 0));
    YNN_RETURN_IF_ERROR(
        ynn_define_static_transpose(subgraph, 4, io_perm, value_id, &v_id, 0));
  }

  // S = Q @ K^T. `ynn_define_dot` contracts the last axis of `a` with the
  // second-to-last axis of `b`, so K needs its last two axes swapped.
  const int32_t k_t_perm[] = {0, 1, 3, 2};
  uint32_t k_t_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(
      ynn_define_static_transpose(subgraph, 4, k_t_perm, k_id, &k_t_id, 0));

  uint32_t scores_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_dot(subgraph, /*num_k_dims=*/1, q_id, k_t_id,
                                     YNN_INVALID_VALUE_ID, &scores_id, 0));

  // P = softmax(scale * S) along the key sequence axis. softmax's `beta`
  // scaling is equivalent to scaling the scores.
  uint32_t probs_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_softmax(subgraph, scores_id, scale, probs_id));

  // O = P @ V. Head-major; transposed back to sequence-major below if needed.
  uint32_t o_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_dot(subgraph, /*num_k_dims=*/1, probs_id, v_id,
                                     YNN_INVALID_VALUE_ID,
                                     transpose_io ? &o_id : &output_id, 0));

  // Convert the head-major output [b, n, t, h] back to sequence-major
  // [b, t, n, h].
  if (transpose_io) {
    YNN_RETURN_IF_ERROR(
        ynn_define_static_transpose(subgraph, 4, io_perm, o_id, &output_id, 0));
  }
  return ynn_status_success;
}

ynn_status define_attention_decode1(ynn_subgraph_t subgraph, uint32_t query_id,
                                    uint32_t key_id, uint32_t value_id,
                                    float scale, uint32_t& output_id) {
  // S = K @ Q^T (see the header comment for why this avoids packing K). Q's
  // last two axes ({..., t=1, h} -> {..., h, t=1}) need swapping so its `h`
  // axis lands in the dot's contracted (second-to-last) position; since t is
  // 1 this is very cheap or free.
  const int32_t q_t_perm[] = {0, 1, 3, 2};
  uint32_t q_t_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(
      ynn_define_static_transpose(subgraph, 4, q_t_perm, query_id, &q_t_id, 0));

  uint32_t scores_ts_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_dot(subgraph, /*num_k_dims=*/1, key_id, q_t_id,
                                     YNN_INVALID_VALUE_ID, &scores_ts_id, 0));

  // scores_ts is [b, n, s, t]; swap back to [b, n, t, s] (again cheap or free,
  // since t == 1) so the rest of the pipeline is unchanged from
  // `define_attention`.
  uint32_t scores_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(ynn_define_static_transpose(subgraph, 4, q_t_perm,
                                                  scores_ts_id, &scores_id, 0));

  // P = softmax(scale * S) along the key sequence axis.
  uint32_t probs_id = YNN_INVALID_VALUE_ID;
  YNN_RETURN_IF_ERROR(define_softmax(subgraph, scores_id, scale, probs_id));

  // O = P @ V.
  return ynn_define_dot(subgraph, /*num_k_dims=*/1, probs_id, value_id,
                        YNN_INVALID_VALUE_ID, &output_id, 0);
}

}  // namespace ynn
