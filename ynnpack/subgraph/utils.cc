// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/utils.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/dot/pack.h"
#include "ynnpack/subgraph/runtime.h"
#include "ynnpack/subgraph/slinky.h"
#include "ynnpack/subgraph/subgraph.h"
#include "slinky/base/arithmetic.h"
#include "slinky/builder/pipeline.h"
#include "slinky/runtime/buffer.h"
#include "slinky/runtime/expr.h"
#include "slinky/runtime/stmt.h"

namespace ynn {

template <typename T>
bool is_broadcast_op(const ynn_node& node) {
  const T* op = std::get_if<T>(&node.op);
  return op && op->axes.any();
}

bool allow_in_place(uint32_t input_id, uint32_t output_id,
                    const ynn_subgraph& subgraph) {
  if (input_id == YNN_INVALID_VALUE_ID) return false;

  const ynn_value& a = subgraph.value(input_id);
  const ynn_value& x = subgraph.value(output_id);

  if (x.rank() != a.rank()) {
    return false;
  }

  if (type_size_bytes(a.type) != type_size_bytes(x.type) ||
      type_element_count(a.type) != type_element_count(x.type)) {
    // The types are not the same size, we can't compute in place.
    return false;
  }

  for (size_t d = 0; d < x.rank(); ++d) {
    if (!a.extents[d].defined() && x.extents[d].defined()) {
      // The input is broadcasted (and the output is not), don't allow computing
      // in place.
      return false;
    }
  }

  const ynn_node* producer = subgraph.get_producer(input_id);
  if (!producer) {
    // This input is not produced in the pipeline, we can't overwrite the
    // input (and slinky wouldn't let us anyways).
    return false;
  }

  if (is_broadcast_op<ynn_node::broadcast>(*producer) ||
      is_broadcast_op<ynn_node::broadcast_like>(*producer)) {
    // We can't compute in place with a broadcast input.
    return false;
  }

  return true;
}

// Make a kernel wrapper for packing the input of a dot kernel, i.e.
// interleaving `tile_k` rows at a time.
// TODO(b/454146513): We should try to combine both pack_b and transpose_a into
// a `split_transpose` op that can handle padding, split, and transpose.
auto make_transpose_a_impl(slinky::index_t tile_k) {
  return [tile_k](slinky::raw_buffer input,
                  slinky::raw_buffer output) -> slinky::index_t {
    const slinky::dim& input_k = input.dim(0);
    const slinky::dim& input_m = input.dim(1);
    const slinky::dim& output_ko = output.dim(0);
    const slinky::dim& output_m = output.dim(1);

    const slinky::index_t elem_size = input.elem_size;
    assert(output_m.stride() == elem_size * tile_k);
    (void)output_m;

    input.slice(0);
    input.slice(0, output_m.min());
    output.slice({0, 1});

    // We need the intersection of the input and output bounds.
    const slinky::index_t m =
        std::min(output_m.end(), input_m.end()) - output_m.begin();
    assert(input_k.min() <= output_ko.min() * tile_k);
    assert(output_ko.min() == 0);
    const slinky::index_t k = std::min(output_ko.end() * tile_k, input_k.end());

    // We're transposing columns of the input to rows of the output, but
    // doing tile_k of them at a time.
    // TODO(b/454131137): Support already transposed inputs here.
    packer p(/*transpose=*/true, elem_size * 8, tile_k, /*tile_n=*/m);

    slinky::for_each_element(
        [&](void* output, const void* input) {
          p.pack(k, m, input_m.stride(), input, output_ko.stride(),
                 /*output_block_stride=*/0, output);
        },
        output, input);
    return 0;
  };
}

// Packing means transposing
// a(k, m, ...) => a([0, tile_k), m, k/tile_k, ...)
uint32_t define_transpose_a(ynn_subgraph& subgraph, slinky::index_t tile_k,
                            uint32_t input_a_id) {
  const ynn_value& a = subgraph.value(input_a_id);

  ynn_value& packed_a = subgraph.new_internal_value();
  packed_a.type = a.type;
  uint32_t packed_a_id = packed_a.id;

  slinky::expr k = a.extent(0);
  slinky::expr m = a.extent(1);

  packed_a.extents = {slinky::ceil_div<slinky::expr>(k, tile_k), m};
  packed_a.extents.insert(packed_a.extents.end(), a.extents.begin() + 2,
                          a.extents.end());

  ynn_node node;
  node.inputs = {input_a_id};
  node.outputs = {packed_a_id};
  node.op = ynn_node::transpose_a{static_cast<size_t>(tile_k)};
  node.create = [tile_k](const ynn_node& node, ynn_runtime& runtime) {
    const ynn_runtime_value& input = runtime.value(node.inputs[0]);
    ynn_runtime_value& output = runtime.value(node.outputs[0]);

    output.make_buffer(runtime, input.buffer->elem_size() * tile_k);
    output.buffer->dim(1).stride = output.buffer->elem_size();
    output.buffer->dim(0).stride =
        output.buffer->dim(1).stride * output.buffer->dim(1).extent();

    // Split + Transpose
    std::vector<slinky::var> dims =
        make_dims(output.buffer->rank(), runtime.symbols);

    slinky::expr m = dims[1];
    slinky::expr ko = dims[0];

    slinky::func::input func_input = {input.buffer};
    func_input.bounds = {
        slinky::min_extent(ko * tile_k, tile_k),
        slinky::point(m),
    };
    for (size_t i = 2; i < dims.size(); ++i) {
      func_input.bounds.push_back(slinky::point(dims[i]));
    }
    // This transpose handles padding the input up to tile_k.
    func_input.input_crop = {
        all_bounds(input.extent(0)),
    };

    slinky::call_stmt::attributes attrs;
    attrs.name = "transpose_a";
    auto func = slinky::func::make(make_transpose_a_impl(tile_k),
                                   {std::move(func_input)},
                                   {{output.buffer, dims}}, std::move(attrs));

    runtime.funcs.push_back(std::move(func));
    return ynn_status_success;
  };
  subgraph.add_node(std::move(node));
  return packed_a_id;
}

}  // namespace ynn
