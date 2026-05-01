#include "test/subgraph/rewrites/subgraph_matcher.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "litert/tensor/arithmetic.h"
#include "litert/tensor/backends/xnnpack/conversion.h"
#include "litert/tensor/datatypes.h"
#include "litert/tensor/tensor.h"
#include "litert/tensor/utils/matchers.h"

namespace {

using ::litert::tensor::Tensor;
using ::litert::tensor::Type;
using ::litert::tensor::XnnpackGraph;
using ::testing::Not;
using ::xnnpack::IsIsomorphicTo;
using XnnTensor = Tensor<litert::tensor::XnnpackMixinTag>;

TEST(SubgraphMatcherTest, CompareAGraphToItselfReturnsTrue) {
  XnnTensor a({.name = "a", .type = Type::kFP32});
  XnnTensor b({.name = "b", .type = Type::kFP32});
  Tensor c = Add(a, b);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<XnnpackGraph> graph,
                                  litert::tensor::BuildXnnpackGraph({c}));

  EXPECT_THAT(graph, IsIsomorphicTo(graph));
}

TEST(SubgraphMatcherTest, CompareGraphWithDifferentOpsReturnsFalse) {
  XnnTensor a({.name = "a", .type = Type::kFP32});
  XnnTensor b({.name = "b", .type = Type::kFP32});
  Tensor c = Add(a, b);
  Tensor d = Mul(a, b);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<XnnpackGraph> graph1,
                                  litert::tensor::BuildXnnpackGraph({c}));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<XnnpackGraph> graph2,
                                  litert::tensor::BuildXnnpackGraph({d}));

  EXPECT_THAT(graph1, Not(IsIsomorphicTo(graph2)));
}

TEST(SubgraphMatcherTest, CompareTwinGraphsReturnsTrue) {
  XnnTensor a({.name = "a", .type = Type::kFP32});
  XnnTensor b({.name = "b", .type = Type::kFP32});
  XnnTensor c({.name = "c", .type = Type::kFP32});
  XnnTensor d({.name = "d", .type = Type::kFP32});
  Tensor e = Add(a, b);
  Tensor f = Add(c, d);

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<XnnpackGraph> graph1,
                                  litert::tensor::BuildXnnpackGraph({e, f}));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<XnnpackGraph> graph2,
                                  litert::tensor::BuildXnnpackGraph({f, e}));

  EXPECT_THAT(graph1, IsIsomorphicTo(graph2));
}

TEST(SubgraphMatcherTest, SortingGraphWithSimilarTailIsStable) {
  XnnTensor a({.name = "a", .type = Type::kFP32});
  XnnTensor b({.name = "b", .type = Type::kFP32});
  Tensor e = Sqrt(Abs(Add(a, b)));
  XnnTensor c({.name = "c", .type = Type::kFP32});
  XnnTensor d({.name = "d", .type = Type::kFP32});
  Tensor f = Sqrt(Abs(Mul(c, d)));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<XnnpackGraph> graph1,
                                  litert::tensor::BuildXnnpackGraph({e, f}));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<XnnpackGraph> graph2,
                                  litert::tensor::BuildXnnpackGraph({f, e}));

  EXPECT_THAT(graph1, IsIsomorphicTo(graph2));
}

TEST(SubgraphMatcherTest, SortingGraphWithSimilarHeadIsStable) {
  XnnTensor a({.name = "a", .type = Type::kFP32});
  XnnTensor b({.name = "b", .type = Type::kFP32});
  Tensor e = Sqrt(Abs(Add(a, b)));
  XnnTensor c({.name = "c", .type = Type::kFP32});
  XnnTensor d({.name = "d", .type = Type::kFP32});
  Tensor f = Rsqrt(Abs(Add(c, d)));

  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<XnnpackGraph> graph1,
                                  litert::tensor::BuildXnnpackGraph({e, f}));
  LRT_TENSOR_ASSERT_OK_AND_ASSIGN(std::unique_ptr<XnnpackGraph> graph2,
                                  litert::tensor::BuildXnnpackGraph({f, e}));

  EXPECT_THAT(graph1, IsIsomorphicTo(graph2));
}

}  // namespace
