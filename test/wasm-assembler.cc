// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <xnnpack/common.h>

#if XNN_PLATFORM_WEB

#include <xnnpack/memory.h>
#include <xnnpack/wasm-assembler.h>

#include <array>
#include <cstdint>
#include <numeric>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::ElementsAreArray;
using ::testing::FloatNear;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::Pointwise;
using ::testing::Sequence;
using ::testing::Test;
using GetIntPtr = int (*)();
using GetPiPtr = float (*)();
using Add5Ptr = int (*)(int);
using AddPtr = int (*)(int, int);
using MaxPtr = AddPtr;
using SumUntil = Add5Ptr;
using DoWhile = Add5Ptr;
using SumUntilManyLocals = int (*)();
using SumArray = int (*)(const int*, int);
using MemCpy = void (*)(int*, const int*, int);
using V128Add = void (*)(const float*, float*);
using V128AddConst = void (*)(const float*, float*);
using V128MAdd = void (*)(const float*, float*);
using V128Shuffle = void (*)(const float*, float*);

namespace xnnpack {
namespace {

constexpr int32_t kExpectedGet5ReturnValue = 5;
constexpr int32_t kPositive = 5;
constexpr int32_t kNegative = -5;
constexpr int32_t kA = 55;
constexpr int32_t kAPlusFive = kA + kExpectedGet5ReturnValue;
constexpr int32_t kB = 42;
constexpr int32_t kExpectedSum = kA + kB;
constexpr int32_t kExpectedSumTwice = 2 * kExpectedSum;
constexpr uint32_t kLargeNumberOfLocals = 129;
constexpr uint32_t kLargeNumberOfFunctions = 235;
constexpr size_t kArraySize = 5;
constexpr std::array<int, kArraySize> kArray = {1, 2, 3, 45, 6};
const int kExpectedArraySum = std::accumulate(kArray.begin(), kArray.end(), 0);
constexpr float kPi = 3.14;

struct Get5Generator : WasmAssembler {
  explicit Get5Generator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt no_locals;
    AddFunc<0>({i32}, "get5", no_locals, [this]() { i32_const(5); });
  }
};

template <typename G, typename F>
struct GeneratorTestSuite {
  using Generator = G;
  using Func = F;
};

template <typename ValueType, const ValueType& expected_value, typename G, typename F>
struct GetValueTestSuite : GeneratorTestSuite<G, F> {
  static void ExpectFuncCorrect(F f) { EXPECT_EQ(f(), expected_value); }
};

struct Get5TestSuite : GetValueTestSuite<int32_t, kExpectedGet5ReturnValue, Get5Generator, GetIntPtr> {};

struct AddGenerator : WasmAssembler {
  explicit AddGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt no_locals;
    AddFunc<2>({i32}, "add", {i32, i32}, no_locals, [this](Local a, Local b) {
      local_get(a);
      local_get(b);
      i32_add();
    });
  }
};

template <typename G, uint32_t kSum>
struct AddTestSuiteTmpl : GeneratorTestSuite<G, AddPtr> {
  static void ExpectFuncCorrect(AddPtr add) { EXPECT_EQ(add(kA, kB), kSum); }
};

struct AddTestSuite : AddTestSuiteTmpl<AddGenerator, kExpectedSum> {};

struct Get5AndAddGenerator : WasmAssembler {
  explicit Get5AndAddGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    GenerateGet5();
    GenerateAdd();
    GenerateGet5();
    GenerateAdd();
  }

 private:
  void GenerateGet5() {
    ValTypesToInt no_locals;
    const std::string& name = kFunctionNames[count++];
    AddFunc<0>({i32}, name.c_str(), {}, no_locals, [this]() { i32_const(5); });
  }

  void GenerateAdd() {
    ValTypesToInt no_locals;
    const std::string& name = kFunctionNames[count++];
    AddFunc<2>({i32}, name.c_str(), {i32, i32}, no_locals, [this](Local a, Local b) {
      local_get(a);
      local_get(b);
      i32_add();
    });
  }

  const std::array<std::string, 4> kFunctionNames = {
    std::string("get5_1"),
    std::string("add_1"),
    std::string("get5_2"),
    std::string("add_2"),
  };

  int count = 0;
};

struct Get5AndAddTestSuite : GeneratorTestSuite<Get5AndAddGenerator, GetIntPtr> {
  static void ExpectFuncCorrect(GetIntPtr get5) {
    int first = (int) get5;
    Get5TestSuite::ExpectFuncCorrect((GetIntPtr) (first + 0));
    AddTestSuite::ExpectFuncCorrect((AddPtr) (first + 1));
    Get5TestSuite::ExpectFuncCorrect((GetIntPtr) (first + 2));
    AddTestSuite::ExpectFuncCorrect((AddPtr) (first + 3));
  }
};

struct AddWithLocalGenerator : WasmAssembler {
  explicit AddWithLocalGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt single_local_int = {{i32, 1}};
    AddFunc<2>({i32}, "add_with_local", single_local_int, [this](Local a, Local b) {
      auto sum = MakeLocal(i32);
      sum = I32Add(a, b);
      local_get(sum);
    });
  }
};

struct AddWithLocalTestSuite : AddTestSuiteTmpl<AddWithLocalGenerator, kExpectedSum> {};

struct AddTwiceGenerator : WasmAssembler {
  explicit AddTwiceGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt two_local_ints = {{i32, 2}};
    AddFunc<2>({i32}, "add_twice", {i32, i32}, two_local_ints, [this](Local a, Local b) {
      auto first = MakeLocal(i32);
      auto second = MakeLocal(i32);
      first = I32Add(a, b);
      second = first;
      second = I32Add(second, a);
      second = I32Add(second, b);
      local_get(second);
    });
  }
};

struct AddTwiceTestSuite : AddTestSuiteTmpl<AddTwiceGenerator, kExpectedSumTwice> {};

struct AddTwiceDeclareInitGenerator : WasmAssembler {
  explicit AddTwiceDeclareInitGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt two_local_ints = {{i32, 2}};
    AddFunc<2>({i32}, "add_twice_declare_init", {i32, i32}, two_local_ints, [this](Local a, Local b) {
      auto first = MakeLocal(I32Add(a, b));
      auto second = MakeLocal(first);
      second = first;
      second = I32Add(second, a);
      second = I32Add(second, b);
      local_get(second);
    });
  }
};

struct AddTwiceDeclareInitTestSuite : AddTestSuiteTmpl<AddTwiceDeclareInitGenerator, kExpectedSumTwice> {};

struct AddTwiceWithScopesGenerator : WasmAssembler {
  explicit AddTwiceWithScopesGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt two_local_ints = {{i32, 2}};
    AddFunc<2>({i32}, "add_twice_with_scopes", {i32, i32}, two_local_ints, [this](Local a, Local b) {
      auto first = MakeLocal(i32);
      {
        auto sum = MakeLocal(i32);
        sum = I32Add(a, b);
        first = sum;
      }
      {
        auto sum = MakeLocal(i32);
        sum = I32Add(a, b);
        first = I32Add(first, sum);
      }
      local_get(first);
    });
  }
};

struct AddTwiceWithScopesTestSuite : AddTestSuiteTmpl<AddTwiceWithScopesGenerator, kExpectedSumTwice> {};

struct Add5CodeGenerator : WasmAssembler {
  explicit Add5CodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt no_locals = {};
    AddFunc<1>({i32}, "add5", {i32}, no_locals, [this](Local a) {
      a = I32Add(I32Const(5), a);
      local_get(a);
    });
  }
};

struct Add5TestSuite : GeneratorTestSuite<Add5CodeGenerator, Add5Ptr> {
  static void ExpectFuncCorrect(Add5Ptr add_five) { EXPECT_EQ(add_five(kA), kAPlusFive); }
};

struct MaxCodeGenerator : WasmAssembler {
  explicit MaxCodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt single_local_int = {{i32, 1}};
    AddFunc<2>({i32}, "max", {i32, i32}, single_local_int, [this](Local a, Local b) {
      auto result = MakeLocal(i32);
      IfElse([&] { I32LtS(a, b); }, [&] { result = b; }, [&] { result = a; });
      local_get(result);
    });
  }
};

template <typename G>
struct GenericMaxTestSuite : GeneratorTestSuite<G, MaxPtr> {
  static void ExpectFuncCorrect(MaxPtr max) {
    EXPECT_EQ(max(2, 3), 3);
    EXPECT_EQ(max(3, 2), 3);
  }
};

struct MaxTestSuite : GenericMaxTestSuite<MaxCodeGenerator> {};

struct MaxIncompleteIfCodeGenerator : WasmAssembler {
  explicit MaxIncompleteIfCodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt no_locals = {};
    AddFunc<2>({i32}, "max_incomplete_if", {i32, i32}, no_locals, [this](Local a, Local b) {
      If([&] { I32LtS(a, b); }, [&] { a = b; });
      local_get(a);
    });
  }
};

struct MaxIncompleteIfTestSuite : GenericMaxTestSuite<MaxIncompleteIfCodeGenerator> {};

struct SumUntilCodeGenerator : WasmAssembler {
  explicit SumUntilCodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt two_local_ints = {{i32, 2}};
    AddFunc<1>({i32}, "SumUntil", {i32}, two_local_ints, [&](Local n) {
      auto i = MakeLocal(i32);
      auto result = MakeLocal(i32);
      While([&] { I32LtS(i, n); },
            [&] {
              result = I32Add(result, i);
              i = I32Add(i, I32Const(1));
            });
      local_get(result);
    });
  }
};

static int ReferenceSumUntil(int n) {
  int i = 0;
  int result = 0;
  while (i < n) {
    result += i;
    i++;
  }
  return result;
}

struct SumUntilTestSuite : GeneratorTestSuite<SumUntilCodeGenerator, SumUntil> {
  static void ExpectFuncCorrect(SumUntil sum_until) {
    static constexpr int kN = 5;
    static constexpr int kNoIters = 0;
    EXPECT_EQ(sum_until(kN), ReferenceSumUntil(kN));
    EXPECT_EQ(sum_until(kNoIters), ReferenceSumUntil(kNoIters));
  }
};

static constexpr uint32_t kNSmall = 27;

template <typename Derived>
struct CanInitWithIterator {
  template <typename LocalsArray>
  void InitSummands(LocalsArray& summands) {
    size_t i = 0;
    for (auto& summand : summands) {
      summand = static_cast<Derived*>(this)->I32Const(i);
      i++;
    }
  }
};

template <typename Derived>
struct CanSumWithIterator {
  template <typename Local, typename LocalsArray>
  void Sum(Local& result, const LocalsArray& summands) {
    for (const auto& summand : summands) {
      result = static_cast<Derived*>(this)->I32Add(result, summand);
    }
  }
};

template <typename Derived>
struct CanInitWithIndex {
  template <typename LocalsArray>
  void InitSummands(LocalsArray& summands) {
    for (size_t i = 0; i < summands.size(); i++) {
      summands[i] = static_cast<Derived*>(this)->I32Const(i);
    }
  }
};

template <typename Derived>
struct CanSumWithIndex {
  template <typename Local, typename LocalsArray>
  void Sum(Local& result, const LocalsArray& summands) {
    for (size_t i = 0; i < summands.size(); i++) {
      result = static_cast<Derived*>(this)->I32Add(result, summands[i]);
    }
  }
};

template <typename Derived>
struct CanGenerateSumUntil {
  void GenerateFunctionBody() {
    auto self = static_cast<Derived*>(this);
    auto& i32 = Derived::i32;
    auto summands = self->MakeLocalsArray(kNSmall, i32);
    auto result = self->MakeLocal(i32);
    self->InitSummands(summands);
    self->Sum(result, summands);
    self->local_get(result);
  }
};

struct SumUntilLocalsArrayWithIteratorCodeGenerator
    : WasmAssembler,
      CanGenerateSumUntil<SumUntilLocalsArrayWithIteratorCodeGenerator>,
      CanSumWithIterator<SumUntilLocalsArrayWithIteratorCodeGenerator>,
      CanInitWithIterator<SumUntilLocalsArrayWithIteratorCodeGenerator> {
  explicit SumUntilLocalsArrayWithIteratorCodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt local_ints_decl = {{i32, kNSmall + 1}};
    AddFunc<0>({i32}, "SumUntilN", {}, local_ints_decl, [&] { this->GenerateFunctionBody(); });
  }
};

template <typename G>
struct SumUntilLocalsArrayTestSuite : GeneratorTestSuite<G, GetIntPtr> {
  static void ExpectFuncCorrect(GetIntPtr sum_until_n) { EXPECT_EQ(sum_until_n(), ReferenceSumUntil(kNSmall)); }
};

struct SumUntilLocalsArrayWithIterator : SumUntilLocalsArrayTestSuite<SumUntilLocalsArrayWithIteratorCodeGenerator> {};

struct SumUntilLocalsArrayWithIndexCodeGenerator : WasmAssembler,
                                                   CanGenerateSumUntil<SumUntilLocalsArrayWithIndexCodeGenerator>,
                                                   CanSumWithIndex<SumUntilLocalsArrayWithIndexCodeGenerator>,
                                                   CanInitWithIndex<SumUntilLocalsArrayWithIndexCodeGenerator> {
  explicit SumUntilLocalsArrayWithIndexCodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt local_ints_decl = {{i32, kNSmall + 1}};
    AddFunc<0>({i32}, "SumUntilN", {}, local_ints_decl, [&] { this->GenerateFunctionBody(); });
  }
};

struct SumUntilLocalsArrayWithIndexTestSuite : SumUntilLocalsArrayTestSuite<SumUntilLocalsArrayWithIndexCodeGenerator> {
};

struct DoWhileCodeGenerator : WasmAssembler {
  explicit DoWhileCodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt two_local_ints = {{i32, 2}};
    AddFunc<1>({i32}, "DoWhile", {i32}, two_local_ints, [&](Local n) {
      auto i = MakeLocal(i32);
      auto result = MakeLocal(i32);
      result = I32Const(kPositive);
      DoWhile(
        [&] {
          result = I32Add(result, result);
          i = I32Add(i, I32Const(1));
        },
        [&] { I32LtS(i, n); });
      local_get(result);
    });
  }
};

struct DoWhileTestSuite : GeneratorTestSuite<DoWhileCodeGenerator, DoWhile> {
  static int ReferenceDoWhile(int n) {
    int result = kPositive;
    int i = 0;
    do {
      result += result;
      i++;
    } while (i < n);
    return result;
  }

  static void ExpectFuncCorrect(DoWhile do_while) {
    EXPECT_EQ(do_while(kPositive), ReferenceDoWhile(kPositive));
    EXPECT_EQ(do_while(kNegative), ReferenceDoWhile(kNegative));
  }
};

struct SumArrayMemory : WasmAssembler {
  explicit SumArrayMemory(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt two_local_ints = {{i32, 2}};
    AddFunc<2>({i32}, "SumArray", {i32, i32}, two_local_ints, [&](Local array, Local n) {
      auto i = MakeLocal(i32);
      auto result = MakeLocal(i32);
      While([&] { I32LtS(i, n); },
            [&] {
              result = I32Add(result, I32Load(array, i));
              i = I32Add(i, I32Const(1));
            });
      local_get(result);
    });
  }
};

struct SumArrayTestSuite : GeneratorTestSuite<SumArrayMemory, SumArray> {
  static void ExpectFuncCorrect(SumArray sum_array) {
    EXPECT_EQ(sum_array(kArray.data(), kArraySize), kExpectedArraySum);
  }
};

struct MemCpyGenerator : WasmAssembler {
  explicit MemCpyGenerator(xnn_code_buffer* bf) : WasmAssembler(bf) {
    ValTypesToInt two_local_ints = {{i32, 2}};
    AddFunc<3>({}, "mymemcpy", {i32, i32, i32}, two_local_ints, [&](Local dst, Local src, Local n) {
      auto i = MakeLocal(i32);
      auto value = MakeLocal(i32);

      While([&] { I32LtS(i, n); },
            [&] {
              value = I32Load(src, i);
              I32Store(dst, i, value);
              i = I32Add(i, I32Const(1));
            });
    });
  }
};

struct MemCpyTestSuite : GeneratorTestSuite<MemCpyGenerator, MemCpy> {
  static void ExpectFuncCorrect(MemCpy mem_cpy) {
    std::array<int, kArraySize> dst;
    mem_cpy(dst.data(), kArray.data(), kArraySize);
    EXPECT_THAT(dst, ElementsAreArray(kArray));
  }
};

struct AddDelayedInitLocalsGenerator : WasmAssembler {
  explicit AddDelayedInitLocalsGenerator(xnn_code_buffer* b) : WasmAssembler(b) {
    ValTypesToInt three_ints = {{i32, 3}};
    AddFunc<2>({i32}, "add_delayed_init", {i32, i32}, three_ints, [this](Local a, Local b) {
      Local c;
      c = MakeLocal(i32);
      Local d;
      d = MakeLocal(i32);
      Local sum;
      c = I32Add(c, a);
      d = I32Add(c, b);
      sum = MakeLocal(i32);
      sum = I32Add(a, sum);
      sum = I32Add(b, sum);
      local_get(sum);
    });
  }
};

struct AddDelayedInitTestSuite : AddTestSuiteTmpl<AddDelayedInitLocalsGenerator, kExpectedSum> {};

struct ManyLocalsGenerator : WasmAssembler {
  explicit ManyLocalsGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt many_ints = {{i32, kLargeNumberOfLocals + 1}};
    AddFunc<0>({i32}, "sum_until_with_many_locals", {}, many_ints, [this]() {
      std::array<Local, kLargeNumberOfLocals> locals;
      for (int i = 0; i < kLargeNumberOfLocals; i++) {
        locals[i] = MakeLocal(i32);
        locals[i] = I32Const(i);
      }
      auto sum = MakeLocal(i32);

      for (int i = 0; i < kLargeNumberOfLocals; i++) {
        sum = I32Add(sum, locals[i]);
      }
      local_get(sum);
    });
  }
};

struct ManyLocalsGeneratorTestSuite : GeneratorTestSuite<ManyLocalsGenerator, SumUntilManyLocals> {
  static void ExpectFuncCorrect(SumUntilManyLocals sum_until) {
    EXPECT_EQ(sum_until(), ReferenceSumUntil(kLargeNumberOfLocals));
  }
};

struct V128AddGenerator : WasmAssembler {
  explicit V128AddGenerator(xnn_code_buffer* bf) : WasmAssembler(bf) {
    ValTypesToInt two_v128 = {{v128, 2}};
    AddFunc<2>({}, "v128_add", {i32, i32}, two_v128, [this](Local src, Local dst) {
      auto a = MakeLocal(v128);
      auto b = MakeLocal(v128);
      a = V128Load(src);
      b = V128Load(src, /*offset=*/16);
      a = F32x4Add(a, b);
      V128Store(dst, a);
    });
  }
};

struct V128AddGeneratorTestSuite : GeneratorTestSuite<V128AddGenerator, V128Add> {
  static void ExpectFuncCorrect(V128Add v128_add) {
    static constexpr std::array<float, 8> kIn = {1, 2, 3, 4, 10, 20, 30, 40};
    static constexpr std::array<float, 4> kExpectedOut = {11, 22, 33, 44};
    std::array<float, 4> out;
    v128_add(kIn.data(), out.data());
    EXPECT_THAT(out, ElementsAreArray(kExpectedOut));
  }
};

struct V128AddConstGenerator : WasmAssembler {
  explicit V128AddConstGenerator(xnn_code_buffer* bf) : WasmAssembler(bf) {
    ValTypesToInt two_v128 = {{v128, 2}};
    AddFunc<2>({}, "v128_add_const", {i32, i32}, two_v128, [this](Local src, Local dst) {
      auto a = MakeLocal(v128);
      auto b = MakeLocal(v128);
      a = V128Load(src);
      b = V128Load32Splat(src, /*offset=*/16);
      a = F32x4Add(a, b);
      V128Store(dst, a);
    });
  }
};

struct V128AddConstGeneratorTestSuite : GeneratorTestSuite<V128AddConstGenerator, V128AddConst> {
  static void ExpectFuncCorrect(V128AddConst v128_add_const) {
    static constexpr std::array<float, 5> kIn = {1, 2, 3, 4, 5};
    static constexpr std::array<float, 4> kExpectedOut = {6, 7, 8, 9};
    std::array<float, 4> out;
    v128_add_const(kIn.data(), out.data());
    EXPECT_THAT(out, ElementsAreArray(kExpectedOut));
  }
};

#if XNN_ARCH_WASMRELAXEDSIMD
struct V128MaddGenerator : WasmAssembler {
  explicit V128MaddGenerator(xnn_code_buffer* bf) : WasmAssembler(bf) {
    ValTypesToInt three_v128 = {{v128, 3}};
    AddFunc<2>({}, "v128_madd", {i32, i32}, three_v128, [this](Local src, Local dst) {
      auto a = MakeLocal(v128);
      auto b = MakeLocal(v128);
      auto c = MakeLocal(v128);
      a = V128Load(src);
      b = V128Load32Splat(src, /*offset=*/16);
      c = V128Load32Splat(src, /*offset=*/20);
      a = F32x4RelaxedMadd(a, b, c);
      V128Store(dst, a);
    });
  }
};

struct V128MaddGeneratorTestSuite : GeneratorTestSuite<V128MaddGenerator, V128MAdd> {
  static void ExpectFuncCorrect(V128MAdd v128_madd) {
    static constexpr std::array<float, 6> kIn = {1, 2, 3, 4, 5, 6};
    static constexpr std::array<float, 4> kExpectedOut = {11, 16, 21, 26};
    std::array<float, 4> out;
    v128_madd(kIn.data(), out.data());
    EXPECT_THAT(out, ElementsAreArray(kExpectedOut));
  }
};
#endif

struct V128AddPiGenerator : WasmAssembler {
  explicit V128AddPiGenerator(xnn_code_buffer* bf) : WasmAssembler(bf) {
    ValTypesToInt two_v128 = {{v128, 2}};
    AddFunc<2>({}, "v128_add_pi", {i32, i32}, two_v128, [this](Local src, Local dst) {
      auto a = MakeLocal(v128);
      auto b = MakeLocal(v128);
      a = V128Load(src);
      b = V128Const(kPi);
      a = F32x4Add(a, b);
      V128Store(dst, a);
    });
  }
};

struct V128AddPiGeneratorTestSuite : GeneratorTestSuite<V128AddPiGenerator, V128AddConst> {
  static void ExpectFuncCorrect(V128AddConst v128_add_const) {
    static constexpr std::array<float, 5> kIn = {1, 2, 3, 4, 5};

    std::array<float, 4> expected_out;
    for (int i = 0; i < 4; i++) expected_out[i] = kIn[i] + kPi;

    std::array<float, 4> out;
    v128_add_const(kIn.data(), out.data());
    EXPECT_THAT(out, Pointwise(FloatNear(1e-6f), expected_out));
  }
};

struct I64x2ShuffleGenerator : WasmAssembler {
  explicit I64x2ShuffleGenerator(xnn_code_buffer* bf) : WasmAssembler(bf) {
    ValTypesToInt two_v128 = {{v128, 2}};
    AddFunc<2>({}, "i64x2_shuffle", {i32, i32}, two_v128, [this](Local src, Local dst) {
      auto a = MakeLocal(v128);
      auto b = MakeLocal(v128);
      a = V128Load(src);
      b = V128Load(src, /*offset=*/16);
      V128Store(dst, I64x2Shuffle(a, b, {0, 0}), /*offset=*/0);
      V128Store(dst, I64x2Shuffle(a, b, {3, 1}), /*offset=*/16);
      V128Store(dst, I64x2Shuffle(a, b, {0, 2}), /*offset=*/32);
    });
  }
};

struct I64x2ShuffleGeneratorTestSuite : GeneratorTestSuite<I64x2ShuffleGenerator, V128Shuffle> {
  static void ExpectFuncCorrect(V128Shuffle v128_shuffle) {
    static constexpr std::array<float, 8> kIn = {1, 2, 3, 4, 5, 6, 7, 8};
    static constexpr std::array<float, 12> kExpectedOut = {1, 2, 1, 2, 7, 8, 3, 4, 1, 2, 5, 6};
    std::array<float, 12> out;
    v128_shuffle(kIn.data(), out.data());
    EXPECT_THAT(out, ElementsAreArray(kExpectedOut));
  }
};

struct GetPiGenerator : WasmAssembler {
  explicit GetPiGenerator(xnn_code_buffer* bf) : WasmAssembler(bf) {
    ValTypesToInt no_locals = {};
    AddFunc<0>({f32}, "get_pi", {}, no_locals, [&] { F32Const(kPi); });
  }
};

struct GetPiTestSuite : GetValueTestSuite<float, kPi, GetPiGenerator, GetPiPtr> {};

struct Get5TrickyLifetimeGenerator : WasmAssembler {
  explicit Get5TrickyLifetimeGenerator(xnn_code_buffer* bf) : WasmAssembler(bf) {
    ValTypesToInt two_locals = {{i32, 2}};
    AddFunc<0>({i32}, "get5", {}, two_locals, [&] {
      Local a;
      {
        auto b = MakeLocal(I32Const(kExpectedGet5ReturnValue));
        a = MakeLocal(b);
      }
      auto c = MakeLocal(I32Const(0));
      local_get(a);
    });
  }
};

struct Get5TrickyTestSuite
    : GetValueTestSuite<int32_t, kExpectedGet5ReturnValue, Get5TrickyLifetimeGenerator, GetIntPtr> {};

struct InvalidCodeGenerator : WasmAssembler {
  ValTypesToInt no_locals;
  explicit InvalidCodeGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    AddFunc<0>({}, "invalid", {}, no_locals, [this]() { i32_const(5); });
  }
};

struct ManyFunctionsGenerator : WasmAssembler {
  explicit ManyFunctionsGenerator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    for (uint32_t func_index = 0; func_index < kLargeNumberOfFunctions; func_index++) {
      AddGet5(func_index);
    }
  }

 private:
  void AddGet5(uint32_t index) {
    function_names[index] = "get5_" + std::to_string(index);
    ValTypesToInt no_locals;
    AddFunc<0>({i32}, function_names[index].c_str(), {}, no_locals, [this]() { i32_const(5); });
  }
  static std::array<std::string, kLargeNumberOfFunctions> function_names;
};
std::array<std::string, kLargeNumberOfFunctions> ManyFunctionsGenerator::function_names = {};

constexpr static size_t kTooLongFunctionLength = XNN_DEFAULT_CODE_BUFFER_SIZE * 2;
struct LongGet5Generator : WasmAssembler {
  explicit LongGet5Generator(xnn_code_buffer* buf) : WasmAssembler(buf) {
    ValTypesToInt one_i32 = {{i32, 1}};
    AddFunc<0>({i32}, "long_get5", one_i32, [this]() {
      auto x = MakeLocal(i32);
      for (size_t i = 0; i < kTooLongFunctionLength; i++) {
        x = I32Const(5);
      }
      i32_const(5);
    });
  }
};

template <typename TestSuite>
class WasmAssemblerTest : public Test {};

}  // namespace

TYPED_TEST_SUITE_P(WasmAssemblerTest);

TYPED_TEST_P(WasmAssemblerTest, ValidCode) {
  using TestSuite = TypeParam;
  using Generator = typename TestSuite::Generator;
  using Func = typename TestSuite::Func;
  static constexpr size_t kBufferSize = 131072;

  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, kBufferSize);

  Generator generator(&b);
  generator.Emit();
  ASSERT_THAT(generator.finalize(), NotNull());

  ASSERT_EQ(xnn_finalize_code_memory(&b), xnn_status_success);
  ASSERT_EQ(Error::kNoError, generator.error());
  auto func = (Func) xnn_first_function_ptr(&b);
  TestSuite::ExpectFuncCorrect(func);

  ASSERT_EQ(xnn_release_code_memory(&b), xnn_status_success);
}

REGISTER_TYPED_TEST_SUITE_P(WasmAssemblerTest, ValidCode);

using WasmAssemblerTestSuits =
  testing::Types<Get5TestSuite, AddTestSuite, Get5AndAddTestSuite, AddWithLocalTestSuite, AddTwiceTestSuite,
                 AddTwiceDeclareInitTestSuite, AddTwiceWithScopesTestSuite, Add5TestSuite, MaxTestSuite,
                 MaxIncompleteIfTestSuite, SumUntilTestSuite, SumUntilLocalsArrayWithIterator,
                 SumUntilLocalsArrayWithIndexTestSuite, DoWhileTestSuite, SumArrayTestSuite, MemCpyTestSuite,
                 AddDelayedInitTestSuite, ManyLocalsGeneratorTestSuite, V128AddGeneratorTestSuite,
                 V128AddPiGeneratorTestSuite, V128AddConstGeneratorTestSuite, I64x2ShuffleGeneratorTestSuite,
                 GetPiTestSuite, Get5TrickyTestSuite
#if XNN_ARCH_WASMRELAXEDSIMD
                 ,
                 V128MaddGeneratorTestSuite
#endif
                 >;
INSTANTIATE_TYPED_TEST_SUITE_P(WasmAssemblerTestSuits, WasmAssemblerTest, WasmAssemblerTestSuits);

TEST(WasmAssemblerTest, InvalidCode) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);

  InvalidCodeGenerator generator(&b);
  generator.Emit();
  EXPECT_THAT(generator.finalize(), NotNull());
  EXPECT_THAT(xnn_first_function_ptr(&b), XNN_INVALID_FUNCTION_INDEX);
  xnn_release_code_memory(&b);
}

TEST(WasmAssemblerTest, MaxNumberOfFunctionsExceeded) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);

  ManyFunctionsGenerator generator(&b);
  generator.Emit();
  EXPECT_THAT(generator.finalize(), IsNull());
  EXPECT_EQ(generator.error(), Error::kMaxNumberOfFunctionsExceeded);
  xnn_release_code_memory(&b);
}

TEST(WasmAssemblerTest, FunctionBodyDidNotFitIntoCodeBuffer) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);

  LongGet5Generator generator(&b);
  generator.Emit();
  EXPECT_THAT(generator.finalize(), IsNull());
  EXPECT_EQ(generator.error(), Error::kOutOfMemory);
  xnn_release_code_memory(&b);
}

namespace {
class WasmOpsTest : public internal::V128WasmOps<WasmOpsTest>,
                    public internal::I32WasmOps<WasmOpsTest>,
                    public internal::ControlFlowWasmOps<WasmOpsTest>,
                    public internal::LocalWasmOps<WasmOpsTest>,
                    public internal::MemoryWasmOps<WasmOpsTest>,
                    public Test {
 public:
  MOCK_METHOD(void, Encode8Impl, (byte), (const));
  MOCK_METHOD(void, EncodeU32Impl, (uint32_t), (const));
  MOCK_METHOD(void, EncodeS32Impl, (int32_t), (const));

 protected:
  template <typename Op>
  void TestV128BinaryOp(uint32_t opcode, Op&& op) {
    ExpectEncodeVectorOpcode(opcode);
    auto result = std::mem_fn(op)(*this, v128_value_, v128_value_);
    EXPECT_EQ(result.type, v128);
  }

  void ExpectCallEncode8(byte opcode) { EXPECT_CALL(*this, Encode8Impl(opcode)).Times(1).InSequence(sequence_); }

  void ExpectCallEncodeU32(uint32_t value) { EXPECT_CALL(*this, EncodeU32Impl(value)).Times(1).InSequence(sequence_); }

  void ExpectCallEncodeS32(uint32_t value) { EXPECT_CALL(*this, EncodeS32Impl(value)).Times(1).InSequence(sequence_); }

  void ExpectEncodeVectorOpcode(uint32_t opcode) {
    ExpectCallEncode8(0xFD);
    ExpectCallEncodeU32(opcode);
  }

  Sequence sequence_;
  ValueOnStack v128_value_{v128, this};
  ValueOnStack i32_value_{i32, this};
  ValueOnStack f32_value_{f32, this};

  Local i32_local_{i32, i32_local_index, false, this};

  static constexpr uint32_t i32_local_index = 152;
  static constexpr uint32_t kLogAlignment = 3;
  static constexpr uint32_t kAlignment = 1 << 3;
  static constexpr uint32_t kOffset = 16;
};

class V128StoreLaneWasmOpTest : public WasmOpsTest {
 protected:
  void SetStoreLaneExpectations(byte expected_opcode) {
    ExpectEncodeVectorOpcode(expected_opcode);
    ExpectCallEncodeU32(kLogAlignment);
    ExpectCallEncodeU32(kOffset);
    ExpectCallEncode8(kLane);
  }

  static constexpr uint8_t kLane = 1;
};
}  // namespace

TEST_F(WasmOpsTest, F32x4Mul) {
  TestV128BinaryOp(0xE6, &WasmOpsTest::F32x4Mul);
}

TEST_F(V128StoreLaneWasmOpTest, Store32Lane) {
  SetStoreLaneExpectations(0x5A);
  V128Store32Lane(v128_value_, v128_value_, kLane, kOffset, kAlignment);
}

TEST_F(V128StoreLaneWasmOpTest, Store64Lane) {
  SetStoreLaneExpectations(0x5B);
  V128Store64Lane(v128_value_, v128_value_, kLane, kOffset, kAlignment);
}

TEST_F(WasmOpsTest, I32Ne) {
  ExpectCallEncode8(0x47);
  I32Ne(i32_value_, i32_value_);
}

TEST_F(WasmOpsTest, I32GeU) {
  ExpectCallEncode8(0x4F);
  I32GeU(i32_value_, i32_value_);
}

TEST_F(WasmOpsTest, V128Load64Splat) {
  ExpectEncodeVectorOpcode(0x0A);
  ExpectCallEncodeU32(kLogAlignment);
  ExpectCallEncodeU32(kOffset);

  V128Load64Splat(i32_value_, kOffset, kAlignment);
}

TEST_F(WasmOpsTest, V128F32x4Pmax) {
  TestV128BinaryOp(0xEB, &WasmOpsTest::F32x4Pmax);
}

TEST_F(WasmOpsTest, V128F32x4Pmin) {
  TestV128BinaryOp(0xEA, &WasmOpsTest::F32x4Pmin);
}

TEST_F(WasmOpsTest, V128F32x4Eq) {
  TestV128BinaryOp(0x41, &WasmOpsTest::F32x4Eq);
}

TEST_F(WasmOpsTest, V128Andnot) {
  TestV128BinaryOp(0x4F, &WasmOpsTest::V128Andnot);
}

TEST_F(WasmOpsTest, I32x4MaxS) {
  TestV128BinaryOp(0xB8, &WasmOpsTest::I32x4MaxS);
}

TEST_F(WasmOpsTest, F32x4Splat) {
  ExpectEncodeVectorOpcode(0x13);
  F32x4Splat(f32_value_);
}

TEST_F(WasmOpsTest, I32x4Splat) {
  ExpectEncodeVectorOpcode(0x11);
  I32x4Splat(i32_value_);
}

TEST_F(WasmOpsTest, I32x4Shuffle) {
  static constexpr std::array<uint8_t, 4> kLanes = {1, 2, 3, 4};
  ExpectEncodeVectorOpcode(0x0D);
  for (auto lane : internal::MakeLanesForI8x16Shuffle(kLanes.data(), kLanes.size())) {
    ExpectCallEncode8(lane);
  }
  I32x4Shuffle(v128_value_, v128_value_, kLanes);
}

TEST_F(WasmOpsTest, Return) {
  ExpectCallEncode8(0x0F);
  Return();
}

TEST_F(WasmOpsTest, Tee) {
  ExpectCallEncode8(0x22);
  ExpectCallEncodeU32(i32_local_index);
  local_tee(i32_local_);
}

TEST_F(WasmOpsTest, Select) {
  ExpectCallEncode8(0x1B);
  Select(i32_value_, i32_value_, i32_value_);
}

TEST_F(WasmOpsTest, I32NeZ) {
  ExpectCallEncode8(0x41);
  ExpectCallEncodeS32(0);
  ExpectCallEncode8(0x47);
  I32NeZ(i32_value_);
}


#if XNN_ARCH_WASMRELAXEDSIMD
TEST_F(WasmOpsTest, F32x4RelaxedMin) {
  TestV128BinaryOp(0x10D, &WasmOpsTest::F32x4RelaxedMin);
}

TEST_F(WasmOpsTest, F32x4RelaxedMax) {
  TestV128BinaryOp(0x10E, &WasmOpsTest::F32x4RelaxedMax);
}
#endif

using ::xnnpack::internal::LocalsManager;

TEST(WasmAssembler, LocalsManager) {
  ValType i32(1);
  ValType f32(20);
  ValType f16(10);
  constexpr uint32_t params = 4;
  ValTypesToInt local_declaration = {{i32, 2}, {f32, 0}, {f16, 3}};

  LocalsManager manager;
  manager.ResetLocalsManager(params, local_declaration);
  EXPECT_EQ(manager.GetNewLocalIndex(f16), 6);
  EXPECT_EQ(manager.GetNewLocalIndex(f16), 7);
  EXPECT_EQ(manager.GetNewLocalIndex(f16), 8);
  manager.DestructLocal(f16, 7);
  EXPECT_EQ(manager.GetNewLocalIndex(f16), 7);
  // assert fails
  // manager.GetNewLocalIndex(f32);
  EXPECT_EQ(manager.GetNewLocalIndex(i32), 4);
}

}  // namespace xnnpack
#endif
