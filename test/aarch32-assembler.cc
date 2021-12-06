#include <xnnpack/aarch32-assembler.h>

#include <gtest/gtest.h>

#define EXPECT_INSTR_EQ(expected, call) \
  a.reset();                            \
  call;                                 \
  EXPECT_EQ(expected, *a.start())

namespace xnnpack {
namespace aarch32 {
TEST(AArch32Assembler, DataProcessingRegister) {
  Assembler a;
  EXPECT_INSTR_EQ(0xE0810002, a.add(r0, r1, r2));
}
}  // namespace aarch32
}  // namespace xnnpack
