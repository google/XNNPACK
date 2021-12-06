#include <xnnpack/aarch32-assembler.h>

#include <gtest/gtest.h>

#define EXPECT_INSTR_EQ(expected, call) \
  a.reset();                            \
  call;                                 \
  EXPECT_EQ(expected, *a.start())

#define EXPECT_ERROR(expected, call) \
  a.reset();                         \
  call;                              \
  EXPECT_EQ(expected, a.error());

namespace xnnpack {
namespace aarch32 {
TEST(AArch32Assembler, InstructionEncoding) {
  Assembler a;

  EXPECT_INSTR_EQ(0xE0810002, a.add(r0, r1, r2));

  EXPECT_INSTR_EQ(0xE92D0FF0, a.push({r4, r5, r6, r7, r8, r9, r10, r11}));
  EXPECT_ERROR(Error::kInvalidOperand, a.push({}));
  EXPECT_ERROR(Error::kInvalidOperand, a.push({r1}));
}

TEST(AArch32Assembler, CoreRegisterList) {
  EXPECT_EQ(0x3, CoreRegisterList({r0, r1}));
  EXPECT_EQ(0xFC00, CoreRegisterList({r10, r11, r12, r13, r14, r15}));

  EXPECT_FALSE(CoreRegisterList({}).has_more_than_one_register());
  EXPECT_FALSE(CoreRegisterList({r0}).has_more_than_one_register());
  EXPECT_FALSE(CoreRegisterList({r1}).has_more_than_one_register());
  EXPECT_TRUE(CoreRegisterList({r0, r1}).has_more_than_one_register());
}
}  // namespace aarch32
}  // namespace xnnpack
