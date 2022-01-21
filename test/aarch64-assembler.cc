// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/aarch64-assembler.h>
#include <xnnpack/allocator.h>
#include <xnnpack/common.h>

#include "assembler-helpers.h"
#include <gtest/gtest.h>

namespace xnnpack {
namespace aarch64 {

TEST(AArch64Assembler, Initialization) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);
  Assembler a(&b);
  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(AArch64Assembler, BaseInstructionEncoding) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);
  Assembler a(&b);

  CHECK_ENCODING(0xA9403FEE, a.ldp(x14, x15, mem[sp]));
  CHECK_ENCODING(0xA8C13FEE, a.ldp(x14, x15, mem[sp], 16));
  CHECK_ENCODING(0xA9413FEE, a.ldp(x14, x15, mem[sp, 16]));
  CHECK_ENCODING(0xA9603FEE, a.ldp(x14, x15, mem[sp, -512]));
  CHECK_ENCODING(0xA95FBFEE, a.ldp(x14, x15, mem[sp, 504]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldp(x14, x15, mem[sp], 15));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldp(x14, x15, mem[sp], -520));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldp(x14, x15, mem[sp], 512));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldp(x14, x15, mem[sp, 16], 16));

  CHECK_ENCODING(0xF9400BE8, a.ldr(x8, mem[sp, 16]));
  CHECK_ENCODING(0xF97FFFE8, a.ldr(x8, mem[sp, 32760]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(x8, mem[sp, -8]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(x8, mem[sp, 7]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(x8, mem[sp, 32768]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(x8, MemOperand(sp, 16, AddressingMode::kPostIndex)));

  CHECK_ENCODING(0xF98000A0, a.prfm(PLDL1KEEP, mem[x5]));
  EXPECT_ERROR(Error::kInvalidOperand, a.prfm(PLDL1KEEP, mem[x5, -8]));
  EXPECT_ERROR(Error::kInvalidOperand, a.prfm(PLDL1KEEP, mem[x5, 32761]));

  CHECK_ENCODING(0xD65F03C0, a.ret());

  CHECK_ENCODING(0xCB020083, a.sub(x3, x4, x2));

  CHECK_ENCODING(0xF1008040, a.subs(x0, x2, 32));
  CHECK_ENCODING(0xF13FFC40, a.subs(x0, x2, 4095));
  EXPECT_ERROR(Error::kInvalidOperand, a.subs(x0, x2, -32));
  EXPECT_ERROR(Error::kInvalidOperand, a.subs(x0, x2, 4096));

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(AArch64Assembler, SIMDInstructionEncoding) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);
  Assembler a(&b);

  CHECK_ENCODING(0x4E25D690, a.fadd(v16.v4s(), v20.v4s(), v5.v4s()));
  EXPECT_ERROR(Error::kInvalidOperand, a.fadd(v16.v4s(), v20.v4s(), v5.v2s()));

  CHECK_ENCODING(0x4E30F7E3, a.fmax(v3.v4s(), v31.v4s(), v16.v4s()));
  EXPECT_ERROR(Error::kInvalidOperand, a.fmax(v3.v8h(), v31.v4s(), v16.v4s()));

  CHECK_ENCODING(0x4EB1F7C2, a.fmin(v2.v4s(), v30.v4s(), v17.v4s()));
  EXPECT_ERROR(Error::kInvalidOperand, a.fmin(v2.v4s(), v30.v16b(), v17.v4s()));

  CHECK_ENCODING(0x4F801290, a.fmla(v16.v4s(), v20.v4s(), v0.s()[0]));
  EXPECT_ERROR(Error::kInvalidOperand, a.fmla(v16.v4s(), v20.v2s(), v0.s()[0]));
  EXPECT_ERROR(Error::kInvalidOperand, a.fmla(v16.v2d(), v20.v2d(), v0.s()[0]));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.fmla(v16.v4s(), v20.v4s(), v0.s()[4]));

  CHECK_ENCODING(0x0CDF7060, a.ld1({v0.v8b()}, mem[x3], 8));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld1({v0.v8b()}, mem[x3], 16));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld1({v0.v16b()}, mem[x3], 8));

  CHECK_ENCODING(0x0CDFA060, a.ld1({v0.v8b(), v1.v8b()}, mem[x3], 16));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld1({v0.v8b(), v1.v8b()}, mem[x3], 32));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld1({v0.v16b(), v1.v16b()}, mem[x3], 16));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld1({v0.v8b(), v2.v8b()}, mem[x3], 16));

  CHECK_ENCODING(0x4CDF61F0, a.ld1({v16.v16b(), v17.v16b(), v18.v16b()}, mem[x15], 48));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld1({v16.v8b(), v17.v16b(), v18.v16b()}, mem[x15], 48));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld1({v16.v16b(), v17.v16b(), v18.v8b()}, mem[x15], 48));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld1({v16.v16b(), v17.v16b(), v18.v16b()}, mem[x15], 24));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld1({v16.v8b(), v17.v8b(), v18.v8b()}, mem[x15], 48));

  CHECK_ENCODING(0xACC154B4, a.ldp(q20, q21, mem[x5], 32));
  CHECK_ENCODING(0xACE054B4, a.ldp(q20, q21, mem[x5], -1024));
  CHECK_ENCODING(0xACDFD4B4, a.ldp(q20, q21, mem[x5], 1008));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldp(q20, q21, mem[x5], 15));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldp(q20, q21, mem[x5], -1040));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldp(q20, q21, mem[x5], 1024));

  CHECK_ENCODING(0x3CC10460, a.ldr(q0, mem[x3], 16));
  CHECK_ENCODING(0x3CCFF460, a.ldr(q0, mem[x3], 255));
  CHECK_ENCODING(0x3CD00460, a.ldr(q0, mem[x3], -256));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(q0, mem[x3], -257));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(q0, mem[x3], 256));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(q0, mem[x3, 16], 16));

  CHECK_ENCODING(0x4D60C902, a.ld2r({v2.v4s(), v3.v4s()}, mem[x8]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld2r({v2.v4s(), v3.v4s()}, mem[x8, 16]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld2r({v2.v4s(), v4.v4s()}, mem[x8, 16]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld2r({v2.v4s(), v3.v8b()}, mem[x8]));

  CHECK_ENCODING(0x4F000405, a.movi(v5.v4s(), 0));
  CHECK_ENCODING(0x4F008405, a.movi(v5.v8h(), 0));
  CHECK_ENCODING(0x4F00E405, a.movi(v5.v16b(), 0));
  EXPECT_ERROR(Error::kUnimplemented, a.movi(v5.v16b(), 0xFF));

  CHECK_ENCODING(0x4C82746F, a.st1({v15.v8h()}, mem[x3], x2));

  CHECK_ENCODING(0x4C95AA8F, a.st1({v15.v4s(), v16.v4s()}, mem[x20], x21));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v15.v4s(), v17.v4s()}, mem[x20], x21));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v15.v4s(), v16.v8h()}, mem[x20], x21));

  CHECK_ENCODING(0x4C8E60D0, a.st1({v16.v16b(), v17.v16b(), v18.v16b() }, mem[x6], x14));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v15.v16b(), v17.v16b(), v18.v16b()}, mem[x6], x14));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v16.v16b(), v17.v16b(), v18.v4s()}, mem[x6], x14));

  CHECK_ENCODING(0x4C812FB4, a.st1({v20.v2d(), v21.v2d(), v22.v2d(), v23.v2d()}, mem[x29], x1));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v20.v2d(), v21.v2d(), v22.v2d(), v23.v2s()}, mem[x29], x1));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v20.v2d(), v21.v2d(), v22.v2d(), v27.v2d()}, mem[x29], x1));

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(AArch64Assembler, Label) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);
  Assembler a(&b);

  Label l1;
  a.movi(v0.v4s(), 0);

  // Branch to unbound label.
  auto b1 = a.offset<uint32_t*>();
  a.b_eq(l1);

  a.movi(v1.v4s(), 0);

  auto b2 = a.offset<uint32_t*>();
  a.b_ne(l1);

  a.movi(v2.v4s(), 0);

  a.bind(l1);

  // Check that b1 and b2 are both patched after binding l1.
  EXPECT_INSTR(0x54000080, *b1);
  EXPECT_INSTR(0x54000041, *b2);

  a.movi(v3, 0);

  // Branch to bound label.
  auto b3 = a.offset<uint32_t*>();
  a.b_hi(l1);
  auto b4 = a.offset<uint32_t*>();
  a.b_hs(l1);
  auto b5 = a.offset<uint32_t*>();
  a.b_lo(l1);

  EXPECT_INSTR(0x54FFFFE8, *b3);
  EXPECT_INSTR(0x54FFFFC2, *b4);
  EXPECT_INSTR(0x54FFFFA3, *b5);

  // Binding a bound label is an error.
  a.bind(l1);
  EXPECT_ERROR(Error::kLabelAlreadyBound, a.bind(l1));

  // Check for bind failure due to too many users of label.
  Label lfail;
  a.reset();
  // Arbitrary high number of users that we probably won't support.
  for (int i = 0; i < 1000; i++) {
    a.b_eq(lfail);
  }
  EXPECT_EQ(Error::kLabelHasTooManyUsers, a.error());

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(AArch64Assembler, Tbnz) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);
  Assembler a(&b);

  Label l1;
  a.movi(v0.v4s(), 0);

  // Branch to unbound label.
  auto b1 = a.offset<uint32_t*>();
  a.tbnz(x0, 4, l1);

  a.movi(v1.v4s(), 0);
  a.bind(l1);

  EXPECT_INSTR(0x37200040, *b1);

  a.movi(v2.v4s(), 0);

  // Branch to bound label.
  auto b2 = a.offset<uint32_t*>();
  a.tbnz(x1, 6, l1);

  EXPECT_INSTR(0x3737FFE1, *b2);

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

}  // namespace aarch64
}  // namespace xnnpack
