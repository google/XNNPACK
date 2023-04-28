// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <random>

#include <gtest/gtest.h>

#include <xnnpack/aarch64-assembler.h>
#include <xnnpack/common.h>
#include <xnnpack/memory.h>
#include <xnnpack/microparams.h>
#include <xnnpack/microparams-init.h>
#include "assembler-helpers.h"


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

  CHECK_ENCODING(0x91008041, a.add(x1, x2, 32));
  CHECK_ENCODING(0x913FFC41, a.add(x1, x2, 4095));
  EXPECT_ERROR(Error::kInvalidOperand, a.add(x1, x2, 4096));

  CHECK_ENCODING(0x8B040069, a.add(x9, x3, x4));

  CHECK_ENCODING(0xB1002069, a.adds(x9, x3, 8));

  CHECK_ENCODING(0xF2400869, a.ands(x9, x3, 7));
  // Any immediate other than 7 is not supported.
  EXPECT_ERROR(Error::kInvalidOperand, a.ands(x9, x3, 8));

  CHECK_ENCODING(0x94000001, a.bl(4));
  CHECK_ENCODING(0x97FFFF80, a.bl(-512));
  EXPECT_ERROR(Error::kInvalidOperand, a.bl(3));
  EXPECT_ERROR(Error::kLabelOffsetOutOfBounds, a.bl(128 * 1024 * 1204 + 4));  // > 128MB
  EXPECT_ERROR(Error::kLabelOffsetOutOfBounds, a.bl(-128 * 1024 * 1204 - 4));  // < -128MB

  CHECK_ENCODING(0xD63F0100, a.blr(x8));

  CHECK_ENCODING(0xF100081F, a.cmp(x0, 2));
  EXPECT_ERROR(Error::kInvalidOperand, a.cmp(x0, 4096));

  CHECK_ENCODING(0xEB0C02DF, a.cmp(x22, x12));

  CHECK_ENCODING(0x9A8F322E, a.csel(x14, x17, x15, kLO));

  CHECK_ENCODING(0xD4400000, a.hlt());

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

  CHECK_ENCODING(0xB8408488, a.ldr(w8, mem[x4], 8));
  CHECK_ENCODING(0xB84FF488, a.ldr(w8, mem[x4], 255));
  CHECK_ENCODING(0xB8500488, a.ldr(w8, mem[x4], -256));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(w8, mem[x4], 256));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(w8, mem[x4], -257));

  CHECK_ENCODING(0xF8408488, a.ldr(x8, mem[x4], 8));
  CHECK_ENCODING(0xF84FF488, a.ldr(x8, mem[x4], 255));
  CHECK_ENCODING(0xF8500488, a.ldr(x8, mem[x4], -256));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(x8, mem[x4], 256));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(x8, mem[x4], -257));

  CHECK_ENCODING(0xD29BD5A9, a.mov(x9, 0xDEAD));

  CHECK_ENCODING(0xAA0303E9, a.mov(x9, x3));

  CHECK_ENCODING(0xF29BD5A9, a.movk(x9, 0xDEAD, 0));
  CHECK_ENCODING(0xF2BBD5A9, a.movk(x9, 0xDEAD, 16));
  CHECK_ENCODING(0xF2DBD5A9, a.movk(x9, 0xDEAD, 32));
  CHECK_ENCODING(0xF2FBD5A9, a.movk(x9, 0xDEAD, 48));
  // Not divisible by 16.
  EXPECT_ERROR(Error::kInvalidOperand, a.movk(x9, 0xDEAD, 1));
  // Out of range, max shift is 48.
  EXPECT_ERROR(Error::kInvalidOperand, a.movk(x9, 0xDEAD, 64));

  CHECK_ENCODING(0xD503201F, a.nop());

  CHECK_ENCODING(0xF98000A0, a.prfm(kPLDL1KEEP, mem[x5]));
  CHECK_ENCODING(0xF98020A0, a.prfm(kPLDL1KEEP, mem[x5, 64]));
  EXPECT_ERROR(Error::kInvalidOperand, a.prfm(kPLDL1KEEP, mem[x5, -8]));
  EXPECT_ERROR(Error::kInvalidOperand, a.prfm(kPLDL1KEEP, mem[x5, 32761]));

  CHECK_ENCODING(0xF9800210, a.prfm(kPSTL1KEEP, mem[x16]));
  EXPECT_ERROR(Error::kInvalidOperand, a.prfm(kPSTL1KEEP, mem[x5, -8]));
  EXPECT_ERROR(Error::kInvalidOperand, a.prfm(kPSTL1KEEP, mem[x5, 32761]));

  CHECK_ENCODING(0xD65F03C0, a.ret());

  CHECK_ENCODING(0xCB020083, a.sub(x3, x4, x2));

  CHECK_ENCODING(0xD1003083, a.sub(x3, x4, 12));
  CHECK_ENCODING(0xD13FFC83, a.sub(x3, x4, 4095));
  EXPECT_ERROR(Error::kInvalidOperand, a.sub(x0, x2, 4096));  // Out of bounds.

  CHECK_ENCODING(0xA90457F4, a.stp(x20, x21, mem[sp, 64]));
  CHECK_ENCODING(0xA98457F4, a.stp(x20, x21, mem[sp, 64]++));
  CHECK_ENCODING(0xA91FD7F4, a.stp(x20, x21, mem[sp, 504]));
  CHECK_ENCODING(0xA92057F4, a.stp(x20, x21, mem[sp, -512]));
  EXPECT_ERROR(Error::kInvalidOperand, a.stp(x20, x21, mem[sp, 3]));
  EXPECT_ERROR(Error::kInvalidOperand, a.stp(x20, x21, mem[sp, 512]));
  EXPECT_ERROR(Error::kInvalidOperand, a.stp(x20, x21, mem[sp, -520]));

  CHECK_ENCODING(0xF80FFFF4, a.str(x20, mem[sp, 255]++));
  CHECK_ENCODING(0xF81B0FF4, a.str(x20, mem[sp, -80]++));
  CHECK_ENCODING(0xF8100FF4, a.str(x20, mem[sp, -256]++));
  CHECK_ENCODING(0xF90003F4, a.str(x20, mem[sp, 0]));
  CHECK_ENCODING(0xF93FFFF4, a.str(x20, mem[sp, 32760]));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(sp, mem[sp, -257]++));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(sp, mem[sp, 256]++));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(sp, mem[sp, 3]));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(sp, mem[sp, -1]));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(sp, mem[sp, 32768]));

  CHECK_ENCODING(0xF1008040, a.subs(x0, x2, 32));
  CHECK_ENCODING(0xF13FFC40, a.subs(x0, x2, 4095));
  EXPECT_ERROR(Error::kInvalidOperand, a.subs(x0, x2, -32));
  EXPECT_ERROR(Error::kInvalidOperand, a.subs(x0, x2, 4096));

  CHECK_ENCODING(0xF240043F, a.tst(x1, 3));
  CHECK_ENCODING(0xF2400C3F, a.tst(x1, 15));
  CHECK_ENCODING(0xF240103F, a.tst(x1, 31));
  EXPECT_ERROR(Error::kUnimplemented, a.tst(x1, 32));

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(AArch64Assembler, SIMDInstructionEncoding) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);
  Assembler a(&b);

  CHECK_ENCODING(0x5E0C07DE, a.dup(s30, v30.s()[1]));
  EXPECT_ERROR(Error::kInvalidOperand, a.dup(s30, v30.s()[4]));
  EXPECT_ERROR(Error::kInvalidOperand, a.dup(s30, v30.d()[1]));

  CHECK_ENCODING(0x5E180610, a.dup(d16, v16.d()[1]));
  EXPECT_ERROR(Error::kInvalidOperand, a.dup(d16, v16.d()[2]));
  EXPECT_ERROR(Error::kInvalidOperand, a.dup(d16, v16.s()[1]));

  CHECK_ENCODING(0x4E0204C4, a.dup(v4.v8h(), v6.h()[0]));
  CHECK_ENCODING(0x4E0604C5, a.dup(v5.v8h(), v6.h()[1]));

  CHECK_ENCODING(0x4EA0F8B0, a.fabs(v16.v4s(), v5.v4s()));
  EXPECT_ERROR(Error::kInvalidOperand, a.fabs(v16.v4s(), v5.v2s()));

  CHECK_ENCODING(0x4E521610, a.fadd(v16.v8h(), v16.v8h(), v18.v8h()));

  CHECK_ENCODING(0x4E25D690, a.fadd(v16.v4s(), v20.v4s(), v5.v4s()));
  EXPECT_ERROR(Error::kInvalidOperand, a.fadd(v16.v4s(), v20.v4s(), v5.v2s()));

  CHECK_ENCODING(0x4E5037E3, a.fmax(v3.v8h(), v31.v8h(), v16.v8h()));

  CHECK_ENCODING(0x4E30F7E3, a.fmax(v3.v4s(), v31.v4s(), v16.v4s()));
  EXPECT_ERROR(Error::kInvalidOperand, a.fmax(v3.v8h(), v31.v4s(), v16.v4s()));

  CHECK_ENCODING(0x4ED137C2, a.fmin(v2.v8h(), v30.v8h(), v17.v8h()));

  CHECK_ENCODING(0x4EB1F7C2, a.fmin(v2.v4s(), v30.v4s(), v17.v4s()));
  EXPECT_ERROR(Error::kInvalidOperand, a.fmin(v2.v4s(), v30.v16b(), v17.v4s()));

  CHECK_ENCODING(0x4F001290, a.fmla(v16.v8h(), v20.v8h(), v0.h()[0]));
  CHECK_ENCODING(0x4F101290, a.fmla(v16.v8h(), v20.v8h(), v0.h()[1]));
  CHECK_ENCODING(0x4F001A90, a.fmla(v16.v8h(), v20.v8h(), v0.h()[4]));
  // Only lane indices 0 to 7 are valid.
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.fmla(v16.v8h(), v20.v8h(), v0.h()[8]));
  // Only the first 15 vector registers can be used for half-precision.
  EXPECT_ERROR(Error::kInvalidOperand, a.fmla(v16.v8h(), v20.v8h(), v16.h()[0]));

  CHECK_ENCODING(0x4F801290, a.fmla(v16.v4s(), v20.v4s(), v0.s()[0]));
  EXPECT_ERROR(Error::kInvalidOperand, a.fmla(v16.v4s(), v20.v2s(), v0.s()[0]));
  EXPECT_ERROR(Error::kInvalidOperand, a.fmla(v16.v2d(), v20.v2d(), v0.s()[0]));
  EXPECT_ERROR(Error::kInvalidLaneIndex, a.fmla(v16.v4s(), v20.v4s(), v0.s()[4]));

  CHECK_ENCODING(0x6E29DC61, a.fmul(v1.v4s(), v3.v4s(), v9.v4s()));
  EXPECT_ERROR(Error::kInvalidOperand, a.fmul(v16.v4s(), v20.v4s(), v5.v2s()));

  CHECK_ENCODING(0x4E181C8E, a.ins(v14.d()[1], x4));
  CHECK_ENCODING(0x4E181C93, a.ins(v19.d()[1], x4));

  CHECK_ENCODING(0x6EA0FBC2, a.fneg(v2.v4s(), v30.v4s()));
  EXPECT_ERROR(Error::kInvalidOperand, a.fneg(v2.v4s(), v30.v16b()));

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

  CHECK_ENCODING(0x4DDF8520, a.ld1({v0.d()}, 1, mem[x9], 8));
  CHECK_ENCODING(0x4DDF4120, a.ld1({v0.h()}, 4, mem[x9], 2));

  CHECK_ENCODING(0x6D433FEE, a.ldp(d14, d15, mem[sp, 48]));
  CHECK_ENCODING(0x6DC33FEE, a.ldp(d14, d15, mem[sp, 48]++));
  CHECK_ENCODING(0x6CC427E8, a.ldp(d8, d9, mem[sp], 64));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldp(d14, d15, mem[sp, 7]));

  CHECK_ENCODING(0xACC154B4, a.ldp(q20, q21, mem[x5], 32));
  CHECK_ENCODING(0xACE054B4, a.ldp(q20, q21, mem[x5], -1024));
  CHECK_ENCODING(0xACDFD4B4, a.ldp(q20, q21, mem[x5], 1008));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldp(q20, q21, mem[x5], 15));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldp(q20, q21, mem[x5], -1040));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldp(q20, q21, mem[x5], 1024));

  CHECK_ENCODING(0x7C402460, a.ldr(h0, mem[x3], 2));

  CHECK_ENCODING(0xFD4020B0, a.ldr(d16, mem[x5, 64]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(d16, mem[x5, 32768]));

  CHECK_ENCODING(0xFC408460, a.ldr(d0, mem[x3], 8));

  CHECK_ENCODING(0xBD400106, a.ldr(s6, mem[x8]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(s6, mem[x6, 16384]));

  CHECK_ENCODING(0xBC404460, a.ldr(s0, mem[x3], 4));

  CHECK_ENCODING(0x3DC004B9, a.ldr(q25, mem[x5, 16]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(q25, mem[x5, -16]));  // Negative offset.
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(q25, mem[x5, 17]));  // Not multiple of 16.
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(q25, mem[x5, 65536]));  // Out of range.

  CHECK_ENCODING(0x3CC10460, a.ldr(q0, mem[x3], 16));
  CHECK_ENCODING(0x3CCFF460, a.ldr(q0, mem[x3], 255));
  CHECK_ENCODING(0x3CD00460, a.ldr(q0, mem[x3], -256));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(q0, mem[x3], -257));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(q0, mem[x3], 256));
  EXPECT_ERROR(Error::kInvalidOperand, a.ldr(q0, mem[x3, 16], 16));

  CHECK_ENCODING(0x4D40C904, a.ld1r({v4.v4s()}, mem[x8]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld1r({v4.v4s(), v5.v4s()}, mem[x8]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld1r({v4.v4s()}, mem[x8, 16]));

  CHECK_ENCODING(0x4D60C902, a.ld2r({v2.v4s(), v3.v4s()}, mem[x8]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld2r({v2.v4s(), v3.v4s()}, mem[x8, 16]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld2r({v2.v4s(), v4.v4s()}, mem[x8]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld2r({v2.v4s(), v3.v8b()}, mem[x8]));

  CHECK_ENCODING(0x4D40E906, a.ld3r({v6.v4s(), v7.v4s(), v8.v4s()}, mem[x8]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld3r({v6.v4s(), v7.v4s(), v8.v4s()}, mem[x8, 16]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld3r({v6.v4s(), v7.v4s(), v9.v4s()}, mem[x8]));
  EXPECT_ERROR(Error::kInvalidOperand, a.ld3r({v6.v4s(), v7.v2s(), v8.v4s()}, mem[x8]));

  CHECK_ENCODING(0x4EB21E50, a.mov(v16.v16b(), v18.v16b()));
  CHECK_ENCODING(0x0EB21E50, a.mov(v16.v8b(), v18.v8b()));
  EXPECT_ERROR(Error::kInvalidOperand, a.mov(v16.v16b(), v18.v8b()));

  CHECK_ENCODING(0x4E183DC3, a.mov(x3, v14.d()[1]));
  CHECK_ENCODING(0x4E083D02, a.mov(x2, v8.d()[0]));
  EXPECT_ERROR(Error::kInvalidOperand, a.mov(x3, v14.d()[2]));

  CHECK_ENCODING(0x4F000405, a.movi(v5.v4s(), 0));
  CHECK_ENCODING(0x4F008405, a.movi(v5.v8h(), 0));
  CHECK_ENCODING(0x4F00E405, a.movi(v5.v16b(), 0));
  EXPECT_ERROR(Error::kUnimplemented, a.movi(v5.v16b(), 0xFF));

  CHECK_ENCODING(0x0C9F786F, a.st1({v15.v2s()}, mem[x3], 8));
  CHECK_ENCODING(0x4C9F786F, a.st1({v15.v4s()}, mem[x3], 16));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v15.v4s()}, mem[x3], 8));

  CHECK_ENCODING(0x0C9FA86F, a.st1({v15.v2s(), v16.v2s()}, mem[x3], 16));
  CHECK_ENCODING(0x4C9FA86F, a.st1({v15.v4s(), v16.v4s()}, mem[x3], 32));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v15.v4s(), v16.v4s()}, mem[x3], 16));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v15.v4s(), v17.v4s()}, mem[x3], 16));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v15.v4s(), v16.v2s()}, mem[x3], 16));

  CHECK_ENCODING(0x0C9F686F, a.st1({v15.v2s(), v16.v2s(), v17.v2s()}, mem[x3], 24));
  CHECK_ENCODING(0x4C9F686F, a.st1({v15.v4s(), v16.v4s(), v17.v4s()}, mem[x3], 48));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v15.v4s(), v16.v4s(), v17.v4s()}, mem[x3], 24));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v15.v4s(), v17.v4s(), v17.v4s()}, mem[x3], 48));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v15.v4s(), v16.v2s(), v17.v4s()}, mem[x3], 48));

  CHECK_ENCODING(0x0C9F286F, a.st1({v15.v2s(), v16.v2s(), v17.v2s(), v18.v2s()}, mem[x3], 32));
  CHECK_ENCODING(0x4C9F286F, a.st1({v15.v4s(), v16.v4s(), v17.v4s(), v18.v4s()}, mem[x3], 64));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v15.v4s(), v16.v4s(), v17.v4s(), v18.v4s()}, mem[x3], 32));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v15.v4s(), v17.v4s(), v17.v4s(), v18.v4s()}, mem[x3], 64));
  EXPECT_ERROR(Error::kInvalidOperand, a.st1({v15.v4s(), v16.v2s(), v17.v4s(), v18.v4s()}, mem[x3], 64));

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

  CHECK_ENCODING(0x6D012FEA, a.stp(d10, d11, mem[sp, 16]));
  CHECK_ENCODING(0x6D202FEA, a.stp(d10, d11, mem[sp, -512]));
  CHECK_ENCODING(0x6D1FAFEA, a.stp(d10, d11, mem[sp, 504]));
  EXPECT_ERROR(Error::kInvalidOperand, a.stp(d10, d11, mem[sp, -520]));
  EXPECT_ERROR(Error::kInvalidOperand, a.stp(d10, d11, mem[sp, 512]));

  CHECK_ENCODING(0x6D812FEA, a.stp(d10, d11, mem[sp, 16]++));

  CHECK_ENCODING(0xAD0075BC, a.stp(q28, q29, mem[x13]));
  CHECK_ENCODING(0xAD80F5BC, a.stp(q28, q29, mem[x13, 16]++));
  EXPECT_ERROR(Error::kInvalidOperand, a.stp(q28, q28, mem[x13, 7]));

  CHECK_ENCODING(0xAC8144D0, a.stp(q16, q17, mem[x6], 32));
  CHECK_ENCODING(0xAC9FC4D0, a.stp(q16, q17, mem[x6], 1008));
  CHECK_ENCODING(0xACA044D0, a.stp(q16, q17, mem[x6], -1024));
  EXPECT_ERROR(Error::kInvalidOperand, a.stp(q16, q17, mem[x6], 34));
  EXPECT_ERROR(Error::kInvalidOperand, a.stp(q16, q17, mem[x6], 1024));
  EXPECT_ERROR(Error::kInvalidOperand, a.stp(q16, q17, mem[x6], -1040));

  CHECK_ENCODING(0xFC0084D0, a.str(d16, mem[x6], 8));
  CHECK_ENCODING(0x3C8104D0, a.str(q16, mem[x6], 16));
  CHECK_ENCODING(0x3C8FF4D0, a.str(q16, mem[x6], 255));
  CHECK_ENCODING(0x3C9004D0, a.str(q16, mem[x6], -256));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(q16, mem[x6], 256));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(q16, mem[x6], -257));

  CHECK_ENCODING(0xBD0000D0, a.str(s16, mem[x6]));
  CHECK_ENCODING(0xBD3FFCD0, a.str(s16, mem[x6, 16380]));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(s16, mem[x6, 3]));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(s16, mem[x6, -4]));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(s16, mem[x6, 16384]));

  CHECK_ENCODING(0xBC0044D0, a.str(s16, mem[x6], 4));
  CHECK_ENCODING(0xBC0FF4D0, a.str(s16, mem[x6], 255));
  CHECK_ENCODING(0xBC1004D0, a.str(s16, mem[x6], -256));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(s16, mem[x6], 256));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(s16, mem[x6], -257));

  CHECK_ENCODING(0x7D0000D4, a.str(h20, mem[x6]));
  CHECK_ENCODING(0x7D3FFCD4, a.str(h20, mem[x6, 8190]));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(h20, mem[x6, 1]));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(h20, mem[x6, -2]));
  EXPECT_ERROR(Error::kInvalidOperand, a.str(h20, mem[x6, 8192]));

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

TEST(AArch64Assembler, Tbz) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);
  Assembler a(&b);

  Label l1;
  a.movi(v0.v4s(), 0);

  // Branch to unbound label.
  auto b1 = a.offset<uint32_t*>();
  a.tbz(x0, 4, l1);

  a.movi(v1.v4s(), 0);
  a.bind(l1);

  EXPECT_INSTR(0x36200040, *b1);

  a.movi(v2.v4s(), 0);

  // Branch to bound label.
  auto b2 = a.offset<uint32_t*>();
  a.tbz(x1, 6, l1);

  EXPECT_INSTR(0x3637FFE1, *b2);

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(AArch64Assembler, UnconditionalBranch) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);
  Assembler a(&b);

  Label l1;
  a.movi(v0.v4s(), 0);

  // Branch to unbound label.
  auto b1 = a.offset<uint32_t*>();
  a.b(l1);

  a.movi(v1.v4s(), 0);
  a.bind(l1);

  EXPECT_INSTR(0x14000002, *b1);

  a.movi(v2.v4s(), 0);

  // Branch to bound label.
  auto b2 = a.offset<uint32_t*>();
  a.b(l1);

  EXPECT_INSTR(0x17FFFFFF, *b2);

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(AArch64Assembler, Align) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);
  Assembler a(&b);

  a.add(x0, x1, x2);
  a.align(4);
  EXPECT_EQ(0, reinterpret_cast<uintptr_t>(a.offset<uint32_t*>()) & 0x3);
  EXPECT_EQ(4, a.code_size_in_bytes());

  a.align(8);
  EXPECT_EQ(0, reinterpret_cast<uintptr_t>(a.offset<uint32_t*>()) & 0x7);
  EXPECT_EQ(8, a.code_size_in_bytes());

  a.add(x0, x1, x2);
  a.align(8);
  EXPECT_EQ(0, reinterpret_cast<uintptr_t>(a.offset<uint32_t*>()) & 0x7);
  EXPECT_EQ(16, a.code_size_in_bytes());

  a.add(x0, x1, x2);
  EXPECT_EQ(20, a.code_size_in_bytes());

  a.align(16);
  EXPECT_EQ(0, reinterpret_cast<uintptr_t>(a.offset<uint32_t*>()) & 0xF);
  EXPECT_EQ(32, a.code_size_in_bytes());

  a.add(x0, x1, x2);
  a.add(x0, x1, x2);
  EXPECT_EQ(40, a.code_size_in_bytes());

  a.align(16);
  EXPECT_EQ(0, reinterpret_cast<uintptr_t>(a.offset<uint32_t*>()) & 0xF);
  EXPECT_EQ(48, a.code_size_in_bytes());

  // Not power-of-two.
  EXPECT_ERROR(Error::kInvalidOperand, a.align(6));
  // Is power-of-two but is not a multiple of instruction size.
  EXPECT_ERROR(Error::kInvalidOperand, a.align(2));

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(AArch64Assembler, AssembleToEndOfBuffer) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);

  Assembler a1(&b);
  a1.emit32(1);
  a1.finalize();

  // Different assembler, but same code buffer.
  Assembler a2(&b);
  a2.emit32(2);
  a2.finalize();

  // Check that we wrote to end of buffer and did not overwrite.
  uint32_t* p = (uint32_t*) b.start;
  ASSERT_EQ(1, *p);
  ASSERT_EQ(2, *(p+1));

  ASSERT_EQ(8, b.size);

  a2.reset();

  ASSERT_EQ(4, b.size);
  ASSERT_EQ((byte*)b.start + 4, a2.offset());

  a2.emit32(3);
  ASSERT_EQ(3, *(p+1));

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(AArch64Assembler, FinalizeWithError) {
  xnn_code_buffer b;
  xnn_allocate_code_memory(&b, XNN_DEFAULT_CODE_BUFFER_SIZE);

  Assembler a(&b);
  // Write a valid instruction.
  a.add(x1, x2, 32);
  // Then write an invalid instruction.
  a.ldp(x14, x15, mem[sp], 15);
  // Since we have an error, size should not be updated.
  a.finalize();
  ASSERT_EQ(0, b.size);

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

TEST(AArch64Assembler, BindOverflow) {
  xnn_code_buffer b;
  // Requested memory is rounded to page size.
  xnn_allocate_code_memory(&b, 4);
  Assembler a(&b);
  Label l1;
  for (int i = 0; i < b.capacity; i += 1 << kInstructionSizeInBytesLog2) {
    a.add(x0, x0, 2);
  }
  EXPECT_EQ(Error::kNoError, a.error());

  // This is out of bounds, not written.
  a.tbz(x1, 1, l1);
  EXPECT_EQ(Error::kOutOfMemory, a.error());

  a.bind(l1);
  ASSERT_EQ(false, l1.bound);

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&b));
}

#if XNN_ARCH_ARM64 && XNN_PLATFORM_JIT
JitF32HardswishFn GenerateF32Hardswish(MacroAssembler& a, std::vector<VRegister> accs, std::vector<VRegister> tmps) {
  const VRegister sixth = v0.v4s();
  const VRegister three = v1.v4s();
  const VRegister six = v2.v4s();
  const VRegister zero = v3.v4s();

  // Load params.
  a.ld3r({sixth, three, six}, mem[x2]);
  a.movi(zero, 0);
  // Load inputs.
  for (size_t i = 0; i < accs.size(); i++) {
    a.ld1({accs[i].v4s()}, mem[x0], 16);
  }
  a.f32_hardswish(
      sixth, three, six, zero,
      accs.data(),
      accs.size(),
      tmps.data(),
      tmps.size());
  // Write results of hardswish.
  for (size_t i = 0; i < accs.size(); i++) {
    a.st1({accs[i].v4s()}, mem[x1], 16);
  }
  a.ret();

  return reinterpret_cast<JitF32HardswishFn>(a.finalize());
}

class F32HardswishTest : public testing::TestWithParam<std::vector<VRegister>> {};

TEST_P(F32HardswishTest, F32Hardswish) {
  xnn_code_buffer buffer;
  xnn_allocate_code_memory(&buffer, XNN_DEFAULT_CODE_BUFFER_SIZE);
  MacroAssembler assembler(&buffer);

  const std::vector<VRegister> accs = GetParam();
  const std::vector<VRegister> tmps = {v16.v4s(), v17.v4s(), v18.v4s(), v19.v4s()};

  std::random_device random_device;
  std::mt19937 rng(random_device());
  std::uniform_real_distribution<float> f32dist(-6.0f, 6.0f);

  xnn_f32_hswish_params params;
  xnn_init_f32_hswish_scalar_params(&params);

  std::vector<float> input(4 * accs.size());
  std::vector<float> output(4 * accs.size());
  std::vector<float> expected_output(output);

  std::generate(input.begin(), input.end(), [&]{ return f32dist(rng); });
  std::fill(output.begin(), output.end(), std::nanf(""));
  std::fill(expected_output.begin(), expected_output.end(), std::nanf(""));

  // Call generated function.
  JitF32HardswishFn jit_f32_hardswish_fn = GenerateF32Hardswish(assembler, accs, tmps);
  EXPECT_EQ(Error::kNoError, assembler.error());
  xnn_finalize_code_memory(&buffer);
  jit_f32_hardswish_fn(input.data(), output.data(), &params);

  // Compute reference results.
  std::transform(input.begin(), input.end(), expected_output.begin(), hardswish);

  // Verify results.
  for (size_t i = 0; i < output.size(); i++) {
    EXPECT_NEAR(output[i], expected_output[i], std::max(5.0e-6, std::abs(expected_output[i]) * 1.0e-5))
        << "at " << i << " / " << output.size() << ", x[" << i << "] = " << input[i];
  }
  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&buffer));
}

INSTANTIATE_TEST_SUITE_P(
  AArch64Assembler,
  F32HardswishTest,
  testing::Values(
    std::vector<VRegister>({v4.v4s()}),
    std::vector<VRegister>({v4.v4s(), v5.v4s(), v6.v4s(), v7.v4s()}),
    std::vector<VRegister>({v4.v4s(), v5.v4s(), v6.v4s(), v7.v4s(), v20.v4s(), v21.v4s(), v22.v4s(), v23.v4s()})));

typedef void (*MovFn)(uint64_t*);

TEST(MovTest, Mov) {
  xnn_code_buffer buffer;
  xnn_allocate_code_memory(&buffer, XNN_DEFAULT_CODE_BUFFER_SIZE);
  MacroAssembler assm(&buffer);

  uint64_t expected = 0x0123456789ABCDEF;
  assm.Mov(x1, expected);
  assm.str(x1, mem[x0]);
  assm.ret();

  MovFn mov_fn = reinterpret_cast<MovFn>(assm.finalize());
  xnn_finalize_code_memory(&buffer);

  uint64_t out = 0;
  mov_fn(&out);

  ASSERT_EQ(xnn_status_success, xnn_release_code_memory(&buffer));

  EXPECT_EQ(expected, out);
}

#endif  // XNN_ARCH_ARM64 && XNN_PLATFORM_JIT

}  // namespace aarch64
}  // namespace xnnpack
