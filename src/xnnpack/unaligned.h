// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"


XNN_INLINE static uint16_t unaligned_load_s16(const void* address) {
  typedef XNN_UNALIGNED int16_t xnn_unaligned_int16_t;
  return *((const xnn_unaligned_int16_t*) address);
}

XNN_INLINE static uint16_t unaligned_load_u16(const void* address) {
  typedef XNN_UNALIGNED uint16_t xnn_unaligned_uint16_t;
  return *((const xnn_unaligned_uint16_t*) address);
}

XNN_INLINE static float unaligned_load_f32(const void* address) {
  typedef XNN_UNALIGNED float xnn_unaligned_float;
  return *((const xnn_unaligned_float*) address);
}

XNN_INLINE static int32_t unaligned_load_s32(const void* address) {
  typedef XNN_UNALIGNED int32_t xnn_unaligned_int32_t;
  return *((const xnn_unaligned_int32_t*) address);
}

XNN_INLINE static uint32_t unaligned_load_u32(const void* address) {
  typedef XNN_UNALIGNED uint32_t xnn_unaligned_uint32_t;
  return *((const xnn_unaligned_uint32_t*) address);
}

XNN_INLINE static float unaligned_indexed_load_f32(const void* address, size_t index) {
  typedef XNN_UNALIGNED float xnn_unaligned_float;
  return ((const xnn_unaligned_float*) address)[index];
}

XNN_INLINE static uint16_t unaligned_indexed_load_u16(const void* address, size_t index) {
  typedef XNN_UNALIGNED uint16_t xnn_unaligned_uint16_t;
  return ((const xnn_unaligned_uint16_t*) address)[index];
}

XNN_INLINE static int32_t unaligned_indexed_load_s32(const void* address, size_t index) {
  typedef XNN_UNALIGNED int32_t xnn_unaligned_int32_t;
  return ((const xnn_unaligned_int32_t*) address)[index];
}

XNN_INLINE static uint32_t unaligned_indexed_load_u32(const void* address, size_t index) {
  typedef XNN_UNALIGNED uint32_t xnn_unaligned_uint32_t;
  return ((const xnn_unaligned_uint32_t*) address)[index];
}

XNN_INLINE static void unaligned_store_u16(void* address, uint16_t value) {
  typedef XNN_UNALIGNED uint16_t xnn_unaligned_uint16_t;
  *((xnn_unaligned_uint16_t*) address) = value;
}

XNN_INLINE static void unaligned_store_f32(void* address, float value) {
  typedef XNN_UNALIGNED float xnn_unaligned_float;
  *((xnn_unaligned_float*) address) = value;
}

XNN_INLINE static void unaligned_store_s32(void* address, int32_t value) {
  typedef XNN_UNALIGNED int32_t xnn_unaligned_int32_t;
  *((xnn_unaligned_int32_t*) address) = value;
}

XNN_INLINE static void unaligned_store_u32(void* address, uint32_t value) {
  typedef XNN_UNALIGNED uint32_t xnn_unaligned_uint32_t;
  *((xnn_unaligned_uint32_t*) address) = value;
}

XNN_INLINE static void unaligned_indexed_store_f32(void* address, size_t index, float value) {
  typedef XNN_UNALIGNED float xnn_unaligned_float;
  ((xnn_unaligned_float*) address)[index] = value;
}

XNN_INLINE static void unaligned_indexed_store_s32(void* address, size_t index, int32_t value) {
  typedef XNN_UNALIGNED int32_t xnn_unaligned_int32_t;
  ((xnn_unaligned_int32_t*) address)[index] = value;
}

XNN_INLINE static void unaligned_indexed_store_u32(void* address, size_t index, uint32_t value) {
  typedef XNN_UNALIGNED uint32_t xnn_unaligned_uint32_t;
  ((xnn_unaligned_uint32_t*) address)[index] = value;
}

XNN_INLINE static void unaligned_indexed_store_u16(void* address, size_t index, uint16_t value) {
  typedef XNN_UNALIGNED uint16_t xnn_unaligned_uint16_t;
  ((xnn_unaligned_uint16_t*) address)[index] = value;
}

XNN_INLINE static uint64_t unaligned_load_u64(const void* address) {
  typedef XNN_UNALIGNED uint64_t xnn_unaligned_uint64_t;
  return *((const xnn_unaligned_uint64_t*) address);
}
