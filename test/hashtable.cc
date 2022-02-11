// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/hashtable.h>

#include <xnnpack.h>

#include <gtest/gtest.h>

TEST(HashTable, InitAndRelease)
{
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_hash_table table;
  xnn_hash_table_init(&table);
  EXPECT_EQ(0, table.num_entries);
  xnn_hash_table_release(&table);
}

TEST(HashTable, LookupFromEmpty)
{
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_hash_table table;
  xnn_hash_table_init(&table);
  const void* got = xnn_hash_table_lookup(&table, 0);
  EXPECT_EQ(NULL, got);
  xnn_hash_table_release(&table);
}

TEST(HashTable, NullValue)
{
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_hash_table table;
  xnn_hash_table_init(&table);
  void* got = xnn_hash_table_lookup(&table, 0);
  EXPECT_EQ(NULL, got);

  // Insert a NULL value
  EXPECT_EQ(true, xnn_hash_table_insert(&table, 1, 0));
  EXPECT_EQ(1, table.num_entries);
  got = xnn_hash_table_lookup(&table, 1);
  EXPECT_EQ(NULL, got);

  xnn_hash_table_release(&table);
}

TEST(HashTable, Simple)
{
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_hash_table table;
  xnn_hash_table_init(&table);

  size_t value = 1;
  const size_t hash = 1;
  EXPECT_EQ(true, xnn_hash_table_insert(&table, hash, &value));
  const void* got = xnn_hash_table_lookup(&table, hash);
  EXPECT_EQ(value, *((size_t*) got));

  xnn_hash_table_release(&table);
}

TEST(HashTable, InsertSameDoesNotOverwrite)
{
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_hash_table table;
  xnn_hash_table_init(&table);

  size_t old_value = 1;
  const size_t hash = 1;
  EXPECT_EQ(true, xnn_hash_table_insert(&table, hash, &old_value));

  size_t new_value = 2;
  // Insert same hash, but different old_value
  EXPECT_EQ(false, xnn_hash_table_insert(&table, hash, &new_value));

  const void* got = xnn_hash_table_lookup(&table, hash);
  EXPECT_EQ(old_value, *((size_t*) got));
  EXPECT_EQ(1, table.num_entries);

  const size_t missing_entry_hash = 3;
  EXPECT_EQ(NULL, xnn_hash_table_lookup(&table, missing_entry_hash));

  xnn_hash_table_release(&table);
}

TEST(HashTable, InsertManyElements)
{
  EXPECT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  struct xnn_hash_table table;
  xnn_hash_table_init(&table);
  const size_t n = 1000;
  size_t elements[n];
  for (size_t i = 0; i < n; i++) {
    elements[i] = i;
  }

  for (size_t i = 0; i < n; i++) {
    EXPECT_EQ(true, xnn_hash_table_insert(&table, i, &elements[i]));
    // We should be able to find it right after we insert.
    const void* got = xnn_hash_table_lookup(&table, i);
    EXPECT_EQ(elements[i], *((size_t*) got));
  }

  // Check that everything is still there.
  for (size_t i = 0; i < n; i++) {
    const void* got = xnn_hash_table_lookup(&table, i);
    EXPECT_EQ(elements[i], *((size_t*) got));
  }

  EXPECT_EQ(1000, table.num_entries);

  xnn_hash_table_release(&table);
}
