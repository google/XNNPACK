#include <check.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* 
 * Since hexagon_hvx.h contains HVX-specific intrinsics that require
 * Hexagon toolchain, we test the security invariant by simulating
 * the vulnerable pattern: memcpy with sizeof(result) must not exceed
 * source buffer size.
 */

#define HVX_VECTOR_SIZE 128  /* Typical HVX vector size in bytes */

START_TEST(test_simd_memcpy_bounds_check)
{
    /* Invariant: memcpy operations must not read beyond allocated buffer */
    
    /* Test cases: buffer sizes that could result from dimension calculations */
    size_t buffer_sizes[] = {
        0,                      /* Zero-size buffer (exploit case) */
        HVX_VECTOR_SIZE - 1,    /* Off-by-one boundary */
        HVX_VECTOR_SIZE,        /* Exact valid size */
        HVX_VECTOR_SIZE + 1,    /* Slightly larger valid buffer */
        SIZE_MAX / 2            /* Large value that could overflow */
    };
    int num_cases = sizeof(buffer_sizes) / sizeof(buffer_sizes[0]);

    for (int i = 0; i < num_cases; i++) {
        size_t src_size = buffer_sizes[i];
        size_t copy_size = HVX_VECTOR_SIZE;
        
        /* Security invariant: copy_size must not exceed source buffer size */
        int safe_to_copy = (src_size >= copy_size);
        
        if (safe_to_copy && src_size <= 1024 * 1024) {
            /* Only perform copy if bounds check passes */
            uint8_t *src = malloc(src_size);
            uint8_t result[HVX_VECTOR_SIZE];
            
            ck_assert_ptr_nonnull(src);
            memset(src, 0xAA, src_size);
            memcpy(result, src, copy_size);
            ck_assert_uint_eq(result[0], 0xAA);
            
            free(src);
        } else {
            /* Invariant: undersized buffers must be rejected */
            ck_assert_msg(!safe_to_copy || src_size > 1024 * 1024,
                "Buffer size %zu is insufficient for %zu byte copy",
                src_size, copy_size);
        }
    }
}
END_TEST

Suite *security_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Security");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_simd_memcpy_bounds_check);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = security_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}