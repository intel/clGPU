/* Copyright (c) 2017-2018 Intel Corporation
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define IDX(m, n, ld) ((n)*(ld) + (m))

#define SIMD (16)

__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(SIMD, 1, 1)))
__kernel void Ssyr_early_return_simd16x1x1_lower(uint n, float alpha, __global float* x, uint incx, __global float* a, uint lda)
{
    const uint group_m = get_group_id(0);
    const uint group_n = get_group_id(1);
    const uint group_start_m = group_m * SIMD;
    const uint group_start_n = group_n * SIMD;

    const bool upper = group_start_m < group_start_n;
    if (upper) return; // Early return tiles that are in upper part of matrix

    const uint slid = get_sub_group_local_id();
    const uint ind_m = group_start_m + slid;
    const uint ind_n = group_start_n + slid;

    const bool diagonal = group_start_m == group_start_n; // Tile is on diagonal
    const bool last_row = group_start_m + SIMD > n; // Tile is in last row and exceeds target area

    uint index_a = IDX(ind_m, group_start_n, lda);

    float left_x;
    if (ind_m < n) left_x = x[ind_m * incx];

    float right_x;
    if (ind_n < n) right_x = x[ind_n * incx];

    #define SSYR_UPDATE_PROD(prod)         \
        float this_a = a[index_a];         \
        this_a = mad(alpha, prod, this_a); \
        a[index_a] = this_a

    #define SSYR_SIMD_UPDATE(shuffle_idx)                                        \
        float prod = left_x * intel_sub_group_shuffle(right_x, shuffle_idx);     \
        SSYR_UPDATE_PROD(prod)

    if (last_row && diagonal) {
        uint columns = n - group_start_n;
        for (int i = 0; i < columns; i++) {
            if (i <= slid) {
                float prod = left_x * intel_sub_group_shuffle(right_x, i);
                if (ind_m < n) {
                    SSYR_UPDATE_PROD(prod);
                    index_a += lda;
                }
            }
        }
    } else if (last_row) {
        __attribute__((opencl_unroll_hint(SIMD)))
        for (int i = 0; i < SIMD; i++) {
            float prod = left_x * intel_sub_group_shuffle(right_x, i);
            if (ind_m < n) {
                SSYR_UPDATE_PROD(prod);
                index_a += lda;
            }
        }
    } else if (diagonal) {
        __attribute__((opencl_unroll_hint(SIMD)))
        for (int i = 0; i < SIMD; i++) {
            if (i <= slid) {
                SSYR_SIMD_UPDATE(i);
                index_a += lda;
            }
        }
    } else {
        __attribute__((opencl_unroll_hint(SIMD)))
        for (int i = 0; i < SIMD; i++) {
            SSYR_SIMD_UPDATE(i);
            index_a += lda;
        }
    }

    #undef SSYR_SIMD_UPDATE
    #undef SSYR_UPDATE_PROD
}
