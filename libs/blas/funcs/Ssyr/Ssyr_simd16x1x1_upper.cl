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

// Each tile has size 16x16 - calculated by single sub_group
// One work item calculates 1x16 (one row from tile)
__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(SIMD, 1, 1)))
__kernel void Ssyr_simd16x1x1_upper(uint n, float alpha, __global float* x, uint incx, __global float* a, uint lda)
{
    const uint slid = get_sub_group_local_id();

    const uint tiles = (n + SIMD - 1) / SIMD;
    const uint tile_number = get_group_id(0);

    const bool perfect_fit = tiles * SIMD == n;
    
    // Calculate starting coordinates of sub_group tile
    uint sub_group_start_m = 0;
    uint sub_group_start_n = 0;
    if (slid == 0) {
        uint tiles_left = tile_number;
        while (tiles_left + sub_group_start_m >= tiles) {
            tiles_left = tiles_left - tiles + sub_group_start_m;
            sub_group_start_n += 1;
            sub_group_start_m = sub_group_start_n;
        }
        sub_group_start_m += tiles_left;
        sub_group_start_m -= sub_group_start_n;
        sub_group_start_n = tiles - sub_group_start_n - 1;

        sub_group_start_m *= SIMD;
        sub_group_start_n *= SIMD;
    }

    sub_group_start_m = sub_group_broadcast(sub_group_start_m, 0);
    sub_group_start_n = sub_group_broadcast(sub_group_start_n, 0);

    uint ind_m = sub_group_start_m + slid;
    uint ind_n = sub_group_start_n + slid;

    float left_x;
    float right_x;

    uint index_a = IDX(ind_m, sub_group_start_n, lda);

    #define SSYR_SIMD_UPDATE(shuffle_idx)                                    \
        float this_a = a[index_a];                                           \
        float prod = left_x * intel_sub_group_shuffle(right_x, shuffle_idx); \
        this_a = mad(alpha, prod, this_a);                                   \
        a[index_a] = this_a

    // Handle tiles that exceed target area
    if (!perfect_fit && (tiles - 1) * SIMD == sub_group_start_n) {
        if (ind_m < n) left_x = x[ind_m * incx];
        if (ind_n < n) right_x = x[ind_n * incx];

        if (sub_group_start_m == sub_group_start_n) {
            index_a += slid * lda;
            for (uint i = slid, j = ind_n; j < n; j++, i++) {
                SSYR_SIMD_UPDATE(i);
                index_a += lda;
            }
        } else {
            for (uint i = 0; i < n - sub_group_start_n; i++) {
                SSYR_SIMD_UPDATE(i);
                index_a += lda;
            }
        }
    // Tiles that fit into target area:
    // Tiles on diagonal
    } else if (sub_group_start_m == sub_group_start_n) {
        left_x = x[ind_m * incx];
        right_x = left_x;

        index_a += slid * lda;
        for (uint i = slid; i < SIMD; i++) {
            SSYR_SIMD_UPDATE(i);
            index_a += lda;
        }
    // Tiles that fit and are not on a diagonal
    } else {
        left_x = x[ind_m * incx];
        right_x = x[ind_n * incx];

        __attribute__((opencl_unroll_hint(SIMD)))
        for (uint i = 0; i < SIMD; i++) {
            SSYR_SIMD_UPDATE(i);
            index_a += lda;
        }
    }

    #undef SSYR_SIMD_UPDATE
}
