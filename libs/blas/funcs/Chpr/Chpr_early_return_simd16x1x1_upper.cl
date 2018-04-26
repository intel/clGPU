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

#define IDX_PACKED_UPPER(m, n) (((n) + 1) * (n) / 2 + (m))

#define SIMD (16)

// Each sub group calculates tile of size 16x16, one work item calulates 16 values from a row
// Work size is n x n/16 and sub groups in lower part of matrix are returned before calculations
__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(SIMD, 1, 1)))
__kernel void Chpr_early_return_simd16x1x1_upper(uint n, float alpha, __global complex_t* x, uint incx, __global complex_t* a)
{
    const uint group_m = get_group_id(0);
    const uint group_n = get_group_id(1);
    const uint group_start_m = group_m * SIMD;
    const uint group_start_n = group_n * SIMD;

    const bool lower = group_start_m > group_start_n;
    if (lower) return; // Early return tiles that are in lower part of matrix

    const uint slid = get_sub_group_local_id();
    const uint ind_m = group_start_m + slid;
    const uint ind_n = group_start_n + slid;

    const bool diagonal = group_start_m == group_start_n; // Tile is on a diagonal
    const bool last_col = group_start_n + SIMD > n; // Tile is in last column and exceeds target area

    uint index_a = IDX_PACKED_UPPER(ind_m, group_start_n);

    // Increment needed to go to next column in matrix a
    // Iteration is number of current column - group_start_n, that is column number in this tile
    #define NEXT_COL(iteration) (group_start_n + (iteration) + 1)

    if (last_col && diagonal) {
        if (ind_m >= n) return;
        complex_t left_x = x[ind_m * incx];
        complex_t right_x = conjg(left_x);

        const uint columns = n - group_start_n;

        for (uint tile_col = 0; tile_col < columns; tile_col++) {
            if (tile_col >= slid) {
                complex_t this_a = a[index_a];

                complex_t prod = cmul(left_x, intel_sub_group_shuffle(right_x, tile_col));
                this_a = cfmaf(prod, alpha, this_a);
                if (tile_col == slid) this_a.y = 0.f;

                a[index_a] = this_a;
            }

            index_a += NEXT_COL(tile_col);
        }
    } else if (last_col) {
        complex_t left_x = x[ind_m * incx];
        complex_t right_x;
        if (ind_n < n) right_x = conjg(x[ind_n * incx]);

        const uint columns = n - group_start_n;

        for (uint tile_col = 0; tile_col < columns; tile_col++) {
            complex_t this_a = a[index_a];

            complex_t prod = cmul(left_x, intel_sub_group_shuffle(right_x, tile_col));
            this_a = cfmaf(prod, alpha, this_a);

            a[index_a] = this_a;

            index_a += NEXT_COL(tile_col);
        }
    } else if (diagonal) {
        complex_t left_x = x[ind_m * incx];
        complex_t right_x = conjg(left_x);

        __attribute__((opencl_unroll_hint(SIMD)))
        for (uint tile_col = 0; tile_col < SIMD; tile_col++) {
            if (tile_col >= slid) {
                complex_t this_a = a[index_a];

                complex_t prod = cmul(left_x, intel_sub_group_shuffle(right_x, tile_col));
                this_a = cfmaf(prod, alpha, this_a);
                if (tile_col == slid) this_a.y = 0.f;

                a[index_a] = this_a;
            }

            index_a += NEXT_COL(tile_col);
        }
    } else {
        complex_t left_x = x[ind_m * incx];
        complex_t right_x = conjg(x[ind_n * incx]);

        __attribute__((opencl_unroll_hint(SIMD)))
        for (uint tile_col = 0; tile_col < SIMD; tile_col++) {
            complex_t this_a = a[index_a];

            complex_t prod = cmul(left_x, intel_sub_group_shuffle(right_x, tile_col));
            this_a = cfmaf(prod, alpha, this_a);

            a[index_a] = this_a;

            index_a += NEXT_COL(tile_col);
        }
    }

    #undef NEXT_COL
}
