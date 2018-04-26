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

#define SIMD 8

#define IDX(row, col, ld) ((col) * (ld) + (row))
#define IDX_B(row, col, ld, ku) IDX((row) + (ku) - (col), col, ld)

__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(SIMD, 1, 1)))
__kernel void Sgbmv_simd_one_row_ntrans(uint m, uint n, uint kl, uint ku, float alpha, __global float* A, uint lda, __global float* x, uint incx, float beta, __global float* y, uint incy)
{
    const uint gid = get_group_id(0);
    const uint sglid = get_sub_group_local_id();

    const uint row = gid;

    const uint group_start_col = max(row, kl) - kl;
    const uint group_end_col = min(row + ku + 1, n);

    uint col = group_start_col + sglid;

    float prod = 0.f;

    while (col < group_end_col)
    {
        float this_x = x[col * incx];
        float this_A = A[IDX_B(row, col, lda, ku)];

        prod = mad(this_x, this_A, prod);

        col += SIMD;
    }

    float row_prod = sub_group_reduce_add(prod);

    if (sglid == 0)
    {
        float this_y = y[row * incy];
        row_prod *= alpha;
        if (beta != 0.f)
        {
            this_y = mad(beta, this_y, row_prod);
        }
        else
        {
            this_y = row_prod;
        }
        y[row * incy] = this_y;
    }
}
