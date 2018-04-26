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

#ifndef SIMD
#define SIMD 8
#endif

#ifndef VEC_SIZE
#define VEC_SIZE 4
#endif

#include "vector_operations.h"

#define WIDTH ((SIMD) * (VEC_SIZE))

#define IDX(row, col, ld) ((col) * (ld) + (row))
#define IDX_B(row, col, ld, ku) IDX((row) + (ku) - (col), col, ld)

__attribute__((intel_reqd_sub_group_size(SIMD)))
__attribute__((reqd_work_group_size(SIMD, 1, 1)))
#ifdef NOINCX
__kernel void Sgbmv_simd_one_row_trans(uint m, uint n, uint kl, uint ku, float alpha, __global float* A, uint lda, __global float* x, float beta, __global float* y, uint incy)
#else
__kernel void Sgbmv_simd_one_row_trans(uint m, uint n, uint kl, uint ku, float alpha, __global float* A, uint lda, __global float* x, uint incx, float beta, __global float* y, uint incy)
#endif
{
    const uint gid = get_group_id(0);
    const uint sglid = get_sub_group_local_id();

    const uint row = gid;

    const uint group_start_col = max(row, ku) - ku;
    const uint group_end_col = min(row + kl + 1, m);

    uint group_col = group_start_col;
    uint col = group_col + sglid;

    FLOAT_VS prod = 0.f;

    uint A_ind = IDX_B(group_col, row, lda, ku);

    while (group_col + WIDTH <= group_end_col)
    {
#ifdef NOINCX
        FLOAT_VS this_x = AS_FLOAT_VS( BLOCK_READ_VS( (__global uint*)x + group_col ) );
#else
        FLOAT_VS this_x = FLOAT_LIST_LOAD_VS( x, col * incx, SIMD * incx );
#endif
        FLOAT_VS this_A = AS_FLOAT_VS( BLOCK_READ_VS( (__global uint*)A + A_ind ) );
        prod = mad(this_x, this_A, prod);

        group_col += WIDTH;
        col += WIDTH;
        A_ind += WIDTH;
    }

    if (group_col < group_end_col)
    {
        A_ind += sglid;
        __attribute__((opencl_unroll_hint(VEC_SIZE)))
        for (uint elem = 0; elem < VEC_SIZE; ++elem)
        {
            if (col < group_end_col)
            {
#ifdef NOINCX
                float this_x = x[col];
#else
                float this_x = x[col * incx];
#endif
                float this_A = A[A_ind];
                VEC_ACCESS_VS( prod, elem ) = mad(this_x, this_A, VEC_ACCESS_VS( prod, elem ));

                col += SIMD;
                A_ind += SIMD;
            }
        }
    }

    float row_prod = sub_group_reduce_add( VEC_SUM_VS( prod ) );

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
