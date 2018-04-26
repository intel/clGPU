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

#define IDX(row, col, ld) ((col)*(lda) + (row))
#define IDX_B(row, col, ld, ku) IDX((ku) + (row) - (col), col, ld)

__kernel void Sgbmv_ntrans(uint n, uint kl, uint ku, float alpha, __global float* A, uint lda, __global float* x, uint incx, float beta, __global float* y, uint incy)
{
    const uint gid = get_global_id(0);
    const uint row = gid;

    float out_y = 0.f;
    if (beta != 0.f)
    {
        out_y = y[row * incy] * beta;
    }
    // Calculate alpha*A^T*x
    const uint start_col = max(row, kl) - kl;
    const uint end_col = min(row + ku + 1, n);

    float prod = 0.f;

    for (uint col = start_col; col < end_col; ++col) {
        float this_x = x[col * incx];
        float this_A = A[IDX_B(row, col, lda, ku)];
        prod = mad(this_A, this_x, prod);
    }

    out_y = mad(alpha, prod, out_y);
    y[row * incy] = out_y;
}
