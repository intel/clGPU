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

#define MAT_ACCESS(A, r, c, n) A[c * n + r] 

__kernel void Ssyr2_naive_upper(uint n, float alpha, const __constant float* x, int incx, const __constant float* y, int incy, __global float* A, uint lda)
{   
    for (uint row_id = 0; row_id < n; ++row_id)
    {
        for (uint col_id = row_id; col_id < n; ++col_id)
        {
            float rank_delta = fma(x[row_id * incx], y[col_id * incy], y[row_id * incy] * x[col_id * incx]);
            MAT_ACCESS(A, row_id, col_id, lda) = fma(alpha, rank_delta, MAT_ACCESS(A, row_id, col_id, lda));
        }
    }
}
