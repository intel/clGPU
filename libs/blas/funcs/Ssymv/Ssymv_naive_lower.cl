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

 #define MAT_ACCESS(A, col, row, n) A[col * n + row]

__kernel void Ssymv_naive_lower(uint n, float alpha, __global float* A, uint lda, __global float* x, int incx, float beta, __global float* y, int incy)
{
     for (uint i = 0; i < n; ++i)
     {
        float result = 0;
        for (uint j = 0; j < n; ++j)
        {
            if (i >= j)
               result = fma(MAT_ACCESS(A, j, i, lda), x[j * incx], result);
            else
               result = fma(MAT_ACCESS(A, i, j, lda), x[j * incx], result);
        }

        if(beta != 0)
            y[i * incy] = fma(alpha, result, beta * y[i * incy]);
        else
            y[i * incy] = alpha * result;
     }
}
