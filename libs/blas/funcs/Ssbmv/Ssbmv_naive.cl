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

float access(const global float* a, int m, int n, int N) {
    return a[m * N + n];
}

__kernel void Ssbmv_naive(int uplo, int n, int k, float alpha, __global float* A, int lda, __global float* x, int incx, float beta, __global float* y, int incy)
{
    bool isUp = (uplo == 0);

    if(isUp)
    {
        for(int i = 0; i < n; i++)
        {
            float value = x[i * incx] * access(A, i, k, lda);

            for(int j = 1; j < min(n-i, k+1); j++)
            {
                value += x[(i+j) * incx] * access(A, i+j, k-j, lda);
            }

            for(int j = 1; j < min(i+1, k+1); j++)
            {
                value += x[(i-j) * incx] * access(A, i, 0, lda);
            }

            y[i * incy] = alpha * value + beta * y[i * incy];
        }
    }

    if(!isUp)
    {
        for(int i = 0; i < n; i++)
        {
            float value = x[i * incx] * access(A, i, 0, lda);

            for(int j = 1; j < min(k+1, n-i); j++)
            {
                value += x[(i+j) * incx] * access(A, i, j, lda);
            }

            for(int j = 1; j < min(i+1, k+1); j++)
            {
                value += x[(i-j) * incx] * access(A, i, -1*j, lda);
            }

            y[i * incy] = alpha * value + beta * y[i * incy];
        }
    }
}
