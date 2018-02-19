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

__kernel void Cgemv_naive(int transa, int m, int n, complex_t alpha, __global complex_t* A, int lda, __global complex_t* x, int incx, complex_t beta, __global complex_t* y, int incy)
{
    int global_id = get_global_id(0);

    if(transa == 0)
    {
        complex_t l_result = 0;
        for(int i = 0; i<n; ++i)
        {
            l_result += cmul(cmul(alpha, A[i * lda + global_id]), x[i * incx]); 
        }
        y[global_id * incy] = l_result + cmul(beta, y[global_id * incy]); 
    }

    else if(transa == 1)
    {
        complex_t l_result = 0;
        for(int i = 0; i<m; ++i)
        {
            l_result += cmul(cmul(alpha, A[global_id * lda + i]), x[i * incx]); 
        }
        y[global_id * incy] = l_result + cmul(beta, y[global_id * incy]); 
    }

    else if(transa == 2)
    {
        complex_t l_result = 0;
        for(int i = 0; i<m; ++i)
        {
            l_result += cmul(cmul(alpha, conjg(A[global_id * lda + i])), x[i * incx]);
        }
        y[global_id * incy] = l_result + cmul(beta, y[global_id * incy]); 
    }

}
