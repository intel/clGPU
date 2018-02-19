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

#define MAT_ACCESS(A, row, col, N) A[col*N + row]

#define ICLBLAS_OP_N (0)
#define ICLBLAS_OP_T (1)
#define ICLBLAS_OP_C (2)

__kernel void Sgemv_naive(int trans, int m, int n, float alpha, __global float* A, int lda, __global float* x, int incx, float beta, __global float* y, int incy)
{
    /* For non-transpose matrix A*/
    if(trans == ICLBLAS_OP_N)
    {
        for (uint row_id = 0; row_id < m; ++row_id)
        {
            float l_result = 0;
            for (uint col_id = 0; col_id < n; ++col_id)
            {
                l_result = mad(alpha * MAT_ACCESS(A, row_id, col_id, lda), x[col_id * incx], l_result);
            }

            if(beta != 0)
                y[row_id * incy] = l_result + beta * y[row_id * incy];
            else
                y[row_id * incy] = l_result;
        }
    }

    /* For (conj.)-transpose matrix A*/
    if(trans == ICLBLAS_OP_T | trans == ICLBLAS_OP_C)
    {
        for (uint row_id = 0; row_id < n; ++row_id)
        {
            float l_result = 0;
            for (uint col_id = 0; col_id < m; ++col_id)
            {
                l_result = mad(alpha * MAT_ACCESS(A, col_id, row_id, lda), x[col_id * incx], l_result);
            }

            if(beta != 0)
                y[row_id * incy] = l_result + beta * y[row_id * incy];
            else
                y[row_id * incy] = l_result;
        }
    }
}
