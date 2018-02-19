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

__kernel void Ctrmv_naive(int uplo, int trans, int diag, int n, __global complex_t* A, int lda, __global complex_t* x, int incx)
{
    bool isUp = (uplo == 0);
    bool isDiag = (diag == 1);
    bool isNTrans = (trans == 0);

    // Upper-NonTranspoze Matrix
    if(isUp && isNTrans)
    {
        for (int i = 0; i < n; i++)
        {
            x[i * incx] = isDiag ? x[i * incx] : cmul(A[i * lda + i], x[i * incx]);

            for (int k = i + 1; k < n; k++)
            {
                x[i * incx] += cmul(A[i * lda + k], x[k * incx]);
            }
        }
    }

    // Upper-Transpoze Matrix
    if(isUp && !isNTrans)
    {
        // Transpoze
        if(trans == 1)
        {
            for (int i = n - 1; i >= 0; i--)
            {
                x[i * incx] = isDiag ? x[i * incx] : cmul(A[i * lda + i], x[i * incx]);

                for (int k = 0; k < i; k++)
                {
                    x[i * incx] += cmul(A[i * lda + k], x[k * incx]);
                }
            }
        }

        // Hermitian
        if(trans == 2)
        {
            for (int i = n - 1; i >= 0; i--)
            {
                x[i * incx] = isDiag ? x[i * incx] : cmul(conjg(A[i * lda + i]), x[i * incx]);

                for (int k = 0; k < i; k++)
                {
                    x[i * incx] += cmul(conjg(A[i * lda + k]), x[k * incx]);
                }
            }
        }
    }

    // Lower-NonTranspoze Matrix
    if(!isUp && isNTrans)
    {

        for (int i = n - 1; i >= 0; i--)
        {
            x[i * incx] = isDiag ? x[i * incx] : cmul(A[i * lda + i], x[i * incx]);

            for (int k = 0; k < i; k++)
            {
                x[i * incx] += cmul(A[i * lda + k], x[k * incx]);
            }
        }
    }

    // Lower-Transpoze Matrix
    if(!isUp && !isNTrans)
    {
        // Transpoze
        if(trans == 1)
        {
            for (int i = 0; i < n; i++)
            {
                x[i * incx] = isDiag ? x[i * incx] : cmul(A[i * lda + i], x[i * incx]);

                for (int k = i + 1; k < n; k++)
                {
                    x[i * incx] += cmul(A[i * lda + k], x[k * incx]);
                }
            }
        }

        // Hermitian
        if(trans == 2)
        {
            for (int i = 0; i < n; i++)
            {
                x[i * incx] = isDiag ? x[i * incx] : cmul(conjg(A[i * lda + i]), x[i * incx]);

                for (int k = i + 1; k < n; k++)
                {
                    x[i * incx] += cmul(conjg(A[i * lda + k]), x[k * incx]);
                }
            }
        }
    }
}
