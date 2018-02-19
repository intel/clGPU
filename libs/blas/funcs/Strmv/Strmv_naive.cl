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

#define ACCESS(M, column, lda, row) M[(column) * (lda) + (row)]

__kernel void Strmv_naive(int uplo, int trans, int diag, int n, int lda, __global float* A, __global float* x, int incx)
{
    bool isUp = (uplo == 0);
    bool isDiag = (diag == 1);
    bool isNTrans = (trans == 0);

    // Upper-NonTranspoze Matrix
    if(isUp && isNTrans)
    {
        for (int row = 0; row < n; row++)
        {
            x[row * incx] = isDiag ? x[row * incx] : ACCESS(A, row, lda, row) * x[row * incx];

            for (int column = row + 1; column < n; column++)
            {
                x[row * incx] += ACCESS(A, column, lda, row) * x[column * incx];
            }
        }
    }

    // Upper-Transpoze Matrix
    if(isUp && !isNTrans)
    {
        for (int row = n - 1; row >= 0; row--)
        {
            x[row * incx] = isDiag ? x[row * incx] : ACCESS(A, row, lda, row) * x[row * incx];

            for (int column = 0; column < row; column++)
            {
                x[row * incx] += ACCESS(A, row, lda, column) * x[column * incx];
            }
        }
    }

    // Lower-NonTranspoze Matrix
    if(!isUp && isNTrans)
    {
        for (int row = n - 1; row >= 0; row--)
        {
            x[row * incx] = isDiag ? x[row * incx] : ACCESS(A, row, lda, row) * x[row * incx];

            for (int column = 0; column < row; column++)
            {
                x[row * incx] += ACCESS(A, column, lda, row) * x[column * incx];
            }
        }
    }

    // Lower-Transpoze Matrix
    if(!isUp && !isNTrans)
    {
        for (int row = 0; row < n; row++)
        {
            x[row * incx] = isDiag ? x[row * incx] : ACCESS(A, row, lda, row) * x[row * incx];

            for (int column = row + 1; column < n; column++)
            {
                x[row * incx] += ACCESS(A, row, lda, column) * x[column * incx];
            }
        }
    }
}
