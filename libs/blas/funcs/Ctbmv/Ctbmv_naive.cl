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

complex_t access(const global complex_t* a, int m, int n, int N) {
    return a[n * N + m];
}

__kernel void Ctbmv_naive(int uplo, int trans, int diag, int n, int k, __global complex_t* A, int lda, __global complex_t* x, int incx)
{
    bool isUp = (uplo == 0);
    bool isDiag = (diag == 1);
    bool isNTrans = (trans == 0);

    if(isUp && isNTrans)
    {
        for(int i = 0; i < n; i++)
        {
            x[i * incx] = isDiag ? x[i * incx] : cmul(access(A, k, i, lda), x[i * incx]);

            for(int j = 1; j < min(n-i, k+1); j++)
            {
                x[i * incx] += cmul(access(A, k-j, i+j, lda), x[(i+j) * incx]);
            }
        }
    }

    if(isUp && !isNTrans)
    {
        if(trans == 1)
        {
            for(int i = n-1; i >= 0; i--)
            {
                x[i * incx] = isDiag ? x[i * incx] : cmul(access(A, k, i, lda), x[i * incx]);

                for(int j = 1; j < min(i+1, k+1); j++)
                {
                    x[i * incx] += cmul(access(A, k-j, i, lda), x[(i-j) * incx]);
                }
            }
        }

        if(trans == 2)
        {
            for(int i = n-1; i >= 0; i--)
            {
                x[i * incx] = isDiag ? x[i * incx] : cmul(conjg(access(A, k, i, lda)), x[i * incx]);

                for(int j = 1; j < min(i+1, k+1); j++)
                {
                    x[i * incx] += cmul(conjg(access(A, k-j, i, lda)), x[(i-j) * incx]);
                }
            }
        }
    }

    if(!isUp && isNTrans)
    {
        for(int i = n-1; i >= 0; i--)
        {
            x[i * incx] = isDiag ? x[i * incx] : cmul(access(A, 0, i, lda), x[i * incx]);

            for(int j = 1; j < min(i+1, k+1); j++)
            {
                x[i * incx] += cmul(access(A, j, i-j, lda), x[(i-j) * incx]);
            }
        }
    }

    if(!isUp && !isNTrans)
    {
        if(trans == 1)
        { 
            for(int i = 0; i < n; i++)
            {
                x[i * incx] = isDiag ? x[i * incx] : cmul(access(A, 0, i, lda), x[i * incx]);

                for(int j = 1; j < min(k+1, n-i); j++)
                {
                    x[i * incx] += cmul(access(A, j, i, lda), x[j * incx]);
                }
            }        
        }

        if(trans == 2)
        {
            for(int i = 0; i < n; i++)
            {
                x[i * incx] = isDiag ? x[i * incx] : cmul(conjg(access(A, 0, i, lda)), x[i * incx]);

                for(int j = 1; j < min(k+1, n-i); j++)
                {
                    x[i * incx] += cmul(conjg(access(A, j, i, lda)), x[j * incx]);
                }
            }    
        }
    }
}
