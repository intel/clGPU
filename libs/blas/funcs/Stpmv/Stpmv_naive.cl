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

__kernel void Stpmv_naive(int uplo, int trans, int diag, int n, __global float* AP, __global float* x, int incx, __global float* parties)
{
    bool isUp = (uplo == 0);
    bool isDiag = (diag == 1);
    bool isNTrans = (trans == 0);

    if(isUp && isNTrans)
    {
        int kk = 0;

        for(int i = 0; i < n; i++)
        {
            int k = kk + i + 1;
            
            parties[i] = isDiag ? x[i * incx] : x[i * incx] * AP[kk];

            for(int j = i+1; j < n; j++, k += j)
            {
                parties[i] += x[j * incx] * AP[k];
            }

            kk += (i + 2);
        }
    }

    if(isUp && !isNTrans)
    {
        int k = 0;

        for(int i = 0; i < n; i++)
        {
            for(int j = 0; j < i; j++, k++)
            {
                parties[i] += x[j * incx] * AP[k];
            }

            parties[i] += isDiag ? x[i * incx] : x[i * incx] * AP[k];

            k += 1;
        }
    }

    if(!isUp && isNTrans)
    {
        int kk = 0;

        for(int i = 0; i < n; i++)
        {
            int k = kk;
    
            for(int j = 0; j < i; j++, k += (n - j))
            {
                parties[i] += x[j * incx] * AP[k];
            }

            parties[i] += isDiag ? x[i * incx] : x[i * incx] * AP[k];

            kk += 1;
        }
    }

    if(!isUp && !isNTrans)
    {
        int k = 0;

        for(int i = 0; i < n; i++)
        {
            parties[i] = isDiag ? x[i * incx] : x[i * incx] * AP[k];

            k += 1;

            for(int j = i+1; j < n; j++, k++)
            {
                parties[i] += x[j * incx] *  AP[k];
            }
        }
    }

    for(int i = 0; i < n; i++)
    {
        x[i * incx] = parties[i];
    }
}
