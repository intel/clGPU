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

__kernel void Sspmv_naive(int uplo, int n, float alpha, __global float* AP, __global float* x, int incx, float beta, __global float* y, int incy)
{
    bool isUp = (uplo == 0);

    if(isUp)
    {
        int kk1 = 0;
        int kk2 = 0;

        for(int i = 0; i < n; i++)
        {
            int k1 = kk1;
            int k2 = kk2;
            float value = 0.f;
            
            for(int j = i; j < n; j++, k1 += j)
            {
                value += AP[k1] * x[j * incx];
            }

            for(int j = 0; j < i; j++, k2++)
            {
                value += AP[k2] * x[j * incx];
            }

            y[i * incy] = alpha * value + beta * y[i * incy];

            kk1 += (i + 2);
            kk2 += (i + 1);
        }
    }

    if(!isUp)
    {
        int kk1 = 0;
        int kk2 = 0;

        for(int i = 0; i < n; i++)
        {
            int k1 = kk1;
            int k2 = kk2;
            float value = 0.f;
            
            for(int j = i; j < n; j++, k1++)
            {
                value += AP[k1] * x[j * incx];
            }

            for(int j = 0; j < i; j++, k2 += (n - j))
            {
                value += AP[k2] * x[j * incx];
            }

            y[i * incy] = alpha * value + beta * y[i * incy];

            kk1 += (n - i);
            kk2 += 1;
        }
    }
}
