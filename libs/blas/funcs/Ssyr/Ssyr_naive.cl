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

__kernel void Ssyr_naive(int uplo, int n, float alpha, __global float* x, int incx, __global float* a, int lda)
{
    bool ltriangle = uplo == 1;

    if (ltriangle) {
        for (int i=0; i<n; i++) {
            for (int j=i; j<n; j++) {
                a[i*lda + j] += alpha*x[i*incx]*x[j*incx];
            }
        }
    } else {
        for (int i=0; i<n; i++) {
            for (int j=0; j<=i; j++) {
                a[i*lda + j] += alpha*x[i*incx]*x[j*incx];
            }
        }
    }
}
