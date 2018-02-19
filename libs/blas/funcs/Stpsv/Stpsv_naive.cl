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

// ONE WORK_ITEM KERNEL
__kernel void Stpsv_naive(const int uplo, const int trans, const int diag, int n, __global float* ap, __global float* x, int incx)
{
    bool ntrans = trans == 0;
    bool ltriangle = uplo == 1;
    bool ndiag = diag == 0;

    if (ntrans) {
        if (ltriangle) {
            uint starting = 0;
            for (int i=0; i<n; i++) {
                if (ndiag) {
                    x[i*incx] = x[i*incx]/ap[starting];
                }
                float temp = x[i*incx];
                for (int j = 1; j<n-i; j++) {
                    x[(i+j)*incx] -= temp*ap[starting+j];
                }
                starting += n-i;
            }
        } else {
            uint starting = n*(n+1)/2-1;
            for (int i=n-1; i>=0; i--) {
                if (ndiag) {
                    x[i*incx] = x[i*incx]/ap[starting];
                }
                float temp = x[i*incx];
                for (int j = 1; j<i+1; j++) {
                    x[(i-j)*incx] -= temp*ap[starting-j];
                }
                starting -= i+1;
            }
        }
    } else {
        if (ltriangle) {
            uint starting = n*(n+1)/2 - 1;
            for (int i=n-1; i>=0; i--) {
                if (ndiag) {
                    x[i*incx] = x[i*incx]/ap[starting];
                }
                float temp = x[i*incx];
                for (int j=1; j<i+1; j++) {
                    x[(i-j)*incx] -= temp*ap[starting-(n-i-1)*j-j*(j+1)/2];
                }
                starting -= n-i+1;
            }
        } else {
            uint starting = 0;
            for (int i=0; i<n; i++) {
                 if (ndiag) {
                    x[i*incx] = x[i*incx]/ap[starting];
                }
                float temp = x[i*incx];
                for (int j=1; j<n-i; j++) {
                    x[(i+j)*incx] -= temp*ap[starting+i*j+j*(j+1)/2];
                }
                starting += i+2;
            }
        }
    }
}
