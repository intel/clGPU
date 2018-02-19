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
    return a[n*N + m];
}

kernel void Strsv_naive(const int uplo, const int trans, const int diag, const int n,
                          const global float* a, const int lda, global float* x, const int incx)
{
    bool ntrans = trans == 0;
    bool ltriangle = uplo == 1;
    bool ndiag = diag == 0;

    if (ntrans) {
        if (ltriangle) {
            for (int i = 0; i<n; i++) {
                if (ndiag) {
                    x[i*incx] = x[i*incx]/access(a, i, i, lda);
                }
                float temp = x[i*incx];
                for (int j = i+1; j<n; j++) {
                    x[j*incx] -= temp*access(a, j, i, lda);
                }
            }
        } else {
            for (int i = n-1; i>=0; i--) {
                if (ndiag) {
                    x[i*incx] = x[i*incx]/access(a, i, i, lda);
                }
                float temp = x[i*incx];
                for (int j = i-1; j>=0; j--) {
                    x[j*incx] -= temp*access(a, j, i, lda);
                }
            }
        }
    } else {
        if (ltriangle) {
            for (int i = n-1; i>=0; i--) {
                if (ndiag) {
                    x[i*incx] = x[i*incx]/access(a, i, i, lda);
                }
                float temp = x[i*incx];
                for (int j = i-1; j>=0; j--) {
                    x[j*incx] -= temp*access(a, i, j, lda);
                }
            }
        } else {
            for (int i = 0; i<n; i++) {
                if (ndiag) {
                    x[i*incx] = x[i*incx]/access(a, i, i, lda);
                }
                float temp = x[i*incx];
                for (int j = i+1; j<n; j++) {
                    x[j*incx] -= temp*access(a, i, j, lda);
                }
            }
        }
    }

}
