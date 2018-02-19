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

#define ACCESS(a, m, n, N) a[(n)*(N) + (m)]

__kernel void Csymm_naive(int side, int uplo, int m, int n, complex_t alpha, __global complex_t* a, int lda, __global complex_t* b, int ldb, complex_t beta, __global complex_t* c, int ldc)
{
    bool ltriangle = uplo == 1;
    bool left = side == 0;

    int ind_m = get_global_id(0);
    int ind_n = get_global_id(1);

    complex_t axb = 0.f;
    if (ltriangle) {
        if (left) {
            for (int i=0; i<m; i++) {
                if (i <= ind_m) {
                    axb += cmul(ACCESS(a, ind_m, i, lda), ACCESS(b, i, ind_n, ldb));
                }
                else {
                    axb += cmul(ACCESS(a, i, ind_m, lda), ACCESS(b, i, ind_n, ldb));
                }
            }
        } else {
            for (int i=0; i<n; i++) {
                if (i >= ind_n) {
                    axb += cmul(ACCESS(a, i, ind_n, lda), ACCESS(b, ind_m, i, ldb));
                }
                else {
                    axb += cmul(ACCESS(a, ind_n, i, lda), ACCESS(b, ind_m, i, ldb));
                }
            }
        }
    } else {
        if (left) {
            for (int i=0; i<m; i++) {
                if (i >= ind_m) {
                    axb += cmul(ACCESS(a, ind_m, i, lda), ACCESS(b, i, ind_n, ldb));
                }
                else {
                    axb += cmul(ACCESS(a, i, ind_m, lda), ACCESS(b, i, ind_n, ldb));
                }
            }
        } else {
            for (int i=0; i<n; i++) {
                if (i <= ind_n) {
                    axb += cmul(ACCESS(a, i, ind_n, lda), ACCESS(b, ind_m, i, ldb));
                }
                else {
                    axb += cmul(ACCESS(a, ind_n, i, lda), ACCESS(b, ind_m, i, ldb));
                }
            }
        }
    }

    axb = cmul(axb, alpha);
    complex_t this_c = ACCESS(c, ind_m, ind_n, ldc);
    this_c = cmul(this_c, beta) + axb;
    ACCESS(c, ind_m, ind_n, ldc) = this_c;
}
