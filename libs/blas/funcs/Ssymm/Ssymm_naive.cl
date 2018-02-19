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

#define ACCESS(A, m, n, N) A[(n)*(N) + (m)]

__kernel void Ssymm_naive(int side, int uplo, int m, int n, float alpha, __global float* a, int lda, __global float* b, int ldb, float beta, __global float* c, int ldc)
{
    bool leftside = side == 0;
    bool ltriangle = uplo == 1;

    int ind_m = get_global_id(0);
    int ind_n = get_global_id(1);

    if (ltriangle) {
        if (leftside) {
            float value = 0.f;
            for (int i=0; i<m; i++) {
                if (i > ind_m) {
                    value += ACCESS(a, i, ind_m, lda)*ACCESS(b, i, ind_n, ldb);
                } else {
                    value += ACCESS(a, ind_m, i, lda)*ACCESS(b, i, ind_n, ldb);
                }
            }
            value *= alpha;
            ACCESS(c, ind_m, ind_n, ldc) = beta*ACCESS(c, ind_m, ind_n, ldc) + value;
        } else {
            float value = 0.f;
            for (int i=0; i<n; i++) {
                if (i < ind_n) {
                    value += ACCESS(a, ind_n, i, lda)*ACCESS(b, ind_m, i, ldb);
                } else {
                    value += ACCESS(a, i, ind_n, lda)*ACCESS(b, ind_m, i, ldb);
                }
            }
            value *= alpha;
            ACCESS(c, ind_m, ind_n, ldc) = beta*ACCESS(c, ind_m, ind_n, ldc) + value;
        }
    } else {
        if (leftside) {
            float value = 0.f;
            for (int i=0; i<m; i++) {
                if (i < ind_m) {
                    value += ACCESS(a, i, ind_m, lda)*ACCESS(b, i, ind_n, ldb);
                } else {
                    value += ACCESS(a, ind_m, i, lda)*ACCESS(b, i, ind_n, ldb);
                }
            }
            value *= alpha;
            ACCESS(c, ind_m, ind_n, ldc) = beta*ACCESS(c, ind_m, ind_n, ldc) + value;
        } else {
            float value = 0.f;
            for (int i=0; i<n; i++) {
                if (i > ind_n) {
                    value += ACCESS(a, ind_n, i, lda)*ACCESS(b, ind_m, i, ldb);
                } else {
                    value += ACCESS(a, i, ind_n, lda)*ACCESS(b, ind_m, i, ldb);
                }
            }
            value *= alpha;
            ACCESS(c, ind_m, ind_n, ldc) = beta*ACCESS(c, ind_m, ind_n, ldc) + value;
        }
    }
}
