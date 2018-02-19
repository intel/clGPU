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

__kernel void Csyrk_naive(int uplo, int trans, int n, int k, complex_t alpha, __global complex_t* a, int lda, complex_t beta, __global complex_t* c, int ldc)
{
    bool ltriangle = uplo == 1;
    bool ntrans = trans == 0;
    bool conj = trans == 2;

    int ind_m = get_global_id(0);
    int ind_n = get_global_id(1);

    bool upper = ind_m < ind_n;
    bool lower = ind_m > ind_n;

    // Early return for work items in non-referenced parts
    if (ltriangle && upper) return;
    if (!ltriangle && lower) return;

    complex_t value = 0.f;

    if (ntrans) {
            for (int i=0; i<k; i++) {
                value += cmul(ACCESS(a, ind_m, i, lda), ACCESS(a, ind_n, i, lda));
            }
    } else {
            for (int i=0; i<k; i++) {
                value += cmul(ACCESS(a, i, ind_m, lda), ACCESS(a, i, ind_n, lda));
            }
            if (conj) value = conjg(value);
    }

    value = cmul(alpha, value);
    ACCESS(c, ind_m, ind_n, ldc) = cmul(beta, ACCESS(c, ind_m, ind_n, ldc)) + value;
}
