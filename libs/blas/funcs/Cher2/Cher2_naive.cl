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

__kernel void Cher2_naive(int uplo, complex_t alpha, __global complex_t* x, int incx, __global complex_t* y, int incy, __global complex_t* a, int lda)
{
    bool ltriangle = uplo == 1;

    int ind_m = get_global_id(0);
    int ind_n = get_global_id(1);

    bool upper = ind_m < ind_n;
    bool lower = ind_m > ind_n;

    if (ltriangle && upper) return;
    if (!ltriangle && lower) return;
    
    complex_t result = ACCESS(a, ind_m, ind_n, lda);
    result += cmul(alpha, cmul(x[ind_m*incx], conjg(y[ind_n*incy])));
    result += cmul(conjg(alpha), cmul(y[ind_m*incy], conjg(x[ind_n*incx])));
    if (ind_m == ind_n) result.y = 0.f;
    ACCESS(a, ind_m, ind_n, lda) = result;
}
