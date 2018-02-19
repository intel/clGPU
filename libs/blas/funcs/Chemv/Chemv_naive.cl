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

__kernel void Chemv_naive(int uplo, int n, complex_t alpha, __global complex_t* a, int lda, __global complex_t* x, int incx, complex_t beta, __global complex_t* y, int incy)
{
    bool ltriangle = uplo == 1;

    int ind_m = get_global_id(0);

    complex_t out_y = 0.f;
    if (ltriangle) {
        for (int i=0; i<n; i++) {
            if (i < ind_m) {
                out_y += cmul(x[i*incx], ACCESS(a, ind_m, i, lda));
            } else if (i == ind_m) {
                complex_t diag = (complex_t)(ACCESS(a, ind_m, i, lda).x, 0.f);
                out_y += cmul(x[i*incx], diag);
            } else {
                out_y += cmul(x[i*incx], conjg(ACCESS(a, i, ind_m, lda)));
            }
        }
    } else {
        for (int i=0; i<n; i++) {
            if (i > ind_m) {
                out_y += cmul(x[i*incx], ACCESS(a, ind_m, i, lda));
            } else if (i == ind_m) {
                complex_t diag = (complex_t)(ACCESS(a, ind_m, i, lda).x, 0.f);
                out_y += cmul(x[i*incx], diag);
            } else {
                out_y += cmul(x[i*incx], conjg(ACCESS(a, i, ind_m, lda)));
            }
        }
    }
    out_y = cmul(out_y, alpha);
    y[ind_m*incy] = cmul(beta, y[ind_m*incy]) + out_y;
}
