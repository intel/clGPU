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

// TODO: Move to separate header file after include support
#define ICLBLAS_FILL_MODE_LOWER (1)
//
#define IDX(m, n, ld) ((n)*(ld) + (m))

__kernel void Csyr_naive(int uplo, uint n, complex_t alpha, __global complex_t* x, uint incx, __global complex_t* a, uint lda)
{
    bool ltriangle = uplo == ICLBLAS_FILL_MODE_LOWER;

    if (ltriangle) {
        for (uint col=0; col<n; col++) {
            complex_t tmp_prod = cmul(alpha, x[col*incx]);
            for (uint row=col; row<n; row++) {
                a[IDX(row, col, lda)] += cmul(tmp_prod, x[row*incx]);
            }
        }
    } else {
        for (uint col=0; col<n; col++) {
            complex_t tmp_prod = cmul(alpha, x[col*incx]);
            for (uint row=0; row<=col; row++) {
                a[IDX(row, col, lda)] += cmul(tmp_prod, x[row*incx]);
            }
        }
    }
}
