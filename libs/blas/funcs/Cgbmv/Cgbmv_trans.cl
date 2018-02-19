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

#define ACCESS(a, m, n, lda) a[(n)*(lda) + (m)]

__kernel void Cgbmv_trans(int m, int kl, int ku, complex_t alpha, __global complex_t* a, int lda, __global complex_t* x, int incx, complex_t beta, __global complex_t* y, int incy)
{
    int gid = get_global_id(0);

    complex_t out_y = cmul(y[gid*incy], beta);
    // Calculate alpha * A^T * x
    complex_t tmp_x = 0.f;
    for (int j=max(gid-ku, 0); j<min(gid+kl+1, m); j++) {
        tmp_x += cmul(ACCESS(a, ku+j-gid, gid, lda), x[j*incx]);
    }
    out_y += cmul(alpha, tmp_x);
    y[gid*incy] = out_y;
}
