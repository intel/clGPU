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

__kernel void Chpmv_naive(int uplo, int n, complex_t alpha, __global complex_t* a, __global complex_t* x, int incx, complex_t beta, __global complex_t* y, int incy)
{
    bool ltriangle = uplo == 1;

    int gid = get_global_id(0);

    complex_t out_y = 0.f;

    if (ltriangle) {
        int ind = gid;
        for (int i=0; i<n; i++) {
            if (i<gid) {
                out_y += cmul(a[ind], x[i*incx]);
                ind += n-i-1;
            } else if (i==gid) {
                out_y += cmul((complex_t)(a[ind].x, 0.f), x[i*incx]);
                ind++;
            } else {
                out_y += cmul(conjg(a[ind]), x[i*incx]);
                ind++;
            }
        }
    } else {
        int ind = (gid+1)*gid/2;
        for (int i=0; i<n; i++) {
            if (i<gid) {
                out_y += cmul(conjg(a[ind]), x[i*incx]);
                ind++;
            } else if (i == gid) {
                out_y += cmul((complex_t)(a[ind].x, 0.f), x[i*incx]);
                ind += i+1;
            } else {
                out_y += cmul(a[ind], x[i*incx]);
                ind += i+1;
            }
        }
    }

    out_y = cmul(alpha, out_y);
    out_y += cmul(beta, y[gid*incy]);
    y[gid*incy] = out_y;
}
