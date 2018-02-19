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

__kernel void Chpr2_naive(int uplo, int n, complex_t alpha, __global complex_t* x, int incx, __global complex_t* y, int incy, __global complex_t* a)
{
    bool ltriangle = uplo == 1;

    if (ltriangle) {
        for (int i=0, k=0; i<n; i++) {
            a[k] += cmul(alpha, cmul(x[i*incx], conjg(y[i*incy])));
            a[k] += cmul(conjg(alpha), cmul(y[i*incy], conjg(x[i*incx])));
            a[k].y = 0.f;
            k++;
            for (int j=i+1; j<n; j++, k++) {
                a[k] += cmul(alpha, cmul(x[j*incx], conjg(y[i*incy])));
                a[k] += cmul(conjg(alpha), cmul(y[j*incy], conjg(x[i*incx])));
            }
        }
    } else {
        for (int i=0, k=0; i<n; i++) {
            for (int j=0; j<i; j++, k++) {
                a[k] += cmul(alpha, cmul(x[j*incx], conjg(y[i*incy])));
                a[k] += cmul(conjg(alpha), cmul(y[j*incy], conjg(x[i*incx])));
            }
            a[k] += cmul(alpha, cmul(x[i*incx], conjg(y[i*incy])));
            a[k] += cmul(conjg(alpha), cmul(y[i*incy], conjg(x[i*incx])));
            a[k].y = 0.f;
            k++;
        }
    }
}
