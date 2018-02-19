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

__kernel void Crot_naive(int n, __global complex_t* x, int incx, __global complex_t* y, int incy, float c, complex_t s)
{
    for (uint i = 0; i < n; i++)
    { 
        complex_t _x = cfmaf(x[i * incx], c, cmul(s, y[i * incy]));
        y[i * incy] = cfmaf(y[i * incy], c, cmul(cneg(s), x[i * incx]));
        x[i * incx] = _x;
    }
}
