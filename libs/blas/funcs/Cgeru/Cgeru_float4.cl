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

__kernel void Cgeru_float4(uint m, complex_t alpha, __global complex_t* x, uint incx, __global complex_t* y, uint incy, __global float* a, uint lda)
{
    const uint ind_x = get_global_id(0) * 2;
    const uint ind_y = get_global_id(1);

    bool over = ind_x + 2 > m;

    if (over) {
        float2 this_x = x[ind_x * incx];
        float2 this_y = y[ind_y * incy];
        uint index_a = ind_y * lda + ind_x;

        float2 this_a = vload2(index_a, a);
        float2 prod = cmul(this_x, this_y);
        this_a += cmul(alpha, prod);
        vstore2(this_a, index_a, a);
    } else {
        float4 this_x = (float4)(x[ind_x * incx], x[ind_x * incx + incx]);
        float2 this_y = y[ind_y * incy];
        uint index_a = ind_y * lda + ind_x;
        index_a /= 2;

        float4 this_a = vload4(index_a, a);
        float4 prod = (float4)(cmul(this_x.xy, this_y), cmul(this_x.zw, this_y));
        this_a += (float4)(cmul(alpha, prod.xy), cmul(alpha, prod.zw));
        vstore4(this_a, index_a, a);
    }
}
