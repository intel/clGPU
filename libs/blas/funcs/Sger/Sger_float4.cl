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

#define VEC_SIZE 4

#define VLIST2(tab, start, inc) (float2)(tab[(start)], tab[(start) + (inc)])
#define VLIST4(tab, start, inc) (float4)(VLIST2(tab, start, inc), VLIST2(tab, (start) + 2 * (inc), inc))

// Each work item calculates 4x1 tile using float4 where possible
__kernel void Sger_float4(uint m, float alpha, __global float* x, uint incx, __global float* y, uint incy, __global float* a, uint lda)
{
    lda /= VEC_SIZE;
    const uint ind_m = get_global_id(0);
    const uint ind_n = get_global_id(1);

    uint index_a = ind_n * lda + ind_m;

    bool over = (ind_m + 1) * VEC_SIZE > m;

    // Tile exceeds -> use float
    if (over) {
        index_a *= VEC_SIZE;
        float this_y = y[ind_n * incy];
        for (uint row = ind_m * VEC_SIZE; row < m; row++) {
            float this_x = x[row * incx];
            float this_a = a[index_a];
            float prod = this_x * this_y;
            this_a = mad(alpha, prod, this_a);
            a[index_a] = this_a;
            index_a++;
        }
    // Tile does not exceed -> use float4
    } else {
        float4 this_x = VLIST4(x, ind_m * VEC_SIZE * incx, incx);
        float this_y = y[ind_n * incy];

        float4 this_a = vload4(index_a, a);
        float4 prod = (float4)this_y * this_x;
        this_a = mad((float4)alpha, prod, this_a);
        vstore4(this_a, index_a, a);
    }
}
