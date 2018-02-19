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

/* Access for Lower symmetric matrix (packed) coordinates from normal matrix coordinates */
#define LS_ACCESS(A, i, j, N) A[(i+((2*N-j+1)*j)/2) - (1*j)]

__kernel void Sspr2_naive_lower(int n, float alpha, __global float* x, int incx, __global float* y, int incy, __global float* AP)
{
    int row_id = get_global_id(0);
    int col_id = get_global_id(1);

    float res_1;
    float res_2;

    if(row_id >= col_id)
    {
        res_1 = alpha * x[row_id * incx] * y[col_id * incy];
        res_2 = alpha * y[row_id * incy] * x[col_id * incx];
        LS_ACCESS(AP, row_id, col_id, n) = res_1 + res_2 + LS_ACCESS(AP, row_id, col_id, n);
    }
}
