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

/* Access for Upper symmetric matrix (packed) coordinates from normal matrix coordinates */
#define US_ACCESS(A, i, j, N) A[i+(j*(j+1))/2]

__kernel void Sspr_naive_upper(int n, float alpha, __global float* x, int incx, __global float* AP)
{
    int row_id = get_global_id(0);
    int col_id = get_global_id(1);

    float res_1; 

    if(col_id >= row_id)
    {
        res_1 = alpha * x[row_id * incx] * x[col_id * incx];
        US_ACCESS(AP, row_id, col_id, n) = res_1 + US_ACCESS(AP, row_id, col_id, n);
    }
}
