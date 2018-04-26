// Copyright (c) 2017-2018 Intel Corporation
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//      http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#define MAT_ACCESS(A, col, row, n) A[col * n + row]

#define SIMD_WIDTH 16

/* We're lunching 1 thread for every row. Then in one row, we use 16 threads, where one thread
/* calculate element n, n+SIMD_WIDTH and so on. Then they're added using sub_group_reduce_add.
*/

__attribute__((intel_reqd_sub_group_size(SIMD_WIDTH)))
__attribute__((reqd_work_group_size(1, SIMD_WIDTH, 1)))
__kernel void Sgemv_opt_simd16(uint trans, uint m, uint n, float alpha, __global float* A, uint lda, __global float* x, int incx, float beta, __global float* y, int incy)
{
    uint row_id = get_global_id(0);
    uint col_id = get_global_id(1);
   
    float thread_res = 0;
    float subgr_acc;

    for (uint col_loop_id = col_id; col_loop_id < n; col_loop_id += SIMD_WIDTH)
        thread_res = fma(MAT_ACCESS(A, col_loop_id, row_id, lda), x[col_loop_id * incx], thread_res);

    subgr_acc = sub_group_reduce_add(thread_res);

    if(col_id == 0)
    {   
        if(beta != 0)
            y[row_id * incy] = fma(beta, y[row_id * incy], alpha * subgr_acc);
        else
            y[row_id * incy] = alpha * subgr_acc;
    }
}
