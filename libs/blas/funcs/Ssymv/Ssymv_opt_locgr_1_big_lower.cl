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

#define GLOBAL_SIZE 256
#define LOCAL_SIZE GLOBAL_SIZE

__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
__kernel void Ssymv_opt_locgr_1_big_lower(uint n, float alpha, __global float* A, uint lda, __global float* x, int incx, float beta, __global float* y, int incy)
{
     uint local_id = get_local_id(0);

     /* Local Group [Only One Group] */ 
    __local float acc [LOCAL_SIZE];
    float loop_acc;


    //Do calculations for every row
    __attribute__((opencl_unroll_hint))
    for(uint row_id = 0; row_id<n; ++row_id)
    {
        loop_acc = 0.f;

        //Now in one row, every thread calculates one or more items. Threads in local group will be added using reduction
        __attribute__((opencl_unroll_hint))
        for (uint l_global_id = local_id; l_global_id<n; l_global_id += GLOBAL_SIZE)
        {
            
           if(row_id >= l_global_id) /* Upper Test */
               loop_acc = fma(MAT_ACCESS(A, l_global_id, row_id, lda), x[l_global_id * incx], loop_acc);
           else
               loop_acc = fma(MAT_ACCESS(A, row_id, l_global_id, lda), x[l_global_id * incx], loop_acc);
           
        }

        acc[local_id] = loop_acc;

        /* Wait for all threads in local group */
        barrier(CLK_LOCAL_MEM_FENCE);

        /* Reduction of local group */ 
        __attribute__((opencl_unroll_hint))
        for(uint offset = LOCAL_SIZE / 2; offset > 0; offset = offset / 2) 
        {
            if (local_id < offset) 
            {
            float val_1 = acc[local_id + offset];
            float val_2 = acc[local_id];

            acc[local_id] = val_1 + val_2;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

       /* Wait for all threads in local group */
       barrier(CLK_LOCAL_MEM_FENCE);
   
      /* Updating vector y with new calculated value  */
      if (local_id == 0) 
      {
          if(beta != 0)
              y[row_id * incy] = fma(alpha, acc[0], beta * y[row_id * incy]);
          else
              y[row_id * incy] = alpha * acc[0];
      }
      
      /* Ending Calculation for one row, jumping to next row if exists */
    }
}
