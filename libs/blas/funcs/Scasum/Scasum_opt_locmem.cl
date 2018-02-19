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

/* More work per work item, use of local memory and reduction */

#define GLOBAL_SIZE 256
#define LOCAL_SIZE GLOBAL_SIZE

__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
__kernel void Scasum_opt_locmem(int n, __global complex_t* x, __global float* result)
{
    __local float reduction_buf[LOCAL_SIZE];

     uint local_index = get_local_id(0);

     float acc = 0;

     //I still need this value, because it is used for this loop
     uint gid = local_index;
     __attribute__((opencl_unroll_hint))
     while (gid < n) 
     {
         acc += scabs1(x[gid]);

         gid += GLOBAL_SIZE;
     }
 
      reduction_buf[local_index] = acc;

      barrier(CLK_LOCAL_MEM_FENCE);

      __attribute__((opencl_unroll_hint))
      for(uint offset = LOCAL_SIZE / 2; offset > 0; offset /= 2) 
      {
        if (local_index < offset) 
        {
            float other = reduction_buf[local_index + offset];
            float mine = reduction_buf[local_index];

            reduction_buf[local_index] = other + mine;

        }
        barrier(CLK_LOCAL_MEM_FENCE);
      }

      if (local_index == 0) 
      {
        result[0]  = reduction_buf[0];
      }  
}