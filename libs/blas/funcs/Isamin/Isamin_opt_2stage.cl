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

#define GLOBAL_SIZE 128
#define LOCAL_SIZE GLOBAL_SIZE

typedef struct /* Index and Value type that holds index and value used in this kernel */
{
    uint index; 
    float value; 
} iav_type;

__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
__kernel void Isamin_opt_2stage(int lengthh, __global float* buffer, __global int* result)
{
  __local iav_type scratch[LOCAL_SIZE];

 uint local_index = get_local_id(0);
 uint global_index = get_global_id(0);

 iav_type accumulator;
 accumulator.index = global_index;
 accumulator.value = fabs(buffer[global_index]);

 __attribute__((opencl_unroll_hint))
  while (global_index < lengthh) 
  {
      iav_type element;
      element.value = fabs(buffer[global_index]);
      element.index = global_index;

      if(accumulator.value > element.value)
       {
            accumulator.value = element.value;
            accumulator.index = element.index;
       }
        global_index += GLOBAL_SIZE;
  }

 
  scratch[local_index] = accumulator;

  barrier(CLK_LOCAL_MEM_FENCE);

  __attribute__((opencl_unroll_hint))
  for(uint offset = LOCAL_SIZE / 2; offset > 0; offset /= 2) 
  {
    if (local_index < offset) 
    {
        iav_type other = scratch[local_index + offset];
        iav_type mine = scratch[local_index];

        if(mine.value > other.value | ((mine.value == other.value) & (other.index < mine.index)) )
        {
          scratch[local_index] = other;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (local_index == 0) 
  {
    result[0]  = scratch[0].index;
  }
  
}
