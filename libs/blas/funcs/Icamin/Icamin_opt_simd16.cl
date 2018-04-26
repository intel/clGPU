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

typedef struct /* Index and Value type that holds index and value used in this kernel */
{
    uint index; 
    float value; 
} iav_type;

/* Calculates first index with minimum value withing a subgroup, 
 * res_min (in_out) - min value within a subgroup
 * res_index (in_out) - first index with maximum value within a subgroup
 * simd_val (in) - iav_type value from simd thread 
*/
 #define SUB_GROUP_MIN_VALUE_MIN_INDEX(res_min, res_index, simd_val)\
{\
    res_min = sub_group_reduce_min ( simd_val.value ); \
    res_index = sub_group_reduce_min ( simd_val.value == res_min ? simd_val.index : UINT_MAX  ); \
}

#define SIMD_WIDTH 16
#define SOUBGROUP_NUMER 16

#define LOCAL_SIZE 256

/* There are 16 subgroups, every subgroup with 16 threads. Every subgroup calculates it's own max value with smallest index
 * and writing it to local memory. Then one subgroup taking whole local memory of size 16 and doing the final calucations
 */
__attribute__((intel_reqd_sub_group_size(SIMD_WIDTH)))
__attribute__((reqd_work_group_size(LOCAL_SIZE, 1, 1)))
__kernel void Icamin_opt_simd16(int n, __global complex_t* x, int incx, __global int* result)
{
    uint subgr_id = get_sub_group_id();
    uint subgr_local_id = get_sub_group_local_id();
    uint global_id = get_global_id(0);

    //Shared local memory with size of subgroups
    __local iav_type SLM[SOUBGROUP_NUMER];

    //Thread result
    iav_type res;
    res.index = global_id;
    res.value = scabs1(x[global_id * incx]);

    global_id += LOCAL_SIZE;

    //Calculating result for this thread.
    while (global_id < n) 
    {
      iav_type element;
      element.value = scabs1(x[global_id * incx]);
      element.index = global_id;

      if(element.value < res.value)
       {
            res.value = element.value;
            res.index = element.index;
       }
        global_id += LOCAL_SIZE;
    }
    
    //Get min value withing subgroup
    float min_val;
    uint min_index;
    SUB_GROUP_MIN_VALUE_MIN_INDEX(min_val, min_index, res);

    //Writing subgroup result to SLM
    if(subgr_local_id == 0)
    {
        SLM[subgr_id].value = min_val;
        SLM[subgr_id].index = min_index;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //One Subgroup taking care of SLM and writing final result
    if(subgr_id == 0)
    {
        res.index = SLM[subgr_local_id].index;
        res.value = SLM[subgr_local_id].value;

        SUB_GROUP_MIN_VALUE_MIN_INDEX(min_val, min_index, res);

        if(subgr_local_id == 0)
            result[0] = min_index;
    }
}
