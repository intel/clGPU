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

typedef struct /* Index and Value type that holds index and value used in this kernel */
{
    uint index; 
    float value; 
} iav_type;

/* Calculates first index with maximum value withing a subgroup, 
 * res_max (in_out) - max value within a subgroup
 * res_index (in_out) - first index with maximum value within a subgroup
 * simd_val (in) - iav_type value from simd thread 
*/
 #define SUB_GROUP_MAX_VALUE_MIN_INDEX(res_max, res_index, simd_val)\
{\
    res_max = sub_group_reduce_max ( simd_val.value ); \
    res_index = sub_group_reduce_min ( simd_val.value == res_max ? simd_val.index : UINT_MAX  ); \
}

#define SIMD_WIDTH 16
#define LOCAL_GROUP_SIZE 256
#define LOCAL_GROUP_NUMBER 256

#define WORKING_THREADS LOCAL_GROUP_SIZE * LOCAL_GROUP_NUMBER

__attribute__((intel_reqd_sub_group_size(SIMD_WIDTH)))
__attribute__((reqd_work_group_size(LOCAL_GROUP_SIZE, 1, 1)))
__kernel void Icamax_opt_simd16_2stage(int n, __global complex_t* x, int incx, __global float2* tempResult)
{
    uint subgr_id = get_sub_group_id();
    uint subgr_local_id = get_sub_group_local_id(); 
    uint global_id = get_global_id(0);
    uint locgr_id = get_group_id(0);

    //Shared local memory with size of subgroups (LOCAL_GROUP_SIZE / SIMD_WIDTH)
    __local iav_type SLM[16];

    //Thread result
    iav_type res;
    res.index = global_id;
    res.value = scabs1(x[global_id * incx]);

    global_id += WORKING_THREADS;

    //Calculating result for this thread.
    while (global_id < n) 
    {
      iav_type element;
      element.value = scabs1(x[global_id * incx]);
      element.index = global_id;

      if(element.value > res.value)
       {
            res.value = element.value;
            res.index = element.index;
       }
        global_id += WORKING_THREADS;
    }
    
    //Get max value withing subgroup
    float max_val;
    uint min_index;
    SUB_GROUP_MAX_VALUE_MIN_INDEX(max_val, min_index, res);

    //Writing subgroup result to SLM
    if(subgr_local_id == 0)
    {
        SLM[subgr_id].value = max_val;
        SLM[subgr_id].index = min_index;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //One Subgroup taking care of SLM and writing local group result
    if(subgr_id == 0)
    {
        res.index = SLM[subgr_local_id].index;
        res.value = SLM[subgr_local_id].value;

        SUB_GROUP_MAX_VALUE_MIN_INDEX(max_val, min_index, res);

        if(subgr_local_id == 0)
        {
            tempResult[locgr_id].x = min_index;
            tempResult[locgr_id].y = max_val;
        }
    }
}

//Second stage. Takes 256 elements results from earch local group and calculating final result
__attribute__((intel_reqd_sub_group_size(SIMD_WIDTH)))
__attribute__((reqd_work_group_size(LOCAL_GROUP_SIZE, 1, 1)))
__kernel void Icamax_opt_simd16_2stage_2(uint n, __global float2* tempRes, __global int* result)
{
    uint subgr_id = get_sub_group_id();
    uint subgr_local_id = get_sub_group_local_id();
    uint global_id = get_global_id(0);

    //Shared local memory with size of subgroups
    __local iav_type SLM[16];

    //Thread result
    iav_type res;
    res.index = tempRes[global_id].x;
    res.value = tempRes[global_id].y;
    
    //Get max value withing subgroup
    float max_val;
    uint min_index;
    SUB_GROUP_MAX_VALUE_MIN_INDEX(max_val, min_index, res);

    //Writing subgroup result to SLM
    if(subgr_local_id == 0)
    {
        SLM[subgr_id].value = max_val;
        SLM[subgr_id].index = min_index;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    //One Subgroup taking care of SLM and writing final result
    if(subgr_id == 0)
    {
        res.index = SLM[subgr_local_id].index;
        res.value = SLM[subgr_local_id].value;

        SUB_GROUP_MAX_VALUE_MIN_INDEX(max_val, min_index, res);

        if(subgr_local_id == 0)
            result[0] = min_index;
    }
}
