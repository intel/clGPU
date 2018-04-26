#ifdef GPU

/* Copyright (c) 2016-2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

   OpenCL Kernel used for RESNET50 1x1 convolution - implemented as a matrix multiply operation
   Matrix A (MxK) - Activation, Matrix B (KxN)- Weights, Matrix C (MxN)- Output
   M dimension = Batch_Size*H*W
   K dimension = Ci ( Number of input  channels )
   N dimension = Ko ( Number of output channels )

   This kernel uses a tile-size = 14x2. This is the size of output-tile computed by each work-item.
   Kernel uses two different tile-sizes - 14x2 for most work-groups and 7x2 for last work-groups.
   Kernel Configuration given below 
   
   local_workgroup_size = (8,1,1)
   global_grid_size     = (N/2,M/14,1)
   
   Kernel Build options: K4 = K/4
                         K_ = K
                         M_ = M
                         N_ = N
                         TX = 8
                         TY = 1
						 -cl-no-subgroup-ifp

   Kernel Arguments: 
          Activation matrix in OpenCL buffer format
          Weight matrix in OpenCL image format
          Output in OpenCL buffer format 
          Other arguments - refer to kernel declaration ( __kernel void gemm_buf_14x2 )
   
      This kernel can be used  for RESNET50 1x1 convolution layers where H,W = 7,7. 

 */


#include "gemm_common.cl"

#define MULT(C_, A_, i_)                        \
    DOT4i_LO(C_.s0, B00, A_, i_ + 0);           \
    DOT4i_HI(C_.s0, B00, A_, i_ + 1);           \
    DOT4i_LO(C_.s0, B10, A_, i_ + 2);           \
    DOT4i_HI(C_.s0, B10, A_, i_ + 3);           \
    DOT4i_LO(C_.s1, B01, A_, i_ + 0);           \
    DOT4i_HI(C_.s1, B01, A_, i_ + 1);           \
    DOT4i_LO(C_.s1, B11, A_, i_ + 2);           \
    DOT4i_HI(C_.s1, B11, A_, i_ + 3);

#define local_y get_local_id(1)
#define local_z get_local_id(2)

void gemm_14x16(const __global float *A,
              __read_only image2d_t B,
              __global float *C,
              const float alpha,
              const float beta,
              uint K,
              uint N,
              uint M,
              uint group_x,
              uint group_y,
              uint local_x)
{
    const uint TILE_M = 14 * TY;
    const uint TILE_K = 8;
    const uint TILE_N = 16;

    /*
      // Defined as pre-compiler macros:
      const uint K4 = K >> 2;
    */

    //      i.j [14.2]
    float2 C0 = 0.0f;
    float2 C1 = 0.0f;
    float2 C2 = 0.0f;
    float2 C3 = 0.0f;
    float2 C4 = 0.0f;
    float2 C5 = 0.0f;
    float2 C6 = 0.0f;
    float2 C7 = 0.0f;
    float2 C8 = 0.0f;
    float2 C9 = 0.0f;
    float2 C10 = 0.0f;
    float2 C11 = 0.0f;
    float2 C12 = 0.0f;
    float2 C13 = 0.0f;

     /* Each Work-Item reads a float4 activation and float8 weight
       each subgroup will read 8xfloat4 activations and 8x8 weights for a SIMD8 width */

    uint lxd4 = local_x >> 2;
    uint lxm4 = local_x % 4;

    uint i = TILE_M * group_y + local_y * 16 + lxd4;

    __global const float4 *A_load = (__global const float4*)&A[i*K_ + (lxm4<<2)];

    uint j = group_x << 4;

    // YX->KN
    int2 coordB = (int2)(j * sizeof(uint), local_z * 16);

    uint k4 = local_z*4;

    do {

        // 512 MADs

        int2 coordBTemp = coordB;

        float8 B00 = as_float8(intel_sub_group_block_read8(B, coordBTemp));
        coordBTemp.x += 8 * sizeof(uint);

        float8 B01 = as_float8(intel_sub_group_block_read8(B, coordBTemp));
        coordB.y += 8;

        coordBTemp = coordB;
        float8 B10 = as_float8(intel_sub_group_block_read8(B, coordBTemp));
        coordBTemp.x += 8 * sizeof(uint);

        float8 B11 = as_float8(intel_sub_group_block_read8(B, coordBTemp));
        coordB.y += 1*16 - 8;
        float4 A0 = A_load[K4*0 + k4];
        float4 A2 = A_load[K4*2 + k4];
        float4 A4 = A_load[K4*4 + k4];
        float4 A6 = A_load[K4*6 + k4];

        MULT(C0, A0, 0);
        MULT(C1, A0, 4);
        MULT(C2, A2, 0);
        MULT(C3, A2, 4);
        MULT(C4, A4, 0);
        MULT(C5, A4, 4);
        MULT(C6, A6, 0);
        MULT(C7, A6, 4);

        A0 = A_load[K4*8 + k4];
        A2 = A_load[K4*10 + k4];
        A4 = A_load[K4*12 + k4];

        MULT(C8, A0, 0);
        MULT(C9, A0, 4);
        MULT(C10, A2, 0);
        MULT(C11, A2, 4);
        MULT(C12, A4, 0);
        MULT(C13, A4, 4);

        k4 += 4*1;
    } while (k4 < K4);

     /*14x2 output per Work-Item*/

    uint y0 = group_y * TILE_M + (local_y << 4);
    __global uint *C_write = (__global uint*) &C[group_x * TILE_N + y0 * N_];
    intel_sub_group_block_write2(&C_write[0*N_], as_uint2(C0));
    intel_sub_group_block_write2(&C_write[1*N_], as_uint2(C1));
    intel_sub_group_block_write2(&C_write[2*N_], as_uint2(C2));
    intel_sub_group_block_write2(&C_write[3*N_], as_uint2(C3));
    intel_sub_group_block_write2(&C_write[4*N_], as_uint2(C4));
    intel_sub_group_block_write2(&C_write[5*N_], as_uint2(C5));
    intel_sub_group_block_write2(&C_write[6*N_], as_uint2(C6));
    intel_sub_group_block_write2(&C_write[7*N_], as_uint2(C7));
    intel_sub_group_block_write2(&C_write[8*N_], as_uint2(C8));
    intel_sub_group_block_write2(&C_write[9*N_], as_uint2(C9));
    intel_sub_group_block_write2(&C_write[10*N_], as_uint2(C10));
    intel_sub_group_block_write2(&C_write[11*N_], as_uint2(C11));
    intel_sub_group_block_write2(&C_write[12*N_], as_uint2(C12));
    intel_sub_group_block_write2(&C_write[13*N_], as_uint2(C13));
}

void gemm_7x16(const __global float *A,
              __read_only image2d_t B,
              __global float *C,
              const float alpha,
              const float beta,
              uint K,
              uint N,
              uint M,
              uint group_x,
              uint group_y,
              uint local_x)
{
    const uint TILE_M = 14 * TY;
    const uint TILE_K = 8;
    const uint TILE_N = 16;

    /*
      // Defined as pre-compiler macros:
      const uint K4 = K >> 2;
    */

    //      i.j [7.2]
    float2 C0 = 0.0f;
    float2 C1 = 0.0f;
    float2 C2 = 0.0f;
    float2 C3 = 0.0f;
    float2 C4 = 0.0f;
    float2 C5 = 0.0f;
    float2 C6 = 0.0f;

    uint lxd4 = local_x >> 2;
    uint lxm4 = local_x % 4;

    uint i = TILE_M * group_y + local_y * 16 + lxd4;

    __global const float4 *A_load = (__global const float4*)&A[i*K_ + (lxm4<<2)];

    uint j = group_x << 4;

    // YX->KN
    int2 coordB = (int2)(j * sizeof(uint), local_z * 16);

    uint k4 = local_z*4;

    do {

        // 512 MADs

        int2 coordBTemp = coordB;

        float8 B00 = as_float8(intel_sub_group_block_read8(B, coordBTemp));
        coordBTemp.x += 8 * sizeof(uint);

        float8 B01 = as_float8(intel_sub_group_block_read8(B, coordBTemp));
        coordB.y += 8;

        coordBTemp = coordB;
        float8 B10 = as_float8(intel_sub_group_block_read8(B, coordBTemp));
        coordBTemp.x += 8 * sizeof(uint);

        float8 B11 = as_float8(intel_sub_group_block_read8(B, coordBTemp));
        coordB.y += 1*16 - 8;
        float4 A0 = A_load[K4*0 + k4];
        float4 A2 = A_load[K4*2 + k4];
        float4 A4 = A_load[K4*4 + k4];
        float4 A6 = A_load[K4*6 + k4];

        MULT(C0, A0, 0);
        MULT(C1, A0, 4);
        MULT(C2, A2, 0);
        MULT(C3, A2, 4);
        MULT(C4, A4, 0);
        MULT(C5, A4, 4);
        MULT(C6, A6, 0);

        k4 += 4*1;
    } while (k4 < K4);


    uint y0 = group_y * TILE_M + (local_y << 4);
    __global uint *C_write = (__global uint*) &C[group_x * TILE_N + y0 * N_];
    intel_sub_group_block_write2(&C_write[0*N_], as_uint2(C0));
    intel_sub_group_block_write2(&C_write[1*N_], as_uint2(C1));
    intel_sub_group_block_write2(&C_write[2*N_], as_uint2(C2));
    intel_sub_group_block_write2(&C_write[3*N_], as_uint2(C3));
    intel_sub_group_block_write2(&C_write[4*N_], as_uint2(C4));
    intel_sub_group_block_write2(&C_write[5*N_], as_uint2(C5));
    intel_sub_group_block_write2(&C_write[6*N_], as_uint2(C6));
}

__attribute__((reqd_work_group_size(8, TY, 1)))
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void gemm_buf_14x2 (const __global float *A,
                             __read_only image2d_t B,
                             __global float *C,
                             const float alpha,
                             const float beta,
                             uint K,
                             uint N,
                             uint M

                                )
{
    /*
      // Defined as pre-compiler macros:
      const uint K4 = K >> 2;
    */


#define group_x get_group_id(0)
#define group_y get_group_id(1)


#define local_x get_local_id(0)

    if (group_y == 3) {
        gemm_7x16(A, B, C, alpha, beta, K, N, M, group_x, group_y, local_x);
    } else {
        gemm_14x16(A, B, C, alpha, beta, K, N, M, group_x, group_y, local_x);
    }

}

#endif