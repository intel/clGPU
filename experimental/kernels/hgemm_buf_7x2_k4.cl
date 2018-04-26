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

   OpenCL Kernel  used for RESNET50 1x1 convolution - implemented as a matrix multiply operation
   Matrix A (MxK) - Activation, Matrix B (KxN)- Weights, Matrix C (MxN)- Output
   M dimension = Batch_Size*H*W
   K dimension = Ci ( Number of input channels )
   N dimension = Ko ( Number of output channels )

   This is a FP16 kernel. It uses a tile-size = 7x2. This is the size of output-tile computed by each work-item.
   Kernel uses "K-Slicing" technique to split work along K-Dimension
   Kernel Configuration given below 
   
   local_workgroup_size = (16,1,4)
   global_grid_size    = (N/2,M/7,4)
   
   Kernel Build options: K4 = K/4
                         K_ = K
                         M_ = M
                         N_ = N
                         TX = 16
                         TY = 1
						 -cl-no-subgroup-ifp
   Kernel Arguments: 
          Activation matrix in OpenCL buffer format
          Weight matrix in OpenCL image format
          Output in OpenCL buffer format 
          Other arguments - refer to kernel declaration ( __kernel void hgemm_buf_7x2_k4 )

      Host side must have data in FP16 format for activations,weights,outputs.
      This kernel can be used  for RESNET50 1x1 convolution layers where H,W = 7,7. 

 */



#pragma OPENCL EXTENSION cl_khr_fp16 : enable

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


#define local_z get_local_id(2)

void gemm_7x32(const __global half *A,
              __read_only image2d_t B,
              __global half *C,
              const float alpha,
              const float beta,
              uint K,
              uint N,
              uint M,
              uint group_x,
              uint group_y,
              uint local_x,
              __local half4 *R)
{
    const uint TILE_M = 7 * TY;
    const uint TILE_K = 16;
    const uint TILE_N = 32;

    /*
      // Defined as pre-compiler macros:
      const uint K8 = K >> 3;
    */

    //      i.j [7.2]
    half2 C0 = 0.0f;
    half2 C1 = 0.0f;
    half2 C2 = 0.0f;
    half2 C3 = 0.0f;
    half2 C4 = 0.0f;
    half2 C5 = 0.0f;
    half2 C6 = 0.0f;

 /* Each Work-Item reads a half4 activation and half8 weight
       each subgroup will read 16xhalf4 activations and 16x8 weights for a SIMD16 width */

    uint lxd4 = local_x >> 2;
    uint lxm4 = local_x % 4;

    uint ig = TILE_M * group_y;

    // Ensure i0 does not go out of bounds.
    uint i0 = ig + (lxd4 % 7);

    __global const half4 *A_load_0 = (__global const half4*)&A[i0*K_ + (lxm4<<2)];

    uint j = group_x << 5;

    // YX->KN
    int2 coordB = (int2)(j * sizeof(short), local_z * 16);

    uint k4 = local_z*4;

    do {

        // 512 MADs
        int2 coordBTemp = coordB;

        half8 B00 = as_half8(intel_sub_group_block_read_us8(B, coordBTemp));
        coordBTemp.y += 8;
        half8 B10 = as_half8(intel_sub_group_block_read_us8(B, coordBTemp));

        coordBTemp = coordB;
        coordBTemp.x += 16 * sizeof(ushort);

        half8 B01 = as_half8(intel_sub_group_block_read_us8(B, coordBTemp));
        coordBTemp.y += 8;
        half8 B11 = as_half8(intel_sub_group_block_read_us8(B, coordBTemp));
        coordB.y += 4*16;
        half4 A0 = A_load_0[K4*0 + k4];
        half4 A4 = A_load_0[K4*4 + k4];

        MULT(C0, A0, 0);
        MULT(C1, A0, 4);
        MULT(C2, A0, 8);
        MULT(C3, A0, 12);
        MULT(C4, A4, 0);
        MULT(C5, A4, 4);
        MULT(C6, A4, 8);

        k4 += 4*4;
    } while (k4 < K4);


    __local half4 *R_write = &R[local_z*128 + local_x];

    R_write[0*16] = (half4)(C0.s0, C0.s1, C1.s0, C1.s1);
    R_write[1*16] = (half4)(C2.s0, C2.s1, C3.s0, C3.s1);
    R_write[2*16] = (half4)(C4.s0, C4.s1, C5.s0, C5.s1);
    R_write[3*16] = (half4)(C6.s0, C6.s1, 0.0f, 0.0f);

    __local const half4 *R_read_0 = &R[local_z*16 + local_x];
    __local const half4 *R_read_1 = &R_read_0[16*4];

    uint y0 = group_y * TILE_M + local_z*2;
    __global ushort *C_write = (__global ushort*) &C[group_x * TILE_N + y0 * N_];

    barrier(CLK_LOCAL_MEM_FENCE);

    /* Reduce partial sums of different hardware threads*/

// thread writes: [2, 2, 2, 1]
    if (local_z <= 2) {

        half4 R0
            = R_read_0[0*128]
            + R_read_0[1*128]
            + R_read_0[2*128]
            + R_read_0[3*128];

        intel_sub_group_block_write_us2(&C_write[0*N_], as_ushort2(R0.s01));
        intel_sub_group_block_write_us2(&C_write[1*N_], as_ushort2(R0.s23));
    }
// last 2 write: 2

       half4 R0
            = R_read_0[0*128]
            + R_read_0[1*128]
            + R_read_0[2*128]
            + R_read_0[3*128];

        intel_sub_group_block_write_us2(&C_write[0*N_], as_ushort2(R0.s01));
}

__attribute__((reqd_work_group_size(16, TY, 4)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hgemm_buf_7x2_k4 (const __global half *A,
                             __read_only image2d_t B,
                             __global half *C,
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

    __local half4 R[8*16*4];

    gemm_7x32(A, B, C, alpha, beta, K, N, M, group_x, group_y, local_x, R);

}

#endif