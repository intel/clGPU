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

   This is a FP16 kernel. It uses a tile-size = 12x1. This is the size of output-tile computed by each work-item.
   Kernel uses two different tile-sizes - 12x1 for most work-groups and 4x1 for last work-groups.

   Kernel Configuration given below 
   
   local_workgroup_size = (16,1,1)
   global_grid_size    = (N/1,M/12,1)
   
   Kernel Build options: K8 = K/8
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
          Other arguments - refer to kernel declaration ( __kernel void hgemm_buf_12x1 )

      Host side must have data in FP16 format for activations,weights,outputs.
      This kernel can be used  for RESNET50 1x1 convolution layers where H,W = 14,14. 

 */



#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#include "gemm_common.cl"

#define MULT(C_, A_, i_)                   \
    DOT8i(C_,  B0, A_, i_ + 0);            \
    DOT8i(C_,  B8, A_, i_ + 1);            \
    DOT8i(C_, B16, A_, i_ + 2);            \
    DOT8i(C_, B24, A_, i_ + 3);

#define local_z get_local_id(2)

void gemm_12x16(const __global half *A,
              __read_only image2d_t B,
              __global half *C,
              const float alpha,
              const float beta,
              uint K,
              uint N,
              uint M,
              uint group_x,
              uint group_y,
              uint local_x)
{
    const uint TILE_M = 12 * TY;
    const uint TILE_K = 32;
    const uint TILE_N = 16;

    /*
      // Defined as pre-compiler macros:
      const uint K8 = K >> 3;
    */

    //      i.j [12.2]
    half8 C0 = 0.0f;
    half8 C8 = 0.0f;

   /* Each Work-Item reads a half8 activation and half8 weight
       each subgroup will read 16xhalf8 activations and 16x8 weights for a SIMD16 width */
	
    uint lxd4 = local_x >> 2;
    uint lxm4 = local_x % 4;

    uint ig = TILE_M * group_y;

    // Ensure i0 does not go out of bounds.
    uint i0 = ig + (lxd4 % 12);

    uint i8 = ig + 8 + (lxd4 % 4);

    __global const half8 *A_load_0 = (__global const half8*)&A[i0*K_ + (lxm4<<3)];

    __global const half8 *A_load_8 = (__global const half8*)&A[i8*K_ + (lxm4<<3)];

    uint j = group_x << 4;

    // YX->KN
    int2 coordB = (int2)(j * sizeof(short), local_z * 32);

    uint k8 = local_z*4;

    do {

        // 512 MADs

        half8 B0 = as_half8(intel_sub_group_block_read_us8(B, coordB));
        coordB.y += 8;
        half8 B8 = as_half8(intel_sub_group_block_read_us8(B, coordB));
        coordB.y += 8;

        half8 B16 = as_half8(intel_sub_group_block_read_us8(B, coordB));
        coordB.y += 8;
        half8 B24 = as_half8(intel_sub_group_block_read_us8(B, coordB));
        coordB.y += (32*1 - 24);
        half8 A0 = A_load_0[K8*0 + k8];
        half8 A4 = A_load_0[K8*4 + k8];

        MULT(C0.s0, A0, 0);
        MULT(C0.s1, A0, 4);
        MULT(C0.s2, A0, 8);
        MULT(C0.s3, A0, 12);
        MULT(C0.s4, A4, 0);
        MULT(C0.s5, A4, 4);
        MULT(C0.s6, A4, 8);
        MULT(C0.s7, A4, 12);

        A0 = A_load_8[K8*0 + k8];

        MULT(C8.s0, A0, 0);
        MULT(C8.s1, A0, 4);
        MULT(C8.s2, A0, 8);
        MULT(C8.s3, A0, 12);

        k8 += 4*1;
    } while (k8 < K8);


    uint y0 = group_y * TILE_M;
    __global half *C_write = &C[group_x * TILE_N + y0 * N_ + local_x];
    C_write[0*N_] = C0.s0;
    C_write[1*N_] = C0.s1;
    C_write[2*N_] = C0.s2;
    C_write[3*N_] = C0.s3;
    C_write[4*N_] = C0.s4;
    C_write[5*N_] = C0.s5;
    C_write[6*N_] = C0.s6;
    C_write[7*N_] = C0.s7;
    C_write[8*N_] = C8.s0;
    C_write[9*N_] = C8.s1;
    C_write[10*N_] = C8.s2;
    C_write[11*N_] = C8.s3;
}

void gemm_4x16(const __global half *A,
              __read_only image2d_t B,
              __global half *C,
              const float alpha,
              const float beta,
              uint K,
              uint N,
              uint M,
              uint group_x,
              uint group_y,
              uint local_x)
{
    const uint TILE_M = 12 * TY;
    const uint TILE_K = 32;
    const uint TILE_N = 16;

    /*
      // Defined as pre-compiler macros:
      const uint K8 = K >> 3;
    */

    //      i.j [4.2]
    half8 C0 = 0.0f;

    uint lxd4 = local_x >> 2;
    uint lxm4 = local_x % 4;

    uint ig = TILE_M * group_y;

    // Ensure i0 does not go out of bounds.
    uint i0 = ig + (lxd4 % 4);

    __global const half8 *A_load_0 = (__global const half8*)&A[i0*K_ + (lxm4<<3)];

    uint j = group_x << 4;

    // YX->KN
    int2 coordB = (int2)(j * sizeof(short), local_z * 32);

    uint k8 = local_z*4;

    do {

        // 512 MADs

        half8 B0 = as_half8(intel_sub_group_block_read_us8(B, coordB));
        coordB.y += 8;
        half8 B8 = as_half8(intel_sub_group_block_read_us8(B, coordB));
        coordB.y += 8;

        half8 B16 = as_half8(intel_sub_group_block_read_us8(B, coordB));
        coordB.y += 8;
        half8 B24 = as_half8(intel_sub_group_block_read_us8(B, coordB));
        coordB.y += (32*1 - 24);
        half8 A0 = A_load_0[K8*0 + k8];

        MULT(C0.s0, A0, 0);
        MULT(C0.s1, A0, 4);
        MULT(C0.s2, A0, 8);
        MULT(C0.s3, A0, 12);

        k8 += 4*1;
    } while (k8 < K8);


    uint y0 = group_y * TILE_M;
    __global half *C_write = &C[group_x * TILE_N + y0 * N_ + local_x];
    C_write[0*N_] = C0.s0;
    C_write[1*N_] = C0.s1;
    C_write[2*N_] = C0.s2;
    C_write[3*N_] = C0.s3;
}

__attribute__((reqd_work_group_size(16, TY, 1)))
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void hgemm_buf_12x1 (const __global half *A,
                             __read_only image2d_t B,
                             __global half *C,
                             const float alpha,
                             const float beta,
                             uint K,
                             uint N,
                             uint M

                                )
{
    


#define group_x get_group_id(0)
#define group_y get_group_id(1)


#define local_x get_local_id(0)

    if (group_y == 16) {
        gemm_4x16(A, B, C, alpha, beta, K, N, M, group_x, group_y, local_x);
    } else {
        gemm_12x16(A, B, C, alpha, beta, K, N, M, group_x, group_y, local_x);
    }

}

#endif
