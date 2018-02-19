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

#define IDX(m, n, ld) ((n)*(ld) + (m))
#define IDX_C(m, n, ld) ((m) * (ld) + (n))

#define M_THREADS (16)
#define N_THREADS (1)
#define K_THREADS (1)

#define M_PER_THREAD (1)
#define N_PER_THREAD (1)
#define K_PER_THREAD (4)

#define M_TILE ((M_THREADS)*(M_PER_THREAD))
#define N_TILE ((M_THREADS)*(N_PER_THREAD))
#define K_TILE ((K_THREADS)*(K_PER_THREAD))

__attribute__((intel_reqd_sub_group_size(M_THREADS)))
__attribute__((reqd_sub_group_size(M_THREADS)))
__attribute__((reqd_work_group_size(M_THREADS, N_THREADS, K_THREADS)))
__kernel void Ssymm_simd1_16x16_left_up(uint m, uint n, float alpha, __global float* a, uint lda, __global float* b, uint ldb, float beta, __global float* c, uint ldc)
{
    // We will load to arow tile 1x4 from matrix A. From matrix B we will load tile 4x1 into bcol.
    // Then using intel_sub_group_shuffle each work_item will have access to full atile of size 16x4.
    // Using this shared atile and private bcol it will calculate matrix product of size 16x1 and save it to ccol.
    // Each case for atile below, on and above diagonal is split as it speeds up the computations and allows
    // for more flexible aproach to each case.
    ldb /= 4;
    ldc /= 4;
    float arow[M_PER_THREAD * K_PER_THREAD];
    float bcol[N_PER_THREAD * K_PER_THREAD];
    float ccol[M_TILE * N_PER_THREAD];
    
    __attribute__((opencl_unroll_hint(M_TILE * N_PER_THREAD)))
    for (int i=0; i<M_TILE * N_PER_THREAD; i++) {
        ccol[i] = 0.f;
    }

    const uint local_id = get_sub_group_local_id();
    const uint group_start_m = get_group_id(0)*M_TILE;
    const uint group_start_n = get_group_id(1)*N_TILE;
    const uint ind_m = group_start_m + local_id*M_PER_THREAD;
    const uint ind_n = group_start_n + local_id*N_PER_THREAD;

    uint current_k = 0;
    // Case for atile below diagonal
    for (; current_k + K_TILE <= group_start_m; current_k += K_TILE) {
        // Load arow
        __attribute__((opencl_unroll_hint(M_PER_THREAD)))
        for (int i=0; i<M_PER_THREAD; i++) {
            float4 this_a = vload4(IDX(current_k/4, ind_m + i, lda/4), a);
            arow[IDX_C(i, 0, K_PER_THREAD)] = this_a.x;
            arow[IDX_C(i, 1, K_PER_THREAD)] = this_a.y;
            arow[IDX_C(i, 2, K_PER_THREAD)] = this_a.z;
            arow[IDX_C(i, 3, K_PER_THREAD)] = this_a.w;
        }
        // Load bcol
        __attribute__((opencl_unroll_hint(N_PER_THREAD)))
        for (int i=0; i<N_PER_THREAD; i++) {
            float4 this_b = vload4(IDX(current_k/4, ind_n + i, ldb), b);
            bcol[IDX_C(i, 0, K_PER_THREAD)] = this_b.x;
            bcol[IDX_C(i, 1, K_PER_THREAD)] = this_b.y;
            bcol[IDX_C(i, 2, K_PER_THREAD)] = this_b.z;
            bcol[IDX_C(i, 3, K_PER_THREAD)] = this_b.w;
        }
        // Multiply
        __attribute__((opencl_unroll_hint(M_THREADS)))
        for (int i=0; i<M_THREADS; i++) {
            __attribute__((opencl_unroll_hint(K_PER_THREAD)))
            for (int j=0; j<K_PER_THREAD; j++) {
                float this_a = intel_sub_group_shuffle(arow[IDX_C(0, j, K_PER_THREAD)], i);
                ccol[IDX_C(i, 0, N_PER_THREAD)] = mad(this_a, bcol[IDX_C(0, j, K_PER_THREAD)], ccol[IDX_C(i, 0, N_PER_THREAD)]);
            }
        }
    }
    // Case for atile on diagonal
    for (int i=0; i<M_TILE/K_TILE; i++, current_k += K_TILE) {
        // Load arow
        __attribute__((opencl_unroll_hint(M_PER_THREAD)))
        for (int j=0; j<M_PER_THREAD; j++) {
            __attribute__((opencl_unroll_hint(K_PER_THREAD)))
            for (int k = 0; k<K_PER_THREAD; k++) {
                int index;
                if (current_k + k >= ind_m + j ) {
                    index = IDX(ind_m + j, current_k + k, lda);
                } else {
                    index = IDX(current_k + k, ind_m + j, lda);
                }
                arow[IDX_C(j, k, K_PER_THREAD)] = a[index];
            }
        }
        // Load bcol
        __attribute__((opencl_unroll_hint(N_PER_THREAD)))
        for (int i=0; i<N_PER_THREAD; i++) {
            float4 this_b = vload4(IDX(current_k/4, ind_n + i, ldb), b);
            bcol[IDX_C(i, 0, K_PER_THREAD)] = this_b.x;
            bcol[IDX_C(i, 1, K_PER_THREAD)] = this_b.y;
            bcol[IDX_C(i, 2, K_PER_THREAD)] = this_b.z;
            bcol[IDX_C(i, 3, K_PER_THREAD)] = this_b.w;
        }
        // Multiply
        __attribute__((opencl_unroll_hint(M_THREADS)))
        for (int i=0; i<M_THREADS; i++) {
            __attribute__((opencl_unroll_hint(K_PER_THREAD)))
            for (int j=0; j<K_PER_THREAD; j++) {
                float this_a = intel_sub_group_shuffle(arow[IDX_C(0, j, K_PER_THREAD)], i);
                ccol[IDX_C(i, 0, N_PER_THREAD)] = mad(this_a, bcol[IDX_C(0, j, K_PER_THREAD)], ccol[IDX_C(i, 0, N_PER_THREAD)]);
            }
        }
    }
    // Case for atile above diagonal
    // NOTE: When reorganized to load float4 and abbandon previous sub_group tiling convension achives speed up of ~1%
    for (; current_k + K_TILE <= m; current_k += K_TILE) {
        // Load arow
        __attribute__((opencl_unroll_hint(K_PER_THREAD)))
        for (int i=0; i<K_PER_THREAD; i++) {
            float this_a = a[IDX(ind_m, current_k + i, lda)];
            arow[IDX_C(0, i, K_PER_THREAD)] = this_a;
        }
        // Load bcol
        __attribute__((opencl_unroll_hint(N_PER_THREAD)))
        for (int i=0; i<N_PER_THREAD; i++) {
            float4 this_b = vload4(IDX(current_k/4, ind_n + i, ldb), b);
            bcol[IDX_C(i, 0, K_PER_THREAD)] = this_b.x;
            bcol[IDX_C(i, 1, K_PER_THREAD)] = this_b.y;
            bcol[IDX_C(i, 2, K_PER_THREAD)] = this_b.z;
            bcol[IDX_C(i, 3, K_PER_THREAD)] = this_b.w;
        }
        // Multiply
        __attribute__((opencl_unroll_hint(M_THREADS)))
        for (int i=0; i<M_THREADS; i++) {
            __attribute__((opencl_unroll_hint(K_PER_THREAD)))
            for (int j=0; j<K_PER_THREAD; j++) {
                float this_a = intel_sub_group_shuffle(arow[IDX_C(0, j, K_PER_THREAD)], i);
                ccol[IDX_C(i, 0, N_PER_THREAD)] = mad(this_a, bcol[IDX_C(0, j, K_PER_THREAD)], ccol[IDX_C(i, 0, N_PER_THREAD)]);
            }
        }
    }
    // Handle leftovers
    for (; current_k < m; current_k += 1) {
        // Load only one column to arow
        __attribute__((opencl_unroll_hint(M_PER_THREAD)))
        for (int i=0; i<M_PER_THREAD; i++) {
            int index;
            if (current_k >= ind_m + i ) {
                index = IDX(ind_m + i, current_k, lda);
            } else {
                index = IDX(current_k, ind_m + i, lda);
            }
            arow[IDX_C(i, 0, K_PER_THREAD)] = a[index];
        }
        // Load only one row to bcol
        __attribute__((opencl_unroll_hint(N_PER_THREAD)))
        for (int i=0; i<N_PER_THREAD; i++) {
            bcol[IDX_C(i, 0, K_PER_THREAD)] = b[IDX(current_k, ind_n + i, ldb*4)];
        }
        // Multiply
        __attribute__((opencl_unroll_hint(M_THREADS)))
        for (int i=0; i<M_THREADS; i++) {
            float this_a = intel_sub_group_shuffle(arow[IDX_C(0, 0, K_PER_THREAD)], i);
            ccol[IDX_C(i, 0, N_PER_THREAD)] = mad(this_a, bcol[IDX_C(0, 0, K_PER_THREAD)], ccol[IDX_C(i, 0, N_PER_THREAD)]);
        }
    }
    // Save calculated result
    __attribute__((opencl_unroll_hint(N_TILE/4)))
    for (int i=0; i<N_TILE/4; i++) {
        float4 calculated_c = (float4)(ccol[IDX_C(i*4, 0, N_PER_THREAD)], ccol[IDX_C(i*4 + 1, 0, N_PER_THREAD)],
                                       ccol[IDX_C(i*4 + 2, 0, N_PER_THREAD)], ccol[IDX_C(i*4 + 3, 0, N_PER_THREAD)]);
        float4 this_c = vload4(IDX(group_start_m/4 + i, ind_n, ldc), c);
        this_c *= (float4)beta;
        this_c = mad(calculated_c, (float4)alpha, this_c);
        vstore4(this_c, IDX(group_start_m/4 + i, ind_n, ldc), c);
    }
}
