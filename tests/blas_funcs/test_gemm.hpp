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

#pragma once
#include <gtest/gtest.h>
#include "test_helpers.hpp"


template<iclblasOperation_t transa = ICLBLAS_OP_N>
int get_linear_index(int r, int c, int ld)
{
    return r * ld + c;
}

template<>
inline int get_linear_index<ICLBLAS_OP_N>(int r, int c, int ld)
{
    return c * ld + r;
}

namespace detail {
template <iclblasOperation_t trans>
struct cpu_conj_helper
{
    /// Returns conjugate value if necessary (depending on operation selected).
    template <typename data_type>
    static data_type apply(data_type val)
    {
        return val;
    }
};

template <>
struct cpu_conj_helper<ICLBLAS_OP_C>
{
    template <typename data_type>
    static data_type apply(data_type val)
    {
        using iclgpu::tests::blas_conj; // Allow ADL candidates.

        return blas_conj(val);
    }
};

template <bool beta_zero>
struct cpu_plane_helper
{
    // Calculates plane operation in precision of accumulator and writes it to result:
    //
    // result = acc * alpha + result * beta
    //
    // or (when beta_zero is true), it writes scaled accumulator as result (do not read result):
    //
    // result = acc * alpha
    template <typename data_type, typename acc_type>
    static void plane_update_or_set(data_type& result, acc_type acc, data_type alpha, data_type beta)
    {
        result = static_cast<data_type>(acc * static_cast<acc_type>(alpha) + static_cast<acc_type>(result) * static_cast<acc_type>(beta));
    }
};

template <>
struct cpu_plane_helper<true>
{
    template <typename data_type, typename acc_type>
    static void plane_update_or_set(data_type& result, acc_type acc, data_type alpha, data_type)
    {
        result = static_cast<data_type>(acc * static_cast<acc_type>(alpha));
    }
};

template <
    typename data_type,
    typename acc_type,
    iclblasOperation_t transa,
    iclblasOperation_t transb,
    bool beta_zero
>
void cpu_gemm_spec(
    const int m,
    const int n,
    const int k,
    const data_type alpha,
    const data_type* A,
    const int lda,
    const data_type* B,
    const int ldb,
    const data_type beta,
    data_type* C,
    const int ldc)
{
    constexpr acc_type acc_val_zero = static_cast<acc_type>(0);

    for (int mi = 0; mi < m; ++mi)
    {
        for (int ni = 0; ni < n; ++ni)
        {
            acc_type acc = acc_val_zero;
            for (int ki = 0; ki < k; ++ki)
            {
                acc += static_cast<acc_type>(cpu_conj_helper<transa>::apply(A[get_linear_index<transa>(mi, ki, lda)])) *
                       static_cast<acc_type>(cpu_conj_helper<transb>::apply(B[get_linear_index<transb>(ki, ni, ldb)]));
            }

            cpu_plane_helper<beta_zero>::plane_update_or_set(C[get_linear_index(mi, ni, ldc)], acc, alpha, beta);
        }
    }
}

} // detail

template<typename data_type, typename acc_type = iclgpu::tests::accumulator_type_t<data_type>>
void cpu_gemm(
    const iclblasOperation_t transa,
    const iclblasOperation_t transb,
    const int m,
    const int n,
    const int k,
    const data_type alpha,
    const data_type* A,
    const int lda,
    const data_type* B,
    const int ldb,
    const data_type beta,
    data_type* C,
    const int ldc)
{
    constexpr data_type data_val_zero = static_cast<data_type>(0);

    if (beta == data_val_zero)
    {
        if (transa == ICLBLAS_OP_N)
        {
            if (transb == ICLBLAS_OP_N)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_N, ICLBLAS_OP_N, true>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            else if (transb == ICLBLAS_OP_T)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_N, ICLBLAS_OP_T, true>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            else if (transb == ICLBLAS_OP_C)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_N, ICLBLAS_OP_C, true>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        else if (transa == ICLBLAS_OP_T)
        {
            if (transb == ICLBLAS_OP_N)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_T, ICLBLAS_OP_N, true>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            else if (transb == ICLBLAS_OP_T)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_T, ICLBLAS_OP_T, true>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            else if (transb == ICLBLAS_OP_C)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_T, ICLBLAS_OP_C, true>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        else if (transa == ICLBLAS_OP_C)
        {
            if (transb == ICLBLAS_OP_N)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_C, ICLBLAS_OP_N, true>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            else if (transb == ICLBLAS_OP_T)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_C, ICLBLAS_OP_T, true>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            else if (transb == ICLBLAS_OP_C)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_C, ICLBLAS_OP_C, true>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
    else
    {
        if (transa == ICLBLAS_OP_N)
        {
            if (transb == ICLBLAS_OP_N)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_N, ICLBLAS_OP_N, false>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            else if (transb == ICLBLAS_OP_T)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_N, ICLBLAS_OP_T, false>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            else if (transb == ICLBLAS_OP_C)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_N, ICLBLAS_OP_C, false>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        else if (transa == ICLBLAS_OP_T)
        {
            if (transb == ICLBLAS_OP_N)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_T, ICLBLAS_OP_N, false>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            else if (transb == ICLBLAS_OP_T)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_T, ICLBLAS_OP_T, false>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            else if (transb == ICLBLAS_OP_C)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_T, ICLBLAS_OP_C, false>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        else if (transa == ICLBLAS_OP_C)
        {
            if (transb == ICLBLAS_OP_N)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_C, ICLBLAS_OP_N, false>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            else if (transb == ICLBLAS_OP_T)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_C, ICLBLAS_OP_T, false>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
            else if (transb == ICLBLAS_OP_C)
                detail::cpu_gemm_spec<data_type, acc_type, ICLBLAS_OP_C, ICLBLAS_OP_C, false>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
    }
}


namespace iclgpu { namespace tests {

template <class Func>
struct test_gemm : test_base_GEMMM<Func>
{
    using test_base = test_base_GEMMM<Func>;

    using data_type     = typename test_base::data_type;
    using data_arr_type = typename test_base::data_arr_type;


    data_arr_type A;
    data_arr_type B;
    data_arr_type C;

    data_arr_type C_ref;

    void init_values() override
    {
        const int sda = this->transa == ICLBLAS_OP_N ? this->k : this->m;
        const int sdb = this->transb == ICLBLAS_OP_N ? this->n : this->k;

        A = get_random_vector<data_type>(sda * this->lda);
        B = get_random_vector<data_type>(sdb * this->ldb);
        C = get_random_vector<data_type>(this->n * this->ldc);

        C_ref = C;
    }

    typename Func::params get_params() override
    {
        return
        {
            this->transa,
            this->transb,
            this->m,
            this->n,
            this->k,
            this->alpha,
            this->A.data(),
            this->lda,
            this->B.data(),
            this->ldb,
            this->beta,
            this->C.data(),
            this->ldc
        };
    }
    typename Func::params get_params_ref() override
    {
        return
        {
            this->transa,
            this->transb,
            this->m,
            this->n,
            this->k,
            this->alpha,
            this->A.data(),
            this->lda,
            this->B.data(),
            this->ldb,
            this->beta,
            this->C_ref.data(),
            this->ldc
        };
    }
};

}}
