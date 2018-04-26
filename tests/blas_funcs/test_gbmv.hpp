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

namespace iclgpu { namespace tests {

template<typename data_type, typename acc_type = accumulator_type_t<data_type>>
void cpu_gbmv(
    const iclblasOperation_t trans,
    const int m,
    const int n,
    const int kl,
    const int ku,
    const data_type alpha,
    const data_type* A,
    const int lda,
    const data_type* x,
    const int incx,
    const data_type beta,
    data_type* y,
    const int incy)
{
    auto x_ptr_orig = x;
    if (incx < 0)
    {
        x_ptr_orig -= trans == ICLBLAS_OP_N ? (n - 1) * incx : (m - 1) * incx;
    }
    auto y_ptr = y;
    if (incy < 0)
    {
        y_ptr -= trans == ICLBLAS_OP_N ? (m - 1) * incy : (n - 1) * incy;
    }

    int max_row = trans == ICLBLAS_OP_N ? m : n;
    int max_col = trans == ICLBLAS_OP_N ? n : m;

    for (int row = 0; row < max_row; ++row)
    {
        const int start_col = trans == ICLBLAS_OP_N ? std::max(row - kl, 0) : std::max(row - ku, 0);
        const int end_col = trans == ICLBLAS_OP_N ? std::min(row + ku + 1, max_col) : std::min(row + kl + 1, max_col);

        auto prod = acc_type(0);

        auto x_ptr = x_ptr_orig + start_col * incx;

        for (int col = start_col; col < end_col; ++col)
        {
            auto this_x = static_cast<acc_type>(*x_ptr);
            int index_A = trans == ICLBLAS_OP_N ? IDX_B(row, col, lda, ku) : IDX_B(col, row, lda, ku);
            auto this_A = static_cast<acc_type>(A[index_A]);
            if (trans == ICLBLAS_OP_C)
            {
                this_A = blas_conj(this_A);
            }

            prod += this_x * this_A;

            x_ptr += incx;
        }
        auto this_y = static_cast<acc_type>(*y_ptr);

        this_y *= static_cast<acc_type>(beta);
        prod *= static_cast<acc_type>(alpha);
        this_y += prod;

        *y_ptr = static_cast<data_type>(this_y);

        y_ptr += incy;
    }
}

template<class Func>
struct test_gbmv : test_base_GBMVV<Func>
{
    using data_type = typename test_base_GBMVV<Func>::data_type;
    using data_arr_type = typename test_base_GBMVV<Func>::data_arr_type;

    data_type alpha;
    data_type beta;

    data_arr_type y_ref;

    void init_values() override
    {
        alpha = get_random_scalar<data_type>();
        beta = get_random_scalar<data_type>();

        y_ref = this->y;
    }

    typename Func::params get_params() override
    {
        return
        {
            this->trans,
            this->m,
            this->n,
            this->kl,
            this->ku,
            alpha,
            this->A.data(),
            this->lda,
            this->x.data(),
            this->incx,
            beta,
            this->y.data(),
            this->incy
        };
    }

    typename Func::params get_params_ref() override
    {
        return
        {
            this->trans,
            this->m,
            this->n,
            this->kl,
            this->ku,
            alpha,
            this->A.data(),
            this->lda,
            this->x.data(),
            this->incx,
            beta,
            this->y_ref.data(),
            this->incy
        };
    }
};

}}
