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

template<typename data_type, bool is_hermitian = false, typename acc_type = accumulator_type_t<data_type>>
void cpu_symv(
    const iclblasFillMode_t uplo,
    const int n,
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
        x_ptr_orig -= (n - 1) * incx;
    }
    auto y_ptr_orig = y;
    if (incy < 0)
    {
        y_ptr_orig -= (n - 1) * incy;
    }

    auto alpha_acc = static_cast<acc_type>(alpha);
    auto beta_acc = static_cast<acc_type>(beta);

    auto y_ptr = y_ptr_orig;
    for (int row = 0; row < n; ++row)
    {
        auto x_ptr = x_ptr_orig;
        auto prod = acc_type(0);

        for (int col = 0; col < n; ++col)
        {
            auto this_x = static_cast<acc_type>(*x_ptr);

            acc_type this_A;
            if ((uplo == ICLBLAS_FILL_MODE_UPPER && col < row) || (uplo == ICLBLAS_FILL_MODE_LOWER && col > row))
            {
                this_A = static_cast<acc_type>(A[IDX(col, row, lda)]);
                if (is_hermitian)
                {
                    this_A = blas_conj(this_A);
                }
            }
            else
            {
                this_A = static_cast<acc_type>(A[IDX(row, col, lda)]);
            }

            // Imaginary part of diagonal elements need not be set and is assumed 0
            if (is_hermitian && row == col)
            {
                this_A = zeroe_imag(this_A);
            }

            prod += this_A * this_x;

            x_ptr += incx;
        }
        auto this_y = static_cast<acc_type>(*y_ptr);
        this_y = beta_acc * this_y + alpha_acc * prod;
        *y_ptr = static_cast<data_type>(this_y);

        y_ptr += incy;
    }
}

template<class Func>
struct test_symv : test_base_SMVV<Func>
{
    using data_type = typename test_base_SMVV<Func>::data_type;
    using data_arr_type = typename test_base_SMVV<Func>::data_arr_type;

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
            this->uplo,
            this->n,
            this->alpha,
            this->A.data(),
            this->lda,
            this->x.data(),
            this->incx,
            this->beta,
            this->y.data(),
            this->incy
        };
    }

    typename Func::params get_params_ref() override
    {
        return
        {
            this->uplo,
            this->n,
            this->alpha,
            this->A.data(),
            this->lda,
            this->x.data(),
            this->incx,
            this->beta,
            this->y_ref.data(),
            this->incy
        };
    }
};

} } // namespace iclgpu::tests
