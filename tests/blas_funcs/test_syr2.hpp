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
void cpu_syr2(
    const iclblasFillMode_t uplo,
    const int n,
    const data_type alpha,
    const data_type* x,
    const int incx,
    const data_type* y,
    const int incy,
    data_type* A,
    const int lda)
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

    for (int col = 0; col < n; ++col)
    {
        auto col_x = static_cast<acc_type>(x_ptr_orig[col * incx]);
        auto col_y = static_cast<acc_type>(y_ptr_orig[col * incy]);
        if (is_hermitian)
        {
            col_x = blas_conj(col_x);
            col_y = blas_conj(col_y);
        }

        int start_row = uplo == ICLBLAS_FILL_MODE_UPPER ? 0 : col;
        int end_row = uplo == ICLBLAS_FILL_MODE_UPPER ? col + 1 : n;
        for (int row = start_row; row < end_row; ++row)
        {
            auto row_x = static_cast<acc_type>(x_ptr_orig[row * incx]);
            auto row_y = static_cast<acc_type>(y_ptr_orig[row * incy]);
            auto this_A = static_cast<acc_type>(A[IDX(row, col, lda)]);

            if (is_hermitian)
            {
                this_A += alpha_acc * row_x * col_y + blas_conj(alpha_acc) * row_y * col_x;
            }
            else
            {
                this_A += alpha_acc * (row_x * col_y + row_y * col_x);
            }

            if (is_hermitian && row == col)
            {
                this_A = zeroe_imag(this_A);
            }

            A[IDX(row, col, lda)] = static_cast<data_type>(this_A);
        }
    }
}

template<class Func>
struct test_syr2 : test_base_SMVV<Func>
{
    using data_type = typename test_base_SMVV<Func>::data_type;
    using data_arr_type = typename test_base_SMVV<Func>::data_arr_type;

    data_type alpha;
    data_arr_type A_ref;

    void init_values() override
    {
        alpha = get_random_scalar<data_type>();

        A_ref = this->A;
    }

    typename Func::params get_params() override
    {
        return
        {
            this->uplo,
            this->n,
            this->alpha,
            this->x.data(),
            this->incx,
            this->y.data(),
            this->incy,
            this->A.data(),
            this->lda
        };
    }

    typename Func::params get_params_ref() override
    {
        return
        {
            this->uplo,
            this->n,
            this->alpha,
            this->x.data(),
            this->incx,
            this->y.data(),
            this->incy,
            this->A_ref.data(),
            this->lda
        };
    }

};

} } // namespace iclgpu::tests
